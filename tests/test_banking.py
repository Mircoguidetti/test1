import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry

from costlayer import banking as B
from costlayer.client import AnthropicClient
from costlayer.ledger import Ledger
from costlayer.pricing import SONNET
from costlayer.runner import (
    BANKING_DOMAIN,
    Runner,
    check_domain,
    make_config,
    tasks_by_id,
    tool_kinds,
)
from costlayer.sanitize import find_banned, sanitize_simulation
from costlayer.usersim import install
from tests.fakes import FakeSDK, fake_user_backend, text, thinking

LOCAL = {
    "no_knowledge",
    "full_kb",
    "bm25",
    "bm25_grep",
    "grep_only",
    "terminal_use",
    "terminal_use_write",
}


def banking_tasks():
    return registry.get_tasks_loader(BANKING_DOMAIN)()


# -- 2. retrieval variants ----------------------------------------------------------------------


def test_variants_that_need_no_key_and_no_gold_are_exactly_the_local_ones():
    rows = {r["name"]: r for r in B.all_variants()}
    assert {n for n, r in rows.items() if r["local"]} == LOCAL
    for n, r in rows.items():
        if "embeddings" in n or "alltools" in n:
            assert r["external_keys"], n  # embeddings: OpenAI or OpenRouter
        if "reranker" in n:
            assert "OPENAI_API_KEY" in r["external_keys"], n  # the LLM reranker is OpenAI
    assert rows["golden_retrieval"]["uses_task_gold"] and not rows["golden_retrieval"]["local"]
    assert rows["terminal_use"]["binaries"] == ["srt", "rg", "bwrap", "socat"]
    assert rows["bm25"]["retrieval_tools"] == ["KB_search"]
    assert rows["bm25_grep"]["retrieval_tools"] == ["KB_search", "grep"]
    assert rows["grep_only"]["retrieval_tools"] == ["grep"]
    assert rows["full_kb"]["prompt_tokens"] > 100_000  # the whole knowledge base is in the prompt
    assert rows["bm25"]["prompt_tokens"] < 3_000


@pytest.mark.parametrize(
    "name", ["alltools", "openai_embeddings", "bm25_reranker", "golden_retrieval", "nonsense"]
)
def test_runner_refuses_variants_with_a_key_or_the_oracle(name):
    with pytest.raises(ValueError):
        check_domain(BANKING_DOMAIN, name)


def test_banking_needs_an_explicit_variant_and_retail_takes_none():
    with pytest.raises(ValueError, match="explicit"):
        check_domain(BANKING_DOMAIN, None)  # tau2 would silently default to `alltools`
    with pytest.raises(ValueError):
        check_domain("retail", "bm25")
    check_domain("retail", None)
    check_domain(BANKING_DOMAIN, "bm25")


def test_config_for_banking_sets_the_local_variant_and_retail_is_unchanged():
    c = make_config(SONNET, "default", BANKING_DOMAIN, "grep_only")
    assert c.domain == BANKING_DOMAIN and c.retrieval_config == "grep_only"
    assert c.max_retries == 0 and c.max_steps == 100
    r = make_config(SONNET, "default")
    assert r.domain == "retail" and r.retrieval_config is None


def test_tool_kinds_for_banking():
    k = tool_kinds(BANKING_DOMAIN, "grep_only")
    assert k["grep"] == "READ" and k["log_verification"] == "WRITE"
    assert k["unlock_discoverable_agent_tool"] == "GENERIC"


# -- 3. deterministic split ---------------------------------------------------------------------


def test_banking_order_is_a_seeded_permutation_of_all_97_tasks():
    order = B.banking_order()
    assert len(order) == 97 and sorted(order) == B.task_ids() == sorted(
        t.id for t in banking_tasks()
    )
    assert order == B.banking_order(42) != B.banking_order(43)
    assert order != sorted(order)


def test_banking_order_does_not_depend_on_the_hash_seed():
    code = "from costlayer.banking import banking_order; print(','.join(banking_order()[:12]))"
    outs = set()
    for h in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": h}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1
    assert outs == {",".join(B.banking_order()[:12])}


def test_banking_split_is_disjoint_complete_and_prefix_stable():
    train, test = B.banking_split(30)
    assert len(train) == 30 and len(test) == 67
    assert not set(train) & set(test) and sorted(train + test) == B.task_ids()
    t20, _ = B.banking_split(20)
    t40, _ = B.banking_split(40)
    assert t20 == train[:20] and t40[:30] == train  # a bigger train set only adds tasks
    assert B.banking_split(30) == (train, test)  # deterministic


def test_banking_split_has_no_default_size_and_validates_it():
    with pytest.raises(TypeError):
        B.banking_split()  # the sizes are not chosen here
    for bad in (0, 97, -1, 500):
        with pytest.raises(ValueError):
            B.banking_split(bad)


def test_exclusions_are_removed_before_the_split():
    ex = tuple(B.reward_basis_groups()["ACTION"])
    train, test = B.banking_split(20, exclude=ex)
    assert not set(ex) & set(train + test) and len(train + test) == 88


def test_tasks_by_id_keeps_the_given_order():
    ids = B.banking_order()[:5]
    assert [t.id for t in tasks_by_id(BANKING_DOMAIN, ids)] == ids


# -- 4. sanitizer covers the banking fields -----------------------------------------------------


def test_banking_gold_fields_are_all_recognised_by_the_filter():
    tasks = banking_tasks()
    assert all(t.required_documents for t in tasks)
    for t in tasks:  # the parsed Task objects
        hits = find_banned(t.model_dump())
        assert "/evaluation_criteria" in hits, t.id
        assert "/required_documents" in hits, t.id
    # `annotations` is in the raw task files (null in this data) but tau2's Task model drops it
    raw = sorted(
        Path(os.environ["TAU2_DATA_DIR"]).glob("tau2/domains/banking_knowledge/tasks/task_*.json")
    )
    assert len(raw) == 97
    for f in raw:
        hits = find_banned(json.loads(f.read_text()))
        assert {"/evaluation_criteria", "/required_documents", "/annotations"} <= set(hits), f


def test_sanitized_banking_trace_has_no_gold_even_if_the_simulation_embeds_the_task():
    task = next(t for t in banking_tasks() if t.id == "task_035")
    dump = task.model_dump(mode="json")
    sim = SimulationRun(
        id="s",
        task_id=task.id,
        start_time="t",
        end_time="t",
        duration=1.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        info={"task": dump},
        reward_info=RewardInfo(
            reward=1.0, info={"task": dump}, action_checks=[], nl_assertions=None
        ),
    )
    clean = sanitize_simulation(sim)
    assert set(clean) == {"task_id", "termination_reason", "reward", "messages"}
    blob = json.dumps(clean)
    assert not find_banned(clean)
    for doc_id in task.required_documents:
        assert doc_id not in blob
    for a in task.evaluation_criteria.actions:
        assert a.action_id not in blob
    assert "evaluation_criteria" not in blob and "required_documents" not in blob


# -- 5. reward bases under EvaluationType.ENV ---------------------------------------------------


def test_reward_basis_groups():
    g = B.reward_basis_groups()
    assert {k: len(v) for k, v in g.items()} == {"DB": 87, "ACTION": 9, "DB+NL_ASSERTION": 1}
    assert g["ACTION"] == [
        "task_004",
        "task_008",
        "task_012",
        "task_014",
        "task_032",
        "task_033",
        "task_034",
        "task_035",
        "task_083",
    ]
    assert g["DB+NL_ASSERTION"] == ["task_102"]


def env_reward(task):
    kw = {"retrieval_variant": "grep_only", "task": task, "read_log_allowlist": set()}
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=registry.get_env_constructor(BANKING_DOMAIN),
        task=task,
        full_trajectory=[],
        env_kwargs=kw,
    ).reward


def test_env_evaluation_gives_a_free_pass_to_action_tasks_and_only_to_them():
    by_id = {t.id: t for t in banking_tasks()}
    for tid in B.reward_basis_groups()["ACTION"]:
        assert env_reward(by_id[tid]) == 1.0, tid  # an agent that did nothing scores 1
    for tid in ("task_001", "task_002", "task_003", "task_102"):
        assert env_reward(by_id[tid]) == 0.0, tid


# -- 1. the runner end to end on banking, offline -----------------------------------------------


def test_runner_runs_a_banking_task_offline(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl", "d", 0, 100.0)
    sdk = FakeSDK([[thinking(), text("Hello, how can I help?")]])
    install(led, inner=fake_user_backend(["Hi, I need help", "###STOP###"]))
    r = Runner(
        led,
        AnthropicClient(led, sdk),
        tmp_path / "out",
        tmp_path / "tr",
        domain=BANKING_DOMAIN,
        retrieval_variant="grep_only",
    )
    task = tasks_by_id(BANKING_DOMAIN, ["task_001"])[0]
    s = r.run_once(SONNET, "default", task)
    assert s["status"] == "ok" and s["termination"] == "user_stop" and s["reward"] == 0.0
    assert s["domain"] == BANKING_DOMAIN and s["retrieval_variant"] == "grep_only"
    names = {t["name"] for t in sdk.requests[0]["tools"]}
    assert "grep" in names and "KB_search" not in names and "shell" not in names
    trace = json.loads(next((tmp_path / "tr").glob("*.json")).read_text())
    assert not find_banned(trace)
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert {x["purpose"] for x in rows} == {"agent", "user_sim"}


def test_runner_constructor_rejects_unsafe_banking_setups(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "d", 0, 1.0)
    cli = AnthropicClient(led, FakeSDK([[text("x")]]))
    for variant in (None, "alltools", "golden_retrieval", "bm25_reranker"):
        with pytest.raises(ValueError):
            Runner(led, cli, tmp_path / "o", None, domain=BANKING_DOMAIN, retrieval_variant=variant)
