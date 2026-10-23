import json
import shutil

from costlayer import phase2 as p2
from costlayer.agent import CONFIGS
from costlayer.client import AnthropicClient
from costlayer.ledger import Ledger
from costlayer.migrate import migrate
from costlayer.optimizer import analysis
from costlayer.optimizer.classify_eval import loo_accuracy
from costlayer.optimizer.data import load_runs
from costlayer.pricing import HAIKU, SONNET
from costlayer.report2 import build
from costlayer.runner import Runner, run_name, split_tasks, tool_kinds
from costlayer.usersim import install
from tests.fakes import FakeSDK, fake_user_backend, text, thinking


def write_prior(root, task_id, config, cost=0.02, status="ok", trace_cost=None, reward=1.0):
    """A finished earlier run: summary file plus a trace whose per-call costs add up."""
    runs, traces = root / "runs", root / "traces"
    runs.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    s = {
        "model": SONNET,
        "config": config,
        "task": task_id,
        "attempt": 1,
        "status": status,
        "termination": "user_stop",
        "reward": reward,
        "llm_calls": 2,
        "user_calls": 2,
        "cost_agent": cost,
        "cost_user": 0.001,
        "tokens": {"input": 10, "cache_write": 100, "cache_read": 200, "output": 50},
    }
    name = run_name(SONNET, config, task_id, 1)
    (runs / name).write_text(json.dumps(s))
    tr = {
        "task_id": task_id,
        "termination_reason": "user_stop",
        "reward": reward,
        "messages": [
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "ok",
                "cost": trace_cost if trace_cost is not None else cost,
                "usage": {"output": 50, "thinking_tokens": 3},
            },
        ],
    }
    (traces / name).write_text(json.dumps(tr))
    return runs, traces


def test_reuse_copies_matching_runs_and_runs_only_what_is_missing(tmp_path):
    tasks = split_tasks("train")[:2]
    prior_root = tmp_path / "prior"
    for c in ("default", "E:low"):
        write_prior(prior_root, tasks[0].id, c)
    write_prior(prior_root, tasks[1].id, "default")
    prior = p2.Prior(prior_root / "runs", [prior_root / "traces"])
    out, traces = tmp_path / "out", tmp_path / "tr"
    plan, reused = p2.plan_sweep(tasks, [prior], out, traces)
    assert sorted(reused) == sorted(
        [f"default/{tasks[0].id}", f"E:low/{tasks[0].id}", f"default/{tasks[1].id}"]
    )
    todo = {t.id: cfgs for t, cfgs in plan}
    assert todo[tasks[0].id] == ["E:medium", "T:bt"]
    assert todo[tasks[1].id] == ["E:medium", "E:low", "T:bt"]
    copied = json.loads((out / "runs" / run_name(SONNET, "default", tasks[0].id, 1)).read_text())
    assert copied["reused_from"] and (traces / run_name(SONNET, "default", tasks[0].id, 1)).exists()


def test_reuse_rejects_a_trace_that_does_not_match_the_summary(tmp_path):
    t = split_tasks("train")[0]
    root = tmp_path / "prior"
    write_prior(root, t.id, "default", cost=0.02, trace_cost=0.0066)  # the phase 1 mix-up shape
    write_prior(root, t.id, "E:low", status="infra_error")
    prior = p2.Prior(root / "runs", [root / "traces"])
    assert p2.reusable(prior, SONNET, "default", t.id) is None
    assert p2.reusable(prior, SONNET, "E:low", t.id) is None
    assert p2.reusable(prior, HAIKU, "default", t.id) is None


def test_old_phase1_trace_names_are_found(tmp_path):
    t = split_tasks("train")[0]
    root = tmp_path / "prior"
    write_prior(root, t.id, "default")
    new = root / "traces" / run_name(SONNET, "default", t.id, 1)
    new.rename(root / "traces" / f"default__{t.id}__a1.json")  # phase 1 naming
    prior = p2.Prior(root / "runs", [root / "traces"])
    assert p2.reusable(prior, SONNET, "default", t.id) is not None


def test_sweep_end_to_end_on_a_fake_client(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl", "d2", 2, 9.0)
    sdk = FakeSDK([[thinking(), text("hello")]])
    install(led, inner=fake_user_backend(["Hi", "###STOP###"]))
    out, tr = tmp_path / "out", tmp_path / "tr"
    runner = Runner(led, AnthropicClient(led, sdk), out, tr)
    (out / "tool_kinds.json").write_text(json.dumps(tool_kinds()))
    tasks = split_tasks("train")[:3]
    prior_root = tmp_path / "prior"
    for c in CONFIGS:
        write_prior(prior_root, tasks[0].id, c, cost=0.02 + 0.01 * list(CONFIGS).index(c))
    prior = p2.Prior(prior_root / "runs", [prior_root / "traces"])
    res = p2.run_phase2(runner, out, tr, [prior], n_train=3, rail=1.0)
    assert res["to_run"] == 8 and len(res["reused"]) == 4 and not res["stopped"]
    runs = load_runs(out / "runs", [tr])
    assert len(runs) == 12
    assert {r.config for r in runs if r.task == tasks[0].id} == set(CONFIGS)
    # the model is in every ledger row and in every file name
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert {r["model"] for r in rows} == {SONNET, HAIKU}
    res_a = analysis.analyze(runs, json.loads((out / "tool_kinds.json").read_text()))
    assert res_a["n_tasks"] == 3 and "headroom" in res_a and "chains" in res_a
    md = build(
        res_a, res, None, {"total": 1.0, "calls": 2, "by_purpose": {"agent": 1.0}}, 0.67, "note", []
    )
    for needle in (
        "Cost anatomy",
        "Best single setting",
        "Gate",
        "FEATURE ONLY",
        "What this cannot show",
        "cache-write doubling",
        "Honest expectation",
    ):
        assert needle in md


def test_classifier_loo_uses_haiku_and_counts_correct_labels(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "d", 2, 9.0)
    sdk = FakeSDK([[text("cancel_pending_order")], [text("none")], [text("none")]])
    ex = [
        ("a", "cancel my order", "cancel_pending_order"),
        ("b", "where is it", "none"),
        ("c", "return this", "return_delivered_order_items"),
    ]
    res = loo_accuracy(AnthropicClient(led, sdk), ex, "d")
    assert res["n"] == 3 and abs(res["accuracy"] - 2 / 3) < 1e-9
    req = sdk.requests[0]
    assert req["model"] == HAIKU and req["thinking"] == {"type": "disabled"}
    assert "tools" not in req and req["max_tokens"] == 32
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert {r["purpose"] for r in rows} == {"classifier"}


def test_migration_rebuilds_phase1_files_from_the_ledger(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    old = {
        "model": SONNET,
        "config": "default",
        "task": "7",
        "attempt": 1,
        "status": "ok",
        "reward": 1.0,
        "llm_calls": 2,
        "cost_agent": 0.1,
        "cost_user": 0.0,
        "tokens": {"input": 99, "cache_write": 31000, "cache_read": 1, "output": 9999},
    }
    (runs / "default__7__a1.json").write_text(json.dumps(old))

    def row(m, w):
        return {
            "purpose": "agent",
            "model": m,
            "config": "default",
            "task": "7",
            "attempt": 1,
            "status": "ok",
            "input": 1,
            "cache_write": w,
            "cache_read": 5,
            "output": 7,
            "cost": 0.01,
            "raw_usage": {},
        }

    led = tmp_path / "led.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in (row(SONNET, 15000), row(HAIKU, 16000))))
    changed = migrate(runs, led)
    assert len(changed) == 2
    s = json.loads((runs / run_name(SONNET, "default", "7", 1)).read_text())
    assert s["tokens"]["cache_write"] == 15000  # not 31000
    h = json.loads((runs / run_name(HAIKU, "default", "7", 1)).read_text())
    assert h["status"] == "reconstructed_from_ledger" and h["tokens"]["cache_write"] == 16000
    assert migrate(runs, led) == []  # idempotent
    shutil.rmtree(runs)


def test_measured_costs_and_rail(tmp_path):
    root = tmp_path / "p"
    write_prior(root, "1", "default", cost=0.08)
    write_prior(root, "2", "default", cost=0.10)
    c = p2.measured_costs(root / "runs")
    assert abs(c["default"] - 0.091) < 1e-9
    assert p2.rail_for(0.09) == 1.0 and p2.rail_for(0.5) == 2.5
