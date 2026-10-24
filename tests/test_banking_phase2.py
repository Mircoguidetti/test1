# ruff: noqa: E501
"""Test 2, phase 2 (b2): planner and reuse, the cap rule, the gate, the trace fix, mining, the
classifier, the budget guards, the report and the workflow. No network, no key."""

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from costlayer import banking_phase1 as P1
from costlayer import banking_phase2 as B2
from costlayer import phase2 as p2
from costlayer.agent import BANKING_CONFIGS
from costlayer.budget import Budget
from costlayer.cli import main as cli_main
from costlayer.client import AnthropicClient
from costlayer.cut import OMITTED_LINE, cut_top_items
from costlayer.ledger import Ledger
from costlayer.optimizer import analysis, mining
from costlayer.optimizer.classify_eval import BANKING_SYSTEM, loo_accuracy
from costlayer.optimizer.data import (
    BANKING_CONFIGS as OPT_CONFIGS,
)
from costlayer.optimizer.data import (
    Run,
    load_runs,
    prepare_trace,
)
from costlayer.optimizer.gate import FEATURE_ONLY, GO, KILL, project, verdict_lines
from costlayer.pricing import SONNET
from costlayer.report_banking2 import build, projections
from costlayer.runner import BANKING_DOMAIN, Runner, run_name, tasks_by_id
from costlayer.usersim import install
from tests.fakes import FakeSDK, fake_user_backend, text, thinking, tool_use

ROOT = Path(__file__).resolve().parents[1]
PHASE1_RUNS = ROOT / "results" / "banking-phase-1" / "runs"
CONFIGS = tuple(BANKING_CONFIGS)
KINDS = {
    "KB_search": "READ",
    "grep": "READ",
    "get_user_information_by_id": "READ",
    "get_user_information_by_name": "READ",
    "get_credit_card_accounts_by_user": "READ",
    "log_verification": "WRITE",
    "call_discoverable_agent_tool": "WRITE",
    "unlock_discoverable_agent_tool": "GENERIC",
}


# -- helpers -------------------------------------------------------------------------------------


def item(n: int, doc: str, content: str = "some policy text") -> str:
    return f"{n}. Title {n}\n   ID: {doc}\n   Score: 1.2345\n   Content: {content}\n"


def result(n_items: int, extra: dict[int, str] | None = None) -> str:
    """A KB_search-shaped result with `n_items` items and a timing line."""
    items = [
        item(i, f"doc_{i}", (extra or {}).get(i, "some policy text")) for i in range(1, n_items + 1)
    ]
    return "\n".join(items) + "\n[Timing: retrieval=1ms, reranking=0ms, total=1ms]"


def call(i: int, name: str, **args):
    return {"id": f"c{i}", "name": name, "arguments": args, "requestor": "assistant"}


def asst(calls=None, text_=None, cost=0.01, calls_n=0):
    m = {"role": "assistant", "content": text_, "cost": cost, "usage": {"output": 10}}
    if calls:
        m["tool_calls"] = calls
    return m


def tool(i: int, content: str):
    return {
        "role": "tool",
        "id": f"c{i}",
        "content": content,
        "error": False,
        "requestor": "assistant",
    }


def trace(*msgs, seen=False):
    t = {"messages": [{"role": "assistant", "content": "Hi!"},
                      {"role": "user", "content": "I am Sarah Bosch, my id is user_001"}, *msgs]}  # fmt: skip
    if seen:
        t["seen_in_context"] = True
    return t


class SearchThenAnswerSDK(FakeSDK):
    """Every conversation: the first agent call searches the knowledge base, the next one
    answers (so every run of a sweep has exactly one retrieval result)."""

    def _create(self, **kwargs):
        first_turn = isinstance(kwargs["messages"][-1]["content"], str)
        reply = [thinking(), tool_use("KB_search", query="fees")] if first_turn else [
            thinking(), text("Done.")]  # fmt: skip
        with self.lock:
            self.replies = [reply]
            self.requests.append(copy.deepcopy(kwargs))
            return self._respond(reply)


def banking_runner(tmp_path, replies, user_lines=("Hi", "###STOP###"), cap=50.0, sdk=None):
    led = Ledger(tmp_path / "ledger.jsonl", "d1", "b2", cap, total_cap=cap)
    sdk = sdk or FakeSDK(replies)
    install(led, inner=fake_user_backend(list(user_lines)))
    r = Runner(led, AnthropicClient(led, sdk), tmp_path / "out", tmp_path / "tr",
               domain=BANKING_DOMAIN, retrieval_variant=P1.VARIANT)  # fmt: skip
    return r, led, sdk


# -- 1. planner, reuse rules ---------------------------------------------------------------------


def phase1_priors(tmp_path: Path) -> p2.Prior:
    """The REAL phase 1 run summaries with a trace per run whose per-call costs add up."""
    traces = tmp_path / "b1traces"
    traces.mkdir()
    for f in PHASE1_RUNS.glob("sonnet__*.json"):
        s = json.loads(f.read_text())
        tr = trace(asst([call(1, "KB_search", query="q")], cost=s["cost_agent"]), tool(1, "x"))
        (traces / f.name).write_text(json.dumps(tr))
    return p2.Prior(PHASE1_RUNS, [traces])


def test_the_twelve_train_ids_are_the_first_of_the_order_without_action_tasks():
    ids = B2.train_ids()
    assert len(ids) == B2.N_TRAIN == 12
    assert ids == P1.phase1_order()[:12] and ids[:3] == ["task_073", "task_077", "task_047"]
    assert not set(ids) & set(P1.excluded_ids())


def test_reuse_takes_exactly_the_phase_1_runs_that_match(tmp_path):
    plan = B2.plan_b2([phase1_priors(tmp_path)], tmp_path / "out", tmp_path / "tr")
    assert sorted(plan.reused) == sorted(
        [f"{c}/{t}" for c in ("default", "EC") for t in ("task_073", "task_077", "task_047")]
        + ["E:low/task_073", "C:top3/task_073"]
    )
    todo = {t.id: cfgs for t, cfgs in plan.todo}
    assert "task_073" not in todo  # all four configurations reused
    assert todo["task_077"] == ["E:low", "C:top3"] and todo["task_047"] == ["E:low", "C:top3"]
    assert all(todo[i] == list(CONFIGS) for i in B2.train_ids()[3:])
    assert (plan.n_total, plan.n_todo, len(plan.reused)) == (48, 40, 8)
    assert list(todo)[:2] == ["task_077", "task_047"]  # order kept
    # the reused summaries and traces were copied under the phase 2 folders
    name = run_name(SONNET, "EC", "task_073", 1)
    assert json.loads((tmp_path / "out" / "runs" / name).read_text())["reused_from"]
    assert (tmp_path / "tr" / name).exists()


def test_reuse_rejects_unfinished_mismatching_wrong_model_and_missing_trace(tmp_path):
    runs, traces = tmp_path / "runs", tmp_path / "traces"
    runs.mkdir()
    traces.mkdir()

    def put(model, config, task, status="ok", cost=0.2, trace_cost=0.2, with_trace=True):
        s = {"model": model, "config": config, "task": task, "attempt": 1, "status": status,
             "cost_agent": cost, "cost_user": 0.0, "llm_calls": 1, "reward": 1.0}  # fmt: skip
        name = run_name(model, config, task, 1)
        (runs / name).write_text(json.dumps(s))
        if with_trace:
            tr = trace(asst([call(1, "KB_search", query="q")], cost=trace_cost), tool(1, "x"))
            (traces / name).write_text(json.dumps(tr))

    put(SONNET, "default", "task_073")  # good
    put(SONNET, "E:low", "task_073", trace_cost=0.05)  # costs do not add up
    put(SONNET, "C:top3", "task_073", status="infra_error")
    put(SONNET, "EC", "task_073", with_trace=False)
    put("claude-haiku-5-5", "default", "task_077")  # wrong model
    put(SONNET, "default", "task_999")  # not a train task of the sweep
    prior = p2.Prior(runs, [traces])
    got = {c: p2.reusable(prior, SONNET, c, "task_073") is not None for c in CONFIGS}
    assert got == {"default": True, "E:low": False, "C:top3": False, "EC": False}
    assert p2.reusable(prior, SONNET, "default", "task_077") is None
    plan = B2.plan_b2([prior], tmp_path / "o", tmp_path / "t")
    assert plan.reused == ["default/task_073"]
    # a censored or agent-error run is a finished run and may be reused
    put(SONNET, "E:low", "task_077", status="agent_error")
    put(SONNET, "C:top3", "task_077", status="censored")
    assert p2.reusable(prior, SONNET, "E:low", "task_077") is not None
    assert p2.reusable(prior, SONNET, "C:top3", "task_077") is not None


def test_cost_basis_and_rail_come_from_the_phase_1_summaries():
    b = B2.cost_basis(PHASE1_RUNS)
    assert set(b.mean) == set(CONFIGS)
    assert b.mean["default"] == pytest.approx(0.4504, abs=1e-3)
    assert B2.rail_for(b) == pytest.approx(max(1.0, 5 * b.mean["default"]))  # $2.25
    assert all(b.cover(c) >= 1.5 * b.mean[c] - 1e-12 and b.cover(c) >= b.high[c] for c in CONFIGS)


# -- the cap rule and the pre-flight -------------------------------------------------------------


def test_a_task_is_not_started_if_the_cap_does_not_cover_its_runs():
    started, release = [], threading.Event()
    lock = threading.Lock()

    class StubRunner:
        def run_task(self, model, cfgs, task):
            with lock:
                started.append(task.id)
            release.wait(5)

    ledger = SimpleNamespace(phase_cap=7.0, spent=0.0)
    basis = B2.CostBasis(mean=dict.fromkeys(CONFIGS, 0.5), high=dict.fromkeys(CONFIGS, 0.5))
    assert basis.cover("default") == 0.75  # 1.5 x mean beats the highest observed
    tasks = [(SimpleNamespace(id=f"t{i}"), list(CONFIGS)) for i in range(4)]
    plan = B2.SweepPlan(ids=[t.id for t, _ in tasks], todo=tasks, reused=[])

    out = {}
    th = threading.Thread(
        target=lambda: out.update(B2.run_sweep(StubRunner(), ledger, plan, basis, concurrency=3))
    )
    th.start()
    for _ in range(100):  # wait until two tasks are running and the third was refused
        with lock:
            if len(started) == 2:
                break
        threading.Event().wait(0.05)
    release.set()
    th.join(10)
    # each task needs 4 x 0.75 = 3.0: two fit in 7.0 (6.0 outstanding), the third does not (1.0 left)
    assert sorted(started) == ["t0", "t1"]
    assert out["started"] == ["t0", "t1"] and sorted(out["not_started"]) == ["t2", "t3"]
    assert "cap" in out["stopped"]


def test_the_planner_counts_actual_spend_too():
    ran = []

    class StubRunner:
        def run_task(self, model, cfgs, task):
            ran.append(task.id)
            ledger.spent += 2.0  # each task really costs 2.0

    # the classifier check keeps $0.05 of the cap: 7.05 - 0.05 = 7.0 for the tasks
    ledger = SimpleNamespace(phase_cap=7.05, spent=0.0)
    basis = B2.CostBasis(mean=dict.fromkeys(CONFIGS, 0.5), high=dict.fromkeys(CONFIGS, 0.5))
    tasks = [(SimpleNamespace(id=f"t{i}"), list(CONFIGS)) for i in range(4)]
    plan = B2.SweepPlan(ids=[t.id for t, _ in tasks], todo=tasks, reused=[])
    out = B2.run_sweep(StubRunner(), ledger, plan, basis, concurrency=1)
    # t0: 7.0 left >= 3.0; t1: 5.0 >= 3.0; t2: 3.0 >= 3.0; t3: 1.0 < 3.0 -> stops
    assert ran == ["t0", "t1", "t2"] and out["not_started"] == ["t3"]


def test_preflight_refusal_rules(tmp_path):
    plan = B2.plan_b2([phase1_priors(tmp_path)], tmp_path / "o", tmp_path / "t")
    basis = B2.cost_basis(PHASE1_RUNS)
    ok = B2.preflight(plan, basis, 16.1, 0.0, 0.0, 22.0)
    assert ok["refuse"] == [] and 9 < ok["certain"] < 12.9 and not ok["above_eighty_percent"]
    assert ok["startable_at_cover"] >= 1
    # certain part above the cap left
    assert B2.preflight(plan, basis, 8.0, 0.0, 0.0, 22.0)["refuse"]
    assert "certain part" in B2.preflight(plan, basis, 8.0, 0.0, 0.0, 22.0)["refuse"][0]
    # earlier b2 spending reduces the cap left
    assert B2.preflight(plan, basis, 16.1, 8.0, 8.0, 22.0)["refuse"]
    # nothing left, or the total would be passed
    assert any(
        "no cap left" in r for r in B2.preflight(plan, basis, 16.1, 16.1, 16.1, 22.0)["refuse"]
    )
    assert any("total" in r for r in B2.preflight(plan, basis, 16.1, 0.0, 10.0, 22.0)["refuse"])
    # no reuse (the artifact did not download): 48 runs, the certain part grows by about $2
    none = B2.plan_b2(
        [p2.Prior(PHASE1_RUNS, [tmp_path / "missing"])], tmp_path / "o2", tmp_path / "t2"
    )
    assert none.n_todo == 48 and B2.estimate(none, basis)["certain"] > ok["certain"] + 1.5


def test_the_dry_run_prints_runs_reused_and_cost_and_refuses_an_unaffordable_plan(tmp_path, capsys):
    base = ["banking-dry-run", "--phase", "b2", "--founder-go", "b2"]
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"total": 22.0, "phase_caps": {"b1": 4.0, "b2": 16.1}}))
    none = str(tmp_path / "none")
    # the b1 traces are missing (the artifact download failed): the paid step would refuse
    assert cli_main([*base, "--budget", str(big), "--prior-traces", none]) == 1
    assert "no run would be reused" in capsys.readouterr().out
    assert cli_main([*base, "--budget", str(big), "--prior-traces", none, "--allow-no-reuse"]) == 0
    out = capsys.readouterr().out
    for needle in ("runs: 48 in total, 0 reused from phase 1, 48 to execute",
                   "estimated cost, certain part", "per-run abort rail: $2.25", "task_073"):  # fmt: skip
        assert needle in out
    small = tmp_path / "small.json"
    small.write_text(json.dumps({"total": 22.0, "phase_caps": {"b1": 4.0, "b2": 6.0}}))
    assert (
        cli_main([*base, "--budget", str(small), "--prior-traces", none, "--allow-no-reuse"]) == 1
    )
    assert "REFUSED" in capsys.readouterr().out


def test_b2_guards_need_the_go_and_a_cap_in_budget_json(tmp_path):
    only_b1 = tmp_path / "b1.json"
    only_b1.write_text(json.dumps({"total": 4.0, "phase_caps": {"b1": 4.0}}))
    spend = tmp_path / "spend"
    with pytest.raises(SystemExit, match="no cap"):
        cli_main(["banking-phase", "--phase", "b2", "--founder-go", "b2", "--budget", str(only_b1),
                  "--spend-dir", str(spend), "--dispatch-id", "x"])  # fmt: skip
    committed = str(ROOT / "budget.json")
    for go in ("", "b1", "phase2", "B2"):
        with pytest.raises(SystemExit, match="founder's go"):
            cli_main(["banking-phase", "--phase", "b2", "--founder-go", go, "--budget", committed,
                      "--spend-dir", str(spend), "--dispatch-id", "x"])  # fmt: skip
    assert not spend.exists()  # a refused start writes nothing


# -- 4. the gate, three outcomes -----------------------------------------------------------------


def test_gate_kill_when_the_projection_is_above_70_percent_of_default():
    p = project(cost_default=10.0, cost_global=9.0, chain_cost_share=0.05, headroom=0.0,
                headroom_beats_null=False)  # fmt: skip
    assert p.projected == pytest.approx(8.55) and p.verdict == KILL
    assert p.ratio_default == pytest.approx(0.855) and p.ratio_global == pytest.approx(0.95)


def test_gate_feature_only_when_a_global_setting_already_does_it():
    # global* is 0.60 of default and the chains/headroom add little: below 0.70 but above 0.85 of global*
    p = project(10.0, 6.0, 0.05, 0.0, False)
    assert p.verdict == FEATURE_ONLY and p.ratio_default == pytest.approx(0.57)
    assert p.ratio_global == pytest.approx(0.95)


def test_gate_go_when_the_layer_beats_global_star_by_enough():
    p = project(10.0, 6.0, 0.20, 0.0, False)  # chains remove 20% of global*'s cost
    assert (
        p.verdict == GO
        and p.projected == pytest.approx(4.8)
        and p.ratio_global == pytest.approx(0.8)
    )


def test_gate_takes_the_lower_term_and_ignores_headroom_that_does_not_beat_the_null():
    lower = project(10.0, 6.0, 0.10, 0.30, True)
    assert lower.basis == "per-type headroom" and lower.projected == pytest.approx(4.2)
    chains = project(10.0, 6.0, 0.30, 0.10, True)
    assert chains.basis == "chains" and chains.projected == pytest.approx(4.2)
    ignored = project(10.0, 6.0, 0.00, 0.50, False)
    assert ignored.projected == pytest.approx(6.0) and ignored.basis == "chains"
    assert project(10.0, 6.0, 0.00, -0.2, True).projected == pytest.approx(6.0)


def test_gate_boundaries_are_inclusive():
    # exactly 0.70 of default (and 0.70 of global*): not a KILL, and GO
    assert project(10.0, 10.0, 0.30, 0.0, False).verdict == GO
    assert project(10.0, 10.0, 0.29, 0.0, False).verdict == KILL  # 0.71 of default
    # exactly 0.85 of global* (6 x 0.85 = 5.1): GO; just above: FEATURE ONLY
    assert project(10.0, 6.0, 0.15, 0.0, False).verdict == GO
    assert project(10.0, 6.0, 0.14, 0.0, False).verdict == FEATURE_ONLY


def test_verdict_lines_are_literal_and_have_the_three_numbers():
    lines = verdict_lines(project(10.0, 6.0, 0.20, 0.0, False))
    assert lines[0] == "VERDICT: GO TO PHASE 3"
    assert lines[1].startswith("projected cost: $4.8000")
    assert "projected / default: 0.480" in lines[2] and "projected / global*: 0.800" in lines[3]


# -- 3. the trace shows what the agent saw -------------------------------------------------------


@pytest.mark.parametrize(
    "config,cut", [("default", False), ("E:low", False), ("C:top3", True), ("EC", True)]
)
def test_saved_traces_hold_the_text_the_agent_saw(tmp_path, config, cut):
    replies = [[thinking(), tool_use("KB_search", query="credit card fees")],
               [thinking(), text("Done.")]]  # fmt: skip
    r, led, sdk = banking_runner(tmp_path, replies)
    task = tasks_by_id(BANKING_DOMAIN, ["task_073"])[0]
    s = r.run_once(SONNET, config, task)
    assert s["status"] == "ok", s
    sent = sdk.requests[1]["messages"][-1]["content"][0]["content"]
    tr = json.loads((tmp_path / "tr" / run_name(SONNET, config, task.id, 1)).read_text())
    msg = next(m for m in tr["messages"] if m["role"] == "tool")
    assert tr["seen_in_context"] is True
    assert msg["content"] == sent  # exactly what the model got
    assert msg["raw_chars"] == s["retrieval_chars_raw"] > 0
    if cut:
        assert msg["content"].endswith("more results omitted; refine the query to see them]")
        assert msg["content"].count("\n   ID: ") == 3 and msg["raw_chars"] > len(msg["content"])
        assert s["retrieval_chars_in_context"] == len(msg["content"])
    else:
        assert msg["raw_chars"] == len(msg["content"])  # nothing was cut


def test_a_tool_other_than_retrieval_is_saved_untouched(tmp_path):
    replies = [[thinking(), tool_use("get_current_time")], [thinking(), text("Done.")]]
    r, led, sdk = banking_runner(tmp_path, replies)
    task = tasks_by_id(BANKING_DOMAIN, ["task_073"])[0]
    r.run_once(SONNET, "EC", task)
    tr = json.loads((tmp_path / "tr" / run_name(SONNET, "EC", task.id, 1)).read_text())
    msg = next(m for m in tr["messages"] if m["role"] == "tool")
    assert (
        "raw_chars" not in msg
        and msg["content"] == sdk.requests[1]["messages"][-1]["content"][0]["content"]
    )


def test_retail_traces_are_unchanged_by_the_fix(tmp_path):
    from costlayer.runner import split_tasks
    from tests.test_runner import runner_for

    r, led, sdk = runner_for(tmp_path, [[thinking(), text("hello")]])
    s = r.run_once(SONNET, "default", split_tasks("train")[0])
    tr = json.loads(next((tmp_path / "tr").glob("*.json")).read_text())
    assert s["status"] == "ok" and "seen_in_context" not in tr
    assert set(tr) == {"task_id", "termination_reason", "reward", "messages"}


def test_mining_gives_no_credit_for_values_that_only_an_omitted_item_held():
    """The chain's second call uses an id that appears only in item 5 of the raw result. The
    saved (cut) trace cannot credit it; the raw trace would."""
    raw = result(8, extra={5: "the customer id is user_9999"})
    cut = cut_top_items(raw).text
    assert "user_9999" not in cut and cut.endswith(OMITTED_LINE.format(n=5))

    def chain_trace(search_result: str, seen: bool):
        return trace(
            asst([call(1, "KB_search", query="id")]), tool(1, search_result),
            asst([call(2, "get_user_information_by_id", user_id="user_9999")]), tool(2, "{}"),
            asst(text_="ok"), seen=seen,
        )  # fmt: skip

    assert mining.chains(chain_trace(raw, False), KINDS) == [[0, 1]]  # raw text: credited
    assert mining.chains(chain_trace(cut, True), KINDS) == []  # what the agent saw: not
    stats = mining.chain_stats([chain_trace(cut, True)], KINDS)
    assert stats["removable_calls"] == 0 and stats["cost_share"] == 0.0


def test_mining_rules_for_banking_tools():
    # a chain through call_discoverable_agent_tool (WRITE) never forms; parallel calls are one turn
    t = trace(
        asst([call(1, "KB_search", query="user_001")]), tool(1, result(2, {1: "user_001 is Sarah"})),
        asst([call(2, "call_discoverable_agent_tool", agent_tool_name="x", arguments="user_001")]),
        tool(2, "{}"),
        asst([call(3, "get_user_information_by_id", user_id="user_001"),
              call(4, "get_credit_card_accounts_by_user", user_id="user_001")]),
        tool(3, "{}"), tool(4, "{}"),
        asst(text_="done"),
    )  # fmt: skip
    assert mining.chains(t, KINDS) == []
    ok = trace(
        asst([call(1, "KB_search", query="user_001")]), tool(1, result(2, {1: "user_001 is Sarah"})),
        asst([call(3, "get_user_information_by_id", user_id="user_001"),
              call(4, "get_credit_card_accounts_by_user", user_id="user_001")]),
        tool(3, "{}"), tool(4, "{}"),
        asst(text_="done"),
    )  # fmt: skip
    s = mining.chain_stats([ok], KINDS)
    assert s["removable_calls"] == 1 and s["agent_calls"] == 3  # the parallel pair is ONE turn


def raw_phase1_style_trace(n_items=6):
    """An older cut-configuration trace: the tool message holds the RAW result."""
    raw = result(n_items)
    return trace(asst([call(1, "KB_search", query="q")]), tool(1, raw), asst(text_="ok")), raw


def test_older_cut_traces_are_reconstructed_only_if_the_recorded_sizes_agree():
    tr, raw = raw_phase1_style_trace()
    cut = cut_top_items(raw).text
    good = {"retrieval_chars_raw": len(raw), "retrieval_chars_in_context": len(cut)}
    new, src = prepare_trace(tr, good, "EC")
    assert src == "reconstructed" and new["reconstructed_cut"]
    msg = next(m for m in new["messages"] if m["role"] == "tool")
    assert msg["content"] == cut and msg["raw_chars"] == len(raw)
    assert (
        next(m for m in tr["messages"] if m["role"] == "tool")["content"] == raw
    )  # input untouched
    bad = {"retrieval_chars_raw": len(raw), "retrieval_chars_in_context": len(cut) + 1}
    assert prepare_trace(tr, bad, "C:top3")[1] == "unverified"
    assert prepare_trace(tr, good, "default")[1] == "uncut"  # never cut: raw is what was seen
    assert prepare_trace(trace(seen=True), {}, "EC")[1] == "seen"


# -- 2. analysis on banking configurations, report -----------------------------------------------


def run(config, task, success, cost, calls, tr=None, source="seen", **summary):
    s = {"retrieval_results": 3, "retrieval_chars_raw": 70000, "retrieval_chars_in_context": 20000}
    s.update(summary)
    r = Run(config, task, "ok", 1.0 if success else 0.0, cost, 0.001, calls,
            {"input": 10, "cache_write": 1000, "cache_read": 5000, "output": 500}, tr, s, source)  # fmt: skip
    return r


def synthetic_runs(n=6, cheap_loses=False):
    runs = []
    for i in range(n):
        t = f"task_{i:03d}"
        tr = trace(
            asst([call(1, "log_verification", user_id="u")]), tool(1, "{}"), asst(text_="ok")
        )
        runs += [
            run("default", t, True, 1.00, 20, tr),
            run("E:low", t, not (cheap_loses and i < 3), 0.30, 8 if cheap_loses else 19, tr),
            run("C:top3", t, True, 0.70, 21, tr),
            run("EC", t, True, 0.25, 18, tr),
        ]
    return runs


def test_banking_analysis_selects_global_star_and_builds_the_amendment_table():
    res = analysis.analyze(synthetic_runs(), KINDS, OPT_CONFIGS)
    assert res["n_tasks"] == 6 and res["global_star"] == "EC"
    t = res["table"]
    assert t["default"]["successes"] == 6 and t["default"]["mean_calls_failed"] is None
    assert t["EC"]["mean_agent_cost"] == pytest.approx(0.25)
    assert res["cost_ratio_vs_default"]["EC"][0] == pytest.approx(0.25)
    assert res["retrieval"]["EC"]["raw_tokens"] > res["retrieval"]["EC"]["in_tokens"] > 0
    assert not any(f["coincides"] for f in res["savings_flags"])


def test_a_saving_that_comes_with_fewer_calls_and_fewer_successes_is_flagged():
    res = analysis.analyze(synthetic_runs(cheap_loses=True), KINDS, OPT_CONFIGS)
    flags = {f["config"]: f for f in res["savings_flags"]}
    assert flags["E:low"]["coincides"] and not flags["C:top3"]["coincides"]
    assert (
        res["table"]["E:low"]["failures"] == 3 and res["table"]["E:low"]["mean_calls_failed"] == 8
    )
    assert "E:low" not in res["acceptable"] or res["failures"]["E:low"] <= 0 + 0
    md = build(res, {"ids": res["tasks"], "reused": [], "n_todo": 24}, None,
               {"total": 1.0, "calls": 3, "by_purpose": {"agent": 1.0}}, 0.0,
               Budget(22.0, {"b1": 4.0, "b2": 16.1}), "Console: enter the figure", [])  # fmt: skip
    assert "Savings that coincide with fewer calls AND fewer successes:** `E:low`" in md


def test_mining_uses_only_traces_that_show_what_the_agent_saw():
    runs = synthetic_runs()
    for r in runs:
        if r.config == "EC":
            r.trace_source = "unverified"
    res = analysis.analyze(runs, KINDS, OPT_CONFIGS)
    assert res["mining_sources"] == {"unverified": 6}
    assert res["chains"]["agent_calls"] == 0  # nothing was mined


def test_report_has_everything_the_plan_and_the_amendment_ask_for():
    res = analysis.analyze(synthetic_runs(), KINDS, OPT_CONFIGS)
    clf = {"accuracy": 0.58, "n": 6, "detail": []}
    md = build(res, {"ids": res["tasks"], "reused": ["default/task_000"], "n_todo": 23}, clf,
               {"total": 5.0, "calls": 90, "by_purpose": {"agent": 4.9, "user_sim": 0.1}}, 0.0,
               Budget(22.0, {"b1": 4.0, "b2": 16.1}), "Console figure to be entered", ["task_099"])  # fmt: skip
    for needle in (
        "VERDICT: ", "projected cost: $", "projected / default:", "projected / global*:",
        "| config | runs | successes | failures | agent $/run | cost / default (80% interval)",
        "LLM calls/run, failed runs", "bootstrap over tasks, 10,000 resamples, 80%",
        "No configuration's saving coincides with both fewer calls and fewer successes",
        "Classifier", "58.0%", "task_099",
        "Phase b2 spent (ledger, this dispatch): **$5.0000** of $16.10", "of $22.00",
        "Console reconciliation", "What this cannot show",
        "database state only", "other agents, other retrieval variants or other models",
        "A KEEP earns a larger quality test, never a launch",
        "as the agent saw it", "reconstruction checked by sizes",
    ):  # fmt: skip
        assert needle in md, needle


def test_a_weak_classifier_also_reports_the_projection_without_the_per_type_term():
    res = analysis.analyze(synthetic_runs(), KINDS, OPT_CONFIGS)
    res["headroom"] = {**res["headroom"], "headroom": 0.5, "beats_null": True}
    p = projections(res, {"accuracy": 0.5, "n": 6})
    assert p["literal"].basis == "per-type headroom" and p["per_type_off"].basis == "chains"
    assert p["per_type_off"].projected > p["literal"].projected
    assert projections(res, {"accuracy": 0.9, "n": 6})["per_type_off"] is None


def test_classifier_uses_haiku_with_the_banking_prompt_and_counts_correct_labels(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "d", "b2", 5.0)
    sdk = FakeSDK([[text("log_verification")], [text("none")], [text("none")]])
    ex = [("a", "I want to verify my identity", "log_verification"),
          ("b", "what are the fees?", "none"), ("c", "close my account", "call_discoverable_agent_tool")]  # fmt: skip
    res = loo_accuracy(AnthropicClient(led, sdk), ex, "d", BANKING_SYSTEM)
    assert res["n"] == 3 and res["accuracy"] == pytest.approx(2 / 3)
    req = sdk.requests[0]
    assert req["model"] == "claude-haiku-5-5" and "retail-bank" in req["system"]
    assert req["thinking"] == {"type": "disabled"} and "tools" not in req
    assert {json.loads(x)["purpose"] for x in led.path.read_text().splitlines()} == {"classifier"}


# -- the sweep end to end on a fake client -------------------------------------------------------


def test_sweep_end_to_end_reuses_runs_respects_order_and_feeds_the_analysis(tmp_path):
    ids = B2.train_ids()[:3]
    tasks = tasks_by_id(BANKING_DOMAIN, ids)
    r, led, sdk = banking_runner(tmp_path, None, cap=50.0, sdk=SearchThenAnswerSDK([]))
    basis = B2.CostBasis(mean=dict.fromkeys(CONFIGS, 0.5), high=dict.fromkeys(CONFIGS, 0.5))
    plan = B2.SweepPlan(ids=ids, todo=[(t, list(CONFIGS)) for t in tasks], reused=[])
    r.rail_usd = 2.25
    out = B2.run_sweep(r, led, plan, basis, concurrency=3)
    assert sorted(out["started"]) == sorted(ids) and out["stopped"] is None
    runs = load_runs(tmp_path / "out" / "runs", [tmp_path / "tr"])
    assert len(runs) == 12 and {x.trace_source for x in runs} == {"seen"}
    res = analysis.analyze(runs, {"KB_search": "READ", "log_verification": "WRITE"}, OPT_CONFIGS)
    assert res["n_tasks"] == 3 and set(res["table"]) == set(CONFIGS)
    cut_runs = [x for x in runs if x.config in ("C:top3", "EC")]
    assert all(
        x.summary["cut_results"] == 1 and x.summary["cut_items_omitted"] == 7 for x in cut_runs
    )
    assert all(x.summary["cut_results"] == 0 for x in runs if x.config in ("default", "E:low"))
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert {x["config"] for x in rows} == set(CONFIGS)


# -- the workflow --------------------------------------------------------------------------------


def test_workflow_downloads_the_b1_traces_with_continue_on_error_and_stays_dispatch_only():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "paid-run.yml").read_text())
    triggers = wf.get("on", wf.get(True))
    assert list(triggers) == ["workflow_dispatch"]
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["banking_b1_run_id"]["default"] == "40019775854"
    job = next(
        j for j in wf["jobs"].values() if j.get("if") == "${{ startsWith(inputs.phase, 'b') }}"
    )
    steps = {s.get("name", ""): s for s in job["steps"]}
    dl = steps["Download the b1 traces (b2 reuses those runs)"]
    assert dl["continue-on-error"] is True and dl["if"] == "inputs.phase == 'b2'"
    assert dl["with"]["name"] == "traces-banking-b1-${{ inputs.banking_b1_run_id }}"
    assert dl["with"]["run-id"] == "${{ inputs.banking_b1_run_id }}"
    assert dl["with"]["path"] == "traces/banking-b1-prior"
    names = [s.get("name", "") for s in job["steps"]]
    assert (
        names.index(dl["name"])
        < names.index("Dry run estimate (no key, no cost)")
        < names.index("Paid run")
    )
    text_ = (ROOT / ".github" / "workflows" / "paid-run.yml").read_text()
    assert "--prior-traces traces/banking-b1-prior" in text_


# -- the whole b2 command on a fake client (CLI, spend branch files, report, verdict) ------------------


class AgentAndClassifierSDK(SearchThenAnswerSDK):
    """Agent calls search then answer; the Haiku classifier call (no tools) answers `none`."""

    def _create(self, **kwargs):
        if kwargs["model"] == "claude-haiku-5-5" and "tools" not in kwargs:
            with self.lock:
                self.requests.append(copy.deepcopy(kwargs))
                return self._respond([text("none")])
        return super()._create(**kwargs)


def test_banking_phase_b2_runs_the_whole_flow_and_prints_the_verdict(tmp_path, monkeypatch, capsys):
    from costlayer import banking_run2

    ids = B2.train_ids()[:3]
    monkeypatch.setattr(B2, "train_ids", lambda: ids)  # three tasks keep the test short
    sdk = AgentAndClassifierSDK([])
    monkeypatch.setattr(banking_run2, "AnthropicClient", lambda led: AnthropicClient(led, sdk))
    install(
        Ledger(tmp_path / "x.jsonl", "x", "b2", 1.0), inner=fake_user_backend(["Hi", "###STOP###"])
    )
    monkeypatch.setattr(
        banking_run2, "install",
        lambda led: install(led, inner=fake_user_backend(["Hi", "###STOP###"])),
    )  # fmt: skip
    prior = phase1_priors(tmp_path)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"total": 22.0, "phase_caps": {"b1": 4.0, "b2": 16.1}}))
    spend, out, tr, report = (
        tmp_path / "spend2",
        tmp_path / "res",
        tmp_path / "tr",
        tmp_path / "r.md",
    )
    args = ["banking-phase", "--phase", "b2", "--founder-go", "b2", "--budget", str(budget),
            "--spend-dir", str(spend), "--dispatch-id", "d-1", "--prior-runs", str(prior.runs_dir),
            "--prior-traces", str(prior.trace_dirs[0]), "--out", str(out), "--trace-dir", str(tr),
            "--report", str(report)]  # fmt: skip
    assert cli_main(args) == 0
    # spend2 bookkeeping: a start file that reserves the b2 cap, and a ledger
    start = json.loads((spend / "start-d-1.json").read_text())
    assert start["phase"] == "b2" and start["cap"] == 16.1
    rows = [json.loads(x) for x in (spend / "ledger-d-1.jsonl").read_text().splitlines()]
    assert {x["phase"] for x in rows} == {"b2"} and "classifier" in {x["purpose"] for x in rows}
    # reused runs were copied, the rest was executed (4 runs: E:low and C:top3 on two tasks)
    names = sorted(p.name for p in (out / "runs").glob("sonnet__*.json"))
    assert len(names) == 12  # 8 reused + 4 executed
    summaries = [json.loads((out / "runs" / n).read_text()) for n in names]
    assert sum(1 for s in summaries if s.get("reused_from")) == 8
    assert (out / "tool_kinds.json").exists() and (out / "analysis.json").exists()
    md = report.read_text()
    assert "VERDICT: " in md and "Phase b2 spent" in md and "What this cannot show" in md
    printed = capsys.readouterr().out
    assert "VERDICT: " in printed and "projected / default:" in printed
    # the model of the classifier calls is Haiku and nothing but the three tasks was analysed
    assert json.loads((out / "analysis.json").read_text())["n_tasks"] == 3


def test_a_refused_b2_start_leaves_nothing_behind(tmp_path):
    prior = phase1_priors(tmp_path)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"total": 22.0, "phase_caps": {"b1": 4.0, "b2": 6.0}}))
    spend, out, tr = tmp_path / "spend2", tmp_path / "res", tmp_path / "tr"
    args = ["banking-phase", "--phase", "b2", "--founder-go", "b2", "--budget", str(budget),
            "--spend-dir", str(spend), "--dispatch-id", "d-1", "--prior-runs", str(prior.runs_dir),
            "--prior-traces", str(prior.trace_dirs[0]), "--out", str(out), "--trace-dir", str(tr),
            "--report", str(tmp_path / "r.md")]  # fmt: skip
    with pytest.raises(SystemExit, match="REFUSED: certain part"):
        cli_main(args)
    assert not spend.exists() and not out.exists() and not tr.exists()
    assert not (tmp_path / "r.md").exists()
