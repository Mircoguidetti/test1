"""Offline tests for the failure-anatomy analysis (constructed sanitized traces, no network)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from costlayer.optimizer import failure_anatomy as fa
from costlayer.optimizer.data import Run
from costlayer.sanitize import BANNED_KEYS

KINDS = {"get_order": "READ", "find_user": "READ", "cancel": "WRITE", "KB_search": "READ"}


def call(cid: str, name: str, **args: Any) -> dict[str, Any]:
    return {"id": cid, "name": name, "arguments": args, "requestor": "assistant"}


def trace(steps: list[tuple[float, list[dict[str, Any]]]], errors: set[str] = frozenset()):
    msgs: list[dict[str, Any]] = [{"role": "assistant", "content": "Hi!"}]
    for i, (cost, calls) in enumerate(steps):
        msgs.append(
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": calls,
                "cost": cost,
                "usage": {"output_tokens": 1},
            }
        )
        for c in calls:
            msgs.append(
                {
                    "role": "tool",
                    "id": c["id"],
                    "content": f"result {i}",
                    "error": c["id"] in errors,
                    "requestor": "assistant",
                }
            )
    return {"task_id": "t", "reward": 1.0, "messages": msgs}


def run(config: str, task: str, ok: bool, steps, errors=frozenset()) -> Run:
    tr = trace(steps, set(errors))
    total = sum(c for c, _ in steps)
    return Run(
        config=config,
        task=task,
        status="ok",
        reward=1.0 if ok else 0.0,
        cost_agent=total,
        cost_user=0.0,
        llm_calls=len(steps),
        tokens={},
        trace=tr,
        summary={},
        trace_source="uncut",
    )


def views(runs: list[Run]) -> list[fa.RunView]:
    return fa.make_views(runs)


def test_steps_signals_and_errors():
    r = run(
        "default",
        "a",
        False,
        [
            (1.0, [call("1", "get_order", id="o1")]),
            (1.0, [call("2", "get_order", id="o1")]),  # repeat of step 1
            (1.0, [call("3", "cancel", id="o1")]),
        ],
        errors={"3"},
    )
    (v,) = views([r])
    assert [s.idx for s in v.steps] == [1, 2, 3]
    assert fa.first_repeat_step(v.steps) == 2
    assert fa.error_steps(r.trace, v.steps) == [3]


def test_a_trace_whose_cost_does_not_match_the_summary_is_not_used():
    r = run("default", "a", True, [(1.0, [])])
    r.cost_agent = 2.0
    assert views([r])[0].steps is None


def test_stop_rule_exact_counts_saved_and_sunk_cost():
    # successful default runs cost 2 and 4 -> median 3; m=2 -> threshold 6
    runs = [
        run("default", "a", True, [(1.0, []), (1.0, [])]),
        run("default", "b", True, [(2.0, []), (2.0, [])]),
        run("default", "c", False, [(4.0, []), (4.0, []), (4.0, [])]),  # crosses 6 at 8 -> saves 4
        run("E:low", "a", True, [(5.0, []), (5.0, [])]),  # crosses 6 at 10: wrongly stopped
    ]
    rows = fa.stop_rule(views(runs), 2.0, True, ("default", "E:low"))
    default, low, allc = rows
    assert default["threshold"] == 6.0
    assert default["stopped_failed"] == 1 and default["saved_failed"] == 4.0
    assert default["wrongly_stopped"] == 0
    assert low["wrongly_stopped"] == 1 and low["sunk_wrong"] == 10.0
    assert allc["net"] == 4.0 - 10.0
    assert allc["total_cost"] == 2 + 4 + 12 + 10


def test_stop_rule_without_traces_is_an_upper_bound():
    runs = [
        run("default", "a", True, [(3.0, [])]),
        run("default", "b", False, [(10.0, [])]),
    ]
    for r in runs:
        r.trace = None
    rows = fa.stop_rule(views(runs), 2.0, False, ("default",))
    # threshold 6; the failed run (10) is assumed to be stopped at 6: saves at most 4
    assert rows[0]["saved_failed"] == 4.0 and rows[0]["net"] == 4.0


def test_q1_verdict_needs_both_bars_and_is_undetermined_without_traces():
    good = {"net_share": 0.2, "wrongly_stopped": 0, "successes": 10}
    bad_wrong = {"net_share": 0.2, "wrongly_stopped": 2, "successes": 10}
    low_net = {"net_share": 0.05, "wrongly_stopped": 0, "successes": 10}
    assert fa.q1_verdict({2.0: good}, True)["status"] == "PASS"
    assert fa.q1_verdict({2.0: bad_wrong}, True)["status"] == "FAIL"
    assert fa.q1_verdict({2.0: low_net, 3.0: bad_wrong}, True)["status"] == "FAIL"
    assert fa.q1_verdict({2.0: low_net}, False)["status"].startswith("FAIL")
    assert fa.q1_verdict({2.0: good}, False)["status"].startswith("UNDETERMINED")
    assert fa.q1_verdict({2.0: good, 3.0: low_net}, True)["per_m"][3.0]["pass"] is False


def test_signal_summary_positions():
    runs = [
        run("default", "a", False, [(1.0, [call("1", "get_order", id=1)])] * 3),  # same call twice
        run("default", "b", True, [(1.0, [call("1", "get_order", id=1)])]),
    ]
    sig = fa.signal_summary(views(runs))
    assert sig["repeat/failed"]["with_signal"] == 1
    assert sig["repeat/failed"]["median_step"] == 2
    assert sig["repeat/failed"]["median_cost_fraction"] == pytest.approx(2 / 3)
    assert sig["repeat/successful"]["with_signal"] == 0


def test_q2_counts_only_cross_task_repeats_of_read_only_calls():
    a = run(
        "default",
        "a",
        True,
        [
            (1.0, [call("1", "get_order", id="o1"), call("2", "find_user", n="x")]),
            (1.0, [call("3", "get_order", id="o1")]),  # same task: not a cross-task repeat
            (1.0, [call("4", "cancel", id="o1")]),  # write: never counted
        ],
    )
    b = run(
        "default",
        "b",
        True,
        [
            (2.0, [call("5", "get_order", id="o1"), call("6", "get_order", id="o2")]),
            (1.0, [call("7", "cancel", id="o1")]),
        ],
    )
    r = fa.q2_config(views([a, b]), "default", KINDS, ["a", "b"])
    assert r["read_calls"] == 5  # a: 3 read calls; b: 2
    assert r["repeated"] == 1  # b's get_order o1 was called in task a
    assert r["share_calls"] == pytest.approx(1 / 5)
    assert r["repeat_cost"] == pytest.approx(1.0)  # 2.0 / 2 calls in that LLM call
    assert r["total_cost"] == pytest.approx(6.0)
    # the order matters: with b first, a's get_order o1 is the repeat
    r2 = fa.q2_config(views([a, b]), "default", KINDS, ["b", "a"])
    assert r2["repeated"] == 1 and r2["repeat_cost"] == pytest.approx(0.5)


def test_q2_verdict_bars():
    assert fa.q2_verdict({"share_calls": 0.15, "share_cost": 0.05}) == "PASS"
    assert fa.q2_verdict({"share_calls": 0.14, "share_cost": 0.5}) == "FAIL"
    assert fa.q2_verdict({"share_calls": 0.9, "share_cost": 0.04}) == "FAIL"


def test_q2_result_sets_compare_document_id_sets_across_tasks_only():
    def kb(cid, q):
        return call(cid, "KB_search", q=q)

    def with_result(r: Run, docs: list[str]) -> Run:
        for m in r.trace["messages"]:
            if m["role"] == "tool":
                m["content"] = "\n".join(f"1. T\n   ID: {d}\n   body" for d in docs)
        return r

    a = with_result(
        run("default", "a", True, [(1.0, [kb("1", "x")]), (1.0, [kb("2", "y")])]), ["d1", "d2"]
    )
    b = with_result(run("default", "b", True, [(1.0, [kb("3", "z")])]), ["d2", "d1"])
    c = with_result(run("default", "c", True, [(1.0, [kb("4", "w")])]), ["d9"])
    r = fa.q2_result_sets(views([a, b, c]), "default", ["a", "b", "c"])
    # a: 2 results, the second is a same-task repeat of the first (not counted); b repeats a's set
    assert r["results"] == 4 and r["repeated"] == 1


def test_q3_says_so_when_there_are_too_few_failures():
    runs = [
        run("default", f"t{i}", True, [(1.0, [call("1", "get_order", id=i)])]) for i in range(5)
    ]
    runs.append(run("default", "t9", False, [(1.0, [])]))
    q = fa.q3_early(views(runs))
    assert q["enough"] is False and "fewer than 3" in q["note"]


def test_q3_finds_a_perfect_signal_and_ranks_it_against_the_null():
    runs = []
    for i in range(8):
        ok = i % 2 == 0
        name = "get_order" if ok else "find_user"
        runs.append(run("default", f"t{i}", ok, [(1.0, [call("1", name, id=i)])]))
    q = fa.q3_early(views(runs))
    assert q["enough"] and q["accuracy"] == 1.0 and q["rank"] < 0.05
    assert q == fa.q3_early(views(runs))  # deterministic (fixed seed)


def test_analyze_test_and_report_end_to_end(tmp_path):
    runs_dir, traces = tmp_path / "runs", tmp_path / "traces"
    runs_dir.mkdir()
    traces.mkdir()
    spec = [
        ("default", "a", True, [(1.0, [call("1", "get_order", id="o1")])]),
        ("default", "b", True, [(1.0, [call("1", "get_order", id="o1")])]),
        ("default", "c", False, [(1.0, [call("1", "find_user", n="x")])] * 4),
        ("E:low", "a", True, [(0.5, [call("1", "get_order", id="o1")])]),
    ]
    for cfg, task, ok, steps in spec:
        r = run(cfg, task, ok, steps)
        assert not fa.find_banned(r.trace)
        s = {
            "model": "claude-sonnet-5-5",
            "config": cfg,
            "task": task,
            "attempt": 1,
            "status": "ok",
            "reward": r.reward,
            "llm_calls": r.llm_calls,
            "cost_agent": r.cost_agent,
            "cost_user": 0.0,
            "tokens": {},
        }
        short = s["model"].split("-")[1]
        name = f"{short}__{cfg.replace(':', '-')}__{task}__a1.json"
        (runs_dir / name).write_text(json.dumps(s))
        (traces / name).write_text(json.dumps(r.trace))
    res = fa.analyze_test(
        "T", runs_dir, [traces], KINDS, ("default", "E:low"), ["a", "b", "c"], False
    )
    assert res.n_runs == 4 and res.n_with_steps == 4 and res.q1_verdict and res.q2
    assert res.q1[-1]["failures"] == 1
    out = fa.build_report([res], [{"artifact": "x", "test": "T", "state": "yes"}])
    for needle in (fa.PINNED, "What this cannot show", "Q1 for this test", "Q2 for this test"):
        assert needle in out
    # without traces nothing trace-based is invented
    res2 = fa.analyze_test("T", runs_dir, [], KINDS, ("default", "E:low"), None, False)
    assert res2.q2 is None and res2.signals is None and res2.q3 is None
    assert res2.q1_verdict["status"].startswith(("FAIL", "UNDETERMINED"))
    assert "NOT COMPUTED" in fa.build_report([res2], [])


def test_analysis_never_imports_tau2():
    code = (
        "import sys, costlayer.optimizer.failure_anatomy;"
        "bad=[m for m in sys.modules if m=='tau2' or m.startswith('tau2.')];"
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = (
        Path(__file__).resolve().parents[1] / "costlayer/optimizer/failure_anatomy.py"
    ).read_text()
    assert not re.search(r"(?m)^\s*(import|from)\s+tau2", src)


def test_banned_keys_match_the_sanitizer_and_are_detected():
    assert fa.BANNED_KEYS == BANNED_KEYS
    assert fa.find_banned({"messages": [{"x": {"evaluation_criteria": 1}}]})
    assert not fa.find_banned({"messages": [{"content": "evaluation_criteria in text is fine"}]})
    r = run("default", "a", True, [(1.0, [])])
    r.trace["tasks"] = []
    r.trace["evaluation_criteria"] = {}
    with pytest.raises(AssertionError, match="not sanitized"):
        views([r])
