"""Offline tests for the cost-per-success analysis (constructed run summaries, no network)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from costlayer.optimizer import cps, cps_check, cps_rng

CONFIGS = ("default", "A")


def row(config: str, task: str, cost: float, reward: float, status: str = "ok", **kw: Any):
    return {
        "model": "claude-sonnet-5-5",
        "config": config,
        "task": task,
        "attempt": kw.pop("attempt", 1),
        "status": status,
        "reward": reward,
        "cost_agent": cost,
        "cost_user": 99.0,  # must never be counted
        "llm_calls": kw.pop("calls", 10),
        "termination": "user_stop",
        "tokens": {},
        **kw,
    }


def write(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        (d / f"{i:03d}_{r['config']}_{r['task']}_a{r['attempt']}.json").write_text(json.dumps(r))
    return d


def basic(tmp_path: Path) -> Path:
    return write(
        tmp_path,
        [
            row("default", "t1", 1.0, 1.0),
            row("default", "t2", 3.0, 0.0),  # failed run: cost stays in the numerator
            row("default", "t3", 2.0, 1.0),
            row("A", "t1", 0.5, 1.0),
            row("A", "t2", 0.5, 0.0),
            row("A", "t3", 0.5, 0.0, status="agent_error"),
        ],
    )


def test_cost_per_success_counts_failed_runs_in_the_numerator(tmp_path):
    data = cps.load_test("x", basic(tmp_path), CONFIGS)
    t = cps.point_table(data)
    assert t["default"]["cost"] == 6.0 and t["default"]["successes"] == 2
    assert t["default"]["cps"] == 3.0 and t["default"]["cost_per_run"] == 2.0
    assert t["A"]["cost"] == 1.5 and t["A"]["successes"] == 1 and t["A"]["cps"] == 1.5
    assert t["A"]["cps_ratio"] == pytest.approx(0.5)
    assert t["A"]["cost_ratio"] == pytest.approx(0.25)  # per-run view would say 0.25, CPS 0.5


def test_user_simulator_cost_is_never_counted(tmp_path):
    t = cps.point_table(cps.load_test("x", basic(tmp_path), CONFIGS))
    assert t["default"]["cost"] == 6.0  # cost_user was 99 per run


def test_agent_error_and_censored_are_failures_not_exclusions(tmp_path):
    d = write(
        tmp_path,
        [
            row("default", "t1", 1.0, 1.0),
            row("default", "t2", 1.0, 1.0, status="censored"),
            row("A", "t1", 1.0, 1.0),
            row("A", "t2", 1.0, 0.0, status="agent_error"),
        ],
    )
    t = cps.point_table(cps.load_test("x", d, CONFIGS))
    # a censored run with reward 1 is still not a success
    assert t["default"]["successes"] == 1 and t["default"]["runs"] == 2


def test_infra_error_is_excluded_from_both_sides_and_listed(tmp_path):
    d = write(
        tmp_path,
        [
            row("default", "t1", 1.0, 1.0),
            row("default", "t2", 5.0, 0.0, status="infra_error"),
            row("default", "t2", 2.0, 1.0, attempt=2),  # the rerun counts
            row("A", "t1", 1.0, 1.0),
            row("A", "t2", 1.0, 1.0),
        ],
    )
    data = cps.load_test("x", d, CONFIGS)
    assert data.infra == ["default/t2/a1"]
    t = cps.point_table(data)
    assert t["default"]["cost"] == 3.0 and t["default"]["runs"] == 2
    b = cps_check.summarize(*cps_check.per_task(d, CONFIGS, None), CONFIGS)[0]
    assert b["default"]["cost"] == 3.0


def test_task_without_a_finished_run_in_every_configuration_is_dropped_for_all(tmp_path):
    d = write(
        tmp_path,
        [
            row("default", "t1", 1.0, 1.0),
            row("default", "t2", 1.0, 1.0),
            row("A", "t1", 1.0, 1.0),
            row("A", "t2", 1.0, 0.0, status="infra_error"),
        ],
    )
    data = cps.load_test("x", d, CONFIGS)
    assert data.tasks == ["t1"] and data.dropped == ["t2"]
    assert cps.point_table(data)["default"]["runs"] == 1


def test_unknown_status_and_other_models_are_handled_loudly(tmp_path):
    d = write(tmp_path, [row("default", "t1", 1.0, 1.0, status="weird"), row("A", "t1", 1.0, 1.0)])
    with pytest.raises(ValueError, match="unknown status"):
        cps.load_test("x", d, CONFIGS)
    other = write(
        tmp_path / "o",
        [
            row("default", "t1", 1.0, 1.0, model="claude-haiku-5-5"),
            row("default", "t1", 2.0, 1.0, attempt=2),
            row("A", "t1", 1.0, 1.0),
        ],
    )
    data = cps.load_test("x", other, CONFIGS)
    assert data.other_model == 1 and cps.point_table(data)["default"]["cost"] == 2.0


def test_zero_successes_gives_undefined_never_infinity_or_zero(tmp_path):
    d = write(
        tmp_path,
        [
            row("default", "t1", 1.0, 1.0),
            row("A", "t1", 1.0, 0.0),
        ],
    )
    t = cps.point_table(cps.load_test("x", d, CONFIGS))
    assert t["A"]["cps"] is None and t["A"]["cps_ratio"] is None
    assert t["A"]["cost_ratio"] == 1.0  # the per-run ratio still exists
    assert cps.pick(t, 1, "cps") == "default"  # an undefined CPS never wins


def test_resamples_with_no_success_are_counted_not_dropped(tmp_path):
    rows = [row("default", f"t{i}", 1.0, 1.0 if i == 0 else 0.0) for i in range(4)]
    rows += [row("A", f"t{i}", 1.0, 0.0) for i in range(4)]
    data = cps.load_test("x", write(tmp_path, rows), CONFIGS)
    iv = cps.bootstrap(data)
    assert iv["A"]["cps"]["defined"] == 0 and iv["A"]["cps"]["undefined"] == cps_rng.RESAMPLES
    d = iv["default"]["cps"]
    assert d["defined"] + d["undefined"] == cps_rng.RESAMPLES and d["undefined"] > 0


def test_bootstrap_resamples_the_same_tasks_for_every_configuration(tmp_path):
    # A costs exactly half of default on every task, so in every resample the paired cost ratio
    # is 0.5: it is only constant if the same tasks are drawn for both configurations
    rows = []
    for i in range(6):
        c = float(i + 1)
        rows += [row("default", f"t{i}", c, 1.0), row("A", f"t{i}", c / 2, 1.0)]
    data = cps.load_test("x", write(tmp_path, rows), CONFIGS)
    iv = cps.bootstrap(data)["A"]
    assert iv["cost_ratio"]["low"] == pytest.approx(0.5) and iv["cost_ratio"][
        "high"
    ] == pytest.approx(0.5)
    assert iv["cps_ratio"]["low"] == pytest.approx(0.5)
    # an unpaired bootstrap would have spread the ratio widely
    assert iv["cost_per_run"]["low"] < iv["cost_per_run"]["high"]


def test_selection_rule_and_global_star_by_cps(tmp_path):
    # n=10: allowance floor(0.10*10)=1. B is the cheapest per run but loses 3 tasks (not
    # acceptable); A is cheaper per run and per success and loses 1 (acceptable)
    def mk(cfg, cost, wins):
        return [row(cfg, f"t{i}", cost, 1.0 if i < wins else 0.0) for i in range(10)]

    rows = mk("default", 1.0, 9) + mk("A", 0.95, 8) + mk("B", 0.5, 6)
    data = cps.load_test("x", write(tmp_path, rows), ("default", "A", "B"))
    t = cps.point_table(data)
    assert cps.acceptable(t, 10) == ["default", "A"]
    assert cps.pick(t, 10, "cps") == "default"  # 10/9 = 1.11 vs 9.5/8 = 1.19
    assert cps.pick(t, 10, "cost_per_run") == "A"


def test_regression_cost_rule_a_run_is_counted_exactly_once_by_its_agent_cost(tmp_path):
    d = write(
        tmp_path,
        [
            row("default", "t1", 0.25, 1.0, cost_user=7.0, calls=3),
            row("default", "t1", 0.75, 0.0, attempt=2, cost_user=7.0, calls=5),
            row("A", "t1", 1.0, 1.0),
        ],
    )
    c = cps.load_test("x", d, CONFIGS).cells["default"]["t1"]
    assert (c.cost, c.succ, c.runs, c.calls, c.failed_calls, c.failed_runs) == (1.0, 1, 2, 8, 5, 1)
    assert cps.is_success({"status": "ok", "reward": 1.0})
    assert not cps.is_success({"status": "agent_error", "reward": 1.0})
    assert not cps.is_success({"status": "ok", "reward": 0.0})


def test_implementations_a_and_b_agree_on_a_constructed_dataset(tmp_path):
    rows = []
    for i in range(8):
        for cfg, cost, wins in (("default", 1.0, 5), ("A", 0.7, 4), ("B", 0.6, 1)):
            rows.append(row(cfg, f"t{i}", cost + 0.01 * i, 1.0 if i < wins else 0.0, calls=5 + i))
    d = write(tmp_path, rows)
    cfgs = ("default", "A", "B")
    types = {f"t{i}": "x" if i < 5 else "y" for i in range(8)}
    a = cps.analyze(cps.load_test("x", d, cfgs), types, 0.1)
    b = cps_check.analyze(d, cfgs, None, types, 0.1)
    assert cps_check.compare(a, b) == []


def test_compare_reports_a_difference():
    assert cps_check.compare({"x": 1.0}, {"x": 1.0 + 1e-3})
    assert cps_check.compare({"x": None}, {"x": 1.0})
    assert not cps_check.compare({"x": 1.0, "y": 2}, {"x": 1.0 + 1e-12})


def test_leave_one_out_headroom_reproduces_a_known_case(tmp_path):
    # three tasks of type p where A is strictly better, one of type q: A is learned for p
    rows = []
    for i in range(4):
        rows += [row("default", f"t{i}", 1.0, 1.0), row("A", f"t{i}", 0.5, 1.0)]
    data = cps.load_test("x", write(tmp_path, rows), CONFIGS)
    labels = ["p", "p", "p", "p"]
    cost, succ = cps.loo_policy(data, labels, "cps")
    assert (cost, succ) == (2.0, 4)  # every task chooses A learned from the other three
    assert cps.headroom(data, labels, "default", "cps") == pytest.approx(0.5)
    assert cps.headroom(data, labels, "A", "cps") == pytest.approx(0.0)


def test_optimizer_modules_do_not_import_tau2():
    code = (
        "import sys, costlayer.optimizer.cps, costlayer.optimizer.cps_check,"
        " costlayer.optimizer.cps_report;"
        "bad=[m for m in sys.modules if m=='tau2' or m.startswith('tau2.')];"
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    root = Path(__file__).resolve().parents[1] / "costlayer" / "optimizer"
    for f in root.glob("cps*.py"):
        assert not re.search(r"(?m)^\s*(import|from)\s+tau2", f.read_text()), f


def test_the_committed_report_matches_a_fresh_run():
    root = Path(__file__).resolve().parents[1]
    from costlayer.optimizer import cps_report

    a, b, _ = cps_report.run_both(root)
    assert cps_check.compare(a["retail"], b["retail"]) == []
    assert cps_check.compare(a["banking"], b["banking"]) == []
    assert a["banking"]["table"]["default"]["cps"] == pytest.approx(4.801806 / 2)
