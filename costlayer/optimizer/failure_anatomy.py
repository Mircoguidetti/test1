"""Failure anatomy: what do failed runs cost, how much work repeats across tasks, is there an
early signal. Sanitized traces and run summaries only; no API, no tau2.

Definitions (fixed before looking at any data; the report states them):

Q1 cost of failed runs, per test and configuration.
  * A run is a success if its status is `ok` and its reward is 1; every other finished run
    (`ok` with reward 0, `agent_error`, `censored`) is a failure. `infra_error` runs are not
    finished runs and are left out (counted in the report).
  * Rule R(m): abort a run as soon as its cumulative AGENT cost (after an LLM call) exceeds
    T_m = m x median(total agent cost of the successful `default` runs of that test).
    The call that crosses T_m has been paid. For a run stopped at cumulative cost s with total
    cost C: saved = C - s. A failed run that is stopped saves money; a successful run that is
    stopped is "wrongly stopped".
  * net saving = (sum of `saved` over stopped FAILED runs) - (sum of the cost spent up to the stop,
    s, over WRONGLY STOPPED successful runs: money that buys nothing under the rule). It is shown
    as a share of the total agent cost of all finished runs of the test. The components are
    reported too. Rule R(m) is applied to the runs of all configurations of the test.
  * Without traces only the summaries exist, so per-call costs are unknown. Then `saved` is
    bounded above by C - T_m (the run is assumed to cross T_m exactly at the stop) and the sunk
    cost of a wrongly stopped success is bounded below by T_m: an UPPER bound on the net saving.
  * Signals (traces only): (a) the same tool name with identical arguments called again in the
    same run (position = the repeating call); (b) a tool call of the agent returning an error.
    Position = index of the agent LLM call that issued the call, counted from 1; reported as that
    index, as a fraction of the run's LLM calls and as the fraction of the run's cumulative cost
    spent through that call.

Q2 repetition across tasks, per test and configuration. Read-only = tool kind READ in
  tool_kinds.json (WRITE and GENERIC are never counted). A read-only call is a cross-task repeat
  if it is the FIRST occurrence of its (tool name, canonical arguments) inside its own task and the
  same key was called in an EARLIER task of the sweep order, same configuration. Repetition inside
  one task is excluded (that is caching). Share of calls = repeats / all read-only calls. Share of
  cost = sum of the cost of the repeating calls / total agent cost, where the cost of a call is
  the cost of the LLM call that issued it divided by the number of tool calls in that LLM call.
  Banking also: the share of KB_search and grep results (with at least one document) whose SET of
  returned document ids was already returned, for an earlier task, in the same configuration.
  (Under the cutting configurations the saved result holds the items the agent saw.)

Q3 early signal (informational): the first three tool-call NAMES of a run. Statistic = leave-one-
  task-out accuracy of a lookup table (signature -> majority label of the other tasks' runs, unseen
  signature -> overall majority) at predicting failed/successful over all runs of the test; null =
  200 shuffles (seed 42) of the labels. Rank = (1 + #null >= real) / (1 + 200).

Fixed bars: Q1 passes if for at least one m the net saving is >= 10% of total cost AND at most 1
successful run per 10 is wrongly stopped. Q2 passes if repeated read-only calls are >= 15% of
read-only calls AND >= 5% of agent cost. Both bars are checked per test on the pooled
configurations; the overall answer needs both tests.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from costlayer.optimizer.data import COMPLETE, Run, load_runs

MULTIPLIERS = (1.5, 2.0, 3.0)
Q1_NET_BAR = 0.10
Q1_WRONG_BAR = 0.10  # at most 1 successful run per 10
Q2_CALL_BAR = 0.15
Q2_COST_BAR = 0.05
NULL_SHUFFLES = 200
NULL_SEED = 42
RETRIEVAL_TOOLS = ("KB_search", "grep")
# Gold / task-definition keys that must never be in a trace. Kept here because costlayer.sanitize
# imports tau2; a test keeps this tuple equal to sanitize.BANNED_KEYS.
BANNED_KEYS = (
    "evaluation_criteria",
    "expected_actions",
    "action_checks",
    "env_assertions",
    "nl_assertions",
    "communicate_checks",
    "gt_data_hash",
    "required_documents",
    "annotations",
    "issues",
    "reward_basis",
    "user_scenario",
)
COST_TOLERANCE = 0.02  # trace cost must match the summary within 2% to be used


@dataclass(frozen=True)
class Call:
    name: str
    key: str  # canonical JSON of the arguments


@dataclass
class Step:
    idx: int  # 1-based index of the agent LLM call
    cost: float
    calls: list[Call]
    call_ids: list[str]


def find_banned(obj: Any) -> bool:
    """True if a banned key appears anywhere in a nested structure."""
    if isinstance(obj, dict):
        return any(k in BANNED_KEYS or find_banned(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(find_banned(v) for v in obj)
    return False


# -- reading a trace ----------------------------


def agent_steps(trace: dict[str, Any]) -> list[Step]:
    """One Step per agent LLM call (assistant messages with usage; the greeting has none)."""
    steps: list[Step] = []
    for m in trace["messages"]:
        if m["role"] != "assistant" or not m.get("usage"):
            continue
        tcs = [
            c for c in (m.get("tool_calls") or []) if c.get("requestor", "assistant") == "assistant"
        ]
        steps.append(
            Step(
                idx=len(steps) + 1,
                cost=float(m.get("cost") or 0.0),
                calls=[Call(c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in tcs],
                call_ids=[c["id"] for c in tcs],
            )
        )
    return steps


def error_steps(trace: dict[str, Any], steps: list[Step]) -> list[int]:
    """Step indices of agent tool calls whose result was an error."""
    by_id = {cid: s.idx for s in steps for cid in s.call_ids}
    return sorted(
        by_id[m["id"]]
        for m in trace["messages"]
        if m["role"] == "tool"
        and m.get("error")
        and m.get("requestor", "assistant") == "assistant"
        and m["id"] in by_id
    )


def first_repeat_step(steps: list[Step]) -> int | None:
    seen: set[Call] = set()
    for s in steps:
        for c in s.calls:
            if c in seen:
                return s.idx
            seen.add(c)
    return None


@dataclass
class RunView:
    run: Run
    steps: list[Step] | None  # None: no usable trace for this run

    @property
    def total(self) -> float:
        return self.run.cost_agent

    @property
    def success(self) -> bool:
        return self.run.success


def make_views(runs: list[Run]) -> list[RunView]:
    out = []
    for r in runs:
        steps = None
        if r.trace is not None and r.trace_source != "unverified":
            assert not find_banned(r.trace), "trace is not sanitized (gold or task fields found)"
            st = agent_steps(r.trace)
            if st and abs(sum(s.cost for s in st) - r.cost_agent) <= COST_TOLERANCE * max(
                r.cost_agent, 1e-9
            ):
                steps = st
        out.append(RunView(r, steps))
    return out


# -- Q1 ----------------------------


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def q1_table(views: list[RunView], configs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for c in (*configs, "all"):
        vs = [v for v in views if c == "all" or v.run.config == c]
        failed = [v for v in vs if not v.success]
        total = sum(v.total for v in vs)
        cf = sum(v.total for v in failed)
        rows.append(
            {
                "config": c,
                "runs": len(vs),
                "successes": len(vs) - len(failed),
                "failures": len(failed),
                "total_cost": total,
                "failed_cost": cf,
                "failed_share": cf / total if total else 0.0,
            }
        )
    return rows


def threshold_base(views: list[RunView]) -> float | None:
    """Median total agent cost of the successful `default` runs of the test."""
    return median([v.total for v in views if v.run.config == "default" and v.success])


def stop_run(v: RunView, t: float) -> tuple[float, float] | None:
    """(cost spent until the stop, saved) if the run is stopped, else None. Needs the steps."""
    assert v.steps is not None
    cum = 0.0
    for s in v.steps:
        cum += s.cost
        if cum > t:
            return cum, v.total - cum
    return None


def stop_rule(
    views: list[RunView], m: float, exact: bool, configs: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Rule R(m) for each configuration and for all of them. See the module docstring."""
    base = threshold_base(views)
    if base is None:
        raise ValueError("no successful default run: the threshold is undefined")
    t = m * base
    rows = []
    for c in (*configs, "all"):
        vs = [v for v in views if c == "all" or v.run.config == c]
        total = sum(v.total for v in vs)
        saved = sunk = 0.0
        stopped_failed = wrong = 0
        for v in vs:
            if exact:
                sp = stop_run(v, t)
                spent, sv = sp if sp else (None, 0.0)
            else:  # bounds from the summary alone
                spent, sv = (t, v.total - t) if v.total > t else (None, 0.0)
            if spent is None:
                continue
            if v.success:
                wrong += 1
                sunk += spent
            else:
                stopped_failed += 1
                saved += sv
        n_success = sum(1 for v in vs if v.success)
        net = saved - sunk
        rows.append(
            {
                "config": c,
                "m": m,
                "threshold": t,
                "exact": exact,
                "stopped_failed": stopped_failed,
                "saved_failed": saved,
                "wrongly_stopped": wrong,
                "successes": n_success,
                "sunk_wrong": sunk,
                "net": net,
                "total_cost": total,
                "saved_share": saved / total if total else 0.0,
                "net_share": net / total if total else 0.0,
            }
        )
    return rows


def q1_verdict(all_rows: dict[float, dict[str, Any]], exact: bool) -> dict[str, Any]:
    """`all_rows[m]` is the 'all configurations' row of R(m)."""
    per_m = {}
    for m, r in all_rows.items():
        ok_net = r["net_share"] >= Q1_NET_BAR
        ok_wrong = r["wrongly_stopped"] <= Q1_WRONG_BAR * r["successes"]
        per_m[m] = {"net_ok": ok_net, "wrong_ok": ok_wrong, "pass": ok_net and ok_wrong}
    any_pass = any(x["pass"] for x in per_m.values())
    if exact:
        status = "PASS" if any_pass else "FAIL"
    else:  # upper bound on the net saving
        status = (
            "UNDETERMINED (upper bound passes; needs traces)"
            if any_pass
            else ("FAIL (holds even with the upper bound)")
        )
    return {"status": status, "per_m": per_m}


def signal_summary(views: list[RunView]) -> dict[str, dict[str, Any]]:
    """Signals (a) and (b) for failed and successful runs."""
    out: dict[str, dict[str, Any]] = {}
    for sig in ("repeat", "error"):
        for label, want in (("failed", False), ("successful", True)):
            vs = [v for v in views if v.steps is not None and v.success == want]
            hits = []
            for v in vs:
                assert v.steps is not None and v.run.trace is not None
                idx = (
                    first_repeat_step(v.steps)
                    if sig == "repeat"
                    else (error_steps(v.run.trace, v.steps) or [None])[0]
                )
                if idx is None:
                    continue
                n = len(v.steps)
                cum = sum(s.cost for s in v.steps[:idx])
                hits.append((idx, idx / n, cum / v.total if v.total else 0.0))
            out[f"{sig}/{label}"] = {
                "runs": len(vs),
                "with_signal": len(hits),
                "share": len(hits) / len(vs) if vs else 0.0,
                "median_step": median([h[0] for h in hits]),
                "median_step_fraction": median([h[1] for h in hits]),
                "median_cost_fraction": median([h[2] for h in hits]),
            }
    return out


# -- Q2 ----------------------------


def _ordered(views: list[RunView], config: str, order: list[str] | None) -> list[list[RunView]]:
    """The runs of one configuration grouped by task, in sweep order."""
    by_task: dict[str, list[RunView]] = {}
    for v in views:
        if v.run.config == config and v.steps is not None:
            by_task.setdefault(v.run.task, []).append(v)
    tasks = [t for t in (order or sorted(by_task)) if t in by_task]
    tasks += [t for t in sorted(by_task) if t not in tasks]
    return [by_task[t] for t in tasks]


def q2_config(
    views: list[RunView], config: str, kinds: dict[str, str], order: list[str] | None
) -> dict[str, Any]:
    seen_prior: set[Call] = set()
    read_calls = repeated = unique_in_task = 0
    repeat_cost = total_cost = 0.0
    for group in _ordered(views, config, order):
        task_calls: set[Call] = set()
        for v in group:
            assert v.steps is not None
            for s in v.steps:
                total_cost += s.cost
                n = len(s.calls) or 1
                for c in s.calls:
                    if kinds.get(c.name) != "READ":
                        continue
                    read_calls += 1
                    first_in_task = c not in task_calls
                    task_calls.add(c)
                    if first_in_task:
                        unique_in_task += 1
                        if c in seen_prior:
                            repeated += 1
                            repeat_cost += s.cost / n
        seen_prior |= task_calls
    return {
        "config": config,
        "read_calls": read_calls,
        "repeated": repeated,
        "unique_in_task": unique_in_task,
        "share_calls": repeated / read_calls if read_calls else 0.0,
        "share_calls_of_unique": repeated / unique_in_task if unique_in_task else 0.0,
        "repeat_cost": repeat_cost,
        "total_cost": total_cost,
        "share_cost": repeat_cost / total_cost if total_cost else 0.0,
    }


_DOC_ID = re.compile(r"^   ID: (\S+)", re.MULTILINE)


def q2_result_sets(views: list[RunView], config: str, order: list[str] | None) -> dict[str, Any]:
    """Share of KB_search/grep results whose set of document ids was already returned for an
    earlier task of the same configuration (results with no document are not counted)."""
    seen_prior: set[frozenset[str]] = set()
    total = repeated = 0
    for group in _ordered(views, config, order):
        task_sets: set[frozenset[str]] = set()
        for v in group:
            assert v.run.trace is not None and v.steps is not None
            names = {
                cid: c.name for s in v.steps for cid, c in zip(s.call_ids, s.calls, strict=True)
            }
            for m in v.run.trace["messages"]:
                if (
                    m["role"] != "tool"
                    or m.get("error")
                    or names.get(m["id"]) not in RETRIEVAL_TOOLS
                ):
                    continue
                ids = frozenset(_DOC_ID.findall(m.get("content") or ""))
                if not ids:
                    continue
                total += 1
                first_in_task = ids not in task_sets
                task_sets.add(ids)
                if first_in_task and ids in seen_prior:
                    repeated += 1
        seen_prior |= task_sets
    return {
        "config": config,
        "results": total,
        "repeated": repeated,
        "share": repeated / total if total else 0.0,
    }


def q2_pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reads = sum(r["read_calls"] for r in rows)
    rep = sum(r["repeated"] for r in rows)
    cost = sum(r["total_cost"] for r in rows)
    rcost = sum(r["repeat_cost"] for r in rows)
    return {
        "read_calls": reads,
        "repeated": rep,
        "share_calls": rep / reads if reads else 0.0,
        "share_cost": rcost / cost if cost else 0.0,
    }


def q2_verdict(pooled: dict[str, Any]) -> str:
    ok = pooled["share_calls"] >= Q2_CALL_BAR and pooled["share_cost"] >= Q2_COST_BAR
    return "PASS" if ok else "FAIL"


# -- Q3 ----------------------------


def signature(v: RunView) -> tuple[str, ...]:
    names = [c.name for s in (v.steps or []) for c in s.calls][:3]
    return tuple(names)


def _loo_accuracy(sigs: list[tuple[str, ...]], tasks: list[str], labels: list[bool]) -> float:
    correct = 0
    for t in set(tasks):
        train = [i for i, x in enumerate(tasks) if x != t]
        test = [i for i, x in enumerate(tasks) if x == t]
        overall = sum(labels[i] for i in train) * 2 >= len(train)
        table: dict[tuple[str, ...], list[bool]] = {}
        for i in train:
            table.setdefault(sigs[i], []).append(labels[i])
        for i in test:
            votes = table.get(sigs[i])
            pred = (sum(votes) * 2 > len(votes)) if votes else overall
            correct += pred == labels[i]
    return correct / len(labels)


def q3_early(views: list[RunView]) -> dict[str, Any]:
    vs = [v for v in views if v.steps is not None]
    labels = [v.success for v in vs]
    n_ok, n_fail = sum(labels), len(labels) - sum(labels)
    if n_ok < 3 or n_fail < 3:
        return {
            "runs": len(vs),
            "successes": n_ok,
            "failures": n_fail,
            "enough": False,
            "note": "fewer than 3 successful or fewer than 3 failed runs: nothing can be said",
        }
    sigs = [signature(v) for v in vs]
    tasks = [v.run.task for v in vs]
    real = _loo_accuracy(sigs, tasks, labels)
    rng = random.Random(NULL_SEED)
    null = []
    shuffled = list(labels)
    for _ in range(NULL_SHUFFLES):
        rng.shuffle(shuffled)
        null.append(_loo_accuracy(sigs, tasks, shuffled))
    majority = max(n_ok, n_fail) / len(labels)
    rank = (1 + sum(1 for x in null if x >= real)) / (1 + NULL_SHUFFLES)
    return {
        "runs": len(vs),
        "successes": n_ok,
        "failures": n_fail,
        "enough": True,
        "accuracy": real,
        "majority_baseline": majority,
        "null_mean": sum(null) / len(null),
        "null_max": max(null),
        "rank": rank,
        "distinct_signatures": len(set(sigs)),
    }


# -- one test, end to end ----------------------------


@dataclass
class TestResult:
    name: str
    configs: tuple[str, ...]
    n_runs: int
    n_excluded: int  # runs not finished (infrastructure errors)
    n_with_steps: int
    q1: list[dict[str, Any]]
    base_threshold: float | None
    rules: dict[float, list[dict[str, Any]]] = field(default_factory=dict)
    q1_verdict: dict[str, Any] | None = None
    signals: dict[str, dict[str, Any]] | None = None
    q2: list[dict[str, Any]] | None = None
    q2_pool: dict[str, Any] | None = None
    q2_verdict: str | None = None
    q2_sets: list[dict[str, Any]] | None = None
    q3: dict[str, Any] | None = None
    missing_traces: list[str] = field(default_factory=list)
    trace_sources: dict[str, int] = field(default_factory=dict)


def analyze_test(
    name: str,
    runs_dir: Path,
    trace_dirs: list[Path],
    kinds: dict[str, str],
    configs: tuple[str, ...],
    order: list[str] | None,
    banking: bool,
) -> TestResult:
    all_runs = load_runs(runs_dir, trace_dirs)
    runs = [r for r in all_runs if r.status in COMPLETE]
    views = make_views(runs)
    have = [v for v in views if v.steps is not None]
    exact = len(have) == len(views) and bool(views)
    res = TestResult(
        name=name,
        configs=configs,
        n_runs=len(runs),
        n_excluded=len(all_runs) - len(runs),
        n_with_steps=len(have),
        q1=q1_table(views, configs),
        base_threshold=threshold_base(views),
        missing_traces=[f"{v.run.config}/{v.run.task}" for v in views if v.steps is None],
        trace_sources={k: sum(1 for r in runs if r.trace_source == k) for k in
                       sorted({r.trace_source for r in runs})},
    )  # fmt: skip
    if res.base_threshold is not None:
        for m in MULTIPLIERS:
            res.rules[m] = stop_rule(views, m, exact, configs)
        res.q1_verdict = q1_verdict({m: rows[-1] for m, rows in res.rules.items()}, exact)
    if exact:
        res.signals = signal_summary(views)
        res.q2 = [q2_config(views, c, kinds, order) for c in configs]
        res.q2_pool = q2_pooled(res.q2)
        res.q2_verdict = q2_verdict(res.q2_pool)
        if banking:
            res.q2_sets = [q2_result_sets(views, c, order) for c in configs]
        res.q3 = q3_early(views)
    return res


# -- report ------------------------------------------------------------------------------------------

PINNED = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
NOT_COMPUTED = "NOT COMPUTED: the sanitized traces of this test were not available"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.2f}"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def test_section(t: TestResult) -> list[str]:
    exact = t.n_with_steps == t.n_runs and t.n_runs > 0
    L = [f"## {t.name}", ""]
    L.append(
        f"Finished runs: {t.n_runs} ({t.n_with_steps} with a usable sanitized trace); "
        f"left out as infrastructure errors: {t.n_excluded}. Trace sources: {t.trace_sources}."
    )
    L += ["", "### Q1. Cost of failed runs", ""]
    L += table(
        ["config", "runs", "successes", "failed runs", "total agent cost", "cost of failed", "share"],
        [
            [r["config"], str(r["runs"]), str(r["successes"]), str(r["failures"]),
             usd(r["total_cost"]), usd(r["failed_cost"]), pct(r["failed_share"])]
            for r in t.q1
        ],
    )  # fmt: skip
    L.append("")
    if t.base_threshold is None:
        L.append(
            "No successful `default` run: the threshold of R(m) is undefined, R(m) not computed."
        )
    else:
        mode = (
            "exact (per-call costs from traces)"
            if exact
            else (
                "UPPER BOUND from summaries (per-call costs unknown): each stopped run is assumed to "
                "cross the threshold exactly at the stop"
            )
        )
        L.append(
            f"Stop rule R(m), mode: {mode}. Base = median agent cost of successful `default` runs "
            f"= {usd(t.base_threshold)}."
        )
        L.append("")
        rows = []
        for m, rr in t.rules.items():
            r = rr[-1]
            rows.append([
                f"{m:g}", usd(r["threshold"]), str(r["stopped_failed"]), usd(r["saved_failed"]),
                f"{r['wrongly_stopped']} of {r['successes']}", usd(r["sunk_wrong"]),
                usd(r["net"]), pct(r["net_share"]),
            ])  # fmt: skip
        L += table(
            [
                "m",
                "threshold",
                "failed stopped",
                "saved on failed",
                "successes wrongly stopped",
                "sunk on those",
                "net saving",
                "net / total cost",
            ],
            rows,
        )
        L.append("")
        if t.q1_verdict:
            for m, x in t.q1_verdict["per_m"].items():
                L.append(
                    f"- m={m:g}: net >= {pct(Q1_NET_BAR)}: {x['net_ok']}; wrongly stopped <= 1 per 10 "
                    f"successes: {x['wrong_ok']}"
                )
            L.append(f"- **Q1 for this test: {t.q1_verdict['status']}**")
        L.append("")
        L.append("Per configuration (same threshold):")
        L.append("")
        rows = []
        for m, rr in t.rules.items():
            for r in rr[:-1]:
                rows.append([f"{m:g}", r["config"], str(r["stopped_failed"]),
                             f"{r['wrongly_stopped']} of {r['successes']}", pct(r["net_share"])])  # fmt: skip
        L += table(["m", "config", "failed stopped", "wrongly stopped", "net / cost"], rows)
    L.append("")
    if t.signals is None:
        L += [f"Signals (a) repeated identical call, (b) tool error: {NOT_COMPUTED}.", ""]
    else:
        L += ["First-occurrence signals (step = index of the agent LLM call):", ""]
        L += table(
            ["signal / outcome", "runs", "with signal", "share", "median step",
             "median step fraction", "median cost fraction"],
            [[k, str(v["runs"]), str(v["with_signal"]), pct(v["share"]),
              "n/a" if v["median_step"] is None else f"{v['median_step']:g}",
              pct(v["median_step_fraction"]), pct(v["median_cost_fraction"])]
             for k, v in t.signals.items()],
        )  # fmt: skip
        L.append("")
    L += ["### Q2. Repetition of read-only calls across tasks", ""]
    if t.q2 is None or t.q2_pool is None:
        L += [f"{NOT_COMPUTED}.", ""]
    else:
        L += table(
            ["config", "read-only calls", "cross-task repeats", "share of calls", "share of agent cost"],
            [[r["config"], str(r["read_calls"]), str(r["repeated"]), pct(r["share_calls"]),
              pct(r["share_cost"])] for r in t.q2],
        )  # fmt: skip
        p = t.q2_pool
        L += [
            "",
            f"Pooled: {p['repeated']} of {p['read_calls']} read-only calls = {pct(p['share_calls'])} "
            f"(bar {pct(Q2_CALL_BAR)}); {pct(p['share_cost'])} of agent cost (bar {pct(Q2_COST_BAR)}).",
            f"**Q2 for this test: {t.q2_verdict}**",
            "",
        ]
        if t.q2_sets:
            L += ["Retrieval results (KB_search / grep) whose document-id set was already returned "
                  "for an earlier task:", ""]  # fmt: skip
            L += table(["config", "results with documents", "repeated sets", "share"],
                       [[r["config"], str(r["results"]), str(r["repeated"]), pct(r["share"])]
                        for r in t.q2_sets])  # fmt: skip
            L.append("")
    L += ["### Q3. Early signal from the first three tool-call names (informational)", ""]
    if t.q3 is None:
        L += [f"{NOT_COMPUTED}.", ""]
    elif not t.q3["enough"]:
        L += [
            f"{t.q3['note']} ({t.q3['successes']} successful, {t.q3['failures']} failed runs).",
            "",
        ]
    else:
        q = t.q3
        L += [
            f"{q['runs']} runs ({q['successes']} successful, {q['failures']} failed), "
            f"{q['distinct_signatures']} distinct signatures. Leave-one-task-out accuracy "
            f"{pct(q['accuracy'])}; always-majority baseline {pct(q['majority_baseline'])}; "
            f"shuffled-label null mean {pct(q['null_mean'])}, max {pct(q['null_max'])}; "
            f"rank of the real accuracy among {NULL_SHUFFLES} shuffles: p = {q['rank']:.3f}.",
            "",
        ]
    return L


def answer_lines(results: list[TestResult]) -> list[str]:
    out = []
    for r in results:
        q1 = r.q1_verdict["status"] if r.q1_verdict else "not computed"
        q2 = r.q2_verdict or "NOT COMPUTED (needs the sanitized traces)"
        out.append(f"- {r.name}: Q1 = {q1}; Q2 = {q2}; Q3 = informational.")
    out.append(
        "- Q1 without traces is an upper bound on the net saving: a run can only be stopped if its "
        "total cost exceeds the threshold, which the summaries give exactly. A FAIL under the bound "
        "is therefore a real FAIL; a pass under the bound would need the traces to confirm."
    )
    return out


def build_report(results: list[TestResult], blocked: list[dict[str, str]]) -> str:
    L = ["# Failure anatomy (free analysis, no API call)", ""]
    L += [
        f"Pinned tau2-bench commit: `{PINNED}`. Models: Sonnet 5.5 (agent), Haiku 5.5 (user simulator, "
        "excluded from every number below). Reward: database state (`EvaluationType.ENV`). "
        "Cost spent on this analysis: $0.00 (no API call, no ledger, `budget.json` untouched).",
        "",
        "## Artifacts and sources",
        "",
        "Run summaries come from `results/phase-2/runs` (test 1, retail train sweep: 21 tasks x 4 "
        "configurations) and `results/banking-phase-2/runs` (test 2, banking_knowledge: 12 tasks x 4 "
        "configurations, ACTION tasks excluded). Per-call costs, tool calls and tool results come from "
        "the sanitized trace artifacts:",
        "",
    ]
    L += table(
        ["artifact", "state"],
        [[b["artifact"], b["state"]] for b in blocked] or [["(none needed)", "all traces read"]],
    )
    L += [
        "",
        "Runs with a usable sanitized trace: "
        + "; ".join(f"{r.name.split(',')[0]}: {r.n_with_steps} of {r.n_runs}" for r in results)
        + ".",
        "",
    ]
    L += [
        "## Definitions and bars (fixed before looking at any data)",
        "",
        *[ln for ln in (__doc__ or "").split("\n")[3:]],
        "",
    ]
    L += ["# Results", ""]
    for r in results:
        L += test_section(r)
    L += ["## Q1 and Q2 against the fixed bars", ""]
    rows = []
    for r in results:
        rows.append([r["name"] if isinstance(r, dict) else r.name,
                     r.q1_verdict["status"] if r.q1_verdict else "not computed",
                     r.q2_verdict or "NOT COMPUTED (traces unavailable)"])  # fmt: skip
    L += table(["test", "Q1", "Q2"], rows)
    L += [
        "",
        "Q3 is informational and has no bar. No verdict is drawn here beyond the pass / fail above.",
        "",
        "## What this cannot show",
        "",
        '- Banking `default` solved only 2 of 12 tasks. "Failed" is the normal outcome there, so the '
        "failed-run share is high by construction and says little about a well-tuned production agent.",
        "- Sample sizes are small: 21 retail and 12 banking tasks, 4 configurations each. A threshold "
        "based on the median successful `default` run rests on a handful of runs (see the counts "
        "above); one run moves it.",
        "- Benchmark agents are cleaner than production agents: no stale caches, no flaky tools, no "
        "user who changes the request, one fixed policy. Repetition and failure patterns in "
        "production can be very different, in either direction.",
        "- The four configurations are different policies of the same model, not independent "
        "customers. Pooling them makes the stop rule look more stable than it is.",
        "- A success here is the database-state reward only; what the agent said to the user is not "
        "checked.",
        "- Nothing here says a company would pay for any of this.",
        "",
    ]
    return "\n".join(L)


# -- CLI ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--retail-traces", type=Path, nargs="*", default=[])
    ap.add_argument("--banking-traces", type=Path, nargs="*", default=[])
    ap.add_argument("--blocked", type=str, nargs="*", default=[], help="artifact=host that failed")
    ap.add_argument("--out", type=Path, default=Path("reports/failure-anatomy.md"))
    a = ap.parse_args(argv)
    orders = json.loads((a.results / "task_orders.json").read_text())
    retail = analyze_test(
        "Test 1: retail, 21 train tasks, Sonnet 5.5",
        a.results / "phase-2" / "runs",
        a.retail_traces,
        json.loads((a.results / "phase-2" / "tool_kinds.json").read_text()),
        ("default", "E:medium", "E:low", "T:bt"),
        orders["retail_train_sweep"],
        banking=False,
    )
    bank = analyze_test(
        "Test 2: banking_knowledge (bm25_grep), 12 train tasks, Sonnet 5.5",
        a.results / "banking-phase-2" / "runs",
        a.banking_traces,
        json.loads((a.results / "banking-phase-2" / "tool_kinds.json").read_text()),
        ("default", "E:low", "C:top3", "EC"),
        orders["banking_sweep"],
        banking=True,
    )
    blocked = [{"artifact": b, "test": "-", "state": "download blocked"} for b in a.blocked]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build_report([retail, bank], blocked) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
