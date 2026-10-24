"""Cost per completed task (CPS), implementation A: plain loops over the raw run-summary files.

Definitions (fixed before looking at any number; the report repeats them):

* Data: the per-run summaries in `results/<phase>/runs/*.json`, Sonnet 5.5 rows only.
* Cost of a run = agent (Sonnet) cost only. The user simulator and any classifier are excluded.
* Success = status `ok` and reward 1. Every other FINISHED run (`ok` with reward 0, `agent_error`,
  `censored`) is a failure and its cost stays in the numerator. `infra_error` runs are excluded
  from numerator and denominator and listed. Any other status is an error (nothing is guessed).
* CPS of a configuration = sum of the cost of ALL its finished runs / its number of successes.
  Zero successes: undefined (None), never infinity or zero. CPS ratio = CPS(config)/CPS(default).
* A task is analysed only if every configuration has a finished run on it (otherwise it is
  dropped for all configurations and listed).
* Intervals: bootstrap over TASKS (10,000 resamples, seed 42, `cps_rng`), the same resampled tasks
  for every configuration, every statistic recomputed from the resampled sums. 80% interval =
  10th and 90th percentile (linear interpolation) of the DEFINED resamples; the number of
  undefined resamples is reported, never dropped silently.
* global*: among configurations with failures <= default failures + floor(0.10 x n), the one
  with the lowest CPS (ties: first in the configuration order; undefined CPS never wins).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from costlayer.optimizer import cps_rng

SONNET = "claude-sonnet-5-5"
FINISHED = ("ok", "agent_error", "censored")
INFRA = "infra_error"
LOW_Q, HIGH_Q = 10, 90
STATS = ("cost_per_run", "success_rate", "cps", "cps_ratio", "cost_ratio")


@dataclass
class Cell:
    cost: float
    succ: int
    runs: int
    calls: int  # agent LLM calls over all its finished runs
    failed_calls: int  # ... over its failed runs only
    failed_runs: int


@dataclass
class TestData:
    name: str
    configs: tuple[str, ...]
    tasks: list[str]
    cells: dict[str, dict[str, Cell]]
    infra: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    files: int = 0
    other_model: int = 0


def is_success(s: dict[str, Any]) -> bool:
    return s["status"] == "ok" and float(s.get("reward") or 0.0) >= 1.0


def load_test(
    name: str, runs_dir: Path, configs: tuple[str, ...], order: list[str] | None = None
) -> TestData:
    per: dict[str, dict[str, list[dict[str, Any]]]] = {c: {} for c in configs}
    infra: list[str] = []
    files = other = 0
    for f in sorted(Path(runs_dir).glob("*.json")):
        s = json.loads(f.read_text())
        files += 1
        if s.get("model") != SONNET:
            other += 1
            continue
        if s["status"] == INFRA:
            infra.append(f"{s['config']}/{s['task']}/a{s['attempt']}")
            continue
        if s["status"] not in FINISHED:
            raise ValueError(f"{f.name}: unknown status {s['status']!r}")
        if s["config"] not in per:
            raise ValueError(f"{f.name}: unexpected configuration {s['config']!r}")
        per[s["config"]].setdefault(s["task"], []).append(s)
    every = set.intersection(*(set(per[c]) for c in configs)) if configs else set()
    union = set.union(*(set(per[c]) for c in configs)) if configs else set()
    tasks = [t for t in (order or sorted(every)) if t in every]
    tasks += [t for t in sorted(every) if t not in tasks]
    cells = {
        c: {
            t: Cell(
                math.fsum(r["cost_agent"] for r in per[c][t]),
                sum(1 for r in per[c][t] if is_success(r)),
                len(per[c][t]),
                sum(r["llm_calls"] for r in per[c][t]),
                sum(r["llm_calls"] for r in per[c][t] if not is_success(r)),
                sum(1 for r in per[c][t] if not is_success(r)),
            )
            for t in tasks
        }
        for c in configs
    }
    return TestData(name, configs, tasks, cells, sorted(infra), sorted(union - every), files, other)


# -- point estimates ------------------


def sums(data: TestData, config: str, idx: list[int] | None = None) -> tuple[float, int, int]:
    """(cost, successes, runs) of a configuration over the tasks (or the resampled positions)."""
    cells = [data.cells[config][t] for t in data.tasks]
    if idx is not None:
        cells = [cells[i] for i in idx]
    return math.fsum(c.cost for c in cells), sum(c.succ for c in cells), sum(c.runs for c in cells)


def statistics_of(
    cost: float, succ: int, runs: int, base: tuple[float, int, int] | None
) -> dict[str, float | None]:
    cps = cost / succ if succ else None
    out: dict[str, float | None] = {
        "cost_per_run": cost / runs if runs else None,
        "success_rate": succ / runs if runs else None,
        "cps": cps,
        "cps_ratio": None,
        "cost_ratio": None,
    }
    if base is not None:
        bcost, bsucc, _ = base
        bcps = bcost / bsucc if bsucc else None
        if cps is not None and bcps is not None:
            out["cps_ratio"] = cps / bcps
        if bcost:
            out["cost_ratio"] = cost / bcost
    return out


def point_table(data: TestData) -> dict[str, dict[str, Any]]:
    base = sums(data, "default")
    table = {}
    for c in data.configs:
        cost, succ, runs = sums(data, c)
        row: dict[str, Any] = {
            "runs": runs,
            "successes": succ,
            "failures": runs - succ,
            "cost": cost,
        }
        row.update(statistics_of(cost, succ, runs, base))
        cells = [data.cells[c][t] for t in data.tasks]
        fruns = sum(x.failed_runs for x in cells)
        row["calls_per_run"] = sum(x.calls for x in cells) / runs
        row["calls_per_failed_run"] = sum(x.failed_calls for x in cells) / fruns if fruns else None
        table[c] = row
    return table


def acceptable(table: dict[str, dict[str, Any]], n: int) -> list[str]:
    limit = table["default"]["failures"] + math.floor(0.10 * n)
    return [c for c, r in table.items() if r["failures"] <= limit]


def pick(table: dict[str, dict[str, Any]], n: int, key: str) -> str | None:
    """Cheapest acceptable configuration by `key` (cps or cost_per_run); None if none."""
    best: str | None = None
    for c in acceptable(table, n):
        v = table[c][key]
        if v is not None and (best is None or v < table[best][key]):
            best = c
    return best


# -- bootstrap ------------------


def percentile(sorted_xs: list[float], q: float) -> float:
    pos = (len(sorted_xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def interval(values: list[float | None]) -> dict[str, Any]:
    ok = sorted(v for v in values if v is not None)
    return {
        "low": percentile(ok, LOW_Q) if ok else None,
        "high": percentile(ok, HIGH_Q) if ok else None,
        "defined": len(ok),
        "undefined": len(values) - len(ok),
    }


def bootstrap(
    data: TestData, draws: list[list[int]] | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    draws = draws if draws is not None else cps_rng.resamples(len(data.tasks))
    acc: dict[str, dict[str, list[float | None]]] = {
        c: {s: [] for s in STATS} for c in data.configs
    }
    for idx in draws:
        base = sums(data, "default", idx)
        for c in data.configs:
            cost, succ, runs = sums(data, c, idx)
            st = statistics_of(cost, succ, runs, None if c == "default" else base)
            for s in STATS:
                acc[c][s].append(st[s])
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for c in data.configs:
        out[c] = {}
        for s in STATS:
            if c == "default" and s in ("cps_ratio", "cost_ratio"):
                continue
            out[c][s] = interval(acc[c][s])
    return out


# -- leave-one-task-out per-type headroom ------------------


def _group_choice(data: TestData, group: list[int], metric: str, order: tuple[str, ...]) -> str:
    """Cheapest acceptable configuration on a group of task positions (default if none)."""
    n = len(group)
    stats = {}
    for c in order:
        cells = [data.cells[c][data.tasks[i]] for i in group]
        runs = sum(x.runs for x in cells)
        succ = sum(x.succ for x in cells)
        stats[c] = (math.fsum(x.cost for x in cells), succ, runs - succ)
    limit = stats["default"][2] + math.floor(0.10 * n)
    best: str | None = None
    best_v = 0.0
    for c in order:
        cost, succ, fails = stats[c]
        if fails > limit:
            continue
        v = cost if metric == "cost" else (cost / succ if succ else None)
        if v is not None and (best is None or v < best_v):
            best, best_v = c, v
    return best or "default"


def loo_policy(
    data: TestData, labels: list[str], metric: str, min_group: int = 3
) -> tuple[float, int]:
    """Total (cost, successes) when every task uses the choice learned from the OTHER tasks of its
    type (a type with fewer than `min_group` other tasks gets `default`)."""
    total_cost = 0.0
    total_succ = 0
    costs = []
    for h, task in enumerate(data.tasks):
        group = [i for i in range(len(data.tasks)) if i != h and labels[i] == labels[h]]
        c = (
            _group_choice(data, group, metric, data.configs)
            if len(group) >= min_group
            else "default"
        )
        cell = data.cells[c][task]
        costs.append(cell.cost)
        total_succ += cell.succ
    total_cost = math.fsum(costs)
    return total_cost, total_succ


def headroom(data: TestData, labels: list[str], global_star: str, metric: str) -> float | None:
    """1 - policy / global* in the given metric ('cost': total cost; 'cps': cost per success)."""
    cost, succ = loo_policy(data, labels, metric)
    g_cost, g_succ, _ = sums(data, global_star)
    if metric == "cost":
        return 1 - cost / g_cost
    if not succ or not g_succ:
        return None
    return 1 - (cost / succ) / (g_cost / g_succ)


def headroom_with_null(
    data: TestData, types: dict[str, str], global_star: str, metric: str
) -> dict[str, Any]:
    labels = [types[t] for t in data.tasks]
    real = headroom(data, labels, global_star, metric)
    null = []
    for perm in cps_rng.shuffles(len(labels)):
        v = headroom(data, [labels[p] for p in perm], global_star, metric)
        null.append(v)
    ok = sorted(v for v in null if v is not None)
    p90 = percentile(ok, 90) if ok else None
    return {
        "headroom": real,
        "null_mean": math.fsum(ok) / len(ok) if ok else None,
        "null_p90": p90,
        "null_undefined": len(null) - len(ok),
        "beats_null": real is not None and p90 is not None and real > p90,
    }


# -- everything for one test ------------------


def analyze(data: TestData, types: dict[str, str], chain_share: float) -> dict[str, Any]:
    table = point_table(data)
    n = len(data.tasks)
    g_cps = pick(table, n, "cps")
    g_cost = pick(table, n, "cost_per_run")
    stack_ratio = None
    if g_cps is not None and table[g_cps]["cps_ratio"] is not None:
        stack_ratio = table[g_cps]["cps_ratio"] * (1 - chain_share)
    return {
        "name": data.name,
        "tasks": list(data.tasks),
        "infra_error": data.infra,
        "dropped_tasks": data.dropped,
        "files": data.files,
        "other_model_files": data.other_model,
        "table": table,
        "intervals": bootstrap(data),
        "acceptable": acceptable(table, n),
        "global_star_cps": g_cps,
        "global_star_cost": g_cost,
        "chain_share": chain_share,
        "stacked_cps_ratio": stack_ratio,
        "headroom_cost": headroom_with_null(data, types, g_cost, "cost") if g_cost else None,
        "headroom_cps": headroom_with_null(data, types, g_cps, "cps") if g_cps else None,
    }


def restrict(data: TestData, tasks: list[str]) -> TestData:
    """The same data on a subset of the tasks (used for sensitivity checks)."""
    keep = [t for t in data.tasks if t in set(tasks)]
    cells = {c: {t: data.cells[c][t] for t in keep} for c in data.configs}
    return TestData(data.name, data.configs, keep, cells, data.infra, data.dropped, data.files)
