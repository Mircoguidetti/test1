"""Cost per completed task, implementation B: an independent second computation.

It shares only the definitions (written in `cps.py`) and the random draws (`cps_rng`). The data
path differs (the summaries are read through `costlayer.optimizer.data.load_runs`, i.e. `Run`
objects) and so does the arithmetic: the bootstrap works on multiplicity counts instead of
re-summing resampled lists, percentiles come from `statistics.quantiles`, and the leave-one-out
choice is a `min` over (value, position) keys. `compare` fails loudly on any difference.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from costlayer.optimizer import cps_rng
from costlayer.optimizer.data import Run, load_runs

FINISHED = ("ok", "agent_error", "censored")


def per_task(
    runs_dir: Path,
    configs: tuple[str, ...],
    order: list[str] | None,
    only: set[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, tuple[float, int, int, int, int]]]]:
    """tasks, and {config: {task: (cost, successes, runs)}} over finished runs of all configs."""
    finished = [r for r in load_runs(Path(runs_dir), []) if r.status in FINISHED]
    by: dict[tuple[str, str], list[Run]] = {}
    for r in finished:
        by.setdefault((r.config, r.task), []).append(r)
    present = {c: {t for (cc, t) in by if cc == c} for c in configs}
    common = set.intersection(*present.values())
    if only is not None:
        common &= only
    tasks = [t for t in (order or []) if t in common] + sorted(common - set(order or []))
    out = {
        c: {
            t: (
                sum(r.cost_agent for r in by[(c, t)]),
                sum(1 for r in by[(c, t)] if r.success),
                len(by[(c, t)]),
                sum(r.llm_calls for r in by[(c, t)]),
                sum(r.llm_calls for r in by[(c, t)] if not r.success),
            )
            for t in tasks
        }
        for c in configs
    }
    return tasks, out


def deciles(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (math.nan, math.nan)
    q = statistics.quantiles(xs, n=10, method="inclusive")
    return q[0], q[-1]


def ratio(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def summarize(tasks: list[str], d: dict[str, dict[str, tuple[float, int, int, int, int]]], configs):
    n = len(tasks)
    tot = {c: tuple(sum(d[c][t][k] for t in tasks) for k in range(5)) for c in configs}
    base_cost, base_succ = tot["default"][:2]
    table: dict[str, dict[str, Any]] = {}
    for c in configs:
        cost, succ, runs, calls, fcalls = tot[c]
        cps = ratio(cost, succ)
        bcps = ratio(base_cost, base_succ)
        table[c] = {
            "runs": runs,
            "successes": succ,
            "failures": runs - succ,
            "cost": cost,
            "cost_per_run": cost / runs,
            "success_rate": succ / runs,
            "cps": cps,
            "cps_ratio": None if cps is None or bcps is None else cps / bcps,
            "cost_ratio": cost / base_cost,
            "calls_per_run": calls / runs,
            "calls_per_failed_run": ratio(fcalls, runs - succ),
        }
    limit = table["default"]["failures"] + n // 10
    ok = [c for c in configs if table[c]["failures"] <= limit]

    def best(key: str) -> str | None:
        cand = [(table[c][key], i, c) for i, c in enumerate(configs) if c in ok]
        cand = [x for x in cand if x[0] is not None]
        return min(cand)[2] if cand else None

    return table, ok, best("cps"), best("cost_per_run")


def boot(tasks, d, configs) -> dict[str, dict[str, dict[str, Any]]]:
    draws = cps_rng.resamples(len(tasks))
    cols: dict[str, dict[str, list[float | None]]] = {
        c: {k: [] for k in ("cost_per_run", "success_rate", "cps", "cps_ratio", "cost_ratio")}
        for c in configs
    }
    for idx in draws:
        w = Counter(idx)
        agg = {
            c: tuple(sum(m * d[c][tasks[i]][k] for i, m in w.items()) for k in range(3))
            for c in configs
        }
        bcost, bsucc, _ = agg["default"]
        for c in configs:
            cost, succ, runs = agg[c]
            cols[c]["cost_per_run"].append(cost / runs)
            cols[c]["success_rate"].append(succ / runs)
            cols[c]["cps"].append(ratio(cost, succ))
            if c != "default":
                cols[c]["cost_ratio"].append(cost / bcost)
                a, b = ratio(cost, succ), ratio(bcost, bsucc)
                cols[c]["cps_ratio"].append(None if a is None or b is None else a / b)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for c in configs:
        out[c] = {}
        for k, vals in cols[c].items():
            if not vals:
                continue
            ok = [v for v in vals if v is not None]
            lo, hi = deciles(ok) if ok else (None, None)
            out[c][k] = {
                "low": lo,
                "high": hi,
                "defined": len(ok),
                "undefined": len(vals) - len(ok),
            }
    return out


def choose(tasks, d, configs, group: list[int], metric: str) -> str:
    stats = {
        c: (
            sum(d[c][tasks[i]][0] for i in group),
            sum(d[c][tasks[i]][1] for i in group),
            sum(d[c][tasks[i]][2] for i in group),
        )
        for c in configs
    }
    limit = (stats["default"][2] - stats["default"][1]) + len(group) // 10
    cand = []
    for i, c in enumerate(configs):
        cost, succ, runs = stats[c]
        if runs - succ > limit:
            continue
        v = cost if metric == "cost" else ratio(cost, succ)
        if v is not None:
            cand.append((v, i, c))
    return min(cand)[2] if cand else "default"


def head(tasks, d, configs, labels: list[str], gstar: str, metric: str) -> float | None:
    total_cost = total_succ = 0.0
    for h, t in enumerate(tasks):
        grp = [i for i in range(len(tasks)) if labels[i] == labels[h] and i != h]
        c = choose(tasks, d, configs, grp, metric) if len(grp) >= 3 else "default"
        total_cost += d[c][t][0]
        total_succ += d[c][t][1]
    g_cost = sum(d[gstar][t][0] for t in tasks)
    g_succ = sum(d[gstar][t][1] for t in tasks)
    if metric == "cost":
        return 1 - total_cost / g_cost
    if total_succ == 0 or g_succ == 0:
        return None
    return 1 - (total_cost / total_succ) / (g_cost / g_succ)


def head_null(tasks, d, configs, types, gstar, metric) -> dict[str, Any]:
    labels = [types[t] for t in tasks]
    real = head(tasks, d, configs, labels, gstar, metric)
    null = [
        head(tasks, d, configs, [labels[p] for p in perm], gstar, metric)
        for perm in cps_rng.shuffles(len(tasks))
    ]
    ok = [v for v in null if v is not None]
    p90 = statistics.quantiles(ok, n=10, method="inclusive")[-1] if len(ok) > 1 else None
    return {
        "headroom": real,
        "null_mean": statistics.fmean(ok) if ok else None,
        "null_p90": p90,
        "null_undefined": len(null) - len(ok),
        "beats_null": real is not None and p90 is not None and real > p90,
    }


def analyze(runs_dir, configs, order, types, chain_share, only=None) -> dict[str, Any]:
    tasks, d = per_task(Path(runs_dir), configs, order, only)
    table, ok, g_cps, g_cost = summarize(tasks, d, configs)
    stacked = None
    if g_cps and table[g_cps]["cps_ratio"] is not None:
        stacked = (1 - chain_share) * table[g_cps]["cps_ratio"]
    return {
        "tasks": tasks,
        "table": table,
        "intervals": boot(tasks, d, configs),
        "acceptable": ok,
        "global_star_cps": g_cps,
        "global_star_cost": g_cost,
        "stacked_cps_ratio": stacked,
        "headroom_cost": head_null(tasks, d, configs, types, g_cost, "cost") if g_cost else None,
        "headroom_cps": head_null(tasks, d, configs, types, g_cps, "cps") if g_cps else None,
    }


def compare(a: Any, b: Any, path: str = "", tol: float = 1e-9) -> list[str]:
    """Every difference between two result trees (floats compared to `tol`; dollars also to the
    cent). Keys present only in A (descriptive fields) are ignored; keys only in B are reported."""
    diffs: list[str] = []
    if isinstance(b, dict) and isinstance(a, dict):
        for k in b:
            if k not in a:
                diffs.append(f"{path}/{k}: missing in A")
            else:
                diffs += compare(a[k], b[k], f"{path}/{k}", tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            diffs += compare(x, y, f"{path}[{i}]", tol)
    elif a is None or b is None:
        if a is not b:
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, float) or isinstance(b, float):
        if abs(float(a) - float(b)) > tol * max(1.0, abs(float(a))):
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")
    return diffs
