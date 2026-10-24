"""Phase 2 analysis from sanitized traces: sweep, selection, headroom, mining, gate."""

from __future__ import annotations

from collections import Counter
from typing import Any

from costlayer.optimizer import mining, sweep
from costlayer.optimizer.data import CONFIGS, Run, index, paired_tasks
from costlayer.stats import ratio_ci

GATE_HEADROOM = 0.10
GATE_CALL_SHARE = 0.15
GATE_COST_SHARE = 0.10
CLASSIFIER_MIN = 0.70
MINING_SOURCES = ("seen", "uncut", "reconstructed")


def analyze(
    runs: list[Run], kinds: dict[str, str], configs: tuple[str, ...] = CONFIGS
) -> dict[str, Any]:
    tasks = paired_tasks(runs, configs)
    idx = index(runs)
    res: dict[str, Any] = {"n_tasks": len(tasks), "tasks": tasks}
    if len(tasks) < 3:
        res["error"] = "fewer than 3 complete tasks"
        return res
    res["anatomy"] = sweep.anatomy(runs, tasks, configs)
    res["success"] = sweep.paired_success(runs, tasks, configs)
    fail, cost = sweep.totals(idx, tasks, configs)
    res["failures"], res["cost_agent_total"] = fail, cost
    res["acceptable"] = sweep.acceptable(fail, len(tasks))
    gs = sweep.cheapest_acceptable(idx, tasks, configs)
    res["global_star"] = gs
    base = [idx[(t, "default")].cost_agent for t in tasks]
    res["cost_ratio_vs_default"] = {
        c: ratio_ci(base, [idx[(t, c)].cost_agent for t in tasks])
        for c in configs
        if c != "default"
    }
    types = {t: sweep.task_type(idx, t, kinds, configs) for t in tasks}
    res["types"] = types
    res["type_counts"] = dict(Counter(types.values()))
    res["per_type_choice"] = {
        ty: sweep.choose_for_type(idx, [t for t in tasks if types[t] == ty], configs)
        for ty in sorted(set(types.values()))
    }
    oos = sweep.oos_headroom(idx, tasks, types, gs, configs)
    null = sweep.null_headroom(idx, tasks, types, gs, configs)
    p90 = sweep.percentile(null, 0.9)
    res["headroom"] = {
        **{k: v for k, v in oos.items() if k != "picks"},
        "null_p90": p90,
        "null_mean": sum(null) / len(null),
        "beats_null": oos["headroom"] > p90,
        "oracle_upper_bound": sweep.oracle_headroom(idx, tasks, gs, configs),
    }
    res["table"] = sweep.config_table(runs, tasks, configs)
    res["savings_flags"] = sweep.savings_flags(res["table"])
    res["retrieval"] = sweep.retrieval_stats(runs, tasks, configs)
    # mining only on traces that show what the agent saw (see data.prepare_trace)
    gs_runs = [idx[(t, gs)] for t in tasks]
    res["mining_sources"] = dict(Counter(r.trace_source for r in gs_runs))
    gs_traces = [r.trace for r in gs_runs if r.trace and r.trace_source in MINING_SOURCES]
    res["chains"] = mining.chain_stats(gs_traces, kinds)
    res["fields"] = mining.field_usage(gs_traces, kinds)
    res["missing_traces"] = [f"{c}/{t}" for t in tasks for c in configs if not idx[(t, c)].trace]
    return res


def gate(res: dict[str, Any]) -> dict[str, Any]:
    """Go to phase 3 only if (a) or (b) holds (thresholds fixed in CLAUDE.md)."""
    if "error" in res:
        return {"a": False, "b": False, "pass": False, "why": res["error"]}
    h, c = res["headroom"], res["chains"]
    a = h["headroom"] >= GATE_HEADROOM and h["beats_null"]
    b = c["call_share"] >= GATE_CALL_SHARE and c["cost_share"] >= GATE_COST_SHARE
    return {
        "a": a,
        "b": b,
        "pass": a or b,
        "headroom": h["headroom"],
        "call_share": c["call_share"],
        "cost_share": c["cost_share"],
    }
