"""Sweep analysis: anatomy, paired success, selection rule, out-of-sample headroom and null."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from costlayer.optimizer.data import CONFIGS, Run, index
from costlayer.pricing import SONNET, price_for

NULL_SHUFFLES = 200
NULL_SEED = 42


def acceptable(failures: dict[str, int], n: int) -> list[str]:
    """Configurations whose failures <= default's failures + floor(0.10 n) (dict order kept)."""
    limit = failures["default"] + math.floor(0.10 * n)
    return [c for c in failures if failures[c] <= limit]


def totals(
    runs: dict[tuple[str, str], Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> tuple[dict, dict]:
    fail = {c: sum(0 if runs[(t, c)].success else 1 for t in tasks) for c in configs}
    cost = {c: sum(runs[(t, c)].cost_agent for t in tasks) for c in configs}
    return fail, cost


def cheapest_acceptable(
    runs: dict[tuple[str, str], Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> str:
    fail, cost = totals(runs, tasks, configs)
    return min(acceptable(fail, len(tasks)), key=lambda c: (cost[c], configs.index(c)))


def first_write_tool(trace: dict[str, Any], kinds: dict[str, str]) -> str | None:
    for m in trace["messages"]:
        for c in m.get("tool_calls") or []:
            if kinds.get(c["name"]) == "WRITE":
                return c["name"]
    return None


def task_type(
    runs: dict[tuple[str, str], Run],
    task: str,
    kinds: dict[str, str],
    configs: tuple[str, ...] = CONFIGS,
) -> str:
    """Most common first write tool over the task's traces ('none' if no write); ties go to
    the `default` trace's choice."""
    picks = {}
    for c in configs:
        r = runs[(task, c)]
        picks[c] = (first_write_tool(r.trace, kinds) if r.trace else None) or "none"
    counts = Counter(picks.values())
    top = max(counts.values())
    best = [t for t, n in counts.items() if n == top]
    return picks["default"] if picks["default"] in best else sorted(best)[0]


def choose_for_type(
    runs: dict[tuple[str, str], Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> str:
    """Per-type choice; a type with fewer than 3 tasks gets `default`."""
    if len(tasks) < 3:
        return "default"
    return cheapest_acceptable(runs, tasks, configs)


def oos_headroom(
    runs: dict[tuple[str, str], Run],
    tasks: list[str],
    types: dict[str, str],
    global_star: str,
    configs: tuple[str, ...] = CONFIGS,
) -> dict[str, Any]:
    """Leave-one-task-out: choose the config for the held-out task's type from the OTHER tasks of
    that type, take the measured cost on the held-out task, compare with `global*`."""
    chosen_cost = 0.0
    chosen_fail = 0
    picks = {}
    for t in tasks:
        others = [u for u in tasks if u != t and types[u] == types[t]]
        pick = choose_for_type(runs, others, configs)
        picks[t] = pick
        chosen_cost += runs[(t, pick)].cost_agent
        chosen_fail += 0 if runs[(t, pick)].success else 1
    base = sum(runs[(t, global_star)].cost_agent for t in tasks)
    return {
        "headroom": 1 - chosen_cost / base if base else 0.0,
        "chosen_cost": chosen_cost,
        "global_cost": base,
        "chosen_failures": chosen_fail,
        "picks": picks,
    }


def null_headroom(
    runs: dict[tuple[str, str], Run],
    tasks: list[str],
    types: dict[str, str],
    global_star: str,
    configs: tuple[str, ...] = CONFIGS,
) -> list[float]:
    rng = random.Random(NULL_SEED)
    labels = [types[t] for t in tasks]
    out = []
    for _ in range(NULL_SHUFFLES):
        rng.shuffle(labels)
        shuffled = dict(zip(tasks, labels, strict=True))
        out.append(oos_headroom(runs, tasks, shuffled, global_star, configs)["headroom"])
    return out


def oracle_headroom(
    runs: dict[tuple[str, str], Run],
    tasks: list[str],
    global_star: str,
    configs: tuple[str, ...] = CONFIGS,
) -> float:
    """Upper bound only: the cheapest acceptable-success configuration per task (biased low)."""
    best = 0.0
    for t in tasks:
        ok = [c for c in configs if runs[(t, c)].success or not runs[(t, "default")].success]
        best += min(runs[(t, c)].cost_agent for c in ok)
    base = sum(runs[(t, global_star)].cost_agent for t in tasks)
    return 1 - best / base if base else 0.0


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(q * len(xs)) - 1)]


def anatomy(
    runs: list[Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> dict[str, dict[str, Any]]:
    """Cost shares by token type and per-call output/thinking for each configuration."""
    p = price_for(SONNET)
    idx = index(runs)
    out: dict[str, dict[str, Any]] = {}
    for c in configs:
        rs = [idx[(t, c)] for t in tasks]
        tok = {
            k: sum(r.tokens[k] for r in rs)
            for k in ("input", "cache_write", "cache_read", "output")
        }
        parts = {
            "cache read": tok["cache_read"] * p.cache_read / 1e6,
            "cache write": tok["cache_write"] * p.cache_write / 1e6,
            "uncached input": tok["input"] * p.input / 1e6,
            "output": tok["output"] * p.output / 1e6,
        }
        calls = sum(r.llm_calls for r in rs) or 1
        think = tvis = 0
        have = 0
        for r in rs:
            if r.trace:
                d = call_decomposition(r.trace)
                think += d["thinking"]
                tvis += d["visible_est"]
                have += d["calls"]
        out[c] = {
            "runs": len(rs),
            "mean_agent_cost": sum(r.cost_agent for r in rs) / len(rs),
            "mean_user_cost": sum(r.cost_user for r in rs) / len(rs),
            "calls_per_run": calls / len(rs),
            "shares": {k: v / (sum(parts.values()) or 1) for k, v in parts.items()},
            "output_per_call": tok["output"] / calls,
            "thinking_per_call": think / have if have else None,
            "visible_est_per_call": tvis / have if have else None,
        }
    return out


def call_decomposition(trace: dict[str, Any]) -> dict[str, int]:
    """Sum over the agent's calls of: output tokens, API-reported thinking tokens, and the
    estimated visible output (text plus tool-call JSON, ~3.5 chars per token)."""
    import json

    calls = out = think = vis = 0
    for m in trace["messages"]:
        if m["role"] != "assistant" or not m.get("usage"):
            continue
        calls += 1
        out += m["usage"].get("output", 0)
        think += m["usage"].get("thinking_tokens") or 0
        chars = len(m.get("content") or "")
        for c in m.get("tool_calls") or []:
            chars += len(json.dumps({"name": c["name"], "input": c["arguments"]}))
        vis += -(-chars * 2 // 7)
    return {"calls": calls, "output": out, "thinking": think, "visible_est": vis}


def paired_success(
    runs: list[Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> dict[str, Any]:
    idx = index(runs)
    res: dict[str, Any] = {}
    for c in configs:
        s = sum(idx[(t, c)].success for t in tasks)
        res[c] = {"successes": s, "n": len(tasks)}
        if c != "default":
            res[c]["lost_vs_default"] = [
                t for t in tasks if idx[(t, "default")].success and not idx[(t, c)].success
            ]
            res[c]["won_vs_default"] = [
                t for t in tasks if not idx[(t, "default")].success and idx[(t, c)].success
            ]
    return res


def config_table(
    runs: list[Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> dict[str, dict[str, Any]]:
    """Per configuration, over the paired tasks: cost next to successes, failures, the mean
    number of LLM calls per run and the mean number of calls of the FAILED runs (a cheap
    configuration that ends runs early shows up as fewer calls and fewer successes)."""
    idx = index(runs)
    out: dict[str, dict[str, Any]] = {}
    for c in configs:
        rs = [idx[(t, c)] for t in tasks]
        failed = [r for r in rs if not r.success]
        out[c] = {
            "runs": len(rs),
            "successes": len(rs) - len(failed),
            "failures": len(failed),
            "mean_agent_cost": sum(r.cost_agent for r in rs) / len(rs),
            "total_agent_cost": sum(r.cost_agent for r in rs),
            "mean_calls": sum(r.llm_calls for r in rs) / len(rs),
            "mean_calls_failed": (sum(r.llm_calls for r in failed) / len(failed))
            if failed
            else None,
        }
    return out


def savings_flags(table: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For each non-default configuration: is it cheaper than `default`, does it make fewer
    LLM calls, does it have fewer successes, and do all three coincide?"""
    d = table["default"]
    flags = []
    for c, row in table.items():
        if c == "default":
            continue
        cheaper = row["total_agent_cost"] < d["total_agent_cost"]
        fewer_calls = row["mean_calls"] < d["mean_calls"]
        fewer_successes = row["successes"] < d["successes"]
        flags.append(
            {
                "config": c,
                "cheaper": cheaper,
                "fewer_calls": fewer_calls,
                "fewer_successes": fewer_successes,
                "coincides": cheaper and fewer_calls and fewer_successes,
            }
        )
    return flags


def retrieval_stats(
    runs: list[Run], tasks: list[str], configs: tuple[str, ...] = CONFIGS
) -> dict[str, dict[str, float]]:
    """Per configuration: retrieval results per run and their size (estimated tokens, from the
    sizes the run recorded at the time) as they entered the conversation and as the tool returned
    them. Zeros for domains that do not record them (retail)."""
    idx = index(runs)
    out = {}
    for c in configs:
        rs = [idx[(t, c)].summary for t in tasks]
        n = len(rs)
        out[c] = {
            "results": sum(s.get("retrieval_results", 0) for s in rs) / n,
            "in_tokens": sum(s.get("retrieval_chars_in_context", 0) for s in rs) / n / 3.5,
            "raw_tokens": sum(s.get("retrieval_chars_raw", 0) for s in rs) / n / 3.5,
        }
    return out
