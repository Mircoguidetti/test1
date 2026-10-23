"""Group A levers (1-5): estimated from baseline traces, never part of the verdict."""

from __future__ import annotations

import hashlib
import statistics
from typing import Any

from costlayer.pricing import price_for


def estimate(
    base_runs: list[dict[str, Any]], first_messages: dict[str, str] | None = None
) -> dict[str, Any]:
    """`base_runs`: baseline run summaries with 'steps'. Returns USD estimates per lever."""
    if not base_runs:
        return {}
    model = base_runs[0]["model"]
    p = price_for(model)
    total = sum(r["cost_agent"] for r in base_runs)
    est: dict[str, Any] = {"baseline_agent_cost": round(total, 6), "n_runs": len(base_runs)}

    # 1. Cache-friendly order: a rolling breakpoint on the conversation, so history is read from
    # cache (0.1x) instead of paid fresh. Written increments cost 1.25x.
    saved = 0.0
    for r in base_runs:
        prev_msgs = 0
        for s in r["steps"]:
            fresh = s["usage"]["input"]
            est_total = max(1, s["messages_tokens_est"])
            scale = fresh / est_total if est_total else 1.0
            cached = min(fresh, prev_msgs * scale)
            new = fresh - cached
            saved += (fresh * p.input - cached * p.cache_read - new * p.cache_write) / 1e6
            prev_msgs = s["messages_tokens_est"]
    est["1_cache_order"] = {
        "usd": round(saved, 6),
        "pct": _pct(saved, total),
        "assumption": "rolling cache breakpoint on the last message",
    }

    # 2. Cache duration / duplicate writes: prefix written more than once per domain.
    dup = 0.0
    seen: dict[str, int] = {}
    for r in base_runs:
        first = r["steps"][0]["usage"] if r["steps"] else None
        if first and first["cache_write"] > 0:
            if r["domain"] in seen:
                dup += first["cache_write"] * (p.cache_write - p.cache_read) / 1e6
            seen[r["domain"]] = first["cache_write"]
    est["2_cache_ttl_duplicate_writes"] = {
        "usd": round(dup, 6),
        "pct": _pct(dup, total),
        "assumption": "each domain prefix written once and read afterwards (serialized or "
        "pre-warmed); 1h TTL costs 2x per write and only pays when gaps exceed 5 min",
    }

    # 3. Duplicate task detection: identical first customer messages, and repeated identical
    # tool calls inside one run (upper bound: the cost of one step per repeat).
    dups = 0
    if first_messages:
        hashes = [hashlib.sha256(m.encode()).hexdigest() for m in first_messages.values()]
        dups = len(hashes) - len(set(hashes))
    est["3_duplicate_tasks"] = {
        "usd": round(dups * (total / len(base_runs)), 6),
        "duplicates": dups,
        "assumption": "tau-bench has no real traffic; identical first messages are rare",
    }

    # 4. Batch API: 50% off for non-urgent work. tau-bench is interactive, so the base case is 0.
    est["4_batch_api"] = {
        "usd": 0.0,
        "scenarios_usd": {
            f"{int(s * 100)}%_non_urgent": round(0.5 * s * total, 6) for s in (0.1, 0.3, 0.5)
        },
        "assumption": "interactive conversations cannot be batched; scenarios show the "
        "50% discount on the share of work that could be",
    }

    # 5. Budget per task and loop stop: cap a run at 2x the median successful cost; failed runs
    # beyond the cap are assumed lost anyway.
    ok = [r["cost_agent"] for r in base_runs if r["reward"] == 1.0]
    cap = 2 * statistics.median(ok) if ok else None
    s5 = 0.0
    if cap:
        s5 = sum(
            r["cost_agent"] - cap for r in base_runs if r["reward"] < 1.0 and r["cost_agent"] > cap
        )
    est["5_budget_and_loop_stop"] = {
        "usd": round(s5, 6),
        "pct": _pct(s5, total),
        "cap_usd_per_task": round(cap, 6) if cap else None,
        "assumption": "runs that fail and exceed 2x the median successful cost are stopped at "
        "the cap; successful runs are never cut",
    }
    return est


def _pct(x: float, total: float) -> float:
    return round(100 * x / total, 2) if total else 0.0
