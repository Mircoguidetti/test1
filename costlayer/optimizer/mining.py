"""Mining of `global*` traces: dependent read-only chains (S1) and used tool-result fields (G3)."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

MIN_LEAF = 3  # shorter values (quantities, flags) are too ambiguous to prove provenance


def leaves(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [s for v in x.values() for s in leaves(v)]
    if isinstance(x, list):
        return [s for v in x for s in leaves(v)]
    if isinstance(x, bool) or x is None:
        return []
    s = str(x).lower()
    return [s] if len(s) >= MIN_LEAF else []


def turns(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Each assistant message with tool calls is a turn: calls, results, cost, user text before."""
    msgs = trace["messages"]
    results = {m["id"]: m for m in msgs if m["role"] == "tool"}
    out = []
    user_text = ""
    for m in msgs:
        if m["role"] == "user":
            user_text += " " + (m.get("content") or "").lower()
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "calls": m["tool_calls"],
                    "results": [
                        (results[c["id"]].get("content") or "").lower()
                        for c in m["tool_calls"]
                        if c["id"] in results
                    ],
                    "cost": m.get("cost") or 0.0,
                    "user_text": user_text,
                }
            )
    return out


def grounded(call: dict[str, Any], sources: str) -> bool:
    return all(v in sources for v in leaves(call["arguments"]))


def chains(trace: dict[str, Any], kinds: dict[str, str]) -> list[list[int]]:
    """Maximal runs of consecutive read-only turns where every call's argument values come from
    the user's messages or from the results earlier in the same chain."""
    ts = turns(trace)
    ro = [all(kinds.get(c["name"]) == "READ" for c in t["calls"]) for t in ts]
    found: list[list[int]] = []
    cur: list[int] = []
    for j, t in enumerate(ts):
        if not ro[j]:
            if len(cur) > 1:
                found.append(cur)
            cur = []
            continue
        if cur:
            sources = t["user_text"] + " ".join(r for k in cur for r in ts[k]["results"])
            if all(grounded(c, sources) for c in t["calls"]):
                cur.append(j)
                continue
            if len(cur) > 1:
                found.append(cur)
        cur = [j]
    if len(cur) > 1:
        found.append(cur)
    return found


def chain_stats(traces: list[dict[str, Any]], kinds: dict[str, str]) -> dict[str, Any]:
    """Share of agent LLM calls and of cost that composite tools would remove."""
    calls = removed = 0
    cost = removed_cost = 0.0
    patterns: Counter[str] = Counter()
    for tr in traces:
        ts = turns(tr)
        msgs = [m for m in tr["messages"] if m["role"] == "assistant"]
        calls += len(msgs)
        cost += sum(m.get("cost") or 0.0 for m in msgs)
        for ch in chains(tr, kinds):
            removed += len(ch) - 1
            removed_cost += sum(ts[j]["cost"] for j in ch[1:])
            patterns[" -> ".join("+".join(c["name"] for c in ts[j]["calls"]) for j in ch)] += 1
    return {
        "agent_calls": calls,
        "removable_calls": removed,
        "call_share": removed / calls if calls else 0.0,
        "agent_cost": cost,
        "removable_cost": removed_cost,
        "cost_share": removed_cost / cost if cost else 0.0,
        "top_patterns": patterns.most_common(5),
    }


def field_usage(traces: list[dict[str, Any]], kinds: dict[str, str]) -> dict[str, dict[str, Any]]:
    """For each READ tool returning a JSON object: per top-level key, how often its values show
    up later in the agent's tool-call arguments or text, and how big it is."""
    stats: dict[str, dict[str, list[tuple[bool, int]]]] = {}
    for tr in traces:
        msgs = tr["messages"]
        names = {c["id"]: c["name"] for m in msgs for c in (m.get("tool_calls") or [])}
        for i, m in enumerate(msgs):
            if m["role"] != "tool" or kinds.get(names.get(m["id"], "")) != "READ":
                continue
            try:
                data = json.loads(m.get("content") or "")
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            later = []
            for m2 in msgs[i + 1 :]:
                if m2["role"] == "assistant":
                    later.append((m2.get("content") or "").lower())
                    later += [
                        json.dumps(c["arguments"]).lower() for c in m2.get("tool_calls") or []
                    ]
            blob = "\n".join(later)
            for k, v in data.items():
                used = any(leaf in blob for leaf in leaves(v))
                stats.setdefault(names[m["id"]], {}).setdefault(k, []).append(
                    (used, len(json.dumps(v)))
                )
    out: dict[str, dict[str, Any]] = {}
    for tool, keys in stats.items():
        rows = {
            k: {
                "n": len(v),
                "used_rate": sum(u for u, _ in v) / len(v),
                "avg_chars": sum(c for _, c in v) / len(v),
            }
            for k, v in keys.items()
        }
        total = sum(r["avg_chars"] for r in rows.values()) or 1.0
        unused = sum(
            r["avg_chars"] for r in rows.values() if r["used_rate"] < 0.15 and r["avg_chars"] >= 60
        )
        out[tool] = {
            "keys": rows,
            "unused_char_share": unused / total,
            "observations": max(r["n"] for r in rows.values()),
        }
    return out
