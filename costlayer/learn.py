"""Learn the per-task-type profile from baseline traces of TRAIN tasks only."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from typing import Any

from costlayer.domains import (
    ALWAYS_KEEP_TOOLS,
    READ_ONLY_TOOLS,
    SECTION_TOOLS,
    TAU_BENCH_COMMIT,
)
from costlayer.profile import Profile, TypeProfile

MIN_SUPPORT = 3
FIELD_USE_RATE = 0.15
FIELD_MIN_CHARS = 60


def trace_dict(result: Any) -> dict[str, Any]:
    """Serializable trace of a RunResult (written to the Actions artifact, not to git)."""
    return {
        **result.summary(),
        "messages": result.messages,
        "steps": [vars(s) for s in result.steps],
    }


def tool_sequence(messages: list[dict[str, Any]]) -> list[str]:
    seq = []
    for m in messages:
        if m["role"] == "assistant" and not isinstance(m["content"], str):
            seq += [b["name"] for b in m["content"] if b.get("type") == "tool_use"]
    return seq


def _leaves(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [s for v in x.values() for s in _leaves(v)]
    if isinstance(x, list):
        return [s for v in x for s in _leaves(v)]
    if isinstance(x, bool) or x is None:
        return []
    s = str(x).lower()
    return [s] if len(s) >= 3 else []


def field_evidence(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, tuple[bool, int]]]]:
    """For each dict-valued tool result: {key: (used_later, serialized_chars)}."""
    names: dict[str, str] = {}
    out = []
    for i, m in enumerate(messages):
        if isinstance(m["content"], str):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                names[b["id"]] = b["name"]
            if b.get("type") != "tool_result" or b.get("is_error"):
                continue
            try:
                data = json.loads(b["content"])
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            later = []
            for m2 in messages[i + 1 :]:
                if m2["role"] != "assistant" or isinstance(m2["content"], str):
                    continue
                for b2 in m2["content"]:
                    if b2.get("type") == "tool_use":
                        later.append(json.dumps(b2["input"]).lower())
                    elif b2.get("type") == "text":
                        later.append(b2["text"].lower().replace(",", ""))
            blob = "\n".join(later)
            ev = {
                k: (any(leaf in blob for leaf in _leaves(v)), len(json.dumps(v)))
                for k, v in data.items()
            }
            out.append((names.get(b["tool_use_id"], "?"), ev))
    return out


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(q * len(xs)) - 1)]


def learn_type(
    domain: str, traces: list[dict[str, Any]], field_support: int = MIN_SUPPORT
) -> TypeProfile:
    seqs = [tool_sequence(t["messages"]) for t in traces]
    used = Counter(n for s in seqs for n in set(s))
    tools = sorted(set(used) | ALWAYS_KEEP_TOOLS)
    sections = [h for h, ts in SECTION_TOOLS[domain].items() if ts & set(used)]

    collapsed = [tuple(n for n in s if n != "think") for s in seqs]
    modal, modal_n = Counter(collapsed).most_common(1)[0]
    if modal_n >= max(2, 0.3 * len(traces)):
        path = list(modal)
    else:
        first: dict[str, int] = {}
        for s in collapsed:
            for i, n in enumerate(s):
                first.setdefault(n, i)
        path = [
            n
            for n in sorted(first, key=first.get)  # type: ignore[arg-type]
            if used[n] >= 0.5 * len(traces)
        ]

    ro = READ_ONLY_TOOLS[domain]
    writers = sum(1 for s in seqs if any(n not in ro for n in s))
    effort = "low" if writers < 0.2 * len(traces) else "medium"

    outs = [st["usage"]["output"] for t in traces for st in t["steps"] if st["kind"] != "redo"]
    max_tokens = int(min(8192, max(1024, math.ceil(1.5 * _pct(outs, 0.99) / 256) * 256)))

    peaks = [max(st["messages_tokens_est"] for st in t["steps"]) for t in traces]
    med_peak = statistics.median(peaks)
    compact = int(max(2500, 0.5 * med_peak)) if med_peak >= 4000 else None

    n_steps = [len(t["steps"]) for t in traces]
    max_steps = min(30, math.ceil(1.5 * _pct(n_steps, 0.9)) + 2)

    obs: dict[str, list[dict[str, tuple[bool, int]]]] = {}
    for t in traces:
        for name, ev in field_evidence(t["messages"]):
            obs.setdefault(name, []).append(ev)
    drop: dict[str, list[str]] = {}
    for name, evs in obs.items():
        if name not in READ_ONLY_TOOLS[domain] or len(evs) < field_support:
            continue
        keys = {k for ev in evs for k in ev}
        dk = []
        for k in sorted(keys):
            seen = [ev[k] for ev in evs if k in ev]
            rate = sum(u for u, _ in seen) / len(seen)
            avg_chars = sum(c for _, c in seen) / len(seen)
            if rate < FIELD_USE_RATE and avg_chars >= FIELD_MIN_CHARS:
                dk.append(k)
        if dk:
            drop[name] = dk

    transfer_rate = sum(1 for s in seqs if "transfer_to_human_agents" in s) / len(traces)
    return TypeProfile(
        n_traces=len(traces),
        tools=tools,
        sections=sections,
        path=path,
        effort=effort,
        max_tokens=max_tokens,
        tool_fields=drop,
        compact_after_tokens=compact,
        max_steps=max_steps,
        transfer_rate=transfer_rate,
    )


def learn_profile(traces: list[dict[str, Any]], min_support: int = MIN_SUPPORT) -> Profile:
    """`traces` are baseline runs on train tasks; each needs 'domain', 'task_type', 'reward'."""
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in traces:
        if t["mode"] == "baseline" and t["reward"] == 1.0 and not t.get("error"):
            by.setdefault((t["domain"], t["task_type"]), []).append(t)
    types: dict[str, dict[str, TypeProfile]] = {}
    for (domain, label), ts in sorted(by.items()):
        if label == "other" or len(ts) < min_support:
            continue
        types.setdefault(domain, {})[label] = learn_type(domain, ts, field_support=min_support)
    meta = {
        "tau_bench_commit": TAU_BENCH_COMMIT,
        "n_traces": len(traces),
        "n_successful": sum(len(v) for v in by.values()),
        "min_support": min_support,
        "agent_model": traces[0]["model"] if traces else None,
    }
    return Profile(meta=meta, types=types)
