"""Write reports/phase-N.md from the committed run results and the ledger."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from costlayer.attribution import attribute
from costlayer.group_a import estimate
from costlayer.ledger import PHASE_CAPS_USD, TOTAL_CAP_USD
from costlayer.stats import savings_ci, wilson


def _by_mode(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"baseline": [], "layer": []}
    for r in runs:
        out[r["mode"]].append(r)
    return out


def cache_hit_rate(runs: list[dict[str, Any]]) -> float:
    read = sum(r["tokens"]["cache_read"] for r in runs)
    prompt = sum(
        r["tokens"]["input"] + r["tokens"]["cache_write"] + r["tokens"]["cache_read"] for r in runs
    )
    return read / prompt if prompt else 0.0


def lever_firing(layer_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    applied = [r for r in layer_runs if r["layer_applied"]]
    steps = [s for r in applied for s in r["steps"] if s["applied"]]

    def removed(lever: str) -> int:
        # removed_tokens is the per-request size removed; sum over requests = tokens not sent
        return sum(s["removed_tokens"].get(lever, 0) for s in steps)

    return {
        "6 path / parallel calls": {
            "fired": any(len(s["tools"]) > 1 for s in steps) or bool(applied),
            "evidence": f"{sum(len(s['tools']) > 1 for s in steps)} steps with parallel calls; "
            f"suggested path in system prompt for {len(applied)} runs",
        },
        "7 effort": {"fired": bool(applied), "evidence": f"effort set on {len(applied)} runs"},
        "8a context: policy/system": {
            "fired": removed("context_system") > 0,
            "evidence": f"{removed('context_system')} tokens not sent",
        },
        "8b context: tools": {
            "fired": removed("context_tools") > 0,
            "evidence": f"{removed('context_tools')} tokens not sent",
        },
        "9 tool results trimmed": {
            "fired": sum(r["trimmed"] for r in applied) > 0,
            "evidence": f"{sum(r['trimmed'] for r in applied)} results "
            f"trimmed; {removed('trim')} tokens not sent",
        },
        "10 compaction": {
            "fired": any(r["compacted"] for r in layer_runs),
            "evidence": f"{sum(r['compacted'] for r in layer_runs)} runs compacted; "
            f"{removed('compaction')} tokens not sent",
        },
        "11 output limit": {
            "fired": bool(applied),
            "evidence": f"max_tokens hit on "
            f"{sum(s['stop_reason'] == 'max_tokens' for s in steps)} "
            f"steps",
        },
    }


def build_report(
    phase: int,
    runs: list[dict[str, Any]],
    ledger: dict[str, Any],
    notes: list[str] | None = None,
    first_messages: dict[str, str] | None = None,
) -> str:
    modes = _by_mode(runs)
    base, layer = modes["baseline"], modes["layer"]
    model = runs[0]["model"] if runs else "?"
    L: list[str] = [f"# Phase {phase} report", ""]
    L.append(
        f"Agent model: `{model}`. User simulator and classifier: `claude-haiku-5-5`. "
        f"Runs: {len(base)} baseline, {len(layer)} layer."
    )
    if phase == 1:
        L += [
            "",
            "> Phase 1 runs on Haiku to check the code and the levers. **No conclusion "
            "about savings or success rate may be drawn from these numbers.**",
        ]
    for n in notes or []:
        L += ["", n]

    L += ["", "## Cost per task (agent + classifier, USD)", ""]
    paired = _pair(base, layer)
    if paired:
        b = [x["cost_agent"] + x["cost_classifier"] for x, _ in paired]
        y = [x["cost_agent"] + x["cost_classifier"] for _, x in paired]
        pt, lo, hi = savings_ci(b, y)
        L += [
            "| | mean per task | total |",
            "|---|---:|---:|",
            f"| baseline | {statistics.mean(b):.5f} | {sum(b):.4f} |",
            f"| layer (classifier included) | {statistics.mean(y):.5f} | {sum(y):.4f} |",
            "",
            f"Savings on {len(paired)} paired tasks: **{100 * pt:.1f}%** "
            f"(95% bootstrap CI {100 * lo:.1f}% to {100 * hi:.1f}%).",
        ]
        ub = [x["cost_user"] for x, _ in paired]
        L += [
            f"User-simulator cost (same in both arms, not product cost): "
            f"{statistics.mean(ub):.5f} per task."
        ]
    elif base:
        L += [
            f"Baseline only: mean {statistics.mean(r['cost_agent'] for r in base):.5f} per task "
            f"over {len(base)} tasks."
        ]
    else:
        L += ["No runs."]

    L += [
        "",
        "## Success rate (tau-bench reward)",
        "",
        "| arm | n | success | 95% Wilson CI |",
        "|---|---:|---:|---|",
    ]
    for name, rs in (("baseline", base), ("layer", layer)):
        if rs:
            k = sum(1 for r in rs if r["reward"] == 1.0)
            lo, hi = wilson(k, len(rs))
            L.append(f"| {name} | {len(rs)} | {k / len(rs):.1%} | {lo:.1%} to {hi:.1%} |")

    L += [
        "",
        "## Tokens and cache",
        "",
        "| arm | fresh input | cache write | cache read | output | cache hit rate | steps/task |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rs in (("baseline", base), ("layer", layer)):
        if rs:
            t = {
                k: sum(r["tokens"][k] for r in rs)
                for k in ("input", "cache_write", "cache_read", "output")
            }
            L.append(
                f"| {name} | {t['input']} | {t['cache_write']} | {t['cache_read']} | "
                f"{t['output']} | {cache_hit_rate(rs):.1%} | "
                f"{statistics.mean(r['n_steps'] for r in rs):.1f} |"
            )

    if layer:
        L += ["", "## Layer: fallback and applicability", ""]
        fb = [r for r in layer if r["fallback"]]
        L += [
            f"- Layer applied (task type had a learned profile): "
            f"{sum(r['layer_applied'] for r in layer)}/{len(layer)}",
            f"- Fallback to the full baseline request: {len(fb)}/{len(layer)} runs "
            f"({100 * len(fb) / len(layer):.1f}%)",
        ]
        for reason, n in Counter(r["fallback"] for r in fb).most_common():
            L.append(f"  - {reason}: {n}")
        L += [
            "- Task types: "
            + ", ".join(f"{k}={v}" for k, v in Counter(r["task_type"] for r in layer).most_common())
        ]

        L += ["", "## Do the levers fire?", "", "| lever | fired | evidence |", "|---|---|---|"]
        for k, v in lever_firing(layer).items():
            L.append(f"| {k} | {'yes' if v['fired'] else '**NO**'} | {v['evidence']} |")

    if paired:
        att = attribute([x for x, _ in paired], [y for _, y in paired])
        L += [
            "",
            "## Attribution of agent-cost savings per lever (USD, from per-step token logs)",
            "",
            "| lever | USD |",
            "|---|---:|",
        ]
        for k, v in att.items():
            if k != "max_tokens_hits":
                L.append(f"| {k} | {v:.5f} |")
        L += [
            "",
            f"Steps that hit the output cap: {att.get('max_tokens_hits', 0)}. "
            "Attribution is an estimate: removed-token counts use ~3.5 chars/token; steps and "
            "effort are valued from paired averages.",
        ]

    errs = [r for r in runs if r["error"]]
    L += ["", "## Errors", ""]
    L += [f"- `{r['run_id']}`: {r['error']}" for r in errs] or ["- none"]

    if base:
        ga = estimate(base, first_messages)
        L += [
            "",
            "## Group A levers (estimates from baseline traces; not part of the verdict)",
            "",
            "| lever | est. USD | % of baseline | assumption |",
            "|---|---:|---:|---|",
        ]
        for k in sorted(k for k in ga if k[0].isdigit()):
            v = ga[k]
            L.append(f"| {k} | {v['usd']:.5f} | {v.get('pct', '-')} | {v['assumption']} |")
        sc = ga["4_batch_api"]["scenarios_usd"]
        L += ["", "Batch API scenarios (USD): " + ", ".join(f"{k}: {v}" for k, v in sc.items())]

    cap = PHASE_CAPS_USD.get(phase, 0)
    L += [
        "",
        "## Budget",
        "",
        f"- Phase {phase} spent: ${ledger['by_phase'].get(str(phase), 0.0):.4f} of ${cap:.2f}",
        f"- Total spent so far: ${ledger['total']:.4f} of ${TOTAL_CAP_USD:.2f} hard cap "
        f"({ledger['calls']} API calls in the ledger)",
        "- By purpose: " + ", ".join(f"{k} ${v:.4f}" for k, v in ledger["by_purpose"].items()),
    ]
    return "\n".join(L) + "\n"


def _pair(
    base: list[dict[str, Any]], layer: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    lb = {(r["domain"], r["idx"]): r for r in layer}
    return [(b, lb[(b["domain"], b["idx"])]) for b in base if (b["domain"], b["idx"]) in lb]
