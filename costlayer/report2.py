"""reports/phase-2.md"""

from __future__ import annotations

import json
from typing import Any

from costlayer.ledger import PHASE_CAPS_USD, TOTAL_CAP_USD
from costlayer.optimizer.analysis import (
    CLASSIFIER_MIN,
    GATE_CALL_SHARE,
    GATE_COST_SHARE,
    GATE_HEADROOM,
    gate,
)

PINNED = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
HONEST = (
    "**Honest expectation (every report).** Per-type learning is statistically thin here (7 task "
    "types, 12 to 24 train tasks): most types have 1 to 4 tasks and the per-type effort choice "
    "usually collapses to `global*`. K1 can realistically pass only through fewer agent steps "
    "(S1 composites, S2 prefetch), which are structural and measurable. If only gateway levers "
    "help and they do not beat `global*`, the verdict is FEATURE ONLY."
)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build(
    res: dict[str, Any],
    phase: dict[str, Any],
    classifier: dict[str, Any] | None,
    ledger: dict[str, Any],
    prior_total: float,
    console_note: str,
    dropped: list[str],
) -> str:
    L = ["# Phase 2 report: sweep on train tasks", ""]
    L += [
        f"Benchmark: `sierra-research/tau2-bench` at `{PINNED}` (retail, official train split, "
        "seed-42 order, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`). Agent: "
        "`claude-sonnet-5-5`; user simulator and classifier: `claude-haiku-5-5`. Cost in the "
        "ratios is agent cost only (user simulator excluded).",
        f"Tasks planned: {phase.get('n_train')}; complete in all four configurations: "
        f"{res.get('n_tasks')}; runs reused from phase 1: {len(phase.get('reused', []))}; "
        f"runs executed now: {phase.get('to_run')}.",
        "",
        HONEST,
    ]
    if phase.get("stopped"):
        L += ["", f"**Stopped early:** {phase['stopped']}."]
    if dropped:
        L += [
            "",
            "Tasks dropped from the paired analysis (infrastructure errors): " + ", ".join(dropped),
        ]
    if "error" in res:
        L += ["", f"**Analysis not possible:** {res['error']}."]
        return "\n".join(L) + _tail(ledger, prior_total, console_note)

    L += [
        "",
        "## Phase 1 follow-up: the cache-write doubling and the thinking field",
        "",
        "The doubling in the phase 1 report was a bug in my token bookkeeping, not an effect of "
        "`default` or `E:low`. A run's tokens were summed over all ledger rows with the same "
        "(config, task, attempt), ignoring the model; Haiku and Sonnet had both run `default` on "
        "the same three tasks, so each Sonnet `default` run carried the Haiku run's tokens "
        "(about 15.6K cache-write tokens each: 15.6K + 15.6K = 31K) and the Haiku files were "
        "overwritten. Dollar costs, LLM-call counts, `c(x)` and the allocation were not affected "
        "(they come from per-run spend). Fixed: the model is now part of the key and of the file "
        "names, the phase 1 files were rebuilt from the ledger, and the correction is at the top "
        "of `reports/phase-1.md`. Corrected: cache writes per run are about 15.6K (`default`) "
        "and 15.1K (`E:low`).",
        "",
        "Why the API's thinking field does not explain the output difference: output per call "
        "was 192 (`default`) vs 149 (`E:low`) in phase 1, but the reported thinking tokens were "
        "24 vs 7 per call, so the field covers about 17 of the 43 tokens per call. Most output "
        "is not reported as thinking; it is the visible reply and the tool-call JSON. The table "
        "below splits every call of the sweep into reported thinking, estimated visible output "
        "(text plus tool-call JSON) and the remainder, which is what the field does not explain.",
    ]

    L += [
        "",
        "## Cost anatomy (question 1)",
        "",
        "| config | runs | agent $/run | calls/run | cache read | cache write | uncached "
        "input | output | output/call | reported thinking/call | visible est./call | "
        "remainder/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, a in res["anatomy"].items():
        sh = a["shares"]
        th, vis = a["thinking_per_call"], a["visible_est_per_call"]
        rem = (a["output_per_call"] - th - vis) if th is not None and vis is not None else None
        L.append(
            f"| {c} | {a['runs']} | {a['mean_agent_cost']:.4f} | {a['calls_per_run']:.1f} | "
            + " | ".join(
                pct(sh[k]) for k in ("cache read", "cache write", "uncached input", "output")
            )
            + f" | {a['output_per_call']:.0f} | "
            + (f"{th:.0f}" if th is not None else "n/a")
            + " | "
            + (f"{vis:.0f}" if vis is not None else "n/a")
            + " | "
            + (f"{rem:.0f}" if rem is not None else "n/a")
            + " |"
        )
    L += [
        "",
        "Shares are of agent cost. Visible output is estimated at about 3.5 characters per "
        "token; the remainder is output that is neither reported thinking nor visible text or "
        "tool-call JSON (tokenizer error included).",
    ]

    L += [
        "",
        "## Best single setting (question 2)",
        "",
        "| config | successes | failures | agent $ total | cost / default (80% interval) |",
        "|---|---:|---:|---:|---|",
    ]
    n = res["n_tasks"]
    for c, s in res["success"].items():
        ratio = res["cost_ratio_vs_default"].get(c)
        r = f"{ratio[0]:.2f} ({ratio[1]:.2f} to {ratio[2]:.2f})" if ratio else "1.00"
        L.append(
            f"| {c} | {s['successes']}/{n} | {res['failures'][c]} | "
            f"{res['cost_agent_total'][c]:.4f} | {r} |"
        )
    L += [
        "",
        f"Acceptable (failures <= default + floor(0.10 x {n})): "
        f"{', '.join(res['acceptable'])}. **`global*` = `{res['global_star']}`.**",
    ]
    for c, s in res["success"].items():
        if c != "default":
            L.append(
                f"- `{c}` vs `default`: lost {s['lost_vs_default'] or 'none'}, "
                f"won {s['won_vs_default'] or 'none'}"
            )

    L += [
        "",
        "## Beyond a setting: per-type choice (question 3, gateway levers G1/G2)",
        "",
        "| task type (first write tool) | tasks | per-type choice |",
        "|---|---:|---|",
    ]
    for ty, k in sorted(res["type_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {ty} | {k} | {res['per_type_choice'][ty]} |")
    h = res["headroom"]
    L += [
        "",
        f"Out-of-sample headroom over `global*` (leave one task out): **{pct(h['headroom'])}** "
        f"(chosen cost {h['chosen_cost']:.4f} vs `global*` {h['global_cost']:.4f}; "
        f"{h['chosen_failures']} failures). Null (type labels shuffled 200 times): mean "
        f"{pct(h['null_mean'])}, 90th percentile {pct(h['null_p90'])}; "
        f"{'beats' if h['beats_null'] else 'does NOT beat'} the null. Oracle upper bound "
        f"(biased low, not used by the gate): {pct(h['oracle_upper_bound'])}.",
    ]

    ch = res["chains"]
    L += [
        "",
        f"## Mining on `{res['global_star']}` traces (SDK levers S1/S2 and G3)",
        "",
        f"**(a) Dependent read-only chains.** Composite tools would remove "
        f"{ch['removable_calls']} of {ch['agent_calls']} agent LLM calls "
        f"(**{pct(ch['call_share'])}**) and about **{pct(ch['cost_share'])}** of agent cost "
        f"(measured cost of those calls, ${ch['removable_cost']:.4f} of ${ch['agent_cost']:.4f}).",
    ]
    for pat, k in ch["top_patterns"]:
        L.append(f"- {k}x `{pat}`")
    L += [
        "",
        "**(b) Tool-result fields used later** (values reappear in a later tool call or "
        "reply). Share of result characters in fields used in under 15% of cases:",
        "",
        "| tool | observations | unused char share |",
        "|---|---:|---:|",
    ]
    for tool, f in sorted(res["fields"].items()):
        L.append(f"| {tool} | {f['observations']} | {pct(f['unused_char_share'])} |")
    L += ["", "Retail tool results are small (a few hundred tokens), so G3 has little to gain.", ""]
    if classifier:
        ok = classifier["accuracy"] >= CLASSIFIER_MIN
        L += [
            f"**(c) First-message type classifier** (Haiku, few-shot, leave one out): accuracy "
            f"**{pct(classifier['accuracy'])}** on {classifier['n']} tasks. "
            + (
                "Per-type choices stay on."
                if ok
                else "Below 70%: per-type choices are switched off for the layer."
            )
        ]
    else:
        L += ["**(c) Classifier:** not run."]

    g = gate(res)
    L += [
        "",
        "## Gate",
        "",
        f"- (a) headroom >= {pct(GATE_HEADROOM)} and beats the null: "
        f"{'YES' if g['a'] else 'no'} ({pct(g.get('headroom', 0))})",
        f"- (b) chains remove >= {pct(GATE_CALL_SHARE)} of calls AND >= {pct(GATE_COST_SHARE)} "
        f"of cost: {'YES' if g['b'] else 'no'} ({pct(g.get('call_share', 0))} of calls, "
        f"{pct(g.get('cost_share', 0))} of cost)",
    ]
    if g["pass"]:
        L += ["", "**Gate passed: phase 3 may go ahead, but only on the founder's go.**"]
    else:
        L += [
            "",
            "**Gate not passed: verdict FEATURE ONLY.** A global setting is what this data "
            "supports; spend nothing more.",
        ]
    L += [
        "",
        "Gateway-only savings (G1/G2): `global*` cost "
        f"{res['cost_ratio_vs_default'].get(res['global_star'], (1.0, 1.0, 1.0))[0]:.2f} of "
        f"`default`, per-type headroom on top {pct(h['headroom'])}. Needs-SDK savings (S1): up "
        f"to {pct(ch['cost_share'])} of agent cost.",
    ]

    return "\n".join(L) + _tail(ledger, prior_total, console_note)


def _tail(ledger: dict[str, Any], prior_total: float, console_note: str) -> str:
    cap = PHASE_CAPS_USD[2]
    spent = ledger["total"]
    L = [
        "",
        "## Budget",
        "",
        f"- Phase 2 spent (ledger, this dispatch): **${spent:.4f}** of ${cap:.2f}",
        f"- Total spent so far: ${spent + prior_total:.4f} of ${TOTAL_CAP_USD:.2f} hard cap",
        f"- API calls in the ledger: {ledger['calls']}; by purpose: "
        + ", ".join(f"{k} ${v:.4f}" for k, v in ledger["by_purpose"].items()),
        f"- Console reconciliation (must agree within 5%): {console_note}",
        "",
        "## What this cannot show",
        "",
        "- Whether the agent says the right things: `EvaluationType.ENV` checks only the final "
        "database state and environment assertions.",
        "- That quality is unchanged: with this few tasks a configuration that truly loses "
        "several points of success can still pass the selection rule.",
        "- A real cost saving in production: it is measured on one benchmark with a simulated "
        "user, on a handful of tasks, with the cost ratio's sampling noise shown above.",
        "",
    ]
    return "\n".join(L)


def dumps(res: dict[str, Any]) -> str:
    return json.dumps(res, indent=1, default=str)
