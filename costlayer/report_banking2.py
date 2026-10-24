"""reports/banking-phase-2.md: the sweep, the analysis and the gate (TEST2.md phase 2 and
Amendment 1)."""

from __future__ import annotations

from typing import Any

from costlayer.banking_phase1 import VARIANT
from costlayer.budget import Budget
from costlayer.optimizer.analysis import CLASSIFIER_MIN
from costlayer.optimizer.gate import K1_RATIO, K2_RATIO, Projection, project, verdict_lines
from costlayer.pricing import est_tokens
from costlayer.report import PINNED
from costlayer.runner import ORDER_SEED

HONEST = (
    "**What to expect (every report).** With 12 train tasks per-type learning is not expected to "
    "be reachable: most task types have one or two tasks and a type with fewer than 3 gets "
    "`default`. The gate is computed as written anyway."
)
CANNOT = [
    "Whether the agent says the right things: ENV checks the database state only, and "
    "`task_102`'s language check is dropped.",
    "That quality is unchanged: with 30 to 40 test tasks a layer that truly loses 10 to 15 "
    "points can still pass. A KEEP earns a larger quality test, never a launch.",
    "Anything about other agents, other retrieval variants or other models: this is one "
    "benchmark, one variant, Sonnet 5.5, one simulated user.",
]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def tok(chars: float) -> int:
    return est_tokens("x" * int(chars))


def projections(res: dict[str, Any], classifier: dict[str, Any] | None) -> dict[str, Any]:
    """The literal projection, and (only if the classifier is below 70%) the one with the
    per-type term removed because per-type choices are then switched off."""
    cost = res["cost_agent_total"]
    h, ch = res["headroom"], res["chains"]
    lit = project(
        cost["default"], cost[res["global_star"]], ch["cost_share"], h["headroom"], h["beats_null"]
    )
    out: dict[str, Any] = {"literal": lit, "per_type_off": None}
    if classifier is not None and classifier["accuracy"] < CLASSIFIER_MIN and h["beats_null"]:
        out["per_type_off"] = project(
            cost["default"], cost[res["global_star"]], ch["cost_share"], h["headroom"], False
        )
    return out


def build(
    res: dict[str, Any],
    phase: dict[str, Any],
    classifier: dict[str, Any] | None,
    ledger: dict[str, Any],
    prior_total: float,
    budget: Budget,
    console_note: str,
    dropped: list[str],
    phase_key: str = "b2",
) -> str:
    L: list[str] = ["# Test 2, phase 2 (b2) report: banking_knowledge sweep and gate", ""]
    L += [
        f"Benchmark: `sierra-research/tau2-bench` at `{PINNED}`, domain `banking_knowledge`, "
        f"retrieval variant `{VARIANT}`, tasks in `banking_order(seed={ORDER_SEED}, "
        "exclude=ACTION)` order, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`. Agent "
        "`claude-sonnet-5-5`; user simulator and classifier `claude-haiku-5-5`. Cost in the "
        "ratios is agent cost only (user simulator excluded).",
        f"Train tasks planned: {len(phase.get('ids', []))} (`{', '.join(phase.get('ids', []))}`); "
        f"complete in all four configurations: {res.get('n_tasks')}; runs reused from phase 1: "
        f"{len(phase.get('reused', []))}; runs executed now: {phase.get('n_todo')}.",
        "",
        HONEST,
    ]
    if phase.get("stopped"):
        L += ["", f"**Sweep stopped early:** {phase['stopped']}."]
    if dropped:
        L += [
            "",
            "Tasks dropped from the paired analysis (missing or infrastructure-error runs): "
            + ", ".join(dropped),
        ]
    if "error" in res:
        L += ["", f"**Analysis not possible:** {res['error']}."]
        return "\n".join(L) + _tail(ledger, prior_total, budget, console_note, phase_key)

    proj = projections(res, classifier)
    lit: Projection = proj["literal"]
    L += ["", "## Gate (optimistic projection)", "", "```", *verdict_lines(lit), "```", ""]
    L += [
        "Rule (TEST2.md, fixed): `projected = cost(global*) x (1 - chain_cost_share)`, or "
        "`cost(global*) x (1 - out_of_sample_headroom)` if the headroom beats the null (the lower "
        "is taken); classifier overhead counted as zero. Projected <= "
        f"{K2_RATIO:.2f} of `default` and <= {K1_RATIO:.2f} of `global*`: go to phase 3. "
        f"<= {K2_RATIO:.2f} of `default` but above {K1_RATIO:.2f} of `global*`: FEATURE ONLY. "
        f"Above {K2_RATIO:.2f} of `default`: KILL.",
        "",
        f"Inputs: `global*` = `{res['global_star']}`; agent cost over the {res['n_tasks']} "
        f"paired tasks: `default` ${lit.cost_default:.4f}, `global*` ${lit.cost_global:.4f}; "
        f"chain cost share {pct(lit.chain_cost_share)}; out-of-sample headroom "
        f"{pct(lit.headroom)} ({'beats' if lit.headroom_beats_null else 'does not beat'} the "
        "null).",
    ]
    alt = proj["per_type_off"]
    if alt is not None:
        L += [
            "",
            f"The classifier is at {pct(classifier['accuracy'])}, below "
            f"{pct(CLASSIFIER_MIN)}: per-type choices are switched off for the layer, so the "
            f"headroom term could not be realised. Without it the projection is "
            f"${alt.projected:.4f}, {alt.ratio_default:.3f} of `default` and "
            f"{alt.ratio_global:.3f} of `global*`: **{alt.verdict}**. The literal verdict above "
            "is the one of the rule as written.",
        ]

    # -- the cost table with successes and calls (Amendment 1) -------------------------------
    t = res["table"]
    L += [
        "",
        "## Cost per configuration, with successes and LLM calls",
        "",
        "| config | runs | successes | failures | agent $/run | cost / default (80% interval) "
        "| LLM calls/run | LLM calls/run, failed runs |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for c, row in t.items():
        r = res["cost_ratio_vs_default"].get(c)
        ratio = f"{r[0]:.2f} ({r[1]:.2f} to {r[2]:.2f})" if r else "1.00"
        fcalls = f"{row['mean_calls_failed']:.1f}" if row["mean_calls_failed"] is not None else "-"
        L.append(
            f"| {c} | {row['runs']} | {row['successes']} | {row['failures']} | "
            f"{row['mean_agent_cost']:.4f} | {ratio} | {row['mean_calls']:.1f} | {fcalls} |"
        )
    L += [
        "",
        f"Intervals: bootstrap over tasks, 10,000 resamples, 80%. Paired tasks: {res['n_tasks']}.",
    ]
    flagged = [f for f in res["savings_flags"] if f["coincides"]]
    if flagged:
        L += [
            "",
            "**Savings that coincide with fewer calls AND fewer successes:** "
            + ", ".join(f"`{f['config']}`" for f in flagged)
            + ". For these, part or all of the saving is runs ending early, not cheaper work.",
        ]
    else:
        L += [
            "",
            "**No configuration's saving coincides with both fewer calls and fewer "
            "successes than `default`.**",
        ]
    for f in res["savings_flags"]:
        L.append(
            f"- `{f['config']}`: cheaper than `default`: {'yes' if f['cheaper'] else 'no'}; "
            f"fewer calls per run: {'yes' if f['fewer_calls'] else 'no'}; fewer successes: "
            f"{'yes' if f['fewer_successes'] else 'no'}"
        )

    # -- anatomy -------------------------------------------------------------------------------
    L += [
        "",
        "## Cost anatomy (question 1)",
        "",
        "| config | agent $/run | calls/run | cache read | cache write | uncached input | output "
        "| retrieval results/run | result tokens in context/run (est.) | raw tokens/run (est.) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, a in res["anatomy"].items():
        sh = a["shares"]
        L.append(
            f"| {c} | {a['mean_agent_cost']:.4f} | {a['calls_per_run']:.1f} | "
            + " | ".join(
                pct(sh[k]) for k in ("cache read", "cache write", "uncached input", "output")
            )
            + f" | {res['retrieval'][c]['results']:.1f} | {res['retrieval'][c]['in_tokens']:,.0f} "
            f"| {res['retrieval'][c]['raw_tokens']:,.0f} |"
        )
    L += [
        "",
        "Shares are of agent cost. Result sizes come from the run summaries (characters "
        "recorded at the time, about 3.5 characters per token), never from trace text.",
    ]

    # -- success, selection ----------------------------------------------------------------------
    L += ["", "## Paired success and the selection rule (question 2)", ""]
    for c, s in res["success"].items():
        if c != "default":
            L.append(
                f"- `{c}` vs `default`: lost {s['lost_vs_default'] or 'none'}, won "
                f"{s['won_vs_default'] or 'none'}"
            )
    L += [
        "",
        f"Acceptable (failures <= default + floor(0.10 x {res['n_tasks']}) = "
        f"{res['failures']['default']} + {int(0.1 * res['n_tasks'])}): "
        f"{', '.join(res['acceptable'])}. **`global*` = `{res['global_star']}`.**",
    ]

    # -- types, headroom ---------------------------------------------------------------------------
    h = res["headroom"]
    L += [
        "",
        "## Task types and per-type headroom (question 3, gateway)",
        "",
        "| task type (first write tool) | tasks | per-type choice |",
        "|---|---:|---|",
    ]
    for ty, k in sorted(res["type_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {ty} | {k} | {res['per_type_choice'][ty]} |")
    L += [
        "",
        f"Out-of-sample headroom over `global*` (leave one task out): **{pct(h['headroom'])}** "
        f"(chosen cost {h['chosen_cost']:.4f} vs `global*` {h['global_cost']:.4f}; "
        f"{h['chosen_failures']} failures). Null (type labels shuffled 200 times): mean "
        f"{pct(h['null_mean'])}, 90th percentile {pct(h['null_p90'])}; "
        f"{'beats' if h['beats_null'] else 'does NOT beat'} the null. Oracle upper bound "
        f"(biased low, not used by the gate): {pct(h['oracle_upper_bound'])}.",
    ]

    # -- mining ---------------------------------------------------------------------------
    ch = res["chains"]
    src = res["mining_sources"]
    L += [
        "",
        f"## Mining on `{res['global_star']}` traces (dependent read-only chains)",
        "",
        f"Composite tools would remove {ch['removable_calls']} of {ch['agent_calls']} agent LLM "
        f"calls (**{pct(ch['call_share'])}**) and about **{pct(ch['cost_share'])}** of agent "
        f"cost (measured cost of those calls, ${ch['removable_cost']:.4f} of "
        f"${ch['agent_cost']:.4f}). Calls issued in parallel in one turn are not chained; "
        "`call_discoverable_agent_tool` is a WRITE tool and never joins a chain; a call joins "
        "only if every argument value appears in the user's messages or in an earlier result of "
        "the chain, **as the agent saw it** (the cut text under `C:top3`/`EC`, so a value that "
        "only an omitted item contained earns no credit).",
    ]
    for pat, k in ch["top_patterns"]:
        L.append(f"- {k}x `{pat}`")
    L += [
        "",
        "Trace sources used for mining: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(src.items()))
        + ". `seen` = saved by the "
        "runner after the cut fix; `uncut` = a configuration that never cuts; `reconstructed` "
        "= a phase 1 trace (raw tool results; phase 1 files are not rewritten) with the "
        "deterministic cut re-applied in memory and accepted only because the summed raw and "
        "in-context lengths equal the sizes recorded in the run summary; `unverified` runs are "
        "left out of the mining. **Limit:** for reused phase 1 runs the trace text is a "
        "reconstruction checked by sizes, not a record; costs and result sizes of reused runs "
        "always come from the run summary and the ledger.",
    ]

    if classifier:
        ok = classifier["accuracy"] >= CLASSIFIER_MIN
        L += [
            "",
            f"**Classifier** (Haiku, few-shot, leave one out, first message only): "
            f"accuracy **{pct(classifier['accuracy'])}** on {classifier['n']} tasks. "
            + (
                "Per-type choices stay on."
                if ok
                else "Below 70%: per-type choices are switched off."
            ),
        ]
    else:
        L += [
            "",
            "**Classifier:** not run (" + str(phase.get("classifier_note") or "no result") + ").",
        ]
    return "\n".join(L) + _tail(ledger, prior_total, budget, console_note, phase_key)


def _tail(
    ledger: dict[str, Any], prior_total: float, budget: Budget, console_note: str, phase_key: str
) -> str:
    cap = budget.phase_caps.get(phase_key, 0.0)
    spent = ledger["total"]
    L = [
        "",
        "",
        "## Budget",
        "",
        f"- Phase {phase_key} spent (ledger, this dispatch): **${spent:.4f}** of ${cap:.2f}",
        f"- Test 2 total so far: ${spent + prior_total:.4f} of ${budget.total:.2f}",
        f"- API calls in the ledger: {ledger['calls']}; by purpose: "
        + ", ".join(f"{k} ${v:.4f}" for k, v in ledger["by_purpose"].items()),
        f"- Console reconciliation (must agree within 5%): {console_note}",
        "",
        "## What this cannot show",
        "",
    ]
    L += [f"- {c}" for c in CANNOT]
    return "\n".join(L) + "\n"
