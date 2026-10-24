"""reports/banking-phase-1.md: smoke test, cost anatomy and the budget request (TEST2.md)."""

from __future__ import annotations

import statistics
from typing import Any

from costlayer import allocation2
from costlayer.audit import audit
from costlayer.banking_phase1 import VARIANT
from costlayer.budget import Budget
from costlayer.pricing import HAIKU, SONNET, est_tokens
from costlayer.report import PINNED, cost_by_type
from costlayer.runner import ORDER_SEED

CONFIG_ORDER = ("default", "E:low", "C:top3", "EC")


def last_call_context(rows: list[dict[str, Any]], r: dict[str, Any]) -> int | None:
    """Prompt tokens (uncached + cache write + cache read) of the run's last successful agent call."""
    calls = [
        x
        for x in rows
        if x["purpose"] == "agent"
        and x["status"] == "ok"
        and (x.get("model"), x.get("config"), x.get("task"), x.get("attempt"))
        == (r["model"], r["config"], r["task"], r["attempt"])
    ]
    if not calls:
        return None
    x = calls[-1]
    return x["input"] + x["cache_write"] + x["cache_read"]


def tok(chars: int) -> int:
    return est_tokens("x" * chars)


def sonnet_costs(runs: list[dict[str, Any]]) -> dict[str, float]:
    """c(x): mean cost per ok Sonnet run (agent plus user simulator) of each configuration."""
    out = {}
    for cfg in CONFIG_ORDER:
        xs = [
            r["cost_agent"] + r["cost_user"]
            for r in runs
            if r["model"] == SONNET and r["config"] == cfg and r["status"] == "ok"
        ]
        if xs:
            out[cfg] = statistics.mean(xs)
    return out


def build(
    phase_result: dict[str, Any],
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    ledger_total: dict[str, Any],
    prior_total: float,
    budget: Budget,
    console_note: str,
) -> str:
    cap = budget.cap("b1")
    L: list[str] = ["# Test 2, phase 1 (b1) report: banking_knowledge smoke test", ""]
    L += [
        f"Benchmark: `sierra-research/tau2-bench` at `{PINNED}`, domain `banking_knowledge`, "
        f"retrieval variant `{VARIANT}`, tasks in `banking_order(seed={ORDER_SEED}, "
        "exclude=ACTION)` order (88 tasks; the 9 ACTION tasks are a free pass under ENV), "
        "`EvaluationType.ENV`, `max_steps=100`, `max_retries=0`. "
        f"Models: agent `{HAIKU}` (checks) and `{SONNET}`; user simulator `{HAIKU}` in every "
        "configuration. Cost per run is agent plus user simulator unless stated.",
        "",
        "## Verdict of the smoke test",
        "",
    ]
    for name, c in phase_result.get("haiku_checks", {}).items():
        L.append(f"- {'PASS' if c['pass'] else '**FAIL**'}: {name}. " + "; ".join(c["detail"]))
    L.append(f"- MANUAL: ledger total vs Console. {console_note}")
    if phase_result.get("stopped"):
        L += ["", f"**Stopped early:** {phase_result['stopped']}."]
    L += [
        "",
        "First tasks of the order: " + ", ".join(f"`{t}`" for t in phase_result.get("order", [])),
    ]

    L += [
        "",
        "## Runs",
        "",
        "| model | config | task | status | reward | LLM calls | agent $ | user $ | input | "
        "cache write | cache read | output | context at last call | retrieval results "
        "(raw / in context, est. tokens) | cut results | omitted items | termination |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for r in runs:
        t = r["tokens"]
        ctx = last_call_context(rows, r)
        L.append(
            f"| {r['model'].split('-')[1]} | {r['config']} | {r['task']} | {r['status']} | "
            f"{r['reward']:.0f} | {r['llm_calls']} | {r['cost_agent']:.4f} | {r['cost_user']:.4f} "
            f"| {t['input']} | {t['cache_write']} | {t['cache_read']} | {t['output']} | "
            f"{ctx if ctx is not None else 'n/a'} | {r.get('retrieval_results', 0)} "
            f"({tok(r.get('retrieval_chars_raw', 0))} / {tok(r.get('retrieval_chars_in_context', 0))})"
            f" | {r.get('cut_results', 0)} | {r.get('cut_items_omitted', 0)} | {r['termination']} |"
        )

    flagged = [r for r in runs if r["hit_max_steps"] or r["status"] == "censored"]
    L += ["", "### Runs that hit `max_steps` or the abort rail", ""]
    L += [
        f"- {r['model'].split('-')[1]} {r['config']} {r['task']}: "
        f"{'max_steps' if r['hit_max_steps'] else 'abort rail (censored)'}, "
        f"cost ${r['cost_agent'] + r['cost_user']:.4f}"
        for r in flagged
    ] or ["- none"]
    bad = [r for r in runs if r.get("cut_unparsed")]
    L.append(
        "- C:top3 could not parse a retrieval result in: "
        + (", ".join(f"{r['config']}/{r['task']}" for r in bad) if bad else "no run")
    )

    sonnet = [r for r in runs if r["model"] == SONNET and r["status"] == "ok"]
    L += ["", "## Cost anatomy, Sonnet 5.5", ""]
    if sonnet:
        L += [
            "| config | runs | mean $/run | LLM calls/run | cache read | cache write | "
            "uncached input | output | context at last call (mean) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cfg in CONFIG_ORDER:
            rs = [r for r in sonnet if r["config"] == cfg]
            if not rs:
                continue
            tot = {
                k: sum(r["tokens"][k] for r in rs)
                for k in ("input", "cache_write", "cache_read", "output")
            }
            parts = cost_by_type(SONNET, tot)
            s = sum(parts.values()) or 1.0
            ctxs = [c for r in rs if (c := last_call_context(rows, r)) is not None]
            mean_ctx = f"{statistics.mean(ctxs):.0f}" if ctxs else "n/a"
            mean_cost = statistics.mean(r["cost_agent"] + r["cost_user"] for r in rs)
            mean_calls = statistics.mean(r["llm_calls"] for r in rs)
            L.append(
                f"| {cfg} | {len(rs)} | {mean_cost:.4f} | {mean_calls:.1f} | "
                + " | ".join(f"{100 * v / s:.0f}%" for v in parts.values())
                + f" | {mean_ctx} |"
            )
        L += [
            "",
            "Shares are of agent cost (user simulator excluded). Configurations have different "
            "numbers of runs on different tasks: compare them on the first task only (below).",
        ]
        first = phase_result.get("order", [None])[0]
        L += ["", f"### `C:top3` and the number of calls (first task `{first}`, one run each)", ""]
        L += [
            "| config | LLM calls | retrieval results | cut results | omitted items | "
            "result tokens in context (est.) | mean $ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        by = {r["config"]: r for r in sonnet if r["task"] == first}
        for cfg in CONFIG_ORDER:
            if cfg in by:
                r = by[cfg]
                L.append(
                    f"| {cfg} | {r['llm_calls']} | {r.get('retrieval_results', 0)} | "
                    f"{r.get('cut_results', 0)} | {r.get('cut_items_omitted', 0)} | "
                    f"{tok(r.get('retrieval_chars_in_context', 0))} | "
                    f"{r['cost_agent'] + r['cost_user']:.4f} |"
                )
        if "default" in by and "C:top3" in by:
            d, c3 = by["default"], by["C:top3"]
            L += [
                "",
                f"`C:top3` vs `default` on this task: LLM calls {c3['llm_calls']} vs "
                f"{d['llm_calls']}, retrieval calls {c3.get('retrieval_results', 0)} vs "
                f"{d.get('retrieval_results', 0)} (re-queries show up as extra retrieval calls). "
                "One run each: a first look, not a result.",
            ]
    else:
        L.append("No successful Sonnet run.")

    L += ["", "## Budget request (computed, TEST2.md formula)", ""]
    c = sonnet_costs(runs)
    req = allocation2.request(c)
    if "error" in req:
        L.append(f"Not computed: {req['error']} (have: {sorted(c)}).")
    else:
        L += [
            "Measured `c(x)` = mean cost per run, user simulator included: "
            + ", ".join(f"`{k}` ${v:.4f}" for k, v in c.items())
            + ".",
            "",
            f"- `Sigma` = **${req['sigma']:.4f}**; stand-ins `c(global*)` = ${req['c_global_standin']:.4f}"
            f" (cheapest of the four), `c(layer)` = ${req['c_layer_standin']:.4f} (= `c(default)`).",
            "- `T_needed = 1.1 x (4.0 + 20 x Sigma / 0.8 + (10 x c(default) + 30 x (c(default) + "
            f"c(global*) + c(layer))) / 0.8)` = **${req['t_needed']:.2f}**.",
            f"- At that total: `N_train` = {req['n_train']}, `N_test` = {req['n_test']} (the minimums; "
            f"sum {req['n_train'] + req['n_test']} <= 88: {'OK' if req['tasks_ok'] else 'NO'}).",
            f"- Without `EC` (three configurations): `T_needed` = ${req['t_needed_without_ec']:.2f}.",
            "",
            "If the total is lower, what each option buys (`N_train` with `N_test` held at 30; "
            "`N_test` with `N_train` held at 20; a 0 means that side cannot be afforded):",
            "",
            "| configurations | total | N_train (at N_test=30) | N_test (at N_train=20) | both minimums |",
            "|---|---:|---:|---:|---|",
        ]
        for o in req["options"]:
            L.append(
                f"| {o['configs']} | ${o['total']:.2f} ({o['at']}) | {o['n_train_at_min_test']} | "
                f"{o['n_test_at_min_train']} | {'yes' if o['meets_both_minimums'] else 'no'} |"
            )
        L += [
            "",
            "Caveat: the means rest on a few runs and, for `E:low` and `C:top3`, on the first task "
            "only, while `default` and `EC` may include tasks 2 and 3. Task cost varies a lot. "
            f"On the first task alone the same formula gives: {first_task_t_needed(runs, phase_result)}",
            "",
            "**Stop. The coordinator sets the budget.**",
        ]

    L += [
        "",
        "## OpenAI (and other non-ours) defaults in tau2",
        "",
        "| default | value | refs outside config.py | status |",
        "|---|---|---:|---|",
    ]
    for row in audit():
        L.append(f"| `{row['name']}` | {row['value']} | {len(row['uses'])} | {row['note']} |")

    spent = ledger_total["total"]
    L += [
        "",
        "## Budget",
        "",
        f"- Phase b1 spent (ledger): **${spent:.4f}** of ${cap:.2f}",
        f"- Total spent so far (test 2): ${spent + prior_total:.4f} of ${budget.total:.2f}",
        f"- API calls in the ledger: {ledger_total['calls']}; by purpose: "
        + ", ".join(f"{k} ${v:.4f}" for k, v in ledger_total["by_purpose"].items()),
        "- Console reconciliation: the ledger total for the UTC day beside the Console figure must "
        f"agree within 5%. {console_note}",
        "",
        "## What this cannot show",
        "",
        "- Whether the agent says the right things (ENV checks the database state only; "
        "`task_102`'s language check is dropped).",
        "- That quality is unchanged: with 30 to 40 test tasks a layer that truly loses 10 to 15 "
        "points can still pass. A KEEP earns a larger quality test, never a launch.",
        "- Anything about other agents, other retrieval variants or other models: this is one "
        "benchmark, one variant, Sonnet 5.5, one simulated user.",
        "- Anything about success rates or savings: this phase has 1 to 3 tasks. It checks the "
        "pipeline and measures costs for the budget request. Haiku numbers say nothing about Sonnet.",
        "",
    ]
    return "\n".join(L)


def first_task_t_needed(runs: list[dict[str, Any]], phase_result: dict[str, Any]) -> str:
    first = (phase_result.get("order") or [None])[0]
    c = {
        r["config"]: r["cost_agent"] + r["cost_user"]
        for r in runs
        if r["model"] == SONNET and r["task"] == first and r["status"] == "ok"
    }
    req = allocation2.request(c)
    return (
        f"`T_needed` = ${req['t_needed']:.2f}."
        if "error" not in req
        else f"not computed ({req['error']})."
    )
