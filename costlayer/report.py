"""Phase 1 report: numbers measured in the smoke test and the allocation table."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from costlayer import allocation
from costlayer.audit import audit
from costlayer.ledger import PHASE_CAPS_USD, TOTAL_CAP_USD
from costlayer.pricing import HAIKU, SONNET, est_tokens, price_for
from costlayer.runner import ORDER_SEED

PINNED = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"


def load_runs(out_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "runs").glob("*.json"))]


def visible_tokens(trace: dict[str, Any]) -> int:
    """Visible output (text plus tool-call JSON) of the agent's messages, estimated."""
    n = 0
    for m in trace["messages"]:
        if m["role"] == "assistant":
            n += est_tokens(m.get("content") or "")
            for c in m.get("tool_calls") or []:
                n += est_tokens(json.dumps({"name": c["name"], "input": c["arguments"]}))
    return n


def cost_by_type(model: str, t: dict[str, int]) -> dict[str, float]:
    p = price_for(model)
    return {
        "cache read": t["cache_read"] * p.cache_read / 1e6,
        "cache write": t["cache_write"] * p.cache_write / 1e6,
        "uncached input": t["input"] * p.input / 1e6,
        "output (thinking included)": t["output"] * p.output / 1e6,
    }


def mean_cost(runs: list[dict[str, Any]], config: str, model: str) -> float | None:
    xs = [
        r["cost_agent"] + r["cost_user"]
        for r in runs
        if r["config"] == config and r["model"] == model and r["status"] == "ok"
    ]
    return statistics.mean(xs) if xs else None


def build(
    phase_result: dict[str, Any],
    runs: list[dict[str, Any]],
    ledger_total: dict[str, Any],
    prior_total: float,
    traces_dir: Path | None,
    console_note: str,
) -> str:
    L: list[str] = ["# Phase 1 report: smoke test", ""]
    L += [
        f"Benchmark: `sierra-research/tau2-bench` at `{PINNED}` (retail, official train split, "
        f"seed-{ORDER_SEED} order, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`). "
        f"User simulator: `{HAIKU}`, same in every configuration. "
        "Cost per run below is agent plus user simulator unless stated.",
        "",
    ]
    L += ["## Verdict of the smoke test", ""]
    hc = phase_result.get("haiku_checks", {})
    for name, c in hc.items():
        L.append(f"- {'PASS' if c['pass'] else '**FAIL**'}: {name}. " + "; ".join(c["detail"]))
    L.append(f"- MANUAL: ledger total vs Console. {console_note}")
    if phase_result.get("stopped"):
        L += ["", f"**Stopped early:** {phase_result['stopped']}."]
    L += [
        "",
        "## Runs",
        "",
        "| model | config | task | status | reward | LLM calls | agent $ | "
        "user $ | input | cache write | cache read | output | termination |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        t = r["tokens"]
        L.append(
            f"| {r['model'].split('-')[1]} | {r['config']} | {r['task']} | {r['status']} | "
            f"{r['reward']:.0f} | {r['llm_calls']} | {r['cost_agent']:.4f} | "
            f"{r['cost_user']:.4f} | {t['input']} | {t['cache_write']} | {t['cache_read']} "
            f"| {t['output']} | {r['termination']} |"
        )

    sonnet = [r for r in runs if r["model"] == SONNET and r["status"] == "ok"]
    if sonnet:
        L += [
            "",
            "## Cost anatomy, Sonnet 5.5 (question 1)",
            "",
            "| config | runs | mean $/run (agent+user) | LLM calls/run | cache read | "
            "cache write | uncached input | output |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cfg in ("default", "E:medium", "E:low", "T:bt"):
            rs = [r for r in sonnet if r["config"] == cfg]
            if not rs:
                continue
            tot = {
                k: sum(r["tokens"][k] for r in rs)
                for k in ("input", "cache_write", "cache_read", "output")
            }
            parts = cost_by_type(SONNET, tot)
            s = sum(parts.values()) or 1.0
            mean = statistics.mean(r["cost_agent"] + r["cost_user"] for r in rs)
            L.append(
                f"| {cfg} | {len(rs)} | {mean:.4f}"
                f" | {statistics.mean(r['llm_calls'] for r in rs):.1f} | "
                + " | ".join(f"{100 * v / s:.0f}%" for v in parts.values())
                + " |"
            )
        L += ["", "Shares are of agent cost (user simulator excluded)."]
        L += ["", "### Thinking tokens (Sonnet)", ""]
        for cfg in ("default", "E:medium", "E:low", "T:bt"):
            rs = [r for r in sonnet if r["config"] == cfg]
            if not rs:
                continue
            out = sum(r["tokens"]["output"] for r in rs)
            calls = sum(r["llm_calls"] for r in rs) or 1
            rep = [
                r["thinking_tokens_reported"]
                for r in rs
                if r["thinking_tokens_reported"] is not None
            ]
            vis = None
            if traces_dir:
                v = []
                for r in rs:
                    f = (
                        Path(traces_dir)
                        / f"{cfg.replace(':', '-')}__{r['task']}__a{r['attempt']}.json"
                    )
                    if f.exists():
                        v.append(visible_tokens(json.loads(f.read_text())))
                vis = sum(v) if v else None
            if rep:
                th = f"{sum(rep) / calls:.0f}/call (reported by the API)"
            elif vis is not None:
                th = (
                    f"~{(out - vis) / calls:.0f}/call (approximate: output minus estimated visible)"
                )
            else:
                th = "n/a"
            L.append(f"- {cfg}: output {out / calls:.0f} tokens/call; thinking {th}")
        d = [r for r in sonnet if r["config"] == "default"]
        lo = [r for r in sonnet if r["config"] == "E:low"]
        if d and lo:
            od = sum(r["tokens"]["output"] for r in d) / sum(r["llm_calls"] for r in d)
            ol = sum(r["tokens"]["output"] for r in lo) / sum(r["llm_calls"] for r in lo)
            cw_d = sum(r["tokens"]["cache_write"] for r in d)
            cw_l = sum(r["tokens"]["cache_write"] for r in lo)
            L += [
                "",
                f"**E:low vs default:** output per call {ol:.0f} vs {od:.0f} "
                f"({100 * (ol / od - 1):+.0f}%); cache-write tokens {cw_l} vs {cw_d}. "
                "Few runs: treat as a first look, not a result.",
            ]
        for cfg in ("E:medium", "T:bt"):
            rs = [r for r in runs if r["model"] == SONNET and r["config"] == cfg]
            ok = bool(rs) and all(r["status"] == "ok" for r in rs)
            L.append(
                f"- `{cfg}` accepted by the API on Sonnet 5.5: "
                + ("yes" if ok else ("not run" if not rs else "**NO** (see runs)"))
            )

    c: dict[str, float] = {}
    for cfg in ("default", "E:medium", "E:low", "T:bt"):
        m = mean_cost(runs, cfg, SONNET)
        if m is not None:
            c[cfg] = m
    L += ["", "## Allocation table (computed, phases 2 and 3)", ""]
    if len(c) == 4:
        a = allocation.allocate(c)
        L += [
            "Measured `c(x)` = mean cost per run, user simulator included: "
            + ", ".join(f"`{k}` ${v:.4f}" for k, v in c.items())
            + ".",
            "",
            f"- Phase 2: `N_train = min(74, floor(0.8 x 9.00 / sum c))` = **{a['n_train_raw']}** "
            f"(minimum 12): {'OK' if a['train_ok'] else '**MINIMUM NOT MET**'}.",
            f"- Phase 3 (stand-ins `c(global*)` = ${a['c_global_standin']:.4f} = "
            f"min(default, E:low, T:bt), `c(layer)` = ${a['c_layer_standin']:.4f} = default +5%): "
            f"`N_test = min(40, floor((0.8 x 13.00 - 10 x c(default)) / (sum)))` = "
            f"**{a['n_test_raw']}** (minimum 20): "
            f"{'OK' if a['test_ok'] else '**MINIMUM NOT MET**'}.",
            "",
            "Caveat: `c(E:medium)` and `c(T:bt)` rest on one run each, and `c(default)`/`c(E:low)` "
            "on up to three. Task cost varies a lot, so these numbers are rough.",
        ]
        if a["options"]:
            L += ["", "**A minimum is not met. The founder decides; options and numbers:**", ""]
            for ph, opts in a["options"].items():
                for name, v in opts.items():
                    L.append(f"- {ph} / {name}: {json.dumps(v)}")
    else:
        L += [
            "Not computed: one of the four Sonnet configurations has no successful run "
            f"(have: {sorted(c)})."
        ]

    L += [
        "",
        "## OpenAI (and other non-ours) defaults in tau2 (phase 0 audit)",
        "",
        "Every default model name in `tau2/config.py`, where else it is referenced, and what we "
        "do about it. The workflow has no OpenAI key, so a stray call fails loudly.",
        "",
        "| default | value | refs outside config.py | status |",
        "|---|---|---:|---|",
    ]
    for row in audit():
        L.append(f"| `{row['name']}` | {row['value']} | {len(row['uses'])} | {row['note']} |")

    L += ["", "## Budget", ""]
    cap = PHASE_CAPS_USD[1]
    spent = ledger_total["total"]
    L += [
        f"- Phase 1 spent (ledger): **${spent:.4f}** of ${cap:.2f}",
        f"- Total spent so far: ${spent + prior_total:.4f} of ${TOTAL_CAP_USD:.2f} hard cap",
        f"- API calls in the ledger: {ledger_total['calls']}; by purpose: "
        + ", ".join(f"{k} ${v:.4f}" for k, v in ledger_total["by_purpose"].items()),
        f"- Console reconciliation: ledger total for the UTC day beside the Console figure "
        f"must agree within 5%. {console_note}",
    ]

    L += [
        "",
        "## What this cannot show",
        "",
        "- Whether the agent says the right things: `EvaluationType.ENV` checks only the final "
        "database state and environment assertions.",
        "- Anything about success rates or savings: this phase has 1 to 3 tasks per "
        "configuration. It only checks that the pipeline, the ledger and the settings work.",
        "- Haiku numbers say nothing about Sonnet.",
        "",
    ]
    return "\n".join(L)
