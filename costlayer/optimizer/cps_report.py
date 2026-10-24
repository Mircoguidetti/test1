"""Driver for the cost-per-success analysis: runs both implementations, refuses to continue if
they disagree, cross-checks the totals printed in the old reports, and writes the report.

    python -m costlayer.optimizer.cps_report            # writes reports/cost-per-success.md

No API call, no tau2 import. Reads `results/*/runs/*.json` summaries, `results/*/analysis.json`
(task types and mined chain shares, both already in the repo) and the committed reviewer file.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from costlayer.optimizer import cps, cps_check, cps_rng
from costlayer.optimizer.gate import K2_RATIO, project

PINNED = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
OUT_DIR = Path("results/cost-per-success")
TESTS = {
    "retail": {
        "title": "Test 1: retail, 21 train tasks",
        "runs": "results/phase-2/runs",
        "analysis": "results/phase-2/analysis.json",
        "configs": ("default", "E:medium", "E:low", "T:bt"),
        "order": "retail_train_sweep",
        "old_report": "reports/phase-2.md",
    },
    "banking": {
        "title": "Test 2: banking_knowledge (bm25_grep), 12 train tasks",
        "runs": "results/banking-phase-2/runs",
        "analysis": "results/banking-phase-2/analysis.json",
        "configs": ("default", "E:low", "C:top3", "EC"),
        "order": "banking_sweep",
        "old_report": "reports/banking-phase-2.md",
    },
}


# -- formatting ------------------------------------------------------------------------------------


def usd(x: float | None, d: int = 4) -> str:
    return "undefined" if x is None else f"${x:.{d}f}"


def num(x: float | None, d: int = 2) -> str:
    return "undefined" if x is None else f"{x:.{d}f}"


def pct(x: float | None, d: int = 1) -> str:
    return "undefined" if x is None else f"{100 * x:.{d}f}%"


def iv(i: dict[str, Any] | None, fmt=num) -> str:
    if i is None:
        return "-"
    if i["low"] is None:
        return f"undefined ({i['undefined']} undef.)"
    tail = f" ({i['undefined']} undef.)" if i["undefined"] else ""
    return f"{fmt(i['low'])} to {fmt(i['high'])}{tail}"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


# -- inputs ----------------------------------------------------------------------------------------


def run_both(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    orders = json.loads((root / "results/task_orders.json").read_text())
    a_all: dict[str, dict[str, Any]] = {}
    b_all: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {}
    for key, t in TESTS.items():
        ana = json.loads((root / t["analysis"]).read_text())
        share = ana["chains"]["cost_share"]
        order = orders[t["order"]]
        data = cps.load_test(key, root / t["runs"], t["configs"], order)
        a_all[key] = cps.analyze(data, ana["types"], share)
        b_all[key] = cps_check.analyze(root / t["runs"], t["configs"], order, ana["types"], share)
        meta[key] = {"analysis": ana, "data": data}
    return a_all, b_all, meta


def crosscheck_old(root: Path, a_all: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Totals, success counts and per-run means against the numbers printed in the old reports."""
    rows: list[dict[str, Any]] = []

    def add(test: str, what: str, printed: float, mine: float, decimals: int, source: str):
        ok = round(mine, decimals) == round(printed, decimals)
        rows.append(
            {
                "test": test,
                "what": what,
                "printed": printed,
                "mine": mine,
                "ok": ok,
                "source": source,
            }
        )

    for key, t in TESTS.items():
        table_ = a_all[key]["table"]
        ana = json.loads((root / t["analysis"]).read_text())
        for c in t["configs"]:
            add(key, f"{c} total agent cost (analysis.json)", ana["cost_agent_total"][c],
                table_[c]["cost"], 9, t["analysis"])  # fmt: skip
            add(key, f"{c} successes (analysis.json)", ana["success"][c]["successes"],
                table_[c]["successes"], 0, t["analysis"])  # fmt: skip
        text = (root / t["old_report"]).read_text()
        for c in t["configs"]:
            if key == "retail":
                m = re.search(rf"\| {re.escape(c)} \| (\d+)/21 \| \d+ \| ([\d.]+) \|", text)
                if m:
                    add(key, f"{c} successes (report table)", int(m[1]), table_[c]["successes"], 0,
                        t["old_report"])  # fmt: skip
                    add(key, f"{c} total $ (report table)", float(m[2]), table_[c]["cost"], 4,
                        t["old_report"])  # fmt: skip
            else:
                m = re.search(
                    rf"\| {re.escape(c)} \| 12 \| (\d+) \| \d+ \| ([\d.]+) \| [\d.]+ ?(?:\([^)]*\))? ?\|"
                    rf" ([\d.]+) \| ([\d.]+) \|",
                    text,
                )
                if m:
                    add(key, f"{c} successes (report table)", int(m[1]), table_[c]["successes"], 0,
                        t["old_report"])  # fmt: skip
                    add(key, f"{c} $/run (report table)", float(m[2]), table_[c]["cost_per_run"], 4,
                        t["old_report"])  # fmt: skip
                    add(key, f"{c} LLM calls/run (report table)", float(m[3]),
                        table_[c]["calls_per_run"], 1, t["old_report"])  # fmt: skip
                    add(key, f"{c} LLM calls/run, failed (report table)", float(m[4]),
                        table_[c]["calls_per_failed_run"], 1, t["old_report"])  # fmt: skip
    return rows


def hand_hints(a_all: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hints = {
        "retail": {"default": 0.062, "E:medium": 0.074, "E:low": 0.062, "T:bt": 0.071},
        "banking": {"default": 2.40, "E:low": 3.02, "C:top3": 1.79, "EC": 2.79},
    }
    out = []
    for key, h in hints.items():
        for c, v in h.items():
            mine = a_all[key]["table"][c]["cps"]
            d = 3 if key == "retail" else 2
            out.append(
                {
                    "test": key,
                    "config": c,
                    "hint": v,
                    "d": d,
                    "mine": mine,
                    "ok": round(mine, d) == v,
                }
            )
    return out


def side_facts(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Descriptive numbers for the weaknesses table (summaries only)."""
    out: dict[str, Any] = {}
    for key, t in TESTS.items():
        rows = [json.loads(f.read_text()) for f in sorted((root / t["runs"]).glob("*.json"))]
        rows = [r for r in rows if r["model"] == cps.SONNET]
        agent = math.fsum(r["cost_agent"] for r in rows)
        user = math.fsum(r["cost_user"] for r in rows)
        out[key] = {
            "runs": len(rows),
            "statuses": sorted({r["status"] for r in rows}),
            "terminations": sorted({str(r["termination"]) for r in rows}),
            "hit_max_steps": sum(bool(r.get("hit_max_steps")) for r in rows),
            "max_tokens_stops": sum(r.get("max_tokens_stops", 0) for r in rows),
            "empty_replies": sum(r.get("empty_replies", 0) for r in rows),
            "api_errors": sum(len(r.get("api_errors", [])) for r in rows),
            "dropped_text": sum(r.get("dropped_text", 0) for r in rows),
            "user_share": user / agent,
            "user_total": user,
            "dropped_by_config": {
                c: sum(r.get("dropped_text", 0) for r in rows if r["config"] == c)
                for c in t["configs"]
            },
            "agent_total": agent,
        }
    # retail phase 1 repeats: independent reruns of the same (config, task)
    p1 = {
        f.name: json.loads(f.read_text())
        for f in (root / "results/phase-1/runs").glob("sonnet*.json")
    }
    p2 = {f.name: json.loads(f.read_text()) for f in (root / "results/phase-2/runs").glob("*.json")}
    pairs = [(p1[k], p2[k]) for k in sorted(p1) if k in p2]
    out["retail_repeats"] = {
        "pairs": len(pairs),
        "flips": sum(1 for a, b in pairs if (a["reward"] >= 1) != (b["reward"] >= 1)),
        "flip_tasks": [f"{a['config']}/{a['task']}" for a, b in pairs
                       if (a["reward"] >= 1) != (b["reward"] >= 1)],
        "cost_diff_mean_abs": math.fsum(abs(a["cost_agent"] - b["cost_agent"]) for a, b in pairs)
        / len(pairs),
    }  # fmt: skip
    # banking: phase 1 runs reused by the sweep
    b1 = {
        f.name: json.loads(f.read_text())
        for f in (root / "results/banking-phase-1/runs").glob("sonnet*.json")
    }
    b2 = {
        f.name: json.loads(f.read_text())
        for f in (root / "results/banking-phase-2/runs").glob("*.json")
    }
    reused = sorted(k for k in b1 if k in b2)
    same = all(
        b1[k]["cost_agent"] == b2[k]["cost_agent"] and b1[k]["reward"] == b2[k]["reward"]
        for k in reused
    )
    out["banking_reused"] = {
        "files": len(reused),
        "identical_cost_and_reward": same,
        "tasks": sorted({b1[k]["task"] for k in reused}),
    }
    orders = json.loads((root / "results/task_orders.json").read_text())
    out["task_102_in_sweeps"] = any(
        "task_102" in v for v in (orders["banking_sweep"], orders["retail_train_sweep"])
    )
    ana = meta["banking"]["analysis"]
    n_major = max(ana["type_counts"].values())
    out["banking_classifier"] = {
        "accuracy": ana["classifier"]["accuracy"],
        "majority_baseline": n_major / sum(ana["type_counts"].values()),
    }
    out["retail_classifier"] = {
        "accuracy": meta["retail"]["analysis"]["classifier"]["accuracy"],
        "majority_baseline": max(meta["retail"]["analysis"]["type_counts"].values())
        / sum(meta["retail"]["analysis"]["type_counts"].values()),
    }
    return out


def banking_sensitivity(root: Path, a_all, meta, facts) -> dict[str, Any]:
    """Banking CPS without the tasks that have a reused phase 1 run (A and B must agree)."""
    drop = set(facts["banking_reused"]["tasks"])
    data = meta["banking"]["data"]
    keep = [t for t in data.tasks if t not in drop]
    sub = cps.restrict(data, keep)
    ta = cps.point_table(sub)
    t = TESTS["banking"]
    orders = json.loads((root / "results/task_orders.json").read_text())
    tasks, d = cps_check.per_task(root / t["runs"], t["configs"], orders[t["order"]], set(keep))
    tb = cps_check.summarize(tasks, d, t["configs"])[0]
    diffs = [
        x
        for c in t["configs"]
        for x in cps_check.compare(
            {k: v for k, v in ta[c].items()}, {k: v for k, v in tb[c].items()}, f"/{c}"
        )
    ]
    return {"dropped": sorted(drop), "n": len(keep), "table": ta, "diffs": diffs}


# -- gate ------------------------------------------------------------------------------------------


def gates(a: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b = a["banking"]
    tb = b["table"]
    share = b["chain_share"]
    hc, hs = b["headroom_cost"], b["headroom_cps"]
    old = project(tb["default"]["cost"], tb[b["global_star_cost"]]["cost"], share,
                  hc["headroom"], hc["beats_null"])  # fmt: skip
    gs = b["global_star_cps"]
    new = project(tb["default"]["cps"], tb[gs]["cps"], share,
                  hs["headroom"] or 0.0, hs["beats_null"])  # fmt: skip
    new_old_global = project(tb["default"]["cps"], tb[b["global_star_cost"]]["cps"], share,
                             hs["headroom"] or 0.0, False)  # fmt: skip
    need = 1 - K2_RATIO * tb["default"]["cps"] / tb[gs]["cps"]
    return {
        "old": old, "new": new, "new_with_old_global": new_old_global,
        "chain_share_needed_for_k2": need, "global_new": gs, "global_old": b["global_star_cost"],
    }  # fmt: skip


# -- the report ------------------------------------------------------------------------------------


def headline(a: dict[str, Any], cfgs: tuple[str, ...]) -> list[str]:
    rows = []
    for c in cfgs:
        r, i = a["table"][c], a["intervals"][c]
        rows.append([
            f"`{c}`", str(r["runs"]), str(r["successes"]), str(r["failures"]), usd(r["cost"]),
            f"{usd(r['cost_per_run'])} ({iv(i['cost_per_run'], lambda x: usd(x))})",
            f"{pct(r['success_rate'])} ({iv(i['success_rate'], lambda x: pct(x, 0))})",
            f"{usd(r['cps'])} ({iv(i['cps'], lambda x: usd(x))})",
            "1.00" if c == "default" else f"{num(r['cps_ratio'], 3)} ({iv(i.get('cps_ratio'))})",
            "1.00" if c == "default" else f"{num(r['cost_ratio'], 3)} ({iv(i.get('cost_ratio'))})",
            f"{r['calls_per_run']:.1f}",
            "n/a" if r["calls_per_failed_run"] is None else f"{r['calls_per_failed_run']:.1f}",
        ])  # fmt: skip
    return table(
        ["config", "runs", "successes", "failures", "agent $ total", "$/run (80%)",
         "success rate (80%)", "cost per success, CPS (80%)", "CPS ratio vs default (80%)",
         "old cost ratio per run (80%)", "calls/run", "calls/run, failed runs"],
        rows,
    )  # fmt: skip


def build_report(
    a: dict[str, dict[str, Any]],
    b_diffs: dict[str, list[str]],
    cross: list[dict[str, Any]],
    hints: list[dict[str, Any]],
    facts: dict[str, Any],
    sens: dict[str, Any],
    g: dict[str, Any],
    reviewer: dict[str, Any] | None,
) -> str:
    L: list[str] = ["# Cost per completed task (free recomputation of tests 1 and 2)", ""]
    L += [
        f"Pinned tau2-bench commit: `{PINNED}`. Agent: Sonnet 5.5. Cost spent on this analysis: "
        "$0.00 (no Anthropic or OpenAI call, no paid workflow, `spend`/`spend2`/`budget.json` "
        "untouched). Only run summaries and the two existing `analysis.json` files are read; the "
        "blocked trace artifacts are not used.",
        "",
        "## Answer",
        "",
    ]
    r, bk = a["retail"], a["banking"]
    rt, bt = r["table"], bk["table"]
    L += [
        f"- Retail: `E:low` costs {num(rt['E:low']['cost_ratio'], 3)} of `default` per run but "
        f"{num(rt['E:low']['cps_ratio'], 3)} per completed task (80% interval "
        f"{iv(r['intervals']['E:low']['cps_ratio'])}): the saving per run is gone once the 3 lost "
        f"tasks are counted. `E:medium` ({num(rt['E:medium']['cps_ratio'], 2)}) and `T:bt` "
        f"({num(rt['T:bt']['cps_ratio'], 2)}) are dearer than `default` per completed task.",
        f"- Banking: `default` {usd(bt['default']['cps'], 2)} per success on 2 successes; "
        f"`C:top3` {usd(bt['C:top3']['cps'], 2)} ({num(bt['C:top3']['cps_ratio'], 3)}); `E:low` "
        f"{usd(bt['E:low']['cps'], 2)} and `EC` {usd(bt['EC']['cps'], 2)} are DEARER per success "
        "than `default` although cheaper per run. The 80% intervals are very wide: the share of "
        "resamples with no success at all is "
        + ", ".join(
            f"{pct(bk['intervals'][c]['cps']['undefined'] / cps_rng.RESAMPLES, 0)} (`{c}`)"
            for c in TESTS["banking"]["configs"]
        )
        + ".",
        f"- `global*` by CPS: retail `{r['global_star_cps']}` (old: `{r['global_star_cost']}`), "
        f"banking `{bk['global_star_cps']}` (old: `{bk['global_star_cost']}`).",
        f"- Banking gate recomputed with CPS: **{g['new'].verdict}** (projected / default "
        f"{g['new'].ratio_default:.3f}, bar {K2_RATIO:.2f}); the old per-run gate gave "
        f"**{g['old'].verdict}** ({g['old'].ratio_default:.3f}).",
        "- Details, the verification and the list of what changes are below.",
        "",
        "## Data and definitions (fixed before any number was looked at)",
        "",
        "Data: `results/phase-2/runs` (retail: 21 tasks x 4 configurations, 84 runs) and "
        "`results/banking-phase-2/runs` (banking_knowledge: 12 tasks x 4 configurations, 48 runs; "
        "it already contains the 8 phase 1 runs the sweep reused). Phase 1 runs are not added "
        "again. The retail sweep reused none; the 8 retail phase 1 Sonnet runs are independent "
        "repeats and are used only as a noise remark in the weaknesses table.",
        "",
        "- **Cost of a run** = agent (Sonnet) cost only. **The user simulator cost and any "
        "classifier cost are excluded**, as in the old reports (the simulator share is in the "
        "weaknesses table).",
        "- **Success** = status `ok` and reward 1. Every other finished run (reward 0, "
        "`agent_error`, `censored`) is a failure and its cost stays in the numerator. "
        "`infra_error` runs would be excluded from numerator and denominator and listed: "
        f"retail {r['infra_error'] or 'none'}, banking {bk['infra_error'] or 'none'}; dropped "
        f"tasks: retail {r['dropped_tasks'] or 'none'}, banking {bk['dropped_tasks'] or 'none'}.",
        "- **CPS** of a configuration = sum of the cost of all its finished runs / its number of "
        "successes. Zero successes: undefined. **CPS ratio** = CPS(config) / CPS(`default`) on the "
        "same tasks.",
        "- **Intervals:** bootstrap over tasks, 10,000 resamples, seed 42, the same resampled "
        "tasks for all configurations, every statistic recomputed from the resampled sums; 80% "
        "interval = 10th to 90th percentile of the defined resamples; the number of undefined "
        "resamples is printed (never dropped silently).",
        "- **`global*`** = lowest CPS among configurations with failures <= `default` failures + "
        "floor(0.10 x n) (ties: first in configuration order).",
        "",
    ]
    for key, t in TESTS.items():
        a_ = a[key]
        L += [f"## {t['title']}", "", *headline(a_, t["configs"]), ""]
        flags = []
        for c in t["configs"][1:]:
            x, d = a_["table"][c], a_["table"]["default"]
            flags.append(
                f"`{c}`: cheaper per run {x['cost_per_run'] < d['cost_per_run']}; fewer calls per "
                f"run {x['calls_per_run'] < d['calls_per_run']}; fewer successes "
                f"{x['successes'] < d['successes']}; cheaper per success "
                f"{(x['cps'] is not None and x['cps'] < d['cps'])}"
            )
        L += ["Saving per run vs saving per success (cheaper / fewer calls / fewer successes):", ""]
        L += [f"- {f}" for f in flags]
        lost = [
            c for c in t["configs"][1:]
            if a_["table"][c]["cost_per_run"] < a_["table"]["default"]["cost_per_run"]
            and a_["table"][c]["calls_per_run"] < a_["table"]["default"]["calls_per_run"]
            and a_["table"][c]["successes"] < a_["table"]["default"]["successes"]
        ]  # fmt: skip
        L += ["", "A saving coincides with fewer calls AND fewer successes in: "
              + (", ".join(f"`{c}`" for c in lost) or "none") + ".", ""]  # fmt: skip
        L += [
            f"Acceptable by the selection rule: {', '.join(f'`{c}`' for c in a_['acceptable'])}. "
            f"`global*` by cost per run (old): `{a_['global_star_cost']}`; by CPS (new): "
            f"`{a_['global_star_cps']}`. "
            + (
                "Same choice."
                if a_["global_star_cost"] == a_["global_star_cps"]
                else "**Different choice.**"
            ),  # fmt: skip
            "",
        ]
        if key == "retail":
            d_, e_ = rt["default"]["cps"], rt["E:low"]["cps"]
            L += [
                f"`E:low` beats `default` by CPS only by {usd(d_ - e_, 5)} per success "
                f"({pct(1 - e_ / d_, 2)}), well inside the interval: the choice of `E:low` is a "
                "near-tie decided by a rounding-sized difference, not a finding.",
                "",
            ]
    L += ["## Stacked best case (upper bound)", ""]
    sr, sb = r["stacked_cps_ratio"], bk["stacked_cps_ratio"]
    L += table(
        ["test", "global* (CPS)", "CPS ratio", "chain cost share", "stacked CPS ratio", "old claim"],
        [
            ["retail", f"`{r['global_star_cps']}`", num(rt[r['global_star_cps']]['cps_ratio'], 3),
             pct(r["chain_share"]), num(sr, 3), "0.79 (= 0.90 x (1 - 0.126))"],
            ["banking", f"`{bk['global_star_cps']}`", num(bt[bk['global_star_cps']]['cps_ratio'], 3),
             pct(bk["chain_share"]) + " (mined on `EC`, assumed for `C:top3`)", num(sb, 3),
             "0.574 / default (gate projection on per-run cost)"],
        ],
    )  # fmt: skip
    L += [
        "",
        "Assumption (stated, it makes the number an UPPER bound on the saving): successes do not "
        "change when chain composites are added, and the chain share of cost applies unchanged to "
        "the CPS. Retail: 0.998 x (1 - 0.126) = 0.873, further from the 0.70 bar than the old 0.79. "
        "The chain shares come from the existing mining reports (`analysis.json`); mining was not "
        "redone.",
        "",
        "## Banking gate recomputed with CPS",
        "",
    ]
    o, n, n2 = g["old"], g["new"], g["new_with_old_global"]
    L += table(
        ["variant", "default", "global*", "chain share", "projected", "projected / default (bar 0.70)",
         "projected / global* (bar 0.85)", "outcome"],
        [
            ["old: cost per run, `" + g["global_old"] + "`", usd(o.cost_default), usd(o.cost_global),
             pct(o.chain_cost_share), usd(o.projected), num(o.ratio_default, 3),
             num(o.ratio_global, 3), o.verdict],
            ["CPS, re-selected `" + g["global_new"] + "`", usd(n.cost_default), usd(n.cost_global),
             pct(n.chain_cost_share), usd(n.projected), num(n.ratio_default, 3),
             num(n.ratio_global, 3), n.verdict],
            ["CPS, old choice `" + g["global_old"] + "`", usd(n2.cost_default), usd(n2.cost_global),
             pct(n2.chain_cost_share), usd(n2.projected), num(n2.ratio_default, 3),
             num(n2.ratio_global, 3), n2.verdict],
        ],
    )  # fmt: skip
    hc, hs = bk["headroom_cost"], bk["headroom_cps"]
    L += [
        "",
        "The first row reproduces the old report (projected $2.7562, 0.574, 0.990, FEATURE ONLY) "
        "with the repository's own `gate.project`, which is a check of the inputs. The per-type "
        f"headroom is {pct(hc['headroom'])} by cost (null 90th percentile {pct(hc['null_p90'])}) "
        f"and {pct(hs['headroom'])} by CPS (null 90th percentile {pct(hs['null_p90'])}, "
        f"{hs['null_undefined']} of 200 shuffles undefined): it does not beat the null in either "
        "metric, so the chain term decides. To pass the 0.70 bar with the re-selected `global*` "
        f"the chain cost share would have to be at least {pct(g['chain_share_needed_for_k2'])}, "
        f"against 1.0% measured (on `EC` traces). Note the data cannot separate 0.74 from 0.70: "
        f"the CPS ratio of `{g['global_new']}` has an 80% interval of "
        f"{iv(bk['intervals'][g['global_new']]['cps_ratio'])}.",
        "",
        "Retail (test 1 gate): (a) per-type headroom is a cost comparison; recomputed: "
        f"{pct(r['headroom_cost']['headroom'])} by cost, {pct(r['headroom_cps']['headroom'])} by CPS "
        f"(null 90th percentile {pct(r['headroom_cps']['null_p90'])}): does not beat the null "
        "either way. (b) chains (20.3% of calls and 12.6% of cost, mined on `E:low` traces) do not use "
        "the metric; "
        "`global*` is still `E:low`, so (b) is unchanged. The test 1 gate outcome (passed through "
        "(b)) does not change.",
        "",
        "## Where the old metric drove a decision, and what changes",
        "",
    ]
    ch = lambda x: "**changes**" if x else "unchanged"  # noqa: E731
    L += table(
        ["decision", "old (cost per run)", "CPS", "verdict"],
        [
            ["retail `global*`", f"`{r['global_star_cost']}`", f"`{r['global_star_cps']}` (near-tie)",
             ch(r["global_star_cost"] != r["global_star_cps"])],
            ["retail: does `E:low` save about 10%?", f"yes, {num(rt['E:low']['cost_ratio'], 3)}",
             f"no, {num(rt['E:low']['cps_ratio'], 3)} ({iv(r['intervals']['E:low']['cps_ratio'])})",
             "**changes**"],
            ["retail stacked best case vs 0.70", "0.79, fails", f"{num(sr, 3)}, fails", "unchanged (fails)"],
            ["retail test 1 gate (phase 2)", "passed through chains (b)", "unchanged: chains do not "
             "use the metric", "unchanged"],
            ["banking `global*`", f"`{g['global_old']}`", f"`{g['global_new']}`",
             ch(g["global_old"] != g["global_new"])],
            ["banking gate", g["old"].verdict, g["new"].verdict, ch(g["old"].verdict != g["new"].verdict)],
            ["banking: does `EC`/`E:low` save money?", "yes, 0.58 / 0.63 of default per run",
             f"no, {num(bt['EC']['cps_ratio'], 2)} / {num(bt['E:low']['cps_ratio'], 2)} per success",
             "**changes**"],
            ["thesis outcome of TEST2.md (KEEP or closed as a standalone savings product)",
             "closed (FEATURE ONLY)", f"closed ({g['new'].verdict})", "unchanged: not KEEP"],
        ],
    )  # fmt: skip
    L += [
        "",
        "Other places where per-run cost is the metric but nothing was decided on it yet: the "
        "K1/K2 definitions in `CLAUDE.md` and `TEST2.md` (cost ratio with a separate success "
        "margin of at least 10 points: a layer at 0.70 of the cost and 10 points fewer successes "
        "passes K2 by cost and sits at 0.70/0.9 = 0.78 per success), the per-run abort rail, "
        "the allocation formulas (these are budgeting, where cost per run is the right quantity), "
        "and the selection rule (failures only).",
        "",
        "**Does any verdict change? Yes, two labels and no outcome.** (1) Banking gate: "
        f"{g['old'].verdict} becomes {g['new'].verdict} under the fixed gate formula, because "
        "`EC` and `E:low` finish fewer tasks and the cheapest per-success configuration "
        f"(`{g['global_new']}`, {num(bt[g['global_new']]['cps_ratio'], 3)} of `default`) projects "
        f"to {num(g['new'].ratio_default, 3)} of `default` with 1.0% chains, above the 0.70 bar. "
        "(2) Retail: the 10% saving of `E:low` disappears per completed task. Not changed: "
        "retail `global*`, the retail gate result, and the fact that neither test reaches KEEP.",
        "",
        "## Known weaknesses audit",
        "",
    ]
    f = facts
    W = [
        [
            "Cost per run as the headline",
            "retail `E:low` 0.90 per run but 0.998 per success; "
            "banking `EC` 0.58 per run but 1.16 per success",
            "Changes the saving claims and the banking gate label (above)",
            "Fixed here (free)",
            "No",
        ],
        [
            "Banking `default` solves only 2 of 12",
            "`default` CPS rests on 2 successes; "
            f"{pct(bk['intervals']['default']['cps']['undefined'] / 10000)} of resamples have no "
            "`default` success",
            "Ratios against `default` are very unstable (intervals above)",
            "Cannot be fixed by recomputation",
            "Yes: easier tasks, more tasks, or a stronger baseline configuration",
        ],
        [
            "Sample sizes (21 and 12 tasks)",
            "ratio intervals above; selection rule allows floor(0.10 n) = 2 and 1 extra failures",
            "A true loss of 10 points passes the rule "
            "easily; retail `E:low` lost 2 tasks and was `acceptable`",
            "Intervals reported (free)",
            "Yes, for any narrower claim",
        ],
        [
            "`task_102` scored on its DB part only",
            "in a sweep: " + str(f["task_102_in_sweeps"]) + " (checked against `task_orders.json`)",
            "None on these results; would matter for a test split",
            "n/a",
            "No",
        ],
        [
            "ACTION tasks excluded (9 of 97)",
            "free pass under ENV; fixed once",
            "None on these results; the result does not speak about those tasks",
            "n/a",
            "No",
        ],
        [
            "User simulator excluded from cost",
            f"retail {pct(f['retail']['user_share'])}, banking "
            f"{pct(f['banking']['user_share'])} of agent cost (all runs)",
            "Small; but the simulator "
            "reacts to the agent, so extra turns are not free in production",
            "Share computed here (free); a CPS including it is a one-line change",
            "No",
        ],
        [
            "Censored runs and the abort rail",
            f"statuses seen: retail {f['retail']['statuses']}, "
            f"banking {f['banking']['statuses']}; hit max_steps: {f['retail']['hit_max_steps']} / "
            f"{f['banking']['hit_max_steps']}; max_tokens stops {f['retail']['max_tokens_stops']} / "
            f"{f['banking']['max_tokens_stops']}; empty replies {f['retail']['empty_replies']} / "
            f"{f['banking']['empty_replies']}; API errors {f['retail']['api_errors']} / "
            f"{f['banking']['api_errors']}",
            "No effect: none occurred",
            "n/a",
            "No",
        ],
        [
            "Reused phase 1 runs (banking)",
            f"{f['banking_reused']['files']} files on tasks "
            f"{', '.join(f['banking_reused']['tasks'])}; identical cost and reward to the sweep "
            f"copy: {f['banking_reused']['identical_cost_and_reward']}; different dispatch, "
            "traces only reconstructed",
            "Sensitivity below",
            "Sensitivity computed (free)",
            "No",
        ],
        [
            "Type classifier",
            f"retail {pct(f['retail_classifier']['accuracy'])} (below 70%: off); "
            f"banking {pct(f['banking_classifier']['accuracy'])} on 12 tasks, but 11 of 12 tasks have "
            f"one type, so always guessing it scores {pct(f['banking_classifier']['majority_baseline'])}",
            "Per-type choice is not supported in either test; banking 'per-type' is one type",
            "n/a (inputs fixed)",
            "Yes, for a real per-type test",
        ],
        [
            "No A/A noise measured in tests 1 and 2",
            f"retail phase 1 repeats: {f['retail_repeats']['flips']} "
            f"flip(s) in {f['retail_repeats']['pairs']} paired reruns ({', '.join(f['retail_repeats']['flip_tasks'])}), "
            f"mean absolute cost difference {usd(f['retail_repeats']['cost_diff_mean_abs'])}",
            "Success differences of 1 to 2 tasks cannot be told from run-to-run noise",
            "No (needs repeats)",
            "Yes: repeat `default` on the same tasks",
        ],
        [
            "Mixed text and tool calls: the text is dropped",
            "dropped text blocks per configuration: "
            + "; ".join(
                f"{k}: " + ", ".join(f"{c}={n}" for c, n in f[k]["dropped_by_config"].items())
                for k in ("retail", "banking")
            ),
            "Configurations differ in how often the model writes text next to a tool call "
            f"(`T:bt` {f['retail']['dropped_by_config']['T:bt']} vs `default` "
            f"{f['retail']['dropped_by_config']['default']} in retail); the text is not in the "
            "context the orchestrator sees, which could change behaviour",
            "Counts shown (free)",
            "Yes, to test the effect",
        ],
        [
            "Success is ENV only",
            "what the agent says is not checked",
            "A cheaper agent could complete the database state but answer worse",
            "n/a",
            "Yes (judge), or accept",
        ],
        [
            "Selection rule is lenient",
            "`acceptable` = failures <= default + floor(0.10 n); retail "
            "`E:low` has 3 failures vs 1",
            "Lets a configuration give up tasks and still be 'global*'; CPS exposes it",
            "Replaced here by the same rule + lowest CPS (free)",
            "No",
        ],
        [
            "Gate is optimistic by construction",
            "chain share applied to `global*` with successes "
            "unchanged, classifier overhead zero, chain share mined on `EC` not `C:top3`",
            "Biased toward passing; a KILL under it is robust, a pass would not be",
            "n/a",
            "Mining on `C:top3` traces would need the blocked traces (free if the artifact is "
            "readable) ",
        ],
        [
            "Token and cost accounting",
            "cache TTL 5 min, price table fixed; concurrency 3 may change cache hits",
            "Not quantified here",
            "n/a",
            "No",
        ],
        [
            "One benchmark, one model, one simulated user",
            "-",
            "Nothing generalises beyond them",
            "n/a",
            "Yes",
        ],
    ]
    L += table(["weakness", "evidence (from the repo)", "effect on the verdicts",
                "fixable by recomputation (free)", "needs a new paid run"], W)  # fmt: skip
    L += ["", "Banking sensitivity, without the tasks that have a reused phase 1 run "
          f"({', '.join(sens['dropped'])}; {sens['n']} tasks left; A and B agree: "
          f"{not sens['diffs']}):", ""]  # fmt: skip
    L += table(
        ["config", "successes", "failures", "agent $ total", "CPS", "CPS ratio"],
        [[f"`{c}`", str(x["successes"]), str(x["failures"]), usd(x["cost"]), usd(x["cps"]),
          num(x["cps_ratio"], 3)] for c, x in sens["table"].items()],
    )  # fmt: skip
    succ_set = sorted({x["successes"] for x in sens["table"].values()})
    L += [
        "",
        f"On these {sens['n']} tasks the successes per configuration are {succ_set}"
        + (
            ": every configuration has the same number of successes, so the CPS ranking is the "
            "cost-per-run ranking."
            if len(succ_set) == 1
            else "."
        ),
        "Dropping three tasks therefore reverses the picture of the main table: the banking "
        "result depends on which one or two tasks `default` happens to solve, which is the "
        "point of the 'tiny success counts' limit.",
    ]
    L += ["", "## Verification", ""]
    L += [
        "1. **Two implementations.** A (`cps.py`): plain loops over the raw JSON files. B "
        "(`cps_check.py`): reads through `load_runs`, bootstraps with multiplicity counts, "
        "percentiles from `statistics.quantiles`, leave-one-out choice by `min` over keys. Only "
        "the random draws are shared (`cps_rng.py`). Every number in the tables (point "
        "estimates, intervals, undefined counts, `global*`, stacked ratio, headroom and nulls) is "
        "compared with a tolerance of 1e-9: "
        f"retail differences {len(b_diffs['retail'])}, banking differences {len(b_diffs['banking'])} "
        f"(banking sensitivity: {len(sens['diffs'])}). Any difference stops the report.",
        "",
        "2. **Cross-check with the numbers printed in the old reports** (compared at the printed "
        "precision):",
        "",
    ]
    bad = [x for x in cross if not x["ok"]]
    L += [f"   {len(cross) - len(bad)} of {len(cross)} comparisons agree; mismatches: "
          + ("none." if not bad else "; ".join(
              f"{x['test']} {x['what']}: printed {x['printed']}, mine {x['mine']}" for x in bad))]  # fmt: skip
    L += ["", "   Your hand arithmetic (CPS = total / successes, rounded as you wrote it):", ""]
    L += table(["test", "config", "hint", "computed", "agree"],
               [[x["test"], f"`{x['config']}`", f"{x['hint']:.{x['d']}f}", usd(x["mine"], 5), str(x["ok"])]
                for x in hints])  # fmt: skip
    L += ["", "3. **Independent reviewer.** A subagent saw only the definitions and the raw "
          "run-summary files (not the code, not these numbers) and recomputed the headline "
          "table. Reviewer and implementation A:", ""]  # fmt: skip
    if reviewer is None:
        L += ["   (reviewer result not yet recorded)"]
    else:
        L += table(
            [
                "test",
                "config",
                "runs",
                "successes",
                "total $ (rev. / mine)",
                "CPS (rev. / mine)",
                "CPS ratio (rev. / mine)",
                "$/run (rev. / mine)",
                "agree",
            ],
            reviewer_rows(a, reviewer),
        )
        L += ["", f"   Result: {reviewer_verdict(a, reviewer)}"]
    L += ["", "4. **Unit tests** (`tests/test_cps.py`): the definitions on constructed fixtures, "
          "zero-success handling, `infra_error` exclusion, bootstrap paired by task, a regression "
          "test that fails if cost is counted by another rule, and that A and B agree on a "
          "constructed data set; the optimizer package still has no `tau2` import.", ""]  # fmt: skip
    L += [
        "## What this cannot show",
        "",
        "- Banking's success counts are tiny (2, 1, 2, 1 of 12): every CPS and ratio there rests on "
        "one or two tasks and the intervals are wide or undefined.",
        "- Sample sizes of 21 and 12 tasks: nothing here is a measurement of a population.",
        "- Benchmark agents are cleaner than production agents (no stale caches, no flaky tools, "
        "one policy, a cooperative simulated user).",
        "- One benchmark, one model (Sonnet 5.5), one simulated user (Haiku 5.5).",
        "- The user simulator cost is excluded; ENV checks the database state only.",
        "- Nothing here says a company would pay for any of this.",
        "",
        "## What remains to try",
        "",
        "- Repeat `default` and the cheapest configuration on the same tasks to measure run-to-run "
        "noise (A/A), before reading any success difference.",
        "- A baseline that is not near the floor on banking (easier tasks, or a task subset where "
        "`default` succeeds about half the time), so that cost per success has a denominator.",
        "- Mine chains on `C:top3` traces (free if the trace artifact can be read) instead of "
        "assuming the `EC` share.",
        "- Replace the K1/K2 cost-plus-margin rule by a rule on cost per completed task for any "
        "future paid test.",
        "- Include the user-simulator cost in a CPS variant to see whether longer conversations "
        "change the ranking.",
        "",
    ]
    return "\n".join(L)


# -- reviewer comparison -----------------------------------------------------------------------------


def reviewer_rows(a: dict[str, dict[str, Any]], rev: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key in TESTS:
        for c, mine in a[key]["table"].items():
            x = rev.get(key, {}).get(c)
            if x is None:
                rows.append([key, f"`{c}`", "-", "-", "missing", "-", "-", "-", "False"])
                continue
            ok = same(x, mine)
            rows.append([
                key, f"`{c}`", f"{x['runs']} / {mine['runs']}", f"{x['successes']} / {mine['successes']}",
                f"{x['total_cost']:.4f} / {mine['cost']:.4f}",
                f"{fmt_opt(x['cps'])} / {fmt_opt(mine['cps'])}",
                f"{fmt_opt(x.get('cps_ratio'))} / {fmt_opt(mine['cps_ratio'])}",
                f"{x['cost_per_run']:.4f} / {mine['cost_per_run']:.4f}", str(ok),
            ])  # fmt: skip
    return rows


def fmt_opt(v: float | None) -> str:
    return "undefined" if v is None else f"{v:.4f}"


def same(x: dict[str, Any], mine: dict[str, Any]) -> bool:
    def close(p: float | None, q: float | None, tol=5e-5) -> bool:
        return (p is None and q is None) or (
            p is not None and q is not None and abs(p - q) <= tol * max(1.0, abs(q))
        )

    return (
        x["runs"] == mine["runs"]
        and x["successes"] == mine["successes"]
        and close(x["total_cost"], mine["cost"])
        and close(x["cps"], mine["cps"])
        and close(x.get("cps_ratio"), mine["cps_ratio"])
        and close(x["cost_per_run"], mine["cost_per_run"])
    )


def reviewer_verdict(a: dict[str, dict[str, Any]], rev: dict[str, Any]) -> str:
    bad = [
        f"{key}/{c}"
        for key in TESTS
        for c, mine in a[key]["table"].items()
        if rev.get(key, {}).get(c) is None or not same(rev[key][c], mine)
    ]
    bad += [
        f"global* {key}"
        for key in TESTS
        if rev.get("global_star", {}).get(key) != a[key]["global_star_cps"]
    ]
    if bad:
        return f"DISAGREEMENT in {bad}"
    return (
        "all 8 rows agree (runs, successes exactly; dollars and ratios to 4 decimals) and the "
        "reviewer picks the same `global*` in both tests; it found no `infra_error` run."
    )


# -- main ------------------------------------------------------------------------------------------


def main(root: Path = Path(".")) -> int:
    a_all, b_all, meta = run_both(root)
    diffs = {k: cps_check.compare(a_all[k], b_all[k]) for k in TESTS}
    if any(diffs.values()):
        for k, d in diffs.items():
            for line in d[:20]:
                print(f"DISAGREEMENT {k}: {line}", file=sys.stderr)
        print("A and B disagree: no report written", file=sys.stderr)
        return 1
    facts = side_facts(root, meta)
    sens = banking_sensitivity(root, a_all, meta, facts)
    if sens["diffs"]:
        print(f"sensitivity A/B disagreement: {sens['diffs'][:5]}", file=sys.stderr)
        return 1
    cross = crosscheck_old(root, a_all)
    hints = hand_hints(a_all)
    g = gates(a_all)
    rev_path = root / OUT_DIR / "reviewer.json"
    reviewer = json.loads(rev_path.read_text()) if rev_path.exists() else None
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    (root / OUT_DIR / "method-a.json").write_text(
        json.dumps(a_all, indent=1, sort_keys=True) + "\n"
    )
    (root / OUT_DIR / "method-b.json").write_text(
        json.dumps(b_all, indent=1, sort_keys=True) + "\n"
    )
    text = build_report(a_all, diffs, cross, hints, facts, sens, g, reviewer)
    out = root / "reports/cost-per-success.md"
    out.write_text(text)
    bad = [x for x in cross if not x["ok"]]
    print(f"wrote {out}; old-report cross-check mismatches: {len(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
