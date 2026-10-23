"""Test 2, phase 1 (`b1`): smoke test and cost anatomy on banking_knowledge (TEST2.md).

Haiku 5.5 first (3 tasks, `default`); Sonnet 5.5 only if every Haiku check passes: the first task
on all four configurations, then tasks 2 and 3 on `default` and `EC` while the cap allows.
Tasks come from `banking_order(seed=42, exclude=ACTION)`; the 9 ACTION tasks are a free pass under
ENV and are excluded for good.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tau2.registry import registry

from costlayer.agent import BANKING_CONFIGS
from costlayer.banking import banking_order, reward_basis_groups
from costlayer.cut import OMITTED_LINE, TOP_K, cut_top_items
from costlayer.ledger import BudgetExceeded, Ledger
from costlayer.phase1 import haiku_checks, ledger_rows
from costlayer.pricing import HAIKU, SONNET
from costlayer.runner import (
    BANKING_DOMAIN,
    FIRST_RAIL_USD,
    ORDER_SEED,
    Runner,
    run_name,
    tasks_by_id,
)
from costlayer.usersim import USER_LLM_ARGS

VARIANT = "bm25_grep"
SONNET_CONFIGS = list(BANKING_CONFIGS)  # default, E:low, C:top3, EC
LATER_CONFIGS = ["default", "EC"]
N_HAIKU_TASKS = 3


def excluded_ids() -> tuple[str, ...]:
    return tuple(reward_basis_groups()["ACTION"])


def phase1_order() -> list[str]:
    """Task ids in the fixed order of test 2 (ACTION tasks excluded before the shuffle)."""
    return banking_order(seed=ORDER_SEED, exclude=excluded_ids())


def probe_retrieval() -> dict[str, Any]:
    """Free local check on the REAL tools of the variant (builds the BM25 index): both tools
    return, the item parser agrees with an independent count of `ID:` lines, and the cut keeps
    exactly `TOP_K` items and the exact appended line. No API call."""
    env = registry.get_env_constructor(BANKING_DOMAIN)(retrieval_variant=VARIANT)
    tools = {t.name: t for t in env.tools.get_tools().values()}
    out: dict[str, Any] = {}
    for name, arg, value in (
        ("KB_search", "query", "credit card fees"),
        ("grep", "pattern", "fee"),
    ):
        res = tools[name](**{arg: value})
        independent = len(re.findall(r"^   ID: ", res, re.MULTILINE))
        cut = cut_top_items(res)
        expected_tail = OMITTED_LINE.format(n=independent - TOP_K)
        out[name] = {
            "returns": isinstance(res, str) and independent > 0,
            "items": independent,
            "parsed_items": cut.total,
            "cut_ok": cut.parsed
            and cut.total == independent
            and (
                independent <= TOP_K
                and cut.text == res
                or independent > TOP_K
                and cut.text.endswith(expected_tail)
                and len(re.findall(r"^   ID: ", cut.text, re.MULTILINE)) == TOP_K
            ),
        }
    return out


def _trace(trace_dir: Path | None, r: dict[str, Any]) -> dict[str, Any] | None:
    if not trace_dir:
        return None
    f = Path(trace_dir) / run_name(r["model"], r["config"], r["task"], r["attempt"])
    return json.loads(f.read_text()) if f.exists() else None


def banking_checks(
    runs: list[dict[str, Any]], rows: list[dict[str, Any]], trace_dir: Path | None
) -> dict[str, dict]:
    """The generic Haiku checks plus the two that are specific to this domain."""
    checks = haiku_checks(runs, rows)
    probe = probe_retrieval()
    checks["BM25 index builds; KB_search and grep return; the item parser agrees with the tool"] = {
        "pass": all(p["returns"] and p["cut_ok"] for p in probe.values()),
        "detail": [f"{k}: {json.dumps(v)}" for k, v in probe.items()],
    }
    user_ok = all(
        r["user_calls"] >= 1 and not r["api_errors"] and r["status"] not in ("api_error",)
        for r in runs
    )
    user_tool_calls = 0
    agent_retrieval = {"KB_search": 0, "grep": 0}
    for r in runs:
        tr = _trace(trace_dir, r)
        for m in (tr or {}).get("messages", []):
            for c in m.get("tool_calls") or []:
                if m["role"] == "user":
                    user_tool_calls += 1
                elif c["name"] in agent_retrieval:
                    agent_retrieval[c["name"]] += 1
    checks["user simulator with user tools works through LiteLLM, thinking disabled"] = {
        "pass": user_ok and USER_LLM_ARGS["thinking"] == {"type": "disabled"},
        "detail": [
            f"user calls per run: {[r['user_calls'] for r in runs]}",
            f"user tool calls in traces: {user_tool_calls}",
            f"agent retrieval calls in traces: {agent_retrieval} (information only)",
        ],
    }
    return checks


def run_banking_phase1(runner: Runner, ledger: Ledger, out_dir: Path) -> dict[str, Any]:
    """Return {'haiku_checks':..., 'sonnet_ran': bool, 'stopped': reason|None, 'order': [...]}."""
    order = phase1_order()
    tasks = tasks_by_id(BANKING_DOMAIN, order[:N_HAIKU_TASKS])
    result: dict[str, Any] = {"sonnet_ran": False, "stopped": None, "order": order[:N_HAIKU_TASKS]}
    runner.rail_usd = FIRST_RAIL_USD
    try:
        haiku_runs = runner.run_block(HAIKU, ["default"], tasks, concurrency=3)
        checks = banking_checks(haiku_runs, ledger_rows(ledger.path), runner.trace_dir)
        result["haiku_checks"] = checks
        if not all(c["pass"] for c in checks.values()):
            result["stopped"] = "a Haiku check failed; Sonnet not started"
            return result
        result["sonnet_ran"] = True
        first = runner.run_task(SONNET, SONNET_CONFIGS, tasks[0])
        if any(r["cut_unparsed"] for r in first):
            result["stopped"] = "C:top3 met a retrieval result it could not parse; not continuing"
            return result
        worst = max(r["cost_agent"] + r["cost_user"] for r in first)
        for t in tasks[1:3]:
            if ledger.spent + 2.5 * worst > ledger.phase_cap:
                result["stopped"] = f"phase cap: not enough budget left for task {t.id}"
                break
            runner.run_task(SONNET, LATER_CONFIGS, t)
    except BudgetExceeded as e:
        result["stopped"] = f"budget: {e}"
    return result


# -- dry run (no key, no cost) ------------------------------------------------------------------
# Nothing is measured on banking_knowledge yet, so the dry run uses planning figures. They come
# from the cost model of TEST2.md, not from data, and phase 1 replaces them with measurements.
# One Sonnet run: about 30 agent calls, a context that grows to about 90K tokens (prefix about
# 5K plus about 13 retrieval results of about 6K) -> cache writes 90K x $2.50/M = $0.23, cache
# reads about 30 x 45K x $0.10/M = $0.14, output about 30 x 500 x $10/M = $0.15, user simulator
# about $0.03: about $0.55, rounded up. The same figure is used for all four configurations (no
# credit is given to C:top3 in advance). A Haiku run is about 1/10 of that, rounded up.
PLAN_SONNET_RUN_USD = 0.60
PLAN_HAIKU_RUN_USD = 0.06


def dry_run_plan() -> dict[str, Any]:
    """Runs and planning cost. The first block (Haiku and task 1) is always attempted; tasks 2
    and 3 run only while the cap allows, so they are not counted against the 80% check."""
    haiku = N_HAIKU_TASKS
    first = len(SONNET_CONFIGS)
    later = 2 * len(LATER_CONFIGS)
    committed = haiku * PLAN_HAIKU_RUN_USD + first * PLAN_SONNET_RUN_USD
    return {
        "haiku_runs": haiku,
        "sonnet_runs_committed": first,
        "sonnet_runs_conditional": later,
        "runs_committed": haiku + first,
        "runs_max": haiku + first + later,
        "cost_committed": committed,
        "cost_max": committed + later * PLAN_SONNET_RUN_USD,
        "tasks": phase1_order()[:N_HAIKU_TASKS],
    }
