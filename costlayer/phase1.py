"""Phase 1: smoke test (cap $2.50). Haiku first; Sonnet only if every Haiku check passes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from costlayer.agent import CONFIGS
from costlayer.ledger import BudgetExceeded, Ledger
from costlayer.pricing import HAIKU, SONNET
from costlayer.runner import FIRST_RAIL_USD, Runner, split_tasks

SONNET_CONFIGS = list(CONFIGS)  # default, E:medium, E:low, T:bt
LATER_CONFIGS = ["default", "E:low"]


def ledger_rows(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    return [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]


def haiku_checks(runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, dict]:
    """The four automatic checks of phase 1 step 1 (the ledger-vs-Console check is manual)."""
    errors = [r for r in runs if r["status"] not in ("ok",)]
    api_text = " ".join(" ".join(r["api_errors"]) + str(r["termination"]) for r in runs).lower()
    cache_ok, detail = True, []
    for r in runs:
        calls = [
            x
            for x in rows
            if x["purpose"] == "agent"
            and x.get("config") == r["config"]
            and x.get("task") == r["task"]
            and x.get("attempt") == r["attempt"]
            and x["status"] == "ok"
        ]
        if len(calls) >= 2:
            ok = calls[1]["cache_read"] > 0
            cache_ok &= ok
            detail.append(f"{r['task'][:12]}: call 2 cache_read={calls[1]['cache_read']}")
    return {
        "no API error": {
            "pass": not errors,
            "detail": [f"{r['task']}: {r['status']} {r['termination']}" for r in errors]
            or ["all runs finished without error"],
        },
        "cache reads above zero from the second call": {
            "pass": cache_ok and bool(detail),
            "detail": detail,
        },
        "no 400 from thinking blocks": {
            "pass": not any(w in api_text for w in ("400", "thinking", "invalid_request")),
            "detail": ["no 400 / thinking error text in run errors"],
        },
        "no call to any OpenAI model": {
            "pass": not os.environ.get("OPENAI_API_KEY") and "openai" not in api_text,
            "detail": ["OPENAI_API_KEY is unset; no OpenAI error text in any run"],
        },
    }


def run_phase1(runner: Runner, ledger: Ledger, out_dir: Path) -> dict[str, Any]:
    """Return {'haiku_checks':..., 'sonnet_ran': bool, 'stopped': reason|None}."""
    train = split_tasks("train")
    result: dict[str, Any] = {"sonnet_ran": False, "stopped": None}
    runner.rail_usd = FIRST_RAIL_USD
    try:
        haiku_runs = runner.run_block(HAIKU, ["default"], train[:3], concurrency=3)
        checks = haiku_checks(haiku_runs, ledger_rows(ledger.path))
        result["haiku_checks"] = checks
        if not all(c["pass"] for c in checks.values()):
            result["stopped"] = "a Haiku check failed; Sonnet not started"
            return result
        result["sonnet_ran"] = True
        # first train task on all four configurations, one after the other (clean cache reads)
        first = runner.run_task(SONNET, SONNET_CONFIGS, train[0])
        worst = max(r["cost_agent"] + r["cost_user"] for r in first)
        for t in train[1:3]:
            if ledger.spent + 2.5 * worst > ledger.phase_cap:
                result["stopped"] = f"phase cap: not enough budget left for task {t.id}"
                break
            runner.run_task(SONNET, LATER_CONFIGS, t)
    except BudgetExceeded as e:
        result["stopped"] = f"budget: {e}"
    return result
