"""One-off fix of the phase 1 run files.

Bug: in phase 1 the token counts of a run were summed over every ledger row with the same
(config, task, attempt), ignoring the model. Haiku and Sonnet ran the same three tasks under
`default`, so Sonnet's `default` rows carried Haiku's tokens too, and the Haiku run files were
overwritten by the Sonnet ones. This script rebuilds the files from the ledger.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from costlayer.pricing import HAIKU, SONNET, price_for
from costlayer.runner import run_name

FIELDS = ("input", "cache_write", "cache_read", "output")


def agent_rows(ledger_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in Path(ledger_path).read_text().splitlines() if x.strip()]
    return [r for r in rows if r["purpose"] == "agent"]


def migrate(runs_dir: Path, ledger_path: Path) -> list[str]:
    runs_dir = Path(runs_dir)
    rows = agent_rows(ledger_path)
    by_key: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["model"], r["config"], r["task"], r["attempt"])].append(r)
    changed = []
    for f in sorted(runs_dir.glob("*.json")):
        s = json.loads(f.read_text())
        if f.name == run_name(s["model"], s["config"], s["task"], s["attempt"]):
            continue  # already migrated
        key = (s["model"], s["config"], s["task"], s["attempt"])
        s["tokens"] = {k: sum(r[k] for r in by_key[key]) for k in FIELDS}
        new = runs_dir / run_name(*key)
        new.write_text(json.dumps(s, indent=1) + "\n")
        f.unlink()
        changed.append(f"{f.name} -> {new.name}")
    # Haiku runs whose file was overwritten: rebuild the numbers the ledger knows
    for key, rs in sorted(by_key.items()):
        model, config, task, attempt = key
        if model != HAIKU or (runs_dir / run_name(*key)).exists():
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        s = {
            "model": model,
            "config": config,
            "task": task,
            "attempt": attempt,
            "status": "reconstructed_from_ledger",
            "termination": None,
            "reward": None,
            "llm_calls": len(ok),
            "cost_agent": round(sum(r["cost"] for r in rs), 6),
            "tokens": {k: sum(r[k] for r in rs) for k in FIELDS},
            "note": "run file was overwritten by the Sonnet run (bug); reward and "
            "termination are unknown",
        }
        (runs_dir / run_name(*key)).write_text(json.dumps(s, indent=1) + "\n")
        changed.append(f"(rebuilt) {run_name(*key)}")
    return changed


def correction_md(runs_dir: Path, ledger_path: Path) -> str:
    """Corrected cost anatomy and thinking numbers for Sonnet, straight from the ledger."""
    rows = [r for r in agent_rows(ledger_path) if r["model"] == SONNET and r["status"] == "ok"]
    per: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        c = per[r["config"]]
        c["calls"] += 1
        for k in FIELDS:
            c[k] += r[k]
        c["think"] += (r["raw_usage"].get("output_tokens_details") or {}).get("thinking_tokens", 0)
        c["cost"] += r["cost"]
    p = price_for(SONNET)
    L = [
        "| config | calls | cache read | cache write | uncached input | output | "
        "output/call | reported thinking/call | non-thinking output/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in ("default", "E:medium", "E:low", "T:bt"):
        c = per.get(cfg)
        if not c:
            continue
        parts = [
            c["cache_read"] * p.cache_read,
            c["cache_write"] * p.cache_write,
            c["input"] * p.input,
            c["output"] * p.output,
        ]
        tot = sum(parts)
        L.append(
            f"| {cfg} | {c['calls']} | "
            + " | ".join(f"{100 * x / tot:.0f}%" for x in parts)
            + f" | {c['output'] / c['calls']:.0f} | {c['think'] / c['calls']:.1f} | "
            f"{(c['output'] - c['think']) / c['calls']:.0f} |"
        )
    runs = [json.loads(f.read_text()) for f in sorted(Path(runs_dir).glob("sonnet__*.json"))]
    cw = {
        cfg: [r["tokens"]["cache_write"] for r in runs if r["config"] == cfg]
        for cfg in ("default", "E:low")
    }
    L += [
        "",
        "Cache-write tokens per run, Sonnet: "
        + "; ".join(f"`{k}` {v} (mean {statistics.mean(v):.0f})" for k, v in cw.items() if v)
        + ".",
    ]
    return "\n".join(L)
