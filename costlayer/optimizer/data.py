"""Load run summaries and sanitized traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from costlayer.pricing import SONNET

COMPLETE = ("ok", "agent_error", "censored")  # runs that count (failures score 0)
CONFIGS = ("default", "E:medium", "E:low", "T:bt")


@dataclass
class Run:
    config: str
    task: str
    status: str
    reward: float
    cost_agent: float
    cost_user: float
    llm_calls: int
    tokens: dict[str, int]
    trace: dict[str, Any] | None
    summary: dict[str, Any]

    @property
    def success(self) -> bool:
        return self.status == "ok" and self.reward >= 1.0


def trace_path(trace_dirs: list[Path], s: dict[str, Any]) -> Path | None:
    cfg = s["config"].replace(":", "-")
    old = f"{cfg}__{s['task']}__a{s['attempt']}.json"  # phase 1 naming
    new = f"{s['model'].split('-')[1]}__{old}"
    for d in trace_dirs:
        for name in (new, old):
            if (Path(d) / name).exists():
                return Path(d) / name
    return None


def load_runs(runs_dir: Path, trace_dirs: list[Path], model: str = SONNET) -> list[Run]:
    out = []
    for f in sorted(Path(runs_dir).glob("*.json")):
        s = json.loads(f.read_text())
        if s.get("model") != model or s["status"] == "reconstructed_from_ledger":
            continue
        tp = trace_path(trace_dirs, s)
        out.append(
            Run(
                config=s["config"],
                task=s["task"],
                status=s["status"],
                reward=float(s.get("reward") or 0.0),
                cost_agent=s["cost_agent"],
                cost_user=s["cost_user"],
                llm_calls=s["llm_calls"],
                tokens=s["tokens"],
                trace=json.loads(tp.read_text()) if tp else None,
                summary=s,
            )
        )
    return out


def paired_tasks(runs: list[Run], configs: tuple[str, ...] = CONFIGS) -> list[str]:
    """Tasks with a complete, non-infrastructure run in every configuration."""
    by: dict[str, dict[str, Run]] = {}
    for r in runs:
        if r.status in COMPLETE:
            by.setdefault(r.task, {})[r.config] = r
    return sorted(t for t, d in by.items() if all(c in d for c in configs))


def index(runs: list[Run]) -> dict[tuple[str, str], Run]:
    return {(r.task, r.config): r for r in runs if r.status in COMPLETE}


def load_tool_kinds(path: Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())
