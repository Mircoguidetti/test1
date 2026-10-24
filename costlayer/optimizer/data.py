"""Load run summaries and sanitized traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from costlayer.cut import CUT_TOOLS, TOP_K, cut_top_items
from costlayer.pricing import SONNET

COMPLETE = ("ok", "agent_error", "censored")  # runs that count (failures score 0)
CONFIGS = ("default", "E:medium", "E:low", "T:bt")  # test 1 (retail)
BANKING_CONFIGS = ("default", "E:low", "C:top3", "EC")  # test 2 (banking_knowledge)
CUT_CONFIGS = ("C:top3", "EC")  # the configurations that cut retrieval results


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
    # How the trace's tool results relate to what the agent saw:
    #   "seen"          saved by the runner after the cut fix (`seen_in_context`), exact
    #   "uncut"         a configuration that never cuts, so raw == seen
    #   "reconstructed" an older cut-configuration trace (raw results) with the deterministic cut
    #                   re-applied in memory and checked against the recorded size fields
    #   "unverified"    an older cut-configuration trace that failed that check: NOT for mining
    #   "none"          no trace
    trace_source: str = "none"

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


def _tool_names(trace: dict[str, Any]) -> dict[str, str]:
    return {c["id"]: c["name"] for m in trace["messages"] for c in (m.get("tool_calls") or [])}


def prepare_trace(
    trace: dict[str, Any], summary: dict[str, Any], config: str
) -> tuple[dict[str, Any], str]:
    """Make a trace show what the agent saw.

    New traces already do (`seen_in_context`). Older traces of a cutting configuration hold the
    RAW retrieval results; the cut is a pure function of the raw text and was applied when the
    result entered the conversation, so it can be re-applied here without rewriting any file.
    The result is only trusted if the summed raw and in-context lengths equal the sizes the run
    recorded at the time (`retrieval_chars_raw`, `retrieval_chars_in_context`).
    """
    if trace.get("seen_in_context"):
        return trace, "seen"
    if config not in CUT_CONFIGS:
        return trace, "uncut"
    names = _tool_names(trace)
    raw_total = in_total = 0
    messages = []
    for m in trace["messages"]:
        if m["role"] == "tool" and names.get(m["id"]) in CUT_TOOLS and not m.get("error"):
            raw = m.get("content") or ""
            cut = cut_top_items(raw, TOP_K)
            raw_total += len(raw)
            in_total += len(cut.text)
            m = {**m, "content": cut.text, "raw_chars": len(raw)}
        messages.append(m)
    ok = (
        summary.get("retrieval_chars_raw") == raw_total
        and summary.get("retrieval_chars_in_context") == in_total
    )
    if not ok:
        return trace, "unverified"
    return {**trace, "messages": messages, "reconstructed_cut": True}, "reconstructed"


def load_runs(runs_dir: Path, trace_dirs: list[Path], model: str = SONNET) -> list[Run]:
    out = []
    for f in sorted(Path(runs_dir).glob("*.json")):
        s = json.loads(f.read_text())
        if s.get("model") != model or s["status"] == "reconstructed_from_ledger":
            continue
        tp = trace_path(trace_dirs, s)
        trace, source = (None, "none")
        if tp:
            trace, source = prepare_trace(json.loads(tp.read_text()), s, s["config"])
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
                trace=trace,
                summary=s,
                trace_source=source,
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
