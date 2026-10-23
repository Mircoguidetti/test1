"""Scripted LLM for dry runs and tests: replays each task's ground-truth actions.

No network and no cost. Tool results come from the real tau-bench environment, so message and
tool-result sizes are realistic; token usage is synthesized from character counts (with a
simple cache model) so the cost estimate has the right shape, not the right decimals.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from costlayer.domains import READ_ONLY_TOOLS, first_consequential_type, load_tasks
from costlayer.llm import Request, Response
from costlayer.pricing import HAIKU, SONNET, Usage, cost_usd, est_tokens

THINK_TOKENS = {"low": 80, "medium": 250, "high": 500, "xhigh": 800, "max": 1200}
DEFAULT_EFFORT = {SONNET: "high", HAIKU: "medium"}


@dataclass
class _Script:
    items: list[tuple[str, Any]]
    first_message: str
    task_type: str
    pos: int = 0
    n: int = 0


@dataclass
class ScriptedLLM:
    seed_note: str = "scripted"
    scripts: dict[str, _Script] = field(default_factory=dict)
    cache: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    calls: int = 0
    total_cost: float = 0.0

    def _script(self, run_id: str) -> _Script:
        if run_id not in self.scripts:
            domain, idx, _mode = run_id.rsplit("-", 2)
            task = load_tasks(domain)[int(idx)]
            items: list[tuple[str, Any]] = []
            ro = READ_ONLY_TOOLS[domain]
            for a in task.actions:
                if a.name == "respond":
                    continue
                if a.name not in ro and a.name != "transfer_to_human_agents":
                    items.append(
                        (
                            "respond",
                            "I will do the following: "
                            + json.dumps(a.kwargs)
                            + ". Do you confirm (yes/no)?",
                        )
                    )
                items.append(("tool", a))
            items.append(("respond", "All done. " + " ".join(task.outputs)))
            self.scripts[run_id] = _Script(
                items, task.instruction[:250], first_consequential_type(domain, task.actions)
            )
        return self.scripts[run_id]

    def create(self, req: Request) -> Response:
        with self.lock:
            self.calls += 1
            sc = self._script(req.run_id)
            if req.purpose == "classifier":
                content: list[dict[str, Any]] = [{"type": "text", "text": sc.task_type}]
                stop, out = "end_turn", 5
            elif req.purpose == "user_sim":
                if len(req.messages) == 1:
                    text = sc.first_message
                elif sc.pos >= len(sc.items):
                    text = "###STOP###"
                else:
                    text = "Yes, please go ahead."
                content = [{"type": "text", "text": text}]
                stop, out = "end_turn", est_tokens(text)
            else:
                content, stop, out = self._agent(req, sc)
            usage = self._usage(req, out)
            cost = cost_usd(req.model, usage)
            self.total_cost += cost
            return Response(content, stop, usage, cost)

    def _agent(self, req: Request, sc: _Script) -> tuple[list[dict[str, Any]], str, int]:
        kind, payload = sc.items[min(sc.pos, len(sc.items) - 1)]
        effort = req.effort or DEFAULT_EFFORT.get(req.model, "high")
        think = THINK_TOKENS[effort]
        blocks: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "sig"}]
        if kind == "tool":
            if payload.name not in {t["name"] for t in req.tools}:
                blocks.append({"type": "text", "text": "I don't have a tool to do that for you."})
                return blocks, "end_turn", think + 20  # blocked: pos does not advance
            sc.n += 1
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{sc.n}",
                    "name": payload.name,
                    "input": payload.kwargs,
                }
            )
            sc.pos += 1
            return blocks, "tool_use", min(req.max_tokens, think + 40)
        blocks.append({"type": "text", "text": payload})
        sc.pos += 1
        return blocks, "end_turn", min(req.max_tokens, think + est_tokens(payload))

    def _usage(self, req: Request, out: int) -> Usage:
        tools_s = json.dumps(req.tools)
        prefix = est_tokens(tools_s) + est_tokens(req.system)
        total = prefix + est_tokens(json.dumps(req.messages))
        if not (req.cache and prefix >= 512):
            return Usage(input_tokens=total, output_tokens=out)
        key = hashlib.sha256((req.model + tools_s + req.system).encode()).hexdigest()
        if key in self.cache:
            return Usage(input_tokens=total - prefix, cache_read_tokens=prefix, output_tokens=out)
        self.cache.add(key)
        return Usage(input_tokens=total - prefix, cache_write_tokens=prefix, output_tokens=out)
