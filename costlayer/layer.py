"""The cost layer: builds each request for one run and watches for a run leaving the learned path.

Levers (CLAUDE.md numbering):
  6  fewer steps: suggested path, parallel independent reads, stop when done
  7  thinking effort per task type
  8  context per task type: policy sections, tools, shorter system prompt
  9  tool results trimmed to the fields the model uses
  10 history compaction at the learned moment
  11 output length limit
Fallback: after the first deviation, every later request is the full baseline request.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from costlayer.domains import ALWAYS_KEEP_TOOLS, READ_ONLY_TOOLS, SECTION_TOOLS, split_policy
from costlayer.llm import Request
from costlayer.pricing import est_tokens
from costlayer.profile import TypeProfile

BASELINE_MAX_TOKENS = 8192
LEVERS = ("path", "effort", "context_system", "context_tools", "trim", "compaction", "max_tokens")

_ONE_CALL_RULE = re.compile(r"(?m)^- You should (?:at most|only) make one tool call at a time.*$")
_PARALLEL_RULE = (
    "- You may issue several independent read-only tool calls in one step. Call any action "
    "that changes the database alone. If you take a tool call, do not respond to the user in "
    "the same step. If you respond to the user, do not make a tool call."
)
_INABILITY = re.compile(
    r"(don't|do not|cannot|can't|unable to|no way to)[^.]{0,40}(tool|access|function)", re.I
)


@dataclass
class StepMeta:
    """What the layer did to one request (feeds the per-lever attribution)."""

    applied: bool
    removed_tokens: dict[str, int] = field(default_factory=dict)
    fallback: bool = False


@dataclass
class Controller:
    domain: str
    wiki: str
    tools: list[dict[str, Any]]
    agent_model: str
    run_id: str
    task_type: str = "other"
    profile: TypeProfile | None = None  # None -> baseline request
    compact_override: int | None = None  # tests/phase 1: force an earlier compaction point

    fallen_back: bool = False
    fallback_reason: str | None = None
    compacted: bool = False
    compaction_removed_tokens: int = 0
    errors_seen: int = 0
    steps: int = 0
    trimmed_calls: set[str] = field(default_factory=set)
    trim_removed_tokens: int = 0

    # ------------------------------------------------------------------ request building

    @property
    def layer_active(self) -> bool:
        return self.profile is not None and not self.fallen_back

    def _baseline_request(self, messages: list[dict[str, Any]]) -> Request:
        return Request(
            purpose="agent",
            run_id=self.run_id,
            model=self.agent_model,
            system=self.wiki,
            messages=messages,
            tools=self.tools,
            max_tokens=BASELINE_MAX_TOKENS,
            cache=True,
        )

    def system_prompt(self) -> str:
        p = self.profile
        assert p is not None
        intro, sections = split_policy(self.wiki)
        mapped = SECTION_TOOLS[self.domain]
        keep = [
            text for heading, text in sections if heading not in mapped or heading in p.sections
        ]
        intro = _ONE_CALL_RULE.sub(_PARALLEL_RULE, intro)
        parts = [intro.rstrip(), *[s.rstrip() for s in keep]]
        extra = [
            "## Working style",
            "- Be concise. When the request is fully handled, say so briefly and stop; "
            "do not offer extra help.",
        ]
        if p.path:
            extra.append(
                "- Typical tool sequence for this kind of request (a guide, not a "
                "rule): " + " -> ".join(p.path) + "."
            )
        return "\n\n".join(parts + ["\n".join(extra)]) + "\n"

    def build_request(self, messages: list[dict[str, Any]]) -> tuple[Request, StepMeta]:
        base = self._baseline_request(messages)
        if not self.layer_active:
            return base, StepMeta(applied=False, fallback=self.fallen_back)
        p = self.profile
        assert p is not None
        tools = [t for t in self.tools if t["name"] in set(p.tools) | ALWAYS_KEEP_TOOLS]
        system = self.system_prompt()
        req = Request(
            purpose="agent",
            run_id=self.run_id,
            model=self.agent_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=p.max_tokens,
            effort=p.effort,
            cache=True,
            parallel_tools=True,
        )
        removed = {
            "context_system": max(0, est_tokens(self.wiki) - est_tokens(system)),
            "context_tools": max(
                0, est_tokens(json.dumps(self.tools)) - est_tokens(json.dumps(tools))
            ),
            "trim": self.trim_removed_tokens,
            "compaction": self.compaction_removed_tokens,
        }
        return req, StepMeta(applied=True, removed_tokens=removed)

    # ------------------------------------------------------------------ fallback

    def fall_back(self, reason: str) -> None:
        if not self.fallen_back:
            self.fallen_back = True
            self.fallback_reason = reason

    def inspect(self, content: list[dict[str, Any]], stop_reason: str) -> str | None:
        """Return a deviation reason if this response must be redone with the baseline request."""
        if not self.layer_active:
            return None
        p = self.profile
        assert p is not None
        if stop_reason == "max_tokens":
            return "max_tokens_hit"
        calls = [b for b in content if b.get("type") == "tool_use"]
        if any(c["name"] == "transfer_to_human_agents" for c in calls) and p.transfer_rate < 0.1:
            return "unexpected_transfer"
        if not calls:
            text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
            if _INABILITY.search(text):
                return "claims_missing_capability"
        return None

    def note_step(self) -> None:
        """Count a committed step; flag a run that is taking much longer than learned."""
        self.steps += 1
        if self.layer_active and self.profile and self.steps > self.profile.max_steps:
            self.fall_back("too_many_steps")

    # ------------------------------------------------------------------ tool calls

    def select_calls(
        self, calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Pick the calls to execute; the rest get an error result (every tool_use needs one)."""
        read_only = READ_ONLY_TOOLS[self.domain]
        if not calls:
            return [], {}
        execute: list[dict[str, Any]] = []
        skipped: dict[str, str] = {}
        for i, c in enumerate(calls):
            if i == 0 or (c["name"] in read_only and all(e["name"] in read_only for e in execute)):
                execute.append(c)
            else:
                skipped[c["id"]] = (
                    "Not executed: call actions that change data alone, one at a "
                    "time, after the read-only calls have returned."
                )
        return execute, skipped

    def process_result(self, call: dict[str, Any], observation: str) -> str:
        """Lever 9; asking for the same data again means the trim was too aggressive."""
        key = call["name"] + json.dumps(call["input"], sort_keys=True)
        if observation.startswith("Error"):
            self.errors_seen += 1
            if self.layer_active and self.errors_seen >= 2:
                self.fall_back("repeated_tool_errors")
            return observation
        if not self.layer_active:
            return observation
        if key in self.trimmed_calls:
            self.fall_back("reread_after_trim")
            return observation
        drop = (self.profile.tool_fields or {}).get(call["name"]) if self.profile else None
        if not drop:
            return observation
        try:
            data = json.loads(observation)
        except ValueError:
            return observation
        if not isinstance(data, dict):
            return observation
        trimmed = {k: v for k, v in data.items() if k not in drop}
        if len(trimmed) == len(data):
            return observation
        out = json.dumps(trimmed)
        self.trimmed_calls.add(key)
        self.trim_removed_tokens += max(0, est_tokens(observation) - est_tokens(out))
        return out

    # ------------------------------------------------------------------ compaction (lever 10)

    def maybe_compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.layer_active or self.compacted or self.profile is None:
            return messages
        threshold = self.compact_override or self.profile.compact_after_tokens
        if not threshold or est_tokens(json.dumps(messages)) < threshold:
            return messages
        out = compact_history(messages, keep_last=4)
        if out is messages:
            return messages
        self.compacted = True
        self.compaction_removed_tokens = max(
            0, est_tokens(json.dumps(messages)) - est_tokens(json.dumps(out))
        )
        return out


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )


def compact_history(messages: list[dict[str, Any]], keep_last: int = 4) -> list[dict[str, Any]]:
    """Replace everything before the last `keep_last` messages by a deterministic digest.

    The retained tail starts at an assistant message so roles keep alternating. Thinking blocks
    of retained assistant messages other than the last are dropped (a rewritten prefix would
    invalidate them anyway).
    """
    cut = len(messages) - keep_last
    while cut > 1 and messages[cut]["role"] != "assistant":
        cut -= 1
    if cut <= 1:
        return messages
    lines: list[str] = []
    names: dict[str, tuple[str, str]] = {}
    for m in messages[:cut]:
        if isinstance(m["content"], str):
            lines.append(f"Customer: {m['content'][:300]}")
            continue
        for b in m["content"]:
            t = b.get("type")
            if t == "text" and m["role"] == "assistant":
                lines.append(f"You: {b['text'][:300]}")
            elif t == "text":
                lines.append(f"Customer: {b['text'][:300]}")
            elif t == "tool_use":
                names[b["id"]] = (b["name"], json.dumps(b["input"], sort_keys=True))
            elif t == "tool_result":
                n, a = names.get(b["tool_use_id"], ("tool", ""))
                body = b["content"] if isinstance(b["content"], str) else _text_of(b["content"])
                lines.append(f"Tool {n}({a}) returned: {body[:400]}")
    digest = {
        "role": "user",
        "content": "[Earlier part of this conversation, compacted]\n"
        + "\n".join(lines)
        + "\n[End of compacted part. Continue from the next message.]",
    }
    tail = [dict(m) for m in messages[cut:]]
    last_assistant = max(i for i, m in enumerate(tail) if m["role"] == "assistant")
    for i, m in enumerate(tail):
        if m["role"] == "assistant" and i != last_assistant and not isinstance(m["content"], str):
            m["content"] = [
                b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")
            ]
    return [digest, *tail]
