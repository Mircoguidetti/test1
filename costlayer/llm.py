"""LLM access. Every call goes through the ledger; live calls only run inside GitHub Actions."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from costlayer.ledger import Ledger
from costlayer.pricing import Usage, worst_case_cost


@dataclass
class Request:
    purpose: str  # "agent" | "user_sim" | "classifier"
    run_id: str
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 8192
    effort: str | None = None  # None = model default
    thinking: dict[str, Any] | None = None  # None = model default
    cache: bool = False  # breakpoints on tools and system prompt
    parallel_tools: bool = False  # allow several tool calls in one step (layer lever 6)


@dataclass
class Response:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage
    cost: float
    latency: float = 0.0


class LLM(Protocol):
    def create(self, req: Request) -> Response: ...


def request_chars(req: Request) -> int:
    return len(req.system) + len(json.dumps(req.tools)) + len(json.dumps(req.messages))


def build_kwargs(req: Request) -> dict[str, Any]:
    """Anthropic Messages API arguments for a request."""
    tools = [dict(t) for t in req.tools]
    if req.cache and tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    system: Any = req.system
    if req.cache:
        system = [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}]
    kwargs: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "system": system,
        "messages": req.messages,
    }
    if tools:
        kwargs["tools"] = tools
        # At most one tool call per step, as in tau-bench's tool-calling agent.
        # `auto` is not forced tool use, so it is accepted by Sonnet 5.5.
        kwargs["tool_choice"] = {
            "type": "auto",
            "disable_parallel_tool_use": not req.parallel_tools,
        }
    if req.effort:
        kwargs["output_config"] = {"effort": req.effort}
    if req.thinking is not None:
        kwargs["thinking"] = req.thinking
    return kwargs


class LiveLLM:
    def __init__(self, ledger: Ledger) -> None:
        if os.environ.get("GITHUB_ACTIONS") != "true":
            raise RuntimeError(
                "Live API calls are only allowed inside GitHub Actions (paid runs go through "
                "the workflow_dispatch workflow). Use --dry-run locally."
            )
        import anthropic

        self.ledger = ledger
        self.client = anthropic.Anthropic(max_retries=3, timeout=300.0)

    def create(self, req: Request) -> Response:
        estimate = worst_case_cost(req.model, request_chars(req), req.max_tokens)
        self.ledger.reserve(estimate)
        t0 = time.time()
        try:
            msg = self.client.messages.create(**build_kwargs(req))
        except BaseException:
            self.ledger.release(estimate)
            raise
        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            output_tokens=u.output_tokens or 0,
        )
        cost = self.ledger.record(
            estimate=estimate,
            run_id=req.run_id,
            purpose=req.purpose,
            model=req.model,
            usage=usage,
        )
        content = [b.model_dump(exclude_none=True) for b in msg.content]
        return Response(content, msg.stop_reason or "", usage, cost, time.time() - t0)
