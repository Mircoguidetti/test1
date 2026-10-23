"""Anthropic Messages API client: every call is reserved, sent, and recorded in the ledger.

Live calls are only allowed inside GitHub Actions. Tests inject a fake `sdk`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from costlayer.context import ApiRequestError, InfraError, RunContext
from costlayer.ledger import Ledger
from costlayer.pricing import Usage, worst_case_cost

MAX_TOKENS = 16000  # same for every call (CLAUDE.md)


@dataclass
class ApiResult:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage
    raw_usage: dict[str, Any]
    cost: float
    thinking_tokens: int | None  # only when the API reports it


def find_thinking_tokens(raw_usage: dict[str, Any]) -> int | None:
    """Look for a thinking-token field anywhere in the raw usage JSON."""
    for key, val in raw_usage.items():
        if "thinking" in key and isinstance(val, int):
            return val
        if isinstance(val, dict):
            found = find_thinking_tokens(val)
            if found is not None:
                return found
    return None


def request_chars(system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> int:
    return len(system) + len(json.dumps(tools)) + len(json.dumps(messages))


def build_kwargs(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    effort: str | None,
    thinking: dict[str, Any] | None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "cache_control": {"type": "ephemeral"},  # automatic caching, on in every configuration
    }
    if tools:
        kwargs["tools"] = tools
    # Never set temperature/top_p/top_k (400 on Sonnet 5.5) and never force tool_choice.
    if effort:
        kwargs["output_config"] = {"effort": effort}
    if thinking is not None:
        kwargs["thinking"] = thinking
    return kwargs


class AnthropicClient:
    def __init__(self, ledger: Ledger, sdk: Any | None = None) -> None:
        self.ledger = ledger
        if sdk is None:
            if os.environ.get("GITHUB_ACTIONS") != "true":
                raise RuntimeError(
                    "Live API calls are only allowed inside GitHub Actions. "
                    "Paid runs go through the workflow_dispatch workflow."
                )
            import anthropic

            sdk = anthropic.Anthropic(max_retries=2, timeout=600.0)
        self.sdk = sdk

    def create(
        self,
        *,
        ctx: RunContext,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str | None = None,
        thinking: dict[str, Any] | None = None,
        max_tokens: int = MAX_TOKENS,
        purpose: str = "agent",
    ) -> ApiResult:
        estimate = worst_case_cost(model, request_chars(system, tools, messages), max_tokens)
        self.ledger.reserve(estimate)  # raises BudgetExceeded before anything is sent
        kwargs = build_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            effort=effort,
            thinking=thinking,
            max_tokens=max_tokens,
        )
        try:
            msg = self.sdk.messages.create(**kwargs)
        except BaseException as e:
            kind = self._classify(e)
            self.ledger.record(
                estimate=estimate,
                ctx=ctx,
                purpose=purpose,
                model=model,
                usage=None,
                status=kind,
                error=f"{type(e).__name__}: {e}",
            )
            if kind == "infra_error":
                raise InfraError(str(e)) from e
            if kind == "api_error":
                ctx.api_errors.append(f"{type(e).__name__}: {e}"[:300])
                raise ApiRequestError(str(e)) from e
            raise
        u = msg.usage
        raw_usage = u.model_dump() if hasattr(u, "model_dump") else dict(u)
        usage = Usage(
            input_tokens=raw_usage.get("input_tokens") or 0,
            cache_write_tokens=raw_usage.get("cache_creation_input_tokens") or 0,
            cache_read_tokens=raw_usage.get("cache_read_input_tokens") or 0,
            output_tokens=raw_usage.get("output_tokens") or 0,
        )
        cost = self.ledger.record(
            estimate=estimate,
            ctx=ctx,
            purpose=purpose,
            model=model,
            usage=usage,
            raw_usage=raw_usage,
        )
        content = [
            b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)
            for b in msg.content
        ]
        return ApiResult(
            content, msg.stop_reason or "", usage, raw_usage, cost, find_thinking_tokens(raw_usage)
        )

    @staticmethod
    def _classify(e: BaseException) -> str:
        name = type(e).__name__
        status = getattr(e, "status_code", None)
        if name in (
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "OverloadedError",
        ) or (isinstance(status, int) and (status == 429 or status >= 500)):
            return "infra_error"
        if status is not None:
            return "api_error"
        return "other_error"
