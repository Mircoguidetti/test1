"""Ledger the user simulator (claude-haiku-5-5 through LiteLLM).

tau2's UserSimulator calls `generate`, imported into `tau2.user.user_simulator`. Patching
`llm_utils.generate` would do nothing, so we replace the name in the simulator's module.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from costlayer.context import ApiRequestError, InfraError, current_context
from costlayer.ledger import Ledger
from costlayer.pricing import HAIKU, Usage, worst_case_cost

USER_LLM = f"anthropic/{HAIKU}"
USER_MODEL = HAIKU
USER_LLM_ARGS: dict[str, Any] = {
    "thinking": {"type": "disabled"},  # accepted by Haiku 5.5 at effort high or below
    "max_tokens": 1024,
    "num_retries": 2,
}
# no temperature: Anthropic models of this generation reject it

_INFRA_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "Timeout",
    "APITimeoutError",
    "ServiceUnavailableError",
    "InternalServerError",
    "OverloadedError",
}


def usage_from_raw(raw: dict[str, Any]) -> Usage:
    """LiteLLM's usage dict (OpenAI style) -> our four token counts."""
    prompt = raw.get("prompt_tokens") or 0
    cached = raw.get("cache_read_input_tokens") or (
        (raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    created = raw.get("cache_creation_input_tokens") or 0
    return Usage(
        input_tokens=max(0, prompt - cached - created),
        cache_write_tokens=created,
        cache_read_tokens=cached,
        output_tokens=raw.get("completion_tokens") or 0,
    )


def install(ledger: Ledger, inner: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """Patch tau2.user.user_simulator.generate. `inner` replaces the real call in tests."""
    import tau2.user.user_simulator as us

    original = inner or us.generate

    def ledgered_generate(
        model: str,
        messages: list[Any],
        tools: Any = None,
        tool_choice: Any = None,
        call_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if not model.split("/")[-1].startswith("claude-"):
            raise RuntimeError(f"Refusing to call non-Anthropic model {model!r}")
        ctx = current_context()
        ctx.check_abort()
        chars = sum(len(json.dumps(getattr(m, "content", "") or "")) for m in messages)
        est = worst_case_cost(USER_MODEL, chars, int(kwargs.get("max_tokens") or 1024))
        ledger.reserve(est)
        try:
            msg = original(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                call_name=call_name,
                **kwargs,
            )
        except BaseException as e:
            infra = type(e).__name__ in _INFRA_NAMES
            ledger.record(
                estimate=est,
                ctx=ctx,
                purpose="user_sim",
                model=USER_MODEL,
                usage=None,
                status="infra_error" if infra else "api_error",
                error=f"{type(e).__name__}: {e}",
            )
            if infra:
                raise InfraError(str(e)) from e
            ctx.api_errors.append(f"{type(e).__name__}: {e}"[:300])
            raise ApiRequestError(str(e)) from e
        raw = dict(msg.usage or {})
        usage = usage_from_raw(raw)
        ctx.user_calls += 1
        msg.cost = ledger.record(
            estimate=est, ctx=ctx, purpose="user_sim", model=USER_MODEL, usage=usage, raw_usage=raw
        )
        return msg

    us.generate = ledgered_generate
    return ledgered_generate
