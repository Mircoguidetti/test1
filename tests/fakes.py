"""Fake Anthropic SDK and fake user-simulator backend (no network)."""

from __future__ import annotations

import copy
import itertools
import threading
from types import SimpleNamespace
from typing import Any

from tau2.data_model.message import AssistantMessage


class Block(dict):
    def model_dump(self, exclude_none: bool = True) -> dict:
        return copy.deepcopy(dict(self))


class FakeUsage(dict):
    def model_dump(self) -> dict:
        return dict(self)


def text(t: str) -> Block:
    return Block(type="text", text=t)


def thinking() -> Block:
    return Block(type="thinking", thinking="", signature="sig-abc")


_ids = itertools.count(1)


def tool_use(name: str, **inp: Any) -> Block:
    return Block(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=inp)


class FakeSDK:
    """`replies` is a list of callables/values consumed in order; the last repeats."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)
        self._seen_prefix = False

    def _create(self, **kwargs: Any) -> Any:
        with self.lock:
            self.requests.append(copy.deepcopy(kwargs))
            i = min(len(self.requests) - 1, len(self.replies) - 1)
            r = self.replies[i]
            first = not self._seen_prefix
            self._seen_prefix = True
        if isinstance(r, BaseException):
            raise r
        content, stop = r if isinstance(r, tuple) else (r, "end_turn")
        usage = FakeUsage(
            input_tokens=50,
            output_tokens=200,
            cache_creation_input_tokens=3000 if first else 0,
            cache_read_input_tokens=0 if first else 3000,
            output_tokens_details={"thinking_tokens": 120},
        )
        return SimpleNamespace(content=list(content), stop_reason=stop, usage=usage)


class StatusError(Exception):
    def __init__(self, status: int, name: str = "APIStatusError") -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.__class__ = type(name, (StatusError,), {})


def fake_user_backend(lines: list[str]):
    """Replacement for tau2's `generate` for the user simulator: replies from `lines`."""
    state: dict[tuple, int] = {}

    def gen(
        model: str,
        messages: list[Any],
        tools: Any = None,
        tool_choice: Any = None,
        call_name: str | None = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        from costlayer.context import current_context

        ctx = current_context()
        key = (ctx.config, ctx.task_id, ctx.attempt)
        i = state.get(key, 0)
        state[key] = i + 1
        line = lines[min(i, len(lines) - 1)]
        return AssistantMessage(
            role="assistant",
            content=line,
            cost=0.0,
            usage={"prompt_tokens": 400, "completion_tokens": 20},
            raw_data={},
        )

    return gen
