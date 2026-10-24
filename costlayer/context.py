"""Per-run context shared (through a thread-local) by the agent, the user-simulator patch and
the ledger, so that every API call is tagged with configuration, task and attempt."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


class InfraError(RuntimeError):
    """429/5xx/timeout/network error that survived the SDK's own retries (not a task failure)."""


class ApiRequestError(RuntimeError):
    """The API rejected a request (4xx). A bug or a setting the model does not accept."""


class RunAborted(RuntimeError):
    """The run's own spend passed the per-run safety rail (censored)."""


@dataclass
class RunContext:
    config: str
    task_id: str
    attempt: int
    abort_above_usd: float = 1.0
    spent: float = 0.0  # actual USD of this run, agent and user simulator
    spent_agent: float = 0.0
    spent_user: float = 0.0
    llm_calls: int = 0  # agent calls
    user_calls: int = 0
    dropped_text: int = 0  # replies that had text next to tool calls (text dropped)
    max_tokens_stops: int = 0
    empty_replies: int = 0
    api_errors: list[str] = field(default_factory=list)
    thinking_tokens_reported: int = 0
    thinking_reported_calls: int = 0
    agent_error_kind: str | None = None
    # retrieval results (KB_search, grep) as they enter the conversation (test 2)
    retrieval_results: int = 0
    retrieval_chars_raw: int = 0  # what the tool returned
    retrieval_chars_in: int = 0  # what the agent's conversation got (smaller under C:top3)
    cut_results: int = 0  # results that were actually cut
    cut_items_omitted: int = 0
    cut_unparsed: int = 0  # results that looked like an item list but could not be parsed
    # what the agent saw for each retrieval result: tool_use id -> {raw_chars, text}; `text` is the
    # (cut) string that entered the conversation, or None when it is identical to the raw result
    seen_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def check_abort(self) -> None:
        if self.spent > self.abort_above_usd:
            raise RunAborted(
                f"run spend ${self.spent:.4f} passed the rail ${self.abort_above_usd:.2f}"
            )

    def tag(self) -> dict[str, Any]:
        return {"config": self.config, "task": self.task_id, "attempt": self.attempt}


_local = threading.local()


def set_context(ctx: RunContext | None) -> None:
    _local.ctx = ctx


def current_context() -> RunContext:
    ctx = getattr(_local, "ctx", None)
    if ctx is None:
        raise RuntimeError("No RunContext set for this thread")
    return ctx
