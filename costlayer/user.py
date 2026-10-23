"""User simulator (the customer): always claude-haiku-5-5, identical in every configuration."""

from __future__ import annotations

from typing import Any

from tau_bench.envs.user import BaseUserSimulationEnv, LLMUserSimulationEnv

from costlayer.llm import LLM, Request
from costlayer.pricing import HAIKU

USER_MODEL = HAIKU
USER_MAX_TOKENS = 512


class AnthropicUser(BaseUserSimulationEnv):
    """Same prompt and protocol as tau-bench's LLM user, but through the ledgered client."""

    def __init__(self, llm: LLM, run_id: str, model: str = USER_MODEL) -> None:
        self.llm = llm
        self.run_id = run_id
        self.model = model
        self.system = ""
        self.messages: list[dict[str, Any]] = []
        self.cost = 0.0

    def _next(self) -> str:
        resp = self.llm.create(
            Request(
                purpose="user_sim",
                run_id=self.run_id,
                model=self.model,
                system=self.system,
                messages=self.messages,
                max_tokens=USER_MAX_TOKENS,
                # No reasoning for the simulated customer; same in every configuration.
                thinking={"type": "disabled"},
            )
        )
        self.cost += resp.cost
        text = "".join(b.get("text", "") for b in resp.content if b.get("type") == "text").strip()
        text = text or "###STOP###"
        self.messages.append({"role": "assistant", "content": text})
        return text

    def reset(self, instruction: str | None = None) -> str:
        # Reuse tau-bench's prompt text verbatim (the method does not use `self`).
        self.system = LLMUserSimulationEnv.build_system_prompt(None, instruction)  # type: ignore[arg-type]
        self.messages = [{"role": "user", "content": "Hi! How can I help you today?"}]
        return self._next()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content or "(no response)"})
        return self._next()

    def get_total_cost(self) -> float:
        return self.cost
