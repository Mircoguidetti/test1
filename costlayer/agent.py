"""Our tau2 agent on the official `anthropic` SDK.

Protocol rules (CLAUDE.md), identical for every configuration:
- the greeting lives in the system prompt; messages start with the user's first message;
- the full raw assistant content (thinking blocks included) is echoed back unchanged;
- if a reply has tool calls, only the tool calls go to the orchestrator (text is dropped,
  and counted); several tool calls come back as ONE user turn, results in order;
- an empty or thinking-only reply, and stop_reason max_tokens, end the run as a failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from tau2.agent.base_agent import AgentError, HalfDuplexAgent
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.registry import registry

from costlayer.client import AnthropicClient
from costlayer.context import current_context
from costlayer.cut import CUT_TOOLS, cut_for_tool

AGENT_NAME = "costlayer_agent"
DEFAULT_GREETING = "Hi! How can I help you today?"


@dataclass(frozen=True)
class Policy:
    """What differs between configurations. Constant inside a task."""

    effort: str | None = None  # None = model default (high on Sonnet 5.5)
    thinking: str | None = None  # None = default adaptive; "between_tools"
    cut_top: int | None = None  # C:top<k>: cut KB_search and grep results to k items (test 2)

    def thinking_param(self) -> dict[str, Any] | None:
        return {"type": self.thinking} if self.thinking else None

    def llm_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {"effort": self.effort, "thinking": self.thinking}
        if self.cut_top is not None:  # retail configurations keep their exact llm_args
            args["cut_top"] = self.cut_top
        return args


# Test 1 (retail). Phase 1 and phase 2 iterate over this dict: do not add keys to it.
CONFIGS: dict[str, Policy] = {
    "default": Policy(),
    "E:medium": Policy(effort="medium"),
    "E:low": Policy(effort="low"),
    "T:bt": Policy(thinking="between_tools"),
}

# Test 2 (banking_knowledge, TEST2.md). E:medium and T:bt are dropped. The value 3 is fixed.
BANKING_CONFIGS: dict[str, Policy] = {
    "default": Policy(),
    "E:low": Policy(effort="low"),
    "C:top3": Policy(cut_top=3),
    "EC": Policy(effort="low", cut_top=3),
}

ALL_CONFIGS: dict[str, Policy] = {**CONFIGS, **BANKING_CONFIGS}


class AgentRunFailure(AgentError):
    """The agent produced something that ends the run as a failure (reward 0)."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


class AgentState(BaseModel):
    system: str
    api_messages: list[dict[str, Any]] = []


def to_api_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        f = t.openai_schema["function"]
        out.append(
            {"name": f["name"], "description": f["description"], "input_schema": f["parameters"]}
        )
    return out


def system_prompt(domain_policy: str, greeting: str) -> str:
    base = SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)
    return f'{base}\n\nYou already greeted the user with: "{greeting}"'


class RuntimeHandles:
    """Process-wide handles set by the runner (the factory signature has no room for them)."""

    client: AnthropicClient | None = None


class AnthropicAgent(HalfDuplexAgent[AgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        model: str,
        policy: Policy,
        client: AnthropicClient,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.policy = policy
        self.client = client
        self.api_tools = to_api_tools(tools)

    def get_init_state(self, message_history: list[Message] | None = None) -> AgentState:
        greeting = DEFAULT_GREETING
        for m in message_history or []:
            if isinstance(m, AssistantMessage) and m.content:
                greeting = m.content  # the orchestrator's opening message
        return AgentState(system=system_prompt(self.domain_policy, greeting))

    # -- incoming message -> API user turn ------------------------------------------------

    @staticmethod
    def _tool_names(state: AgentState) -> dict[str, str]:
        """tool_use id -> tool name from the last assistant turn (a ToolMessage has no name)."""
        for m in reversed(state.api_messages):
            if m["role"] == "assistant":
                return {b["id"]: b["name"] for b in m["content"] if b.get("type") == "tool_use"}
        return {}

    def _tool_result_text(self, ctx: Any, name: str | None, content: str, tool_id: str = "") -> str:
        """What enters the conversation for one tool result. Called once per result, at first
        insertion; the returned string is never rewritten afterwards (cache prefix stays intact).
        For retrieval tools the text the agent actually saw is remembered in the run context, so
        the saved trace can show it (and the raw length) instead of the raw tool output."""
        if name not in CUT_TOOLS:
            return content
        cut = cut_for_tool(name, content, self.policy.cut_top)
        ctx.retrieval_results += 1
        ctx.retrieval_chars_raw += len(content)
        if cut is None:
            ctx.retrieval_chars_in += len(content)
            ctx.seen_results[tool_id] = {"raw_chars": len(content), "text": None}
            return content
        ctx.retrieval_chars_in += len(cut.text)
        ctx.seen_results[tool_id] = {
            "raw_chars": len(content),
            "text": cut.text if cut.text != content else None,
        }
        ctx.cut_results += 1 if cut.omitted else 0
        ctx.cut_items_omitted += cut.omitted
        ctx.cut_unparsed += 0 if cut.parsed else 1
        return cut.text

    def _user_turn(
        self, message: UserMessage | ToolMessage | MultiToolMessage, state: AgentState, ctx: Any
    ) -> dict[str, Any]:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content or "(empty message)"}
        tool_msgs = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        names = self._tool_names(state)
        blocks = []
        for tm in tool_msgs:
            content = tm.content or ""
            if content and not tm.error:
                content = self._tool_result_text(ctx, names.get(tm.id), content, tm.id)
            b: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tm.id,
                "content": content if content else "(empty result)",
            }
            if tm.error:
                b["is_error"] = True
            blocks.append(b)
        return {"role": "user", "content": blocks}

    def generate_next_message(
        self, message: UserMessage | ToolMessage | MultiToolMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        ctx = current_context()
        ctx.check_abort()
        state.api_messages.append(self._user_turn(message, state, ctx))
        t0 = time.perf_counter()
        res = self.client.create(
            ctx=ctx,
            model=self.model,
            system=state.system,
            messages=state.api_messages,
            tools=self.api_tools,
            effort=self.policy.effort,
            thinking=self.policy.thinking_param(),
        )
        ctx.llm_calls += 1
        if res.thinking_tokens is not None:
            ctx.thinking_tokens_reported += res.thinking_tokens
            ctx.thinking_reported_calls += 1
        if res.stop_reason == "max_tokens":
            ctx.max_tokens_stops += 1
            raise self._fail(ctx, "max_tokens", "stop_reason max_tokens (thinking counts)")
        if res.stop_reason == "refusal":
            raise self._fail(ctx, "refusal", "stop_reason refusal")
        text = "\n".join(b["text"] for b in res.content if b.get("type") == "text").strip()
        uses = [b for b in res.content if b.get("type") == "tool_use"]
        if not text and not uses:
            ctx.empty_replies += 1
            raise self._fail(ctx, "empty_reply", "no text and no tool call (thinking only?)")
        state.api_messages.append({"role": "assistant", "content": res.content})  # unchanged
        usage = {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.output_tokens,
            **res.usage.as_dict(),
            "thinking_tokens": res.thinking_tokens,
        }
        raw = {"stop_reason": res.stop_reason, "usage": res.raw_usage}
        if uses:
            if text:
                ctx.dropped_text += 1
            calls = [ToolCall(id=u["id"], name=u["name"], arguments=u["input"]) for u in uses]
            msg = AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=calls,
                cost=res.cost,
                usage=usage,
                raw_data=raw,
                generation_time_seconds=time.perf_counter() - t0,
            )
        else:
            msg = AssistantMessage(
                role="assistant",
                content=text,
                cost=res.cost,
                usage=usage,
                raw_data=raw,
                generation_time_seconds=time.perf_counter() - t0,
            )
        return msg, state

    @staticmethod
    def _fail(ctx: Any, kind: str, detail: str) -> AgentRunFailure:
        ctx.agent_error_kind = kind
        return AgentRunFailure(kind, detail)


def create_costlayer_agent(tools: list[Tool], domain_policy: str, **kwargs: Any) -> AnthropicAgent:
    client = RuntimeHandles.client
    if client is None:
        raise RuntimeError("RuntimeHandles.client is not set")
    args = kwargs.get("llm_args") or {}
    return AnthropicAgent(
        tools,
        domain_policy,
        model=kwargs["llm"],
        policy=Policy(args.get("effort"), args.get("thinking"), args.get("cut_top")),
        client=client,
    )


def register() -> None:
    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(create_costlayer_agent, AGENT_NAME)
