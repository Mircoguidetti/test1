import pytest
from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.registry import registry

from costlayer.agent import (
    CONFIGS,
    AgentRunFailure,
    AnthropicAgent,
    Policy,
    system_prompt,
    to_api_tools,
)
from costlayer.client import AnthropicClient, build_kwargs
from costlayer.context import ApiRequestError, InfraError, RunContext, set_context
from costlayer.ledger import Ledger
from costlayer.pricing import SONNET
from tests.fakes import FakeSDK, StatusError, text, thinking, tool_use


@pytest.fixture
def env_tools():
    return registry.get_env_constructor("retail")().get_tools()


def make(sdk, ledger, env_tools, policy=None):
    policy = policy or Policy()
    ctx = RunContext("default", "t1", 1)
    set_context(ctx)
    agent = AnthropicAgent(env_tools, "POLICY", SONNET, policy, AnthropicClient(ledger, sdk))
    state = agent.get_init_state([])
    return agent, state, ctx


def test_request_shape_has_no_forbidden_params_and_automatic_caching(ledger, env_tools):
    sdk = FakeSDK([[thinking(), text("hello")]])
    agent, state, _ = make(sdk, ledger, env_tools, CONFIGS["E:low"])
    agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    req = sdk.requests[0]
    for bad in ("temperature", "top_p", "top_k", "tool_choice"):
        assert bad not in req
    assert req["cache_control"] == {"type": "ephemeral"} and req["max_tokens"] == 16000
    assert req["output_config"] == {"effort": "low"} and "thinking" not in req
    assert req["messages"] == [{"role": "user", "content": "hi"}]  # greeting is NOT a message
    assert "You already greeted the user with" in req["system"] and "POLICY" in req["system"]


def test_configurations_differ_only_in_effort_and_thinking():
    assert CONFIGS["default"] == Policy()
    kw = build_kwargs(
        model=SONNET,
        system="s",
        messages=[],
        tools=[],
        effort=None,
        thinking=CONFIGS["T:bt"].thinking_param(),
    )
    assert kw["thinking"] == {"type": "between_tools"} and "output_config" not in kw
    assert CONFIGS["E:medium"].effort == "medium"


def test_raw_content_is_echoed_unchanged(ledger, env_tools):
    reply = [thinking(), text("let me look"), tool_use("get_order_details", order_id="#W1")]
    sdk = FakeSDK([reply, [text("done")]])
    agent, state, _ = make(sdk, ledger, env_tools)
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    call = msg.tool_calls[0]
    tool = ToolMessage(id=call.id, role="tool", content='{"x": 1}')
    agent.generate_next_message(tool, state)
    sent = sdk.requests[1]["messages"]
    assert sent[1] == {"role": "assistant", "content": [dict(b) for b in reply]}
    assert sent[1]["content"][0]["signature"] == "sig-abc"
    assert sent[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": call.id, "content": '{"x": 1}'}],
    }


def test_text_next_to_tool_calls_is_dropped_and_counted(ledger, env_tools):
    sdk = FakeSDK([[thinking(), text("I will check"), tool_use("get_user_details", user_id="u")]])
    agent, state, ctx = make(sdk, ledger, env_tools)
    msg, _ = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    assert msg.content is None and msg.tool_calls and ctx.dropped_text == 1
    assert msg.cost and msg.usage["prompt_tokens"] == 50 + 3000


def test_several_tool_calls_come_back_as_one_user_turn_in_order(ledger, env_tools):
    uses = [
        tool_use("get_user_details", user_id="a"),
        tool_use("get_order_details", order_id="#W1"),
    ]
    sdk = FakeSDK([[thinking(), *uses], [text("ok")]])
    agent, state, _ = make(sdk, ledger, env_tools)
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    assert [c.name for c in msg.tool_calls] == ["get_user_details", "get_order_details"]
    multi = MultiToolMessage(
        role="tool",
        tool_messages=[
            ToolMessage(id=u["id"], role="tool", content=f"r{i}", error=(i == 1))
            for i, u in enumerate(uses)
        ],
    )
    agent.generate_next_message(multi, state)
    turn = sdk.requests[1]["messages"][-1]
    assert turn["role"] == "user" and len(turn["content"]) == 2
    assert [b["tool_use_id"] for b in turn["content"]] == [u["id"] for u in uses]
    assert turn["content"][1].get("is_error") is True and "is_error" not in turn["content"][0]


@pytest.mark.parametrize(
    "reply,kind,attr",
    [
        ([thinking()], "empty_reply", "empty_replies"),
        ([], "empty_reply", "empty_replies"),
        (([thinking(), text("cut off")], "max_tokens"), "max_tokens", "max_tokens_stops"),
        (
            ([thinking(), tool_use("calculate", expression="1+")], "max_tokens"),
            "max_tokens",
            "max_tokens_stops",
        ),
    ],
)
def test_bad_replies_end_the_run_as_agent_errors(ledger, env_tools, reply, kind, attr):
    sdk = FakeSDK([reply])
    agent, state, ctx = make(sdk, ledger, env_tools)
    with pytest.raises(AgentRunFailure) as e:
        agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    assert e.value.kind == kind and getattr(ctx, attr) == 1 and ctx.agent_error_kind == kind


def test_infra_errors_are_ledgered_at_zero_cost_and_marked(ledger, env_tools):
    for status, name in ((429, "RateLimitError"), (529, "OverloadedError"), (503, "x")):
        sdk = FakeSDK([StatusError(status, name)])
        agent, state, ctx = make(sdk, ledger, env_tools)
        with pytest.raises(InfraError):
            agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    assert ledger.spent == 0
    rows = [__import__("json").loads(x) for x in ledger.path.read_text().splitlines()]
    assert [r["status"] for r in rows] == ["infra_error"] * 3 and all(r["cost"] == 0 for r in rows)


def test_a_400_is_an_api_error_not_an_infra_error(ledger, env_tools):
    sdk = FakeSDK([StatusError(400, "BadRequestError")])
    agent, state, ctx = make(sdk, ledger, env_tools)
    with pytest.raises(ApiRequestError):
        agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    assert ctx.api_errors


def test_tools_convert_to_the_anthropic_schema(env_tools):
    t = to_api_tools(env_tools)
    assert {x["name"] for x in t} >= {"get_order_details", "transfer_to_human_agents"}
    assert all(set(x) == {"name", "description", "input_schema"} for x in t)


def test_system_prompt_is_identical_across_configurations():
    assert system_prompt("P", "Hi!") == system_prompt("P", "Hi!")


def test_live_client_refuses_outside_actions(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(RuntimeError, match="GitHub Actions"):
        AnthropicClient(Ledger(tmp_path / "l.jsonl", "d", 1, 2.5))
