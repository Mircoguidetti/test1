import pytest

from costlayer.ledger import Ledger
from costlayer.llm import LiveLLM, Request, build_kwargs

TOOLS = [
    {"name": "a", "description": "d", "input_schema": {"type": "object"}},
    {"name": "b", "description": "d", "input_schema": {"type": "object"}},
]


def req(**kw):
    base = dict(
        purpose="agent",
        run_id="r",
        model="claude-sonnet-5-5",
        system="SYS",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        cache=True,
    )
    return Request(**{**base, **kw})


def test_live_llm_refuses_outside_actions(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(RuntimeError, match="GitHub Actions"):
        LiveLLM(Ledger(tmp_path / "l.jsonl", phase=1))


def test_cache_breakpoints_on_system_and_last_tool():
    k = build_kwargs(req())
    assert k["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in k["tools"][0]
    assert k["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in TOOLS[-1]  # input not mutated


def test_no_sampling_params_and_no_forced_tool_use():
    k = build_kwargs(req(effort="low"))
    for bad in ("temperature", "top_p", "top_k"):
        assert bad not in k
    assert k["tool_choice"]["type"] == "auto"
    assert k["tool_choice"]["disable_parallel_tool_use"] is True
    assert k["output_config"] == {"effort": "low"}
    assert "thinking" not in k  # default thinking settings


def test_layer_allows_parallel_calls_and_default_effort_is_unset():
    k = build_kwargs(req(parallel_tools=True))
    assert k["tool_choice"]["disable_parallel_tool_use"] is False
    assert "output_config" not in build_kwargs(req())


def test_user_simulator_and_classifier_have_no_tools_or_cache():
    k = build_kwargs(req(tools=[], cache=False, thinking={"type": "disabled"}))
    assert "tools" not in k and "tool_choice" not in k
    assert k["system"] == "SYS" and k["thinking"] == {"type": "disabled"}
