import json

import pytest

from costlayer import phase1 as p1
from costlayer.client import AnthropicClient
from costlayer.context import RunContext
from costlayer.ledger import BudgetExceeded, Ledger
from costlayer.pricing import HAIKU, SONNET
from costlayer.runner import Runner, run_name, split_tasks, tool_kinds
from costlayer.sanitize import find_banned, sanitize_simulation
from costlayer.usersim import USER_LLM, install, usage_from_raw
from tests.fakes import FakeSDK, StatusError, fake_user_backend, text, thinking


def runner_for(tmp_path, replies, user_lines=("Hi, I need help", "###STOP###"), cap=2.5):
    led = Ledger(tmp_path / "ledger.jsonl", "d1", 1, cap)
    sdk = FakeSDK(replies)
    install(led, inner=fake_user_backend(list(user_lines)))
    return Runner(led, AnthropicClient(led, sdk), tmp_path / "out", tmp_path / "tr"), led, sdk


def test_official_split_and_seed_42_order_are_stable():
    tr, te = split_tasks("train"), split_tasks("test")
    assert (len(tr), len(te)) == (74, 40)
    assert not {t.id for t in tr} & {t.id for t in te}
    assert [t.id for t in split_tasks("train")] == [t.id for t in tr]


def test_tool_kinds_come_from_the_decorators():
    k = tool_kinds()
    assert k["get_order_details"] == "READ" and k["cancel_pending_order"] == "WRITE"
    assert k["calculate"] == "GENERIC"


def test_full_run_through_tau2_orchestrator(tmp_path):
    r, led, sdk = runner_for(tmp_path, [[thinking(), text("Sure, how can I help?")]])
    task = split_tasks("train")[0]
    s = r.run_once(SONNET, "default", task)
    assert s["status"] == "ok" and s["termination"] == "user_stop" and s["reward"] in (0.0, 1.0)
    assert s["llm_calls"] == 1 and s["user_calls"] == 2 and s["hit_max_steps"] is False
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert {x["purpose"] for x in rows} == {"agent", "user_sim"}
    assert all(x["config"] == "default" and x["task"] == task.id for x in rows)
    assert s["cost_agent"] > 0 and s["cost_user"] > 0
    assert s["tokens"]["cache_write"] == 3000 and s["thinking_tokens_reported"] == 120
    trace = json.loads(next((tmp_path / "tr").glob("*.json")).read_text())
    assert not find_banned(trace) and trace["reward"] == s["reward"]
    assert (
        trace["messages"][0]["role"] == "assistant"
    )  # the orchestrator's greeting is in the trace
    assert "evaluation_criteria" not in json.dumps(trace)
    # the greeting is in the system prompt, not in the messages sent to the API
    assert sdk.requests[0]["messages"] == [{"role": "user", "content": "Hi, I need help"}]


def test_user_simulator_uses_haiku_through_the_ledger(tmp_path):
    assert USER_LLM == "anthropic/claude-haiku-5-5"
    r, led, _ = runner_for(tmp_path, [[thinking(), text("hello")]])
    r.run_once(SONNET, "default", split_tasks("train")[0])
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert {x["model"] for x in rows if x["purpose"] == "user_sim"} == {HAIKU}
    assert (
        usage_from_raw(
            {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 40},
            }
        ).input_tokens
        == 60
    )


def test_empty_reply_is_a_failure_with_reward_zero(tmp_path):
    r, led, _ = runner_for(tmp_path, [[thinking()]])
    s = r.run_once(SONNET, "E:low", split_tasks("train")[0])
    assert s["status"] == "agent_error" and s["termination"] == "empty_reply"
    assert s["reward"] == 0.0 and s["empty_replies"] == 1


def test_max_tokens_is_a_failure(tmp_path):
    r, _, _ = runner_for(tmp_path, [([thinking(), text("x")], "max_tokens")])
    s = r.run_once(SONNET, "default", split_tasks("train")[0])
    assert s["status"] == "agent_error" and s["max_tokens_stops"] == 1 and s["reward"] == 0.0


def test_infra_error_is_rerun_once_and_counted(tmp_path):
    r, led, _ = runner_for(
        tmp_path, [StatusError(529, "OverloadedError"), [thinking(), text("hello")]]
    )
    res = r.run_task(SONNET, ["default"], split_tasks("train")[0])
    assert len(res) == 1 and res[0]["attempt"] == 2 and res[0]["status"] == "ok"
    attempts = {json.loads(x)["attempt"] for x in led.path.read_text().splitlines()}
    assert attempts == {1, 2}
    assert (tmp_path / "out" / "runs" / run_name(SONNET, "default", res[0]["task"], 1)).exists()


def test_two_infra_errors_leave_the_run_marked_for_dropping(tmp_path):
    r, _, _ = runner_for(tmp_path, [StatusError(503, "x")])
    res = r.run_task(SONNET, ["default"], split_tasks("train")[0])
    assert res[0]["status"] == "infra_error" and res[0]["attempt"] == 2


def test_per_run_rail_censors_the_run(tmp_path):
    r, _, _ = runner_for(tmp_path, [[thinking(), text("hello")]])
    r.rail_usd = 0.0
    s = r.run_once(SONNET, "default", split_tasks("train")[0])
    assert s["status"] == "censored" and s["reward"] == 0.0


def test_cap_stops_a_block_before_the_limit(tmp_path):
    r, led, _ = runner_for(tmp_path, [[thinking(), text("hello")]], cap=0.0001)
    with pytest.raises(BudgetExceeded):
        r.run_block(SONNET, ["default"], split_tasks("train")[:3], concurrency=3)
    assert led.spent <= 0.0001


def test_context_requires_a_run(tmp_path):
    from costlayer.context import current_context, set_context

    set_context(None)
    with pytest.raises(RuntimeError):
        current_context()
    set_context(RunContext("c", "t", 1))
    assert current_context().config == "c"
    set_context(None)


def test_phase1_flow_runs_haiku_first_then_sonnet(tmp_path):
    r, led, sdk = runner_for(
        tmp_path, [[thinking(), text("hello")]], user_lines=("Hi", "ok", "###STOP###")
    )
    res = p1.run_phase1(r, led, tmp_path / "out")
    assert res["sonnet_ran"] and all(c["pass"] for c in res["haiku_checks"].values()), res
    models = [x["model"] for x in sdk.requests]
    assert models[:3] == [HAIKU] * 3 and SONNET in models
    runs = [json.loads(p.read_text()) for p in (tmp_path / "out" / "runs").glob("*.json")]
    sonnet_cfgs = sorted({x["config"] for x in runs if x["model"] == SONNET})
    assert sonnet_cfgs == ["E:low", "E:medium", "T:bt", "default"]


def test_phase1_stops_before_sonnet_when_a_haiku_check_fails(tmp_path):
    r, led, sdk = runner_for(tmp_path, [StatusError(400, "BadRequestError")])
    res = p1.run_phase1(r, led, tmp_path / "out")
    assert not res["sonnet_ran"] and res["stopped"]
    assert not all(c["pass"] for c in res["haiku_checks"].values())
    assert SONNET not in [x["model"] for x in sdk.requests]


def test_sanitizer_strips_gold_from_a_real_simulation_object():
    from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason

    task = split_tasks("train")[0]
    assert task.evaluation_criteria is not None  # the gold exists in the task...
    sim = SimulationRun(
        id="s",
        task_id=task.id,
        start_time="t",
        end_time="t",
        duration=1.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        reward_info=RewardInfo(reward=1.0, info={"x": 1}),
    )
    clean = sanitize_simulation(sim)
    assert clean == {
        "task_id": task.id,
        "termination_reason": "user_stop",
        "reward": 1.0,
        "messages": [],
    }
    assert find_banned({"a": [{"evaluation_criteria": 1}]}) == ["/a[0]/evaluation_criteria"]
