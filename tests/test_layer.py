import json

from tau_bench.envs.retail.tools import ALL_TOOLS
from tau_bench.envs.retail.wiki import WIKI

from costlayer.domains import SECTION_TOOLS, split_policy
from costlayer.layer import Controller, compact_history
from costlayer.pricing import est_tokens
from costlayer.profile import TypeProfile

TOOLS = [
    {
        "name": t.get_info()["function"]["name"],
        "description": "d",
        "input_schema": {"type": "object"},
    }
    for t in ALL_TOOLS
]


def profile(**kw):
    base = dict(
        n_traces=5,
        tools=["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"],
        sections=["cancel pending order"],
        path=["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"],
        effort="medium",
        max_tokens=1024,
        tool_fields={"get_order_details": ["address"]},
        compact_after_tokens=500,
        max_steps=10,
        transfer_rate=0.0,
    )
    return TypeProfile(**{**base, **kw})


def ctl(p=None):
    return Controller(
        domain="retail",
        wiki=WIKI,
        tools=TOOLS,
        agent_model="claude-haiku-5-5",
        run_id="retail-0-layer",
        task_type="cancel_order",
        profile=p or profile(),
    )


def test_baseline_request_when_no_profile():
    c = ctl()
    c.profile = None
    req, meta = c.build_request([{"role": "user", "content": "hi"}])
    assert req.system == WIKI and len(req.tools) == len(TOOLS)
    assert req.effort is None and req.cache and not req.parallel_tools and not meta.applied


def test_layer_request_levers_6_7_8_11():
    req, meta = ctl().build_request([{"role": "user", "content": "hi"}])
    names = {t["name"] for t in req.tools}
    assert names == {
        "find_user_id_by_name_zip",
        "get_order_details",
        "cancel_pending_order",
        "transfer_to_human_agents",
    }
    assert "Cancel pending order" in req.system
    assert "Return delivered order" not in req.system and "Exchange delivered" not in req.system
    assert "Domain basic" in req.system  # unmapped sections are always kept
    assert "several independent read-only tool calls" in req.system
    assert "at most make one tool call" not in req.system
    assert "find_user_id_by_name_zip -> get_order_details -> cancel_pending_order" in req.system
    assert req.effort == "medium" and req.max_tokens == 1024 and req.parallel_tools
    assert meta.removed_tokens["context_system"] > 0 and meta.removed_tokens["context_tools"] > 0
    assert est_tokens(req.system) < est_tokens(WIKI)


def test_every_mapped_section_exists_in_the_policy():
    _, secs = split_policy(WIKI)
    assert set(SECTION_TOOLS["retail"]) <= {h for h, _ in secs}


def call(name, **kw):
    return {"type": "tool_use", "id": f"id_{name}", "name": name, "input": kw}


def test_parallel_reads_run_but_writes_run_alone():
    c = ctl()
    reads = [call("get_order_details", order_id="#W1"), call("get_user_details", user_id="u")]
    run, skipped = c.select_calls(reads)
    assert len(run) == 2 and not skipped
    mixed = [
        call("get_order_details", order_id="#W1"),
        call("cancel_pending_order", order_id="#W1", reason="x"),
    ]
    run, skipped = c.select_calls(mixed)
    assert [r["name"] for r in run] == [
        "get_order_details"
    ] and "id_cancel_pending_order" in skipped
    run, skipped = c.select_calls([mixed[1], mixed[0]])
    assert [r["name"] for r in run] == [
        "cancel_pending_order"
    ] and "id_get_order_details" in skipped


def test_trim_then_reread_triggers_fallback():
    c = ctl()
    c_call = call("get_order_details", order_id="#W1")
    obs = json.dumps({"order_id": "#W1", "address": {"x": "y" * 100}, "status": "pending"})
    out = c.process_result(c_call, obs)
    assert "address" not in json.loads(out) and c.trim_removed_tokens > 0 and not c.fallen_back
    assert c.process_result(c_call, obs) == obs  # re-read: full result, and the run falls back
    assert c.fallback_reason == "reread_after_trim"
    req, meta = c.build_request([{"role": "user", "content": "hi"}])
    assert req.system == WIKI and meta.fallback and not meta.applied


def test_deviation_signals():
    c = ctl()
    assert c.inspect([{"type": "text", "text": "x"}], "max_tokens") == "max_tokens_hit"
    assert (
        c.inspect([call("transfer_to_human_agents", summary="s")], "tool_use")
        == "unexpected_transfer"
    )
    assert (
        c.inspect([{"type": "text", "text": "I don't have a tool for that."}], "end_turn")
        == "claims_missing_capability"
    )
    assert c.inspect([{"type": "text", "text": "Your order is pending."}], "end_turn") is None
    # a type that is known to transfer does not trip the transfer signal
    assert (
        ctl(profile(transfer_rate=0.5)).inspect(
            [call("transfer_to_human_agents", summary="s")], "tool_use"
        )
        is None
    )


def test_repeated_errors_and_long_runs_fall_back():
    c = ctl()
    c.process_result(call("get_order_details", order_id="a"), "Error: nope")
    assert not c.fallen_back
    c.process_result(call("get_order_details", order_id="b"), "Error: nope")
    assert c.fallback_reason == "repeated_tool_errors"
    d = ctl(profile(max_steps=2))
    for _ in range(3):
        d.note_step()
    assert d.fallback_reason == "too_many_steps"


def make_history(n_pairs):
    msgs = [{"role": "user", "content": "I want to cancel #W1"}]
    for i in range(n_pairs):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {
                        "type": "tool_use",
                        "id": f"t{i}",
                        "name": "get_order_details",
                        "input": {"order_id": f"#W{i}"},
                    },
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "R" * 800}],
            }
        )
    return msgs


def test_compaction_keeps_roles_alternating_and_tool_pairs_valid():
    msgs = make_history(8)
    out = compact_history(msgs, keep_last=4)
    assert len(out) < len(msgs) and out[0]["role"] == "user"
    assert all(a["role"] != b["role"] for a, b in zip(out, out[1:], strict=False))
    assert "get_order_details" in out[0]["content"] and "#W0" in out[0]["content"]
    ids_used = {
        b["id"]
        for m in out
        if m["role"] == "assistant"
        for b in m["content"]
        if b.get("type") == "tool_use"
    }
    ids_answered = {
        b["tool_use_id"]
        for m in out
        if m["role"] == "user" and not isinstance(m["content"], str)
        for b in m["content"]
    }
    assert ids_used == ids_answered
    assert est_tokens(json.dumps(out)) < est_tokens(json.dumps(msgs))
    thinking = [
        b
        for m in out[1:]
        if m["role"] == "assistant"
        for b in m["content"]
        if b["type"] == "thinking"
    ]
    assert len(thinking) == 1  # only the last assistant turn keeps its thinking block


def test_compaction_fires_once_at_threshold():
    c = ctl(profile(compact_after_tokens=1500))
    short = make_history(1)
    assert c.maybe_compact(short) is short and not c.compacted
    long = make_history(8)
    out = c.maybe_compact(long)
    assert c.compacted and out is not long and c.compaction_removed_tokens > 0
    assert c.maybe_compact(long) is long  # only once per run
