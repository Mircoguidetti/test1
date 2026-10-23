import json
import re
import subprocess
import sys
from pathlib import Path

from costlayer.optimizer import analysis, mining, sweep
from costlayer.optimizer.data import CONFIGS, Run, index, paired_tasks

KINDS = {
    "get_user": "READ",
    "get_order": "READ",
    "find_user": "READ",
    "cancel": "WRITE",
    "exchange": "WRITE",
    "calc": "GENERIC",
}


def tc(i, name, **args):
    return {"id": f"c{i}", "name": name, "arguments": args}


def asst(calls=None, text=None, cost=0.01, think=5, out=100):
    m = {
        "role": "assistant",
        "content": text,
        "cost": cost,
        "usage": {"output": out, "thinking_tokens": think},
    }
    if calls:
        m["tool_calls"] = calls
    return m


def tool(i, content):
    return {"role": "tool", "id": f"c{i}", "content": content, "error": False}


def trace(*msgs):
    return {
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I am Yusuf Rossi, zip 19122. Cancel #W2378156"},
            *msgs,
        ]
    }


def run(config, task, success, cost, tr=None):
    return Run(
        config,
        task,
        "ok",
        1.0 if success else 0.0,
        cost,
        0.001,
        5,
        {"input": 10, "cache_write": 1000, "cache_read": 5000, "output": 500},
        tr,
        {},
    )


def make_runs(spec):
    """spec: task -> (type_tool, {config: (success, cost)})"""
    out = []
    for t, (wtool, cfgs) in spec.items():
        tr = trace(asst([tc(1, wtool, order_id="#W1")]), tool(1, "{}")) if wtool else trace()
        for c, (ok, cost) in cfgs.items():
            out.append(run(c, t, ok, cost, tr))
    return out


def all_cfg(default, e_med, e_low, t_bt):
    return {"default": default, "E:medium": e_med, "E:low": e_low, "T:bt": t_bt}


def test_selection_rule_and_global_star():
    spec = {
        f"t{i}": ("cancel", all_cfg((True, 1.0), (True, 0.8), (i != 0, 0.5), (True, 0.9)))
        for i in range(10)
    }
    runs = make_runs(spec)
    idx = index(runs)
    tasks = paired_tasks(runs)
    fail, cost = sweep.totals(idx, tasks)
    assert fail == {"default": 0, "E:medium": 0, "E:low": 1, "T:bt": 0}
    assert sweep.acceptable(fail, 10) == ["default", "E:medium", "E:low", "T:bt"]  # floor(1.0)=1
    assert sweep.cheapest_acceptable(idx, tasks) == "E:low"
    spec["t1"] = ("cancel", all_cfg((True, 1.0), (True, 0.8), (False, 0.5), (True, 0.9)))
    idx2 = index(make_runs(spec))
    fail2, _ = sweep.totals(idx2, tasks)
    assert fail2["E:low"] == 2 and "E:low" not in sweep.acceptable(fail2, 10)
    assert sweep.cheapest_acceptable(idx2, tasks) == "E:medium"


def test_task_type_is_the_most_common_first_write_tool_ties_to_default():
    runs = make_runs({"a": ("cancel", all_cfg((True, 1), (True, 1), (True, 1), (True, 1)))})
    idx = index(runs)
    assert sweep.task_type(idx, "a", KINDS) == "cancel"
    mixed = []
    for c, w in zip(CONFIGS, ("cancel", "cancel", "exchange", "exchange"), strict=True):
        mixed.append(run(c, "m", True, 1.0, trace(asst([tc(1, w)]), tool(1, "{}"))))
    assert sweep.task_type(index(mixed), "m", KINDS) == "cancel"  # tie: default's pick
    none = [run(c, "n", True, 1.0, trace()) for c in CONFIGS]
    assert sweep.task_type(index(none), "n", KINDS) == "none"


def test_types_with_fewer_than_three_tasks_get_default():
    runs = make_runs(
        {
            f"t{i}": ("cancel", all_cfg((True, 1), (True, 1), (True, 0.2), (True, 1)))
            for i in range(2)
        }
    )
    assert sweep.choose_for_type(index(runs), ["t0", "t1"]) == "default"


def test_oos_headroom_is_found_when_types_differ_and_null_is_lower():
    spec = {}
    for i in range(8):  # cancel tasks: E:low is fine and cheap; exchange tasks: E:low fails
        spec[f"c{i}"] = ("cancel", all_cfg((True, 1.0), (True, 0.9), (True, 0.4), (True, 0.95)))
        spec[f"x{i}"] = ("exchange", all_cfg((True, 1.0), (True, 0.5), (False, 0.3), (True, 0.95)))
    runs = make_runs(spec)
    idx = index(runs)
    tasks = paired_tasks(runs)
    types = {t: sweep.task_type(idx, t, KINDS) for t in tasks}
    gs = sweep.cheapest_acceptable(idx, tasks)
    h = sweep.oos_headroom(idx, tasks, types, gs)
    null = sweep.null_headroom(idx, tasks, types, gs)
    assert gs == "E:medium" and h["headroom"] > 0.05
    assert h["headroom"] > sweep.percentile(null, 0.9)
    assert len(null) == 200 and null == sweep.null_headroom(idx, tasks, types, gs)  # seeded


def test_chain_needs_values_from_the_user_or_earlier_results():
    user = "Yusuf Rossi 19122 order #W2378156"
    t = {
        "messages": [
            {"role": "user", "content": user},
            asst([tc(1, "find_user", first="Yusuf", last="Rossi", zip="19122")], cost=0.02),
            tool(1, '{"user_id": "yusuf_rossi_9620"}'),
            asst([tc(2, "get_user", user_id="yusuf_rossi_9620")], cost=0.03),
            tool(2, '{"orders": ["#W2378156"]}'),
            asst([tc(3, "get_order", order_id="#W2378156")], cost=0.04),
            tool(3, '{"status": "delivered"}'),
            asst(text="Your order is delivered."),
        ]
    }
    assert mining.chains(t, KINDS) == [[0, 1, 2]]
    s = mining.chain_stats([t], KINDS)
    assert s["removable_calls"] == 2 and s["agent_calls"] == 4
    assert abs(s["removable_cost"] - 0.07) < 1e-9
    # a call whose argument the model invented is not provable: the chain breaks
    t2 = json.loads(json.dumps(t))
    t2["messages"][5]["tool_calls"][0]["arguments"]["order_id"] = "#W9999999"
    assert mining.chains(t2, KINDS) == [[0, 1]]


def test_write_tools_and_parallel_calls_never_join_a_chain():
    t = {
        "messages": [
            {"role": "user", "content": "order #W111111 and order #W222222"},
            asst(
                [tc(1, "get_order", order_id="#W111111"), tc(2, "get_order", order_id="#W222222")]
            ),
            tool(1, "{}"),
            tool(2, "{}"),
            asst([tc(3, "cancel", order_id="#W111111")]),
            tool(3, "{}"),
            asst([tc(4, "get_order", order_id="#W111111")]),
            tool(4, "{}"),
        ]
    }
    assert mining.chains(t, KINDS) == []  # parallel turn is one call; cancel is WRITE


def test_field_usage_flags_unused_big_fields():
    big = "x" * 300
    msgs = [
        {"role": "user", "content": "hi"},
        asst([tc(1, "get_order", order_id="#W1")]),
        tool(1, json.dumps({"status": "pending", "junk": big, "id": "#W1"})),
        asst(text="Your order is pending."),
    ]
    out = mining.field_usage([{"messages": msgs}], KINDS)["get_order"]
    assert out["keys"]["status"]["used_rate"] == 1.0 and out["keys"]["junk"]["used_rate"] == 0.0
    assert out["unused_char_share"] > 0.8


def test_call_decomposition_splits_output():
    t = trace(asst([tc(1, "get_order", order_id="#W1")], text="ok", think=40, out=150))
    d = sweep.call_decomposition(t)
    assert d["calls"] == 1 and d["output"] == 150 and d["thinking"] == 40 and d["visible_est"] > 0


def test_gate_thresholds():
    base = {
        "headroom": {"headroom": 0.12, "beats_null": True},
        "chains": {"call_share": 0.05, "cost_share": 0.05},
    }
    assert analysis.gate(base)["pass"] and analysis.gate(base)["a"]
    base["headroom"]["beats_null"] = False
    assert not analysis.gate(base)["pass"]
    base["chains"] = {"call_share": 0.16, "cost_share": 0.09}
    assert not analysis.gate(base)["b"]  # needs BOTH
    base["chains"]["cost_share"] = 0.10
    assert analysis.gate(base)["b"] and analysis.gate(base)["pass"]


def test_full_analysis_on_synthetic_runs():
    spec = {}
    for i in range(6):
        spec[f"t{i}"] = ("cancel", all_cfg((True, 1.0), (True, 0.8), (True, 0.5), (True, 0.9)))
    res = analysis.analyze(make_runs(spec), KINDS)
    assert res["n_tasks"] == 6 and res["global_star"] == "E:low"
    assert res["cost_ratio_vs_default"]["E:low"][0] == 0.5
    assert res["type_counts"] == {"cancel": 6}


def test_optimizer_package_has_no_tau2_import_even_at_runtime():
    code = (
        "import sys, costlayer.optimizer.analysis, costlayer.optimizer.classify_eval;"
        "bad=[m for m in sys.modules if m=='tau2' or m.startswith('tau2.')];"
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    root = Path(__file__).resolve().parents[1] / "costlayer" / "optimizer"
    for f in root.rglob("*.py"):
        assert not re.search(r"(?m)^\s*(import|from)\s+tau2", f.read_text()), f
