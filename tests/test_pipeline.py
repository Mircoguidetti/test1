import json

from costlayer.batch import load_runs, run_batch
from costlayer.domains import DOMAINS, TaskRef, load_tasks, make_split
from costlayer.learn import field_evidence, learn_profile
from costlayer.pipeline import Phase1Config, phase1, pick, stop_check
from costlayer.pricing import HAIKU
from costlayer.profile import Profile
from costlayer.report import build_report, lever_firing
from costlayer.sim import ScriptedLLM


def test_gold_replay_scores_one_on_every_task(tmp_path):
    """The env wiring (own user simulator, own agent loop) reproduces tau-bench's reward."""
    llm = ScriptedLLM()
    refs = [TaskRef(d, i) for d in DOMAINS for i in range(len(load_tasks(d)))]
    res = run_batch(refs, ["baseline"], llm, HAIKU, tmp_path / "r.jsonl", workers=1)
    bad = [r.run_id for r in res if r.reward != 1.0 or r.error]
    assert not bad, bad


def test_phase1_uses_train_tasks_only():
    test = set(make_split()["test"])
    assert not set(pick("train", 20)) & test


def test_layer_runs_succeed_and_all_levers_fire(built):
    runs = load_runs(built["dir"] / "runs.jsonl")
    layer = [r for r in runs if r["mode"] == "layer"]
    assert layer and all(r["reward"] == 1.0 and not r["error"] for r in layer)
    fired = lever_firing(layer)
    assert all(v["fired"] for v in fired.values()), fired


def test_layer_is_cheaper_than_baseline_on_scripted_runs(built):
    runs = load_runs(built["dir"] / "runs.jsonl")
    b = sum(r["cost_agent"] for r in runs if r["mode"] == "baseline")
    y = sum(r["cost_agent"] + r["cost_classifier"] for r in runs if r["mode"] == "layer")
    assert y < b


def test_missing_tool_triggers_fallback_and_run_still_succeeds(built, tmp_path):
    # Remove a needed tool from a learned profile: the model "claims" it cannot act, the layer
    # discards that answer, redoes the step with the baseline request and finishes the task.
    prof = Profile.from_json(built["profile"].to_json())
    tp = prof.types["retail"]["exchange_items"]
    tp.tools = [t for t in tp.tools if t != "exchange_delivered_order_items"]
    ref = next(
        r
        for r in built["refs"]
        if r.domain == "retail"
        and any(
            a.name == "exchange_delivered_order_items" for a in load_tasks("retail")[r.idx].actions
        )
    )
    res = run_batch(
        [ref], ["layer"], ScriptedLLM(), HAIKU, tmp_path / "r.jsonl", profile=prof, workers=1
    )[0]
    assert res.fallback == "claims_missing_capability"
    assert res.reward == 1.0
    assert any(s.kind == "redo" for s in res.steps)
    assert not any(
        s.applied
        for s in res.steps[res.steps.index(next(s for s in res.steps if s.kind == "redo")) + 1 :]
    )


def test_unlearned_task_type_runs_as_baseline(built, tmp_path):
    prof = Profile(meta={}, types={})
    res = run_batch(
        built["refs"][:1],
        ["layer"],
        ScriptedLLM(),
        HAIKU,
        tmp_path / "r.jsonl",
        profile=prof,
        workers=1,
    )[0]
    assert not res.layer_applied and res.fallback is None and res.reward == 1.0


def test_profile_roundtrip_and_learning_only_from_successful_baseline(built):
    p = built["profile"]
    assert Profile.from_json(p.to_json()).to_json() == p.to_json()
    traces = [dict(t, reward=0.0) for t in built["traces"]]
    assert learn_profile(traces, min_support=1).types == {}
    layer_only = [dict(t, mode="layer") for t in built["traces"]]
    assert learn_profile(layer_only, min_support=1).types == {}


def test_field_evidence_marks_used_keys():
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "1", "name": "get_order_details", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "1",
                    "content": json.dumps({"status": "pending", "junk": "zzzzzz"}),
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "It is pending."}]},
    ]
    ((name, ev),) = field_evidence(msgs)
    assert name == "get_order_details" and ev["status"][0] and not ev["junk"][0]


def test_stop_rule():
    def pair(cb, cl, rb=1.0, rl=1.0):
        return (
            {"cost_agent": cb, "cost_classifier": 0, "reward": rb},
            {"cost_agent": cl, "cost_classifier": 0, "reward": rl},
        )

    assert stop_check([pair(1, 0.5)] * 19)[0] is False  # too early
    assert stop_check([pair(1, 0.6)] * 20)[0] is False  # 40% savings
    assert stop_check([pair(1, 0.9)] * 20)[0] is True  # 10% < 15%
    mixed = [pair(1, 0.5)] * 14 + [pair(1, 0.5, 1.0, 0.0)] * 6  # 30 points lower
    assert stop_check(mixed)[0] is True


def test_phase1_end_to_end_with_report(tmp_path):
    cfg = Phase1Config(model=HAIKU, per_domain=6, workers=1)
    llm = ScriptedLLM()
    phase1(llm, cfg, tmp_path, tmp_path / "t")
    runs = load_runs(tmp_path / "runs.jsonl")
    assert len(runs) == 24 and (tmp_path / "profile.json").exists()
    ledger = {
        "total": 0.1,
        "calls": 5,
        "by_phase": {"1": 0.1},
        "by_purpose": {"agent": 0.1},
        "by_model": {},
    }
    md = build_report(1, runs, ledger)
    for needle in (
        "Cost per task",
        "Success rate",
        "Do the levers fire",
        "Attribution",
        "Group A",
        "Budget",
        "No conclusion",
    ):
        assert needle in md
