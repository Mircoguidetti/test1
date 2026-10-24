"""Test 2 phase 1: order and exclusions, the flow with a fake client, the report, the budget
request formula, the CLI dry run and the workflow file. No network, no key."""

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from costlayer import allocation2 as A2
from costlayer import banking as B
from costlayer import banking_phase1 as P1
from costlayer.budget import Budget
from costlayer.cli import main as cli_main
from costlayer.client import AnthropicClient
from costlayer.ledger import Ledger
from costlayer.pricing import HAIKU, SONNET
from costlayer.report import load_runs
from costlayer.report_banking import build
from costlayer.runner import BANKING_DOMAIN, Runner, make_config, run_name, tasks_by_id
from costlayer.usersim import install
from tests.fakes import FakeSDK, fake_user_backend, text, thinking, tool_use

ROOT = Path(__file__).resolve().parents[1]
ACTION = [
    "task_004", "task_008", "task_012", "task_014", "task_032",
    "task_033", "task_034", "task_035", "task_083",
]  # fmt: skip


# -- order, split, exclusions --------------------------------------------------------------------


def test_phase_1_order_excludes_the_nine_action_tasks_and_keeps_task_102():
    order = P1.phase1_order()
    assert len(order) == 88 and len(set(order)) == 88
    assert not set(ACTION) & set(order) and "task_102" in order
    assert order == B.banking_order(seed=42, exclude=tuple(ACTION))
    assert order != [t for t in B.banking_order() if t not in ACTION]  # excluded BEFORE the shuffle
    assert P1.excluded_ids() == tuple(ACTION)
    assert order[:3] == ["task_073", "task_077", "task_047"]  # frozen: never change the order


def test_train_is_a_prefix_of_the_order_and_test_is_the_rest():
    ex = P1.excluded_ids()
    train, test = B.banking_split(20, exclude=ex)
    assert train == P1.phase1_order()[:20] and test == P1.phase1_order()[20:]
    assert len(train) + len(test) == 88 and not set(train) & set(test)
    t30, _ = B.banking_split(30, exclude=ex)
    assert t30[:20] == train  # a larger N_train only adds tasks


def test_banking_configs_are_valid_only_for_banking_and_retail_for_retail():
    for c in ("default", "E:low", "C:top3", "EC"):
        make_config(SONNET, c, BANKING_DOMAIN, P1.VARIANT)
    for c in ("E:medium", "T:bt"):  # dropped in test 2
        with pytest.raises(ValueError):
            make_config(SONNET, c, BANKING_DOMAIN, P1.VARIANT)
    for c in ("C:top3", "EC"):
        with pytest.raises(ValueError):
            make_config(SONNET, c)  # retail
    r = make_config(SONNET, "E:low")
    assert r.llm_args_agent == {"effort": "low", "thinking": None}
    b = make_config(SONNET, "EC", BANKING_DOMAIN, P1.VARIANT)
    assert b.llm_args_agent == {"effort": "low", "thinking": None, "cut_top": 3}
    assert b.retrieval_config == "bm25_grep" and b.max_retries == 0 and b.max_steps == 100


def test_the_variant_is_local_and_has_both_retrieval_tools():
    row = B.describe(P1.VARIANT)
    assert row["local"] and row["retrieval_tools"] == ["KB_search", "grep"]
    assert row["external_keys"] == [] and not row["uses_task_gold"]


def test_bm25_dependency_is_installed():
    import rank_bm25  # noqa: F401


# -- the flow ------------------------------------------------------------------------------------


def banking_runner(tmp_path, replies, user_lines=("Hi", "ok", "###STOP###"), cap=4.0):
    led = Ledger(tmp_path / "ledger.jsonl", "d1", "b1", cap, total_cap=4.0)
    sdk = FakeSDK(replies)
    install(led, inner=fake_user_backend(list(user_lines)))
    r = Runner(
        led,
        AnthropicClient(led, sdk),
        tmp_path / "out",
        tmp_path / "tr",
        domain=BANKING_DOMAIN,
        retrieval_variant=P1.VARIANT,
    )
    return r, led, sdk


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("flow")
    r, led, sdk = banking_runner(tmp_path, [[thinking(), text("hello")]])
    res = P1.run_banking_phase1(r, led, tmp_path / "out")
    return tmp_path, r, led, sdk, res


def test_phase_1_flow_haiku_first_then_sonnet_in_the_plan_order(flow):
    tmp_path, _, led, sdk, res = flow
    assert res["sonnet_ran"] and not res["stopped"], res
    assert all(c["pass"] for c in res["haiku_checks"].values()), res["haiku_checks"]
    assert (
        "BM25 index builds; KB_search and grep return; the item parser agrees with the tool"
        in (res["haiku_checks"])
    )
    models = [x["model"] for x in sdk.requests]
    assert models[:6] == [HAIKU] * 6 and set(models[6:]) == {SONNET}  # 3 runs x 2 calls first
    runs = [json.loads(p.read_text()) for p in (tmp_path / "out" / "runs").glob("*.json")]
    haiku = [r for r in runs if r["model"] == HAIKU]
    sonnet = [r for r in runs if r["model"] == SONNET]
    order = res["order"]
    assert order == P1.phase1_order()[:3]
    assert sorted((r["task"], r["config"]) for r in haiku) == sorted((t, "default") for t in order)
    by_task = {t: sorted(r["config"] for r in sonnet if r["task"] == t) for t in order}
    assert by_task[order[0]] == ["C:top3", "E:low", "EC", "default"]
    assert by_task[order[1]] == by_task[order[2]] == ["EC", "default"]
    assert all(
        r["domain"] == BANKING_DOMAIN and r["retrieval_variant"] == "bm25_grep" for r in runs
    )
    assert all("cut_unparsed" in r and r["status"] == "ok" for r in runs)
    # configuration shapes in the requests of the first Sonnet task
    efforts = {
        x.get("output_config", {}).get("effort") for x in sdk.requests if x["model"] == SONNET
    }
    assert efforts == {None, "low"}
    assert (tmp_path / "ledger.jsonl").exists() and led.spent < 4.0


def test_report_has_the_required_sections_and_the_formula_result(flow):
    tmp_path, _, led, _, res = flow
    from costlayer.ledger import summarize
    from costlayer.phase1 import ledger_rows

    runs = load_runs(tmp_path / "out")
    rep = build(
        res, runs, ledger_rows(led.path), summarize([led.path]), 0.0,
        Budget(4.0, {"b1": 4.0}), "console note",
    )  # fmt: skip
    for section in (
        "## Verdict of the smoke test", "## Runs", "## Cost anatomy, Sonnet 5.5",
        "### `C:top3` and the number of calls", "## Budget request", "## Budget",
        "## What this cannot show", "context at last call", "Sigma", "T_needed",
        "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2", "bm25_grep",
        "Whether the agent says the right things", "That quality is unchanged",
        "Anything about other agents", "The coordinator sets the budget",
        "Phase b1 spent (ledger)", "of $4.00",
    ):  # fmt: skip
        assert section in rep, section
    c = {}
    for cfg in A2.CONFIGS:
        xs = [
            r["cost_agent"] + r["cost_user"]
            for r in runs
            if r["model"] == SONNET and r["config"] == cfg
        ]
        c[cfg] = sum(xs) / len(xs)
    expected = f"${A2.t_needed(c):.2f}"
    assert expected in rep


def test_unparsed_retrieval_result_stops_before_tasks_2_and_3(tmp_path):
    # a Sonnet agent that searches once; the tool result is replaced by an unparsable listing
    r, led, sdk = banking_runner(
        tmp_path, [[thinking(), text("hello")]], user_lines=("Hi", "ok", "###STOP###")
    )
    orig = r.run_task

    def run_task(model, configs, task):
        out = orig(model, configs, task)
        if model == SONNET:
            out = [{**o, "cut_unparsed": 1} for o in out]
        return out

    r.run_task = run_task
    res = P1.run_banking_phase1(r, led, tmp_path / "out")
    assert res["sonnet_ran"] and "could not parse" in res["stopped"]
    runs = [json.loads(p.read_text()) for p in (tmp_path / "out" / "runs").glob("*.json")]
    assert {x["task"] for x in runs if x["model"] == SONNET} == {res["order"][0]}


def test_a_failed_haiku_check_stops_before_sonnet(tmp_path):
    from tests.fakes import StatusError

    r, led, sdk = banking_runner(tmp_path, [StatusError(400, "BadRequestError")])
    res = P1.run_banking_phase1(r, led, tmp_path / "out")
    assert not res["sonnet_ran"] and res["stopped"]
    assert SONNET not in [x["model"] for x in sdk.requests]


def test_cap_stops_the_flow_before_the_limit(tmp_path):
    r, led, _ = banking_runner(tmp_path, [[thinking(), text("hello")]], cap=0.0001)
    res = P1.run_banking_phase1(r, led, tmp_path / "out")
    assert "budget" in res["stopped"] and led.spent <= 0.0001


def test_agent_searches_cut_results_inside_a_real_banking_run(tmp_path):
    """End to end through tau2's orchestrator with the real KB_search tool: the C:top3 run gets
    3 items, and since the cut-trace fix the saved trace shows those same 3 items (what the agent
    saw) with the raw length kept as an extra field (the orchestrator itself still saw all 10)."""
    replies = [
        [thinking(), tool_use("KB_search", query="credit card fees")],
        [thinking(), text("I found the policy.")],
    ]
    r, led, sdk = banking_runner(tmp_path, replies, user_lines=("Hi", "###STOP###"))
    task = tasks_by_id(BANKING_DOMAIN, P1.phase1_order()[:1])[0]
    s = r.run_once(SONNET, "C:top3", task)
    assert s["status"] == "ok", s
    sent = sdk.requests[1]["messages"][-1]["content"][0]["content"]
    assert sent.count("\n   ID: ") == 3 and sent.endswith(
        "[7 more results omitted; refine the query to see them]"
    )
    assert (s["retrieval_results"], s["cut_results"], s["cut_items_omitted"]) == (1, 1, 7)
    assert s["retrieval_chars_in_context"] < s["retrieval_chars_raw"] and s["cut_unparsed"] == 0
    trace = json.loads((tmp_path / "tr" / run_name(SONNET, "C:top3", task.id, 1)).read_text())
    tool_msg = next(m for m in trace["messages"] if m["role"] == "tool")
    assert trace["seen_in_context"] is True
    assert tool_msg["content"] == sent  # the trace holds the text the agent saw
    assert tool_msg["content"].count("\n   ID: ") == 3 and tool_msg["raw_chars"] > len(sent)
    assert tool_msg["raw_chars"] == s["retrieval_chars_raw"]


# -- the budget request --------------------------------------------------------------------------


def test_t_needed_follows_the_formula_exactly():
    c = {"default": 0.50, "E:low": 0.40, "C:top3": 0.30, "EC": 0.25}
    sigma = 0.50 + 0.40 + 0.30 + 0.25
    glob = 0.25
    want = 1.1 * (4.0 + 20 * sigma / 0.8 + (10 * 0.50 + 30 * (0.50 + glob + 0.50)) / 0.8)
    assert A2.sigma(c) == pytest.approx(sigma)
    assert A2.t_needed(c) == pytest.approx(want)
    req = A2.request(c)
    assert req["t_needed"] == pytest.approx(want) and (req["n_train"], req["n_test"]) == (20, 30)
    assert req["tasks_ok"] and req["c_global_standin"] == 0.25 and req["c_layer_standin"] == 0.50


def test_options_at_a_lower_total_are_consistent_with_the_formula():
    c = {"default": 0.50, "E:low": 0.40, "C:top3": 0.30, "EC": 0.25}
    full = A2.affordable(c, A2.t_needed(c))
    assert full["meets_both_minimums"]
    assert (full["n_train_at_min_test"], full["n_test_at_min_train"]) == (20, 30)
    # buying exactly the reported sizes never costs more than the total
    for frac in (1.0, 0.8, 0.6):
        t = frac * A2.t_needed(c)
        a = A2.affordable(c, t)
        if a["n_train_at_min_test"]:
            assert A2.total_for(c, a["n_train_at_min_test"], 30) <= t + 1e-9
            assert A2.total_for(c, a["n_train_at_min_test"] + 1, 30) > t
        if a["n_test_at_min_train"]:
            assert A2.total_for(c, 20, a["n_test_at_min_train"]) <= t + 1e-9
            assert A2.total_for(c, 20, a["n_test_at_min_train"] + 1) > t
    low = A2.affordable(c, 0.6 * A2.t_needed(c))
    assert not low["meets_both_minimums"]
    # dropping EC changes Sigma and the stand-in c(global*)
    no_ec = A2.t_needed(c, ("default", "E:low", "C:top3"))
    assert no_ec != A2.t_needed(c)
    assert A2.request({"default": 1.0})["error"]


def test_minimum_sizes_fit_in_88_tasks():
    assert (
        A2.MIN_TRAIN == 20
        and A2.MIN_TEST == 30
        and A2.MIN_TRAIN + A2.MIN_TEST <= A2.MAX_TASKS == 88
    )


# -- dry run and workflow ------------------------------------------------------------------------


def test_dry_run_prints_runs_and_cost_and_checks_the_budget(capsys):
    assert cli_main(["banking-dry-run", "--budget", str(ROOT / "budget.json")]) == 0
    out = capsys.readouterr().out
    assert "7 committed" in out and "11 at most" in out and "estimated cost" in out
    assert "task_073, task_077, task_047" in out and "phase cap $4.00" in out
    plan = P1.dry_run_plan()
    assert plan["runs_committed"] == 7 and plan["runs_max"] == 11
    assert plan["cost_committed"] <= 0.8 * 4.0


def test_dry_run_refuses_when_the_phase_is_already_spent(tmp_path, capsys):
    (tmp_path / "start-x.json").write_text(json.dumps({"phase": "b1", "cap": 4.0, "ts": 0}))
    rc = cli_main(
        ["banking-dry-run", "--budget", str(ROOT / "budget.json"), "--spend-dir", str(tmp_path)]
    )
    assert rc == 1 and "TOO HIGH" in capsys.readouterr().out


def load_wf(name="paid-run.yml"):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def test_workflows_are_dispatch_only_where_the_key_is_readable():
    for f in (ROOT / ".github" / "workflows").glob("*.yml"):
        text_ = f.read_text()
        wf = yaml.safe_load(text_)
        triggers = wf.get("on", wf.get(True))
        if "ANTHROPIC_API_KEY" in text_:
            assert list(triggers) == ["workflow_dispatch"], f.name
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "secrets." not in ci


def test_paid_workflow_has_the_test_2_job_with_its_guards():
    wf = load_wf()
    triggers = wf.get("on", wf.get(True))
    opts = triggers["workflow_dispatch"]["inputs"]["phase"]["options"]
    assert opts == ["1", "2", "b1", "b2", "b3"]
    assert wf["concurrency"] == {"group": "paid-runs", "cancel-in-progress": False}
    assert wf["permissions"]["contents"] == "write"
    jobs = wf["jobs"]
    assert set(jobs) == {"run", "banking"}
    assert (
        "!startsWith" in jobs["run"]["if"]
        and "startsWith(inputs.phase, 'b')" in jobs["banking"]["if"]
    )
    steps = jobs["banking"]["steps"]
    names = [s.get("name", "") for s in steps]
    refuse = steps[names.index("Refuse b2 and b3 without the founder's go")]
    assert "inputs.phase == 'b2'" in refuse["if"] and "inputs.phase == 'b3'" in refuse["if"]
    assert "inputs.founder_go != inputs.phase" in refuse["if"]
    dry = names.index("Dry run estimate (no key, no cost)")
    paid = names.index("Paid run")
    assert (
        dry < paid and "banking-dry-run" in steps[dry]["run"] and "budget.json" in steps[dry]["run"]
    )
    assert "banking-phase" in steps[paid]["run"]
    # the key is read in the paid step only, in both jobs
    for job in jobs.values():
        holders = [
            s["name"] for s in job["steps"] if "ANTHROPIC_API_KEY" in json.dumps(s.get("env", {}))
        ]
        assert holders == ["Paid run"]
        assert "ANTHROPIC_API_KEY" not in json.dumps(job.get("env", {}))
    assert jobs["banking"]["env"]["TAU2_DATA_DIR"].endswith(".tau2-bench/data")
    assert any("rank_bm25" in s.get("run", "") for s in steps)
    assert wf["jobs"]["banking"]["timeout-minutes"] <= 120
    txt = json.dumps(steps)
    assert "origin spend2" in txt and "HEAD:spend2" in txt
    assert not re.search(r"origin spend[^2]", txt) and 'HEAD:spend"' not in txt


def test_test_1_job_steps_are_unchanged():
    try:
        old = subprocess.run(
            ["git", "show", "main:.github/workflows/paid-run.yml"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no main branch in this checkout")
    if "banking" in yaml.safe_load(old)["jobs"]:
        pytest.skip("main already has the test-2 job")
    old_run = yaml.safe_load(old)["jobs"]["run"]
    new_run = load_wf()["jobs"]["run"]
    assert new_run["steps"] == old_run["steps"] and new_run["env"] == old_run["env"]
    assert {k: v for k, v in new_run.items() if k != "if"} == old_run


def test_ci_installs_bm25_and_sets_the_data_dir():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "import rank_bm25" in ci and "TAU2_DATA_DIR" in ci
    assert "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2" in ci
    deps = (ROOT / "pyproject.toml").read_text()
    assert "tau2[knowledge]" in deps
