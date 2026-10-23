"""Test 2 budget mechanism: budget.json guards, the spend2 ledger, test 1 left alone."""

import json
from pathlib import Path

import pytest

from costlayer import budget as B
from costlayer import ledger as L
from costlayer.cli import main as cli_main
from costlayer.ledger import BudgetExceeded, Ledger, plan_dispatch, read_prior, write_start

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path, obj) -> Path:
    p = tmp_path / "budget.json"
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return p


def test_committed_budget_json_has_only_b1_and_the_phase_1_total():
    raw = json.loads((ROOT / "budget.json").read_text())
    assert raw == {"total": 4.0, "phase_caps": {"b1": 4.0}}
    b = B.load_budget(ROOT / "budget.json")
    assert b.total == 4.0 and b.cap("b1") == 4.0
    for missing in ("b2", "b3"):
        with pytest.raises(B.BudgetFileError, match="no cap"):
            b.cap(missing)


@pytest.mark.parametrize(
    "content,msg",
    [
        ("{not json", "not valid JSON"),
        ({"total": 4.0}, "exactly the fields"),
        ({"total": 4.0, "phase_caps": {"b1": 4.0}, "extra": 1}, "exactly the fields"),
        ({"total": 4.0, "phase_caps": {}}, "non-empty"),
        ({"total": 4.0, "phase_caps": {"b4": 1.0}}, "Unknown phase keys"),
        ({"total": 4.0, "phase_caps": {"1": 1.0}}, "Unknown phase keys"),
        ({"total": 0, "phase_caps": {"b1": 4.0}}, "positive"),
        ({"total": -1, "phase_caps": {"b1": 1.0}}, "positive"),
        ({"total": "4", "phase_caps": {"b1": 4.0}}, "must be a number"),
        ({"total": True, "phase_caps": {"b1": 4.0}}, "must be a number"),
        ({"total": 4.0, "phase_caps": {"b1": 0}}, "positive"),
        ({"total": 4.0, "phase_caps": {"b1": 5.0}}, "above the total"),
        ({"total": float("nan"), "phase_caps": {"b1": 1.0}}, "positive"),
    ],
)
def test_invalid_budget_files_are_refused(tmp_path, content, msg):
    p = write(tmp_path, content if isinstance(content, str) else json.dumps(content))
    with pytest.raises(B.BudgetFileError, match=msg):
        B.load_budget(p)


def test_missing_budget_file_is_refused(tmp_path):
    with pytest.raises(B.BudgetFileError, match="not found"):
        B.load_budget(tmp_path / "nope.json")


def test_b2_and_b3_need_their_cap_and_the_matching_go(tmp_path):
    only_b1 = B.load_budget(write(tmp_path, {"total": 4.0, "phase_caps": {"b1": 4.0}}))
    assert B.check_may_start(only_b1, "b1", "") == 4.0  # b1 needs no go
    for ph in ("b2", "b3"):  # cap missing: refused even with the right go
        with pytest.raises(B.BudgetFileError, match="no cap"):
            B.check_may_start(only_b1, ph, ph)
    full = B.load_budget(
        write(tmp_path, {"total": 30.0, "phase_caps": {"b1": 4.0, "b2": 10.0, "b3": 12.0}})
    )
    assert B.check_may_start(full, "b2", "b2") == 10.0
    assert B.check_may_start(full, "b3", "b3") == 12.0
    for ph, go in (("b2", ""), ("b2", "b3"), ("b3", "b2"), ("b3", "phase2"), ("b2", "B2")):
        with pytest.raises(B.BudgetFileError, match="founder's go"):
            B.check_may_start(full, ph, go)
    with pytest.raises(B.BudgetFileError, match="Unknown phase"):
        full.cap("1")


def test_cli_refuses_before_writing_anything(tmp_path, capsys):
    only_b1 = write(tmp_path, {"total": 4.0, "phase_caps": {"b1": 4.0}})
    spend = tmp_path / "spend"
    args = ["banking-phase", "--budget", str(only_b1), "--spend-dir", str(spend)]
    with pytest.raises(SystemExit, match="no cap"):
        cli_main([*args, "--phase", "b2", "--founder-go", "b2", "--dispatch-id", "x"])
    full = write(tmp_path, {"total": 30.0, "phase_caps": {"b1": 4.0, "b2": 10.0}})
    args = ["banking-phase", "--budget", str(full), "--spend-dir", str(spend)]
    with pytest.raises(SystemExit, match="founder's go"):
        cli_main([*args, "--phase", "b2", "--dispatch-id", "x"])
    with pytest.raises(SystemExit, match="not built yet"):  # guards pass, flow is a later phase
        cli_main([*args, "--phase", "b2", "--founder-go", "b2", "--dispatch-id", "x"])
    with pytest.raises(SystemExit, match="REFUSED"):
        cli_main(
            [
                "banking-phase",
                "--budget",
                str(tmp_path / "none.json"),
                "--spend-dir",
                str(spend),
                "--dispatch-id",
                "x",
            ]
        )
    assert not spend.exists()  # no start file was written by any refused call
    with pytest.raises(SystemExit):
        cli_main(
            ["banking-dry-run", "--budget", str(only_b1), "--phase", "b3", "--founder-go", "b3"]
        )


# -- the ledger with test 2's caps ---------------------------------------------------------------


def test_test_1_caps_are_untouched():
    assert L.TOTAL_CAP_USD == 27.0 and L.PHASE_CAPS_USD == {1: 2.50, 2: 9.00, 3: 13.00}


def test_spend2_dispatch_accounting(tmp_path):
    caps, total = {"b1": 4.0, "b2": 10.0}, 14.0
    assert plan_dispatch(tmp_path, "d1", "b1", phase_caps=caps, total_cap=total) == 4.0
    write_start(tmp_path, "d1", "b1", 4.0)
    # a start file with no ledger counts as its whole reserved cap
    assert read_prior(tmp_path).total == 4.0
    with pytest.raises(BudgetExceeded, match="already used"):
        plan_dispatch(tmp_path, "d2", "b1", phase_caps=caps, total_cap=total)
    (tmp_path / "ledger-d1.jsonl").write_text(
        json.dumps({"phase": "b1", "cost": 1.25})
        + "\n"
        + json.dumps({"phase": "b1", "cost": 0.5})
        + "\n"
    )
    prior = read_prior(tmp_path)
    assert prior.total == 1.75 and prior.by_phase == {"b1": 1.75}
    assert plan_dispatch(tmp_path, "d2", "b1", phase_caps=caps, total_cap=total) == pytest.approx(
        2.25
    )
    assert plan_dispatch(tmp_path, "d2", "b2", phase_caps=caps, total_cap=total) == 10.0
    with pytest.raises(BudgetExceeded, match="would pass"):  # 1.75 + 10 > 11
        plan_dispatch(tmp_path, "d2", "b2", phase_caps=caps, total_cap=11.0)
    with pytest.raises(BudgetExceeded, match="no cap"):
        plan_dispatch(tmp_path, "d2", "b3", phase_caps=caps, total_cap=total)


def test_test_1_defaults_still_apply_without_arguments(tmp_path):
    assert plan_dispatch(tmp_path, "d1", 1) == 2.50
    with pytest.raises(BudgetExceeded):
        plan_dispatch(tmp_path, "d1", "b1")  # a test-2 phase key is not a test-1 phase


def test_reserve_trips_at_the_test_2_total_before_any_call(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "d", "b1", 4.0, prior_total=3.9, total_cap=4.0)
    led.reserve(0.05)
    with pytest.raises(BudgetExceeded, match="total cap"):
        led.reserve(0.06)  # 3.9 + 0.05 + 0.06 > 4.0
    led2 = Ledger(tmp_path / "l2.jsonl", "d", "b1", 1.0, prior_total=0.0, total_cap=4.0)
    with pytest.raises(BudgetExceeded, match="phase b1 cap"):
        led2.reserve(1.01)
