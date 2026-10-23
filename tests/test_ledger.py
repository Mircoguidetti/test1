import json
import threading

import pytest

from costlayer.context import RunContext
from costlayer.ledger import (
    PHASE_CAPS_USD,
    TOTAL_CAP_USD,
    BudgetExceeded,
    Ledger,
    plan_dispatch,
    read_prior,
    write_start,
)
from costlayer.pricing import HAIKU, SONNET, Usage, cost_usd, worst_case_cost


def test_prices_follow_the_claude_md_table():
    k = 20_000
    u = Usage(input_tokens=k, cache_write_tokens=k, cache_read_tokens=k, output_tokens=k)
    assert cost_usd(SONNET, u) == pytest.approx(k * (2 + 2.5 + 0.10 + 10) / 1e6)
    assert cost_usd(HAIKU, u) == pytest.approx(k * (0.10 + 0.125 + 0.01 + 0.50) / 1e6)
    assert cost_usd(HAIKU, Usage(input_tokens=200_000)) == pytest.approx(200_000 * 0.50 / 1e6)


def test_caps_are_the_plan_caps():
    assert TOTAL_CAP_USD == 27.0
    assert PHASE_CAPS_USD == {1: 2.5, 2: 9.0, 3: 13.0}
    assert sum(PHASE_CAPS_USD.values()) + 2.5 == TOTAL_CAP_USD


def test_record_arithmetic_and_row_content(ledger, tmp_path):
    ctx = RunContext("default", "task_1", 1)
    est = ledger.reserve(0.5)
    usage = Usage(input_tokens=1000, cache_read_tokens=2000, output_tokens=500)
    c = ledger.record(
        estimate=est,
        ctx=ctx,
        purpose="agent",
        model=SONNET,
        usage=usage,
        raw_usage={"input_tokens": 1000},
    )
    assert c == pytest.approx((1000 * 2 + 2000 * 0.10 + 500 * 10) / 1e6)
    assert ledger.spent == pytest.approx(c) and ctx.spent_agent == pytest.approx(c)
    row = json.loads(ledger.path.read_text())
    assert row["config"] == "default" and row["task"] == "task_1" and row["attempt"] == 1
    assert row["raw_usage"] == {"input_tokens": 1000} and row["cost"] == pytest.approx(c, rel=1e-6)


def test_call_without_response_costs_zero(ledger):
    est = ledger.reserve(0.2)
    c = ledger.record(
        estimate=est,
        ctx=None,
        purpose="agent",
        model=SONNET,
        usage=None,
        status="infra_error",
        error="429",
    )
    assert c == 0 and ledger.spent == 0
    ledger.reserve(2.5)  # the reservation was released


def test_phase_cap_trips_before_the_limit(ledger):
    ledger.reserve(2.0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.6)  # 2.0 in flight + 0.6 > 2.5
    ledger.reserve(0.5)


def test_total_cap_counts_prior_spend(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "d2", phase=3, phase_cap=13.0, prior_total=20.0)
    with pytest.raises(BudgetExceeded):
        led.reserve(7.5)
    led.reserve(6.9)


def test_concurrent_reservations_cannot_overshoot(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "d3", phase=1, phase_cap=2.5)
    granted, refused = [], []

    def worker():
        try:
            granted.append(led.reserve(0.3))
        except BudgetExceeded:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(granted) <= 2.5 and len(granted) == 8 and len(refused) == 12


def test_worst_case_estimate_covers_a_full_call():
    est = worst_case_cost(SONNET, prompt_chars=40_000, max_tokens=16_000)
    assert est >= 16_000 * 10 / 1e6 + 10_000 * 2.5 / 1e6


def test_spend_branch_accounting(tmp_path):
    spend = tmp_path / "spend"
    write_start(spend, "run1", 1, 2.5)  # started, never finished: counts its whole cap
    assert read_prior(spend).total == 2.5
    (spend / "ledger-run1.jsonl").write_text(json.dumps({"cost": 0.7, "phase": 1}) + "\n")
    p = read_prior(spend)
    assert p.total == pytest.approx(0.7) and p.by_phase == {1: pytest.approx(0.7)}
    assert read_prior(spend, exclude="run1").total == 0
    # phase 2 may start: 0.7 + 9.0 <= 27; a second dispatch of phase 1 gets the remainder
    assert plan_dispatch(spend, "run2", 2) == 9.0
    assert plan_dispatch(spend, "run3", 1) == pytest.approx(1.8)


def test_dispatch_refused_when_total_cap_would_pass(tmp_path):
    spend = tmp_path / "spend"
    write_start(spend, "a", 2, 9.0)
    write_start(spend, "b", 3, 13.0)
    write_start(spend, "c", 1, 2.5)  # 24.5 reserved
    with pytest.raises(BudgetExceeded):
        plan_dispatch(spend, "d", 1)  # phase 1 already fully reserved
    (spend / "start-c.json").unlink()
    (spend / "start-a.json").write_text(json.dumps({"phase": 2, "cap": 9.0}))
    write_start(spend, "e", 2, 9.0)  # a second phase-2 dispatch: 9 + 13 + 9 > 27
    with pytest.raises(BudgetExceeded):
        plan_dispatch(spend, "f", 3)
