import pytest

from costlayer.ledger import PHASE_CAPS_USD, TOTAL_CAP_USD, BudgetExceeded, Ledger, summarize
from costlayer.pricing import HAIKU, SONNET, Usage, cost_usd, worst_case_cost


def test_prices_from_claude_md():
    k = 20_000  # a prompt below Haiku's 100K-token price break
    u = Usage(input_tokens=k, cache_write_tokens=k, cache_read_tokens=k, output_tokens=k)
    assert cost_usd(HAIKU, u) == pytest.approx(k * (0.10 + 0.125 + 0.01 + 0.50) / 1e6)
    # Sonnet cache read is 0.20 (0.1x input), see pricing.py
    assert cost_usd(SONNET, u) == pytest.approx(k * (2 + 2.5 + 0.20 + 10) / 1e6)
    long = Usage(input_tokens=200_000)  # Haiku prompts above 100K tokens cost more
    assert cost_usd(HAIKU, long) == pytest.approx(0.1)


def test_caps_trip_before_console_limit():
    assert TOTAL_CAP_USD == 27.0 < 35.0
    assert sum(PHASE_CAPS_USD.values()) == 25.0 <= TOTAL_CAP_USD


def test_ledger_records_and_aborts(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", phase=1)
    est = led.reserve(1.0)
    led.record(
        estimate=est, run_id="r", purpose="agent", model=HAIKU, usage=Usage(output_tokens=1_000_000)
    )  # $0.50
    assert led.spent_total == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        led.reserve(2.6)  # 0.5 + 2.6 > phase cap 3.0
    led.reserve(2.4)  # exactly fits
    with pytest.raises(BudgetExceeded):
        led.reserve(0.2)  # in-flight reservation counts


def test_total_cap_spans_phases(tmp_path):
    p = tmp_path / "l.jsonl"
    Ledger(p, phase=1).record(
        estimate=0, run_id="r", purpose="agent", model=SONNET, usage=Usage(output_tokens=2_000_000)
    )  # $20
    led = Ledger(p, phase=3)  # reloads the history
    assert led.spent_total == pytest.approx(20)
    with pytest.raises(BudgetExceeded):
        led.reserve(7.5)  # 20 + 7.5 > 27
    led.reserve(6.5)
    assert summarize(p)["by_phase"] == {"1": 20.0}


def test_worst_case_estimate_is_conservative():
    est = worst_case_cost(SONNET, prompt_chars=40_000, max_tokens=8192)
    real_worst = (10_000 * 2.5 + 8192 * 10) / 1e6  # 40K chars is ~11K tokens at most
    assert est >= real_worst * 0.9
