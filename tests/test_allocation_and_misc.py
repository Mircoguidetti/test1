import math
import re
from pathlib import Path

from costlayer import allocation as al
from costlayer.audit import audit
from costlayer.report import build, visible_tokens
from costlayer.stats import savings_ci, wilson


def test_allocation_formulas():
    c = {"default": 0.10, "E:medium": 0.08, "E:low": 0.06, "T:bt": 0.07}
    a = al.allocate(c)
    assert a["n_train_raw"] == min(74, math.floor(0.8 * 9 / (0.10 + 0.08 + 0.06 + 0.07)))
    glob = min(0.10, 0.06, 0.07)
    layer = 0.10 * 1.05
    assert a["n_test_raw"] == min(40, math.floor((0.8 * 13 - 10 * 0.10) / (0.10 + glob + layer)))
    assert a["train_ok"] and a["test_ok"] and not a["options"]


def test_minimum_not_met_gives_options_not_a_choice():
    c = {"default": 0.80, "E:medium": 0.70, "E:low": 0.60, "T:bt": 0.70}
    a = al.allocate(c)
    assert not a["train_ok"] and not a["test_ok"]
    assert set(a["options"]) == {"phase2", "phase3"}
    assert any("drop E:medium" in k for k in a["options"]["phase2"])
    assert all("A/A" not in k or "plan change" in k for k in a["options"]["phase3"])


def test_stats():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    p, a, b = savings_ci([1.0] * 20, [0.7] * 20)
    assert round(p, 6) == 0.3 and a == b == p or a <= p <= b


def test_audit_lists_the_openai_defaults():
    names = {r["name"]: r["value"] for r in audit()}
    assert "gpt" in names["DEFAULT_LLM_USER"] and "gpt" in names["DEFAULT_LLM_NL_ASSERTIONS"]


def test_optimizer_package_never_imports_tau2():
    root = Path(__file__).resolve().parents[1] / "costlayer" / "optimizer"
    for f in root.rglob("*.py") if root.exists() else []:
        src = f.read_text()
        assert not re.search(r"(?m)^\s*(import|from)\s+tau2", src), f


def test_visible_tokens_counts_text_and_tool_calls():
    t = {
        "messages": [
            {
                "role": "assistant",
                "content": "x" * 350,
                "tool_calls": [{"name": "a", "arguments": {"k": "v"}}],
            },
            {"role": "user", "content": "ignored"},
        ]
    }
    assert visible_tokens(t) > 100


def test_report_builds_from_empty_and_partial_results():
    led = {"total": 0.0, "calls": 0, "by_purpose": {}, "by_model": {}, "by_config": {}}
    md = build({"haiku_checks": {}, "stopped": "x"}, [], led, 0.0, None, "note")
    for needle in ("Allocation table", "What this cannot show", "OpenAI", "Stopped early"):
        assert needle in md
