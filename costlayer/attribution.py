"""Per-lever savings attribution from per-step token logs (no separate runs per lever)."""

from __future__ import annotations

from typing import Any

from costlayer.pricing import price_for


def _prefix_price(model: str, usage: dict[str, int]) -> float:
    p = price_for(model)
    if usage["cache_read"]:
        return p.cache_read
    if usage["cache_write"]:
        return p.cache_write
    return p.input


def attribute(base_runs: list[dict[str, Any]], layer_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """USD saved per lever over paired runs. Each run is a summary dict plus its 'steps'."""
    if not layer_runs or not base_runs:
        return {}
    model = layer_runs[0]["model"]
    p = price_for(model)
    out = {k: 0.0 for k in ("context_system", "context_tools", "trim", "compaction")}
    for run in layer_runs:
        for st in run["steps"]:
            if not st["applied"]:
                continue
            rm = st["removed_tokens"]
            pp = _prefix_price(model, st["usage"])
            out["context_system"] += rm.get("context_system", 0) * pp / 1e6
            out["context_tools"] += rm.get("context_tools", 0) * pp / 1e6
            out["trim"] += rm.get("trim", 0) * p.input / 1e6
            out["compaction"] += rm.get("compaction", 0) * p.input / 1e6

    def n_steps(runs: list[dict[str, Any]]) -> float:
        return sum(len(r["steps"]) for r in runs)

    b_steps, l_steps = n_steps(base_runs), n_steps(layer_runs)
    b_cost = sum(r["cost_agent"] for r in base_runs)
    b_out = sum(s["usage"]["output"] for r in base_runs for s in r["steps"])
    l_out = sum(s["usage"]["output"] for r in layer_runs for s in r["steps"])
    nb, nl = len(base_runs), len(layer_runs)
    # Steps: fewer steps per task, valued at the baseline's average cost per step.
    steps_saved_per_task = b_steps / nb - l_steps / nl
    out["path"] = steps_saved_per_task * (b_cost / b_steps) * nl if b_steps else 0.0
    # Effort: fewer output tokens per step (thinking included), on the layer's own step count.
    out["effort"] = (
        (b_out / b_steps - l_out / l_steps) * l_steps * p.output / 1e6
        if b_steps and l_steps
        else 0.0
    )
    out["max_tokens_hits"] = sum(
        1 for r in layer_runs for s in r["steps"] if s["stop_reason"] == "max_tokens"
    )
    out["classifier_overhead"] = -sum(r["cost_classifier"] for r in layer_runs)
    actual = sum(r["cost_agent"] for r in base_runs) / nb * nl - sum(
        r["cost_agent"] for r in layer_runs
    )
    named = sum(v for k, v in out.items() if k not in ("max_tokens_hits", "classifier_overhead"))
    out["interaction_residual"] = actual - named
    out["agent_cost_saved_total"] = actual
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}
