"""The one sanitizer between tau2 results and everything else (analysis, optimizer, reports).

It keeps messages, tool calls, tool results, token usage and the final reward. It drops the
embedded tasks, `evaluation_criteria` (gold actions) and anything derived from them
(reward_info details), plus raw API payloads.
"""

from __future__ import annotations

from typing import Any

from tau2.data_model.simulation import SimulationRun

BANNED_KEYS = (
    "evaluation_criteria",
    "expected_actions",
    "action_checks",
    "env_assertions",
    "nl_assertions",
    "communicate_checks",
    "gt_data_hash",
    # banking_knowledge task fields (and the task-level fields that sit next to the gold)
    "required_documents",
    "annotations",
    "issues",
    "reward_basis",
    "user_scenario",
)


def sanitize_message(m: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": getattr(m, "content", None)}
    calls = getattr(m, "tool_calls", None)
    if calls:
        out["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments, "requestor": c.requestor}
            for c in calls
        ]
    if m.role == "tool":
        out["id"] = m.id
        out["error"] = m.error
        out["requestor"] = m.requestor
    if getattr(m, "usage", None):
        out["usage"] = m.usage
    if getattr(m, "cost", None) is not None:
        out["cost"] = m.cost
    return out


def sanitize_simulation(sim: SimulationRun) -> dict[str, Any]:
    return {
        "task_id": sim.task_id,
        "termination_reason": str(getattr(sim.termination_reason, "value", sim.termination_reason)),
        "reward": sim.reward_info.reward if sim.reward_info else None,
        "messages": [sanitize_message(m) for m in sim.messages],
    }


def find_banned(obj: Any, path: str = "") -> list[str]:
    """Paths of banned keys inside a nested structure (used by tests and the runner)."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in BANNED_KEYS:
                hits.append(f"{path}/{k}")
            hits += find_banned(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_banned(v, f"{path}[{i}]")
    return hits
