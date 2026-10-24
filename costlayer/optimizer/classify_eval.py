"""Leave-one-out accuracy of a first-message task-type classifier (few-shot, Haiku)."""

from __future__ import annotations

from typing import Any

from costlayer.client import AnthropicClient
from costlayer.context import RunContext
from costlayer.pricing import HAIKU

SYSTEM = (
    "You label the first message of a retail customer-service conversation with the kind of "
    "request, named after the first database-changing action the agent will need (or 'none' if "
    "it only needs information). Answer with exactly one label from the list and nothing else."
)


BANKING_SYSTEM = (
    "You label the first message of a retail-bank customer-service conversation with the kind of "
    "request, named after the first database-changing action the agent will need (or 'none' if "
    "it only needs information). Answer with exactly one label from the list and nothing else."
)


def prompt(labels: list[str], shots: list[tuple[str, str]], message: str) -> str:
    lines = ["Labels: " + ", ".join(labels), ""]
    for m, lab in shots:
        lines += [f"Message: {m}", f"Label: {lab}", ""]
    lines += [f"Message: {message}", "Label:"]
    return "\n".join(lines)


def loo_accuracy(
    client: AnthropicClient,
    examples: list[tuple[str, str, str]],
    dispatch: str,
    system: str = SYSTEM,
) -> dict[str, Any]:
    """`examples`: (task id, first user message, label). One Haiku call per example."""
    labels = sorted({lab for _, _, lab in examples})
    correct = 0
    detail = []
    for i, (task, msg, lab) in enumerate(examples):
        shots = [(m, lb) for j, (_, m, lb) in enumerate(examples) if j != i]
        ctx = RunContext("classifier-loo", task, 1)
        res = client.create(
            ctx=ctx,
            model=HAIKU,
            system=system,
            messages=[{"role": "user", "content": prompt(labels, shots, msg)}],
            tools=[],
            thinking={"type": "disabled"},
            max_tokens=32,
            purpose="classifier",
        )
        text = "".join(b.get("text", "") for b in res.content if b.get("type") == "text").lower()
        pred = next((x for x in labels if x in text), "?")
        correct += pred == lab
        detail.append({"task": task, "label": lab, "pred": pred})
    n = len(examples)
    return {"n": n, "accuracy": correct / n if n else 0.0, "labels": labels, "detail": detail}


def examples_from_runs(runs: list[Any], res: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(task, first user message, task type) for every analysed task, from the `default` trace."""
    by = {r.task: r for r in runs if r.config == "default" and r.trace}
    out = []
    for t in res["tasks"]:
        tr = by[t].trace if t in by else None
        first = (
            next((m["content"] for m in tr["messages"] if m["role"] == "user"), None)
            if tr
            else None
        )
        if first:
            out.append((t, first, res["types"][t]))
    return out
