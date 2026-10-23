"""Task-type classifier (Haiku). Its cost counts as layer cost."""

from __future__ import annotations

from costlayer.domains import TASK_TYPES
from costlayer.llm import LLM, Request
from costlayer.pricing import HAIKU

CLASSIFIER_MODEL = HAIKU

_HINTS = {
    "cancel_order": "the customer wants to cancel a pending order",
    "modify_pending_order": "change items, address or payment of a pending (not yet shipped) order",
    "return_items": "return items of a delivered order",
    "exchange_items": "exchange items of a delivered order for other variants",
    "address_or_profile": "change the default address in the user profile",
    "info_only": "only asks for information; no change to any order or reservation",
    "book_flight": "book a new flight reservation",
    "modify_reservation": "change flights, cabin, baggage or passengers of an existing reservation",
    "cancel_reservation": "cancel a reservation",
    "other": "none of the above, unclear, or several different requests",
}


def classifier_system(domain: str) -> str:
    lines = "\n".join(f"- {t}: {_HINTS[t]}" for t in TASK_TYPES[domain])
    return (
        f"You label the first message of a {domain} customer-service conversation.\n"
        f"Answer with exactly one label from this list and nothing else:\n{lines}"
    )


def classify(llm: LLM, domain: str, run_id: str, first_user_message: str) -> tuple[str, float]:
    """Return (task type, cost in USD)."""
    resp = llm.create(
        Request(
            purpose="classifier",
            run_id=run_id,
            model=CLASSIFIER_MODEL,
            system=classifier_system(domain),
            messages=[{"role": "user", "content": first_user_message}],
            max_tokens=24,
            thinking={"type": "disabled"},
        )
    )
    text = "".join(b.get("text", "") for b in resp.content if b.get("type") == "text").lower()
    for label in TASK_TYPES[domain]:
        if label in text:
            return label, resp.cost
    return "other", resp.cost
