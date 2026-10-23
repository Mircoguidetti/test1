"""tau-bench domains: tasks, split, environments, tools, policy sections, task types."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from tau_bench.envs import get_env
from tau_bench.envs.base import Env
from tau_bench.envs.user import BaseUserSimulationEnv
from tau_bench.types import Action, Task

DOMAINS = ("retail", "airline")
SPLIT_SEED = 20260
TAU_BENCH_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"

READ_ONLY_TOOLS: dict[str, frozenset[str]] = {
    "retail": frozenset(
        {
            "calculate",
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "get_order_details",
            "get_product_details",
            "get_user_details",
            "list_all_product_types",
            "think",
        }
    ),
    "airline": frozenset(
        {
            "calculate",
            "get_reservation_details",
            "get_user_details",
            "list_all_airports",
            "search_direct_flight",
            "search_onestop_flight",
            "think",
        }
    ),
}
ALWAYS_KEEP_TOOLS = frozenset({"transfer_to_human_agents"})

# Policy section (lower-case heading) -> tools that section governs. Sections that are not
# listed (the intro and "Domain basic") are always kept.
SECTION_TOOLS: dict[str, dict[str, frozenset[str]]] = {
    "retail": {
        "cancel pending order": frozenset({"cancel_pending_order"}),
        "modify pending order": frozenset(
            {
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
            }
        ),
        "return delivered order": frozenset({"return_delivered_order_items"}),
        "exchange delivered order": frozenset({"exchange_delivered_order_items"}),
    },
    "airline": {
        "book flight": frozenset(
            {
                "book_reservation",
                "search_direct_flight",
                "search_onestop_flight",
            }
        ),
        "modify flight": frozenset(
            {
                "update_reservation_flights",
                "update_reservation_baggages",
                "update_reservation_passengers",
                "search_direct_flight",
                "search_onestop_flight",
            }
        ),
        "cancel flight": frozenset({"cancel_reservation"}),
        "refund": frozenset({"cancel_reservation", "send_certificate"}),
    },
}

TASK_TYPES: dict[str, tuple[str, ...]] = {
    "retail": (
        "cancel_order",
        "modify_pending_order",
        "return_items",
        "exchange_items",
        "address_or_profile",
        "info_only",
        "other",
    ),
    "airline": ("book_flight", "modify_reservation", "cancel_reservation", "info_only", "other"),
}

# First consequential tool -> task type; used to label training traces and by the dry-run
# classifier stand-in. The live classifier (Haiku) sees only the first user message.
ACTION_TO_TYPE: dict[str, dict[str, str]] = {
    "retail": {
        "cancel_pending_order": "cancel_order",
        "modify_pending_order_items": "modify_pending_order",
        "modify_pending_order_address": "modify_pending_order",
        "modify_pending_order_payment": "modify_pending_order",
        "return_delivered_order_items": "return_items",
        "exchange_delivered_order_items": "exchange_items",
        "modify_user_address": "address_or_profile",
    },
    "airline": {
        "book_reservation": "book_flight",
        "update_reservation_flights": "modify_reservation",
        "update_reservation_baggages": "modify_reservation",
        "update_reservation_passengers": "modify_reservation",
        "cancel_reservation": "cancel_reservation",
    },
}


def type_from_tools(domain: str, tool_names: list[str]) -> str:
    for name in tool_names:
        t = ACTION_TO_TYPE[domain].get(name)
        if t:
            return t
    return "info_only"


@dataclass(frozen=True, order=True)
class TaskRef:
    domain: str
    idx: int

    @property
    def key(self) -> str:
        return f"{self.domain}-{self.idx}"


def load_tasks(domain: str) -> list[Task]:
    if domain == "retail":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST

        return TASKS_TEST
    from tau_bench.envs.airline.tasks_test import TASKS

    return TASKS


def make_split(seed: int = SPLIT_SEED) -> dict[str, list[TaskRef]]:
    """Half train, half test, per domain, by task id with a fixed seed."""
    rng = random.Random(seed)
    train: list[TaskRef] = []
    test: list[TaskRef] = []
    for domain in DOMAINS:
        ids = list(range(len(load_tasks(domain))))
        rng.shuffle(ids)
        half = len(ids) // 2 + len(ids) % 2
        train += [TaskRef(domain, i) for i in sorted(ids[:half])]
        test += [TaskRef(domain, i) for i in sorted(ids[half:])]
    return {"train": train, "test": test}


def make_env(ref: TaskRef, user: BaseUserSimulationEnv) -> Env:
    env = get_env(
        ref.domain,
        user_strategy="human",
        user_model="unused",
        task_split="test",
        task_index=ref.idx,
    )
    env.user = user  # the simulator is ours (Anthropic API, ledgered), never litellm
    return env


def anthropic_tools(env: Env) -> list[dict[str, Any]]:
    out = []
    for t in env.tools_info:
        f = t["function"]
        out.append(
            {"name": f["name"], "description": f["description"], "input_schema": f["parameters"]}
        )
    return out


def split_policy(wiki: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (intro, [(heading, section_text)]); section_text includes its '## ' line."""
    parts = re.split(r"(?m)^(?=## )", wiki)
    intro = parts[0]
    sections = []
    for p in parts[1:]:
        heading = p.splitlines()[0][3:].strip().lower()
        sections.append((heading, p))
    return intro, sections


def first_consequential_type(domain: str, actions: list[Action]) -> str:
    return type_from_tools(domain, [a.name for a in actions])
