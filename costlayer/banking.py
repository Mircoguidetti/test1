"""banking_knowledge: retrieval variants, a deterministic task order, reward-basis groups.

Nothing here calls an API. The variant table is read from tau2's own registry so that it cannot
drift from the pinned code.
"""

from __future__ import annotations

import json
import random
from typing import Any

from tau2.domains.banking_knowledge import retrieval as R
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import get_db, get_knowledge_base
from tau2.domains.banking_knowledge.retrieval_toolkits import (
    KnowledgeToolsAllTools,
    KnowledgeToolsPlain,
    KnowledgeToolsWithGrep,
    KnowledgeToolsWithKBSearch,
    KnowledgeToolsWithKBSearchAndGrep,
    KnowledgeToolsWithShell,
)
from tau2.knowledge.sandbox_manager import _REQUIRED_BINARIES_LINUX
from tau2.registry import registry

from costlayer.pricing import est_tokens

BANKING = "banking_knowledge"
ORDER_SEED = 42

# Environment variables the pinned tau2 reads for each external backend.
KEY_FOR_EMBEDDER = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
RERANKER_KEY = "OPENAI_API_KEY"  # pointwise_llm_reranker builds an OpenAI client (gpt-5.2)


def requirements(v: R.RetrievalVariant) -> dict[str, Any]:
    """What a variant needs beyond this repository's own dependencies."""
    keys: list[str] = []
    for spec in (v.kb_search, v.kb_search_bm25, v.kb_search_dense):
        if spec is None:
            continue
        if spec.type == "embedding":
            keys.append(KEY_FOR_EMBEDDER.get(spec.embedder_type or "", "EMBEDDING_KEY"))
        if spec.reranker:
            keys.append(RERANKER_KEY)
    uses_bm25 = any(s is not None and s.type == "bm25" for s in (v.kb_search, v.kb_search_bm25))
    return {
        "external_keys": sorted(set(keys)),
        "binaries": list(_REQUIRED_BINARIES_LINUX) if v.shell is not None else [],
        "python": ["rank-bm25"] if uses_bm25 or v.grep is not None or v.kb_search else [],
        "uses_task_gold": v.build_prompt is R.golden_prompt,
    }


def is_local(name: str) -> bool:
    """True if the variant needs no key and does not read the task's gold documents."""
    v = R.RETRIEVAL_VARIANTS[R.RETRIEVAL_VARIANT_ALIASES.get(name, name)]
    r = requirements(v)
    return not r["external_keys"] and not r["uses_task_gold"]


def assert_usable(name: str) -> None:
    """Refuse variants with an external key (a stray call would be paid or fail) or the oracle."""
    if name not in R.RETRIEVAL_VARIANTS:
        raise ValueError(f"Unknown retrieval variant {name!r}")
    v = R.RETRIEVAL_VARIANTS[name]
    r = requirements(v)
    if r["external_keys"]:
        raise ValueError(f"Variant {name!r} needs {r['external_keys']}: not allowed")
    if r["uses_task_gold"]:
        raise ValueError(f"Variant {name!r} puts the task's required documents in the prompt")


def toolkit_class(v: R.RetrievalVariant) -> type:
    """The class `build_tools` would pick (same branching, no pipeline is built)."""
    if v.kb_search_bm25 is not None and v.kb_search_dense is not None and v.shell is not None:
        return KnowledgeToolsAllTools
    if v.shell is not None:
        return KnowledgeToolsWithShell
    if v.kb_search is not None and v.grep is not None:
        return KnowledgeToolsWithKBSearchAndGrep
    if v.kb_search is not None:
        return KnowledgeToolsWithKBSearch
    if v.grep is not None:
        return KnowledgeToolsWithGrep
    return KnowledgeToolsPlain


def agent_tools(v: R.RetrievalVariant, db: TransactionalDB) -> list[Any]:
    """Tool objects the agent would get, built with stub pipelines (nothing is indexed)."""
    cls = toolkit_class(v)
    n_stub = cls.__init__.__code__.co_argcount - 2  # arguments after (self, db)
    return list(cls(db, *([None] * n_stub)).get_tools().values())


def schema_tokens(tools: list[Any]) -> int:
    return est_tokens(json.dumps([t.openai_schema for t in tools]))


def describe(name: str) -> dict[str, Any]:
    v = R.RETRIEVAL_VARIANTS[name]
    db = get_db()
    kb = get_knowledge_base()
    tools = agent_tools(v, db)
    extra = [
        t
        for t in tools
        if t.name in {"KB_search", "KB_search_bm25", "KB_search_dense", "grep", "shell"}
    ]
    row: dict[str, Any] = {
        "name": name,
        "toolkit": toolkit_class(v).__name__,
        "n_tools": len(tools),
        "retrieval_tools": [t.name for t in extra],
        "tool_tokens": schema_tokens(tools),
        **requirements(v),
        "local": is_local(name),
    }
    if v.build_prompt is R.golden_prompt:
        toks = sorted(
            est_tokens(R.build_policy(v, kb, t)) for t in registry.get_tasks_loader(BANKING)()
        )
        row["prompt_tokens"] = toks[len(toks) // 2]
        row["prompt_tokens_range"] = (toks[0], toks[-1])
    else:
        row["prompt_tokens"] = est_tokens(R.build_policy(v, kb))
    return row


def all_variants() -> list[dict[str, Any]]:
    return [describe(n) for n in R.RETRIEVAL_VARIANTS]


def base_tool_names() -> list[tuple[str, str]]:
    """(name, kind) of the tools every variant gives the agent."""
    tools = agent_tools(R.RETRIEVAL_VARIANTS["no_knowledge"], get_db())
    kit = KnowledgeToolsPlain(get_db())
    return sorted((t.name, kit.tool_type(t.name).value.upper()) for t in tools)


# -- deterministic task order and split ------------------------------------------------------


def task_ids() -> list[str]:
    return sorted(t.id for t in registry.get_tasks_loader(BANKING)())


def banking_order(seed: int = ORDER_SEED, exclude: tuple[str, ...] = ()) -> list[str]:
    """All task ids in a fixed pseudo-random order: sorted ids, shuffled with `random.Random(seed)`.

    Independent of file-system order and of PYTHONHASHSEED. Excluded ids are removed BEFORE the
    shuffle, so the order of the others does not depend on which tasks are left out only through
    the list length; choose the exclusions once, then fix them.
    """
    ids = [i for i in task_ids() if i not in set(exclude)]
    random.Random(seed).shuffle(ids)
    return ids


def banking_split(
    n_train: int, seed: int = ORDER_SEED, exclude: tuple[str, ...] = ()
) -> tuple[list[str], list[str]]:
    """Train = first `n_train` ids of the seeded order, test = the rest.

    There is no official split for this domain. The sizes are an argument on purpose (no default):
    because train is a prefix of one fixed order, choosing a different size later never moves a
    task from one side to the other except at the boundary.
    """
    order = banking_order(seed, exclude)
    if not 0 < n_train < len(order):
        raise ValueError(f"n_train must be between 1 and {len(order) - 1}, got {n_train}")
    return order[:n_train], order[n_train:]


# -- reward bases -----------------------------------------------------------------------------


def reward_basis_groups() -> dict[str, list[str]]:
    """Task ids grouped by their reward_basis, e.g. 'ACTION', 'DB', 'DB+NL_ASSERTION'."""
    groups: dict[str, list[str]] = {}
    for t in sorted(registry.get_tasks_loader(BANKING)(), key=lambda t: t.id):
        key = "+".join(sorted(b.value for b in t.evaluation_criteria.reward_basis))
        groups.setdefault(key, []).append(t.id)
    return groups


def markdown_variants(rows: list[dict[str, Any]]) -> str:
    L = [
        "| variant | agent tools beyond the base set | external key | extra binaries | "
        "prompt tokens (est.) | tool-schema tokens (est.) | fixed prefix (est.) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        extra = ", ".join(r["retrieval_tools"]) or "none"
        key = ", ".join(r["external_keys"]) or (
            "oracle: reads task gold" if r["uses_task_gold"] else "none"
        )
        rng = r.get("prompt_tokens_range")
        pt = f"{r['prompt_tokens']:,}" + (f" ({rng[0]:,} to {rng[1]:,})" if rng else "")
        L.append(
            f"| `{r['name']}` | {extra} | {key} | {', '.join(r['binaries']) or '-'} | {pt} | "
            f"{r['tool_tokens']:,} | {r['prompt_tokens'] + r['tool_tokens']:,} |"
        )
    return "\n".join(L)


if __name__ == "__main__":  # python -m costlayer.banking : the variant table of the recon report
    print(markdown_variants(all_variants()))
