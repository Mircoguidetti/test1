"""The `C:top3` lever: cut retrieval results to their first items.

What the real tools return (read from the pinned tau2 code, `retrieval_mixins.py`, and checked
against the real `KB_search` and `grep` with the `bm25_grep` variant, see `tests/test_cut.py`):

    1. <title>\\n   ID: <doc id>\\n   Score: <float, 4 decimals>\\n   Content: <content>\\n
    <item 2 in the same layout>, joined with "\\n" from the previous item, ...
    [Timing: retrieval=..ms, reranking=..ms, total=..ms]      (KB_search only, after the last item)

An item starts with a numbered header (`N. title`, `ID:`, `Score:`, `Content:`) and runs to the
next header. Items are numbered 1, 2, 3, ... in the tool's own order, so the parser accepts a
header only when it carries the next expected number and follows a blank line. That makes a
document whose text happens to contain a numbered line harmless unless it reproduces the whole
four-line header with the next number (no document in the pinned data does, tested).

The cut keeps the first `k` items byte for byte, drops everything after the k-th item (the other
items and, for `KB_search`, the timing line) and appends one line. It is a pure function of the
tool-result string, so it is deterministic, and it is applied once, when the result is first
inserted into the conversation (`agent.py`); earlier messages are never touched again.

Results with `k` or fewer items, results with no items ("No relevant documents found.", "No
matches found for pattern: ...") and every tool other than `KB_search` and `grep` pass unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CUT_TOOLS = frozenset({"KB_search", "grep"})
TOP_K = 3  # fixed before any banking data; never tuned (TEST2.md)
OMITTED_LINE = "[{n} more results omitted; refine the query to see them]"

_HEADER = re.compile(
    r"^(?P<num>[0-9]+)\. [^\n]*\n   ID: [^\n]*\n   Score: -?[0-9]+(?:\.[0-9]+)?\n   Content: ",
    re.MULTILINE,
)
_EMPTY_RESULTS = ("No relevant documents found.", "No matches found for pattern: ")


@dataclass(frozen=True)
class CutResult:
    text: str  # what goes into the conversation
    total: int  # items found in the raw result (0 for an empty result or an unknown format)
    omitted: int  # items cut away
    parsed: bool  # False only if the text looks like a result list but could not be parsed


def item_starts(text: str) -> list[int]:
    """Start offsets of items 1..n, in order. Empty if the text is not a numbered item list."""
    starts: list[int] = []
    for m in _HEADER.finditer(text):
        if int(m.group("num")) != len(starts) + 1:
            continue  # not the next expected number: part of an item's content
        if starts and text[m.start() - 2 : m.start()] != "\n\n":
            continue  # items are separated by a blank line
        if not starts and m.start() != 0:
            continue  # the list must open the result
        starts.append(m.start())
    return starts


def cut_top_items(text: str, k: int = TOP_K) -> CutResult:
    """Cut a `KB_search` or `grep` result string to its first `k` items."""
    starts = item_starts(text)
    n = len(starts)
    if n == 0:
        looks_like_list = text.startswith("1. ") or "\n   ID: " in text
        known_empty = text.startswith(_EMPTY_RESULTS)
        return CutResult(text, 0, 0, parsed=known_empty or not looks_like_list)
    if n <= k:
        return CutResult(text, n, 0, parsed=True)
    kept = text[: starts[k]]  # items 1..k, each with its trailing newline, plus the blank line
    return CutResult(kept + OMITTED_LINE.format(n=n - k), n, n - k, parsed=True)


def cut_for_tool(tool_name: str, text: str, k: int | None) -> CutResult | None:
    """None when the lever does not apply (no cut configured, or not a retrieval tool)."""
    if k is None or tool_name not in CUT_TOOLS:
        return None
    return cut_top_items(text, k)
