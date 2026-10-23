"""C:top3 (costlayer/cut.py): the parser against the REAL KB_search and grep (bm25_grep variant,
built locally, no key, no network) and against synthetic results."""

import re

import pytest
from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_knowledge_base
from tau2.registry import registry

from costlayer.agent import ALL_CONFIGS, BANKING_CONFIGS, CONFIGS, AnthropicAgent, Policy
from costlayer.client import AnthropicClient
from costlayer.context import RunContext, set_context
from costlayer.cut import CUT_TOOLS, OMITTED_LINE, TOP_K, cut_for_tool, cut_top_items, item_starts
from costlayer.pricing import SONNET
from tests.fakes import FakeSDK, text, thinking, tool_use

TRAILER = "\n[Timing: retrieval=12ms, reranking=0ms, total=12ms]"


def line(n: int) -> str:
    return f"[{n} more results omitted; refine the query to see them]"


def item(i: int, content: str = "body", title: str = "Title") -> str:
    return (
        f"{i}. {title} {i}\n   ID: doc_{i:03d}\n   Score: {10 - i / 3:.4f}\n   Content: {content}\n"
    )


def result(n: int, trailer: bool = True, contents: dict[int, str] | None = None) -> str:
    contents = contents or {}
    out = "\n".join(
        item(i, contents.get(i, f"text of item {i}\nsecond line")) for i in range(1, n + 1)
    )
    return out + (f"\n{TRAILER}" if trailer else "")


# -- the real tools -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_tools():
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep")
    return {t.name: t for t in env.tools.get_tools().values()}


QUERIES = [
    "credit card fees",
    "dispute",
    "foreign transaction",
    "overdraft",
    "referral bonus",
    "lost card",
    "wire transfer limits",
    "closing an account",
    "interest rate",
    "x",
]


def independent_count(text_: str) -> int:
    return len(re.findall(r"^   ID: ", text_, re.MULTILINE))


def test_no_knowledge_document_can_be_mistaken_for_an_item_header():
    kb = get_knowledge_base()
    assert len(kb.documents) == 698
    for d in kb.documents.values():
        assert "\n" not in d.title and "\n" not in d.id, d.id
        assert not re.search(r"(?m)^[0-9]+\. [^\n]*\n   ID: ", d.content), d.id
        assert "\n   ID: " not in d.content and "\n   Score: " not in d.content, d.id


@pytest.mark.parametrize("tool,arg", [("KB_search", "query"), ("grep", "pattern")])
def test_parser_agrees_with_the_real_tool_on_item_counts(real_tools, tool, arg):
    patterns = QUERIES if tool == "KB_search" else ["fee", "credit.*card", "dispute|refund", "lost"]
    seen_cut = False
    for q in patterns:
        raw = real_tools[tool](**{arg: q})
        n = independent_count(raw)
        res = cut_top_items(raw)
        assert res.parsed and res.total == n == len(item_starts(raw)), (tool, q)
        if n > TOP_K:
            seen_cut = True
            assert res.omitted == n - TOP_K
            assert independent_count(res.text) == TOP_K
            assert (
                res.text.endswith(line(n - TOP_K)) and res.text.count("more results omitted") == 1
            )
            # the kept part is a byte-for-byte prefix of the raw result, up to the blank line
            assert raw.startswith(res.text[: -len(line(n - TOP_K))])
            assert cut_top_items(res.text).text == res.text  # idempotent
            assert res.text == cut_top_items(raw).text  # deterministic
        else:
            assert res.text == raw and res.omitted == 0
    assert seen_cut


def test_real_kb_search_trailer_is_dropped_on_cut_results_only(real_tools):
    raw = real_tools["KB_search"](query="credit card fees")
    assert "[Timing:" in raw and independent_count(raw) == 10
    assert "[Timing:" not in cut_top_items(raw).text


def test_real_empty_results_pass_unchanged_and_count_as_parsed(real_tools):
    for raw in (
        real_tools["grep"](pattern="zzzqqq123"),
        real_tools["KB_search"](query=""),
    ):
        res = cut_top_items(raw)
        assert res.text == raw and res.total == 0 and res.parsed


# -- synthetic results --------------------------------------------------------------------------


def test_cut_keeps_first_three_items_in_order_and_appends_exactly_one_line():
    raw = result(10)
    res = cut_top_items(raw)
    assert res.omitted == 7 and res.total == 10 and res.parsed
    expected = (
        "\n".join(item(i, f"text of item {i}\nsecond line") for i in (1, 2, 3)) + "\n" + line(7)
    )
    assert res.text == expected
    ids = re.findall(r"ID: (doc_\d+)", res.text)
    assert ids == ["doc_001", "doc_002", "doc_003"]  # the tool's own order, not re-ranked
    assert res.text.splitlines()[-1] == line(7)
    assert res.text.count("\n") == expected.count("\n")  # exactly one extra line, no trailing \n


@pytest.mark.parametrize("n,omitted", [(4, 1), (5, 2), (10, 7), (25, 22)])
def test_count_in_the_line_is_the_number_of_omitted_items(n, omitted):
    res = cut_top_items(result(n))
    assert res.omitted == omitted and res.text.endswith(line(omitted))
    assert OMITTED_LINE.format(n=omitted) == line(omitted)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_three_or_fewer_items_pass_unchanged(n):
    raw = result(n) if n else "No relevant documents found.\n\n" + TRAILER
    assert cut_top_items(raw).text == raw and cut_top_items(raw).omitted == 0


def test_grep_style_result_without_trailer_and_content_ending_in_newlines():
    raw = result(6, trailer=False, contents={3: "ends with blank lines\n\n\n"})
    res = cut_top_items(raw)
    assert res.omitted == 3 and independent_count(res.text) == 3
    assert res.text.endswith("\n" + line(3))


def test_numbered_lines_inside_a_document_do_not_split_an_item():
    tricky = "steps:\n2. do this first\n3. then that\n\n4. and also\n   ID: not a header"
    raw = result(5, contents={1: tricky, 2: "1. nested\n   ID: x\n   Score: 1.0\n   Content: y"})
    res = cut_top_items(raw)
    assert res.total == 5 and res.omitted == 2
    assert tricky in res.text  # item 1 is kept whole


def test_idempotent_and_deterministic_for_many_sizes():
    for n in range(0, 15):
        raw = result(n) if n else "No matches found for pattern: x"
        once = cut_top_items(raw)
        assert cut_top_items(raw) == once
        assert cut_top_items(once.text).text == once.text


def test_unknown_format_passes_unchanged_and_is_reported_not_hidden():
    weird = "1. Title\n   Identifier: a\n   Content: b\n2. Title\n   Identifier: c\n"
    res = cut_top_items(weird)
    assert res.text == weird and res.total == 0 and not res.parsed
    broken = "1. Title\n   ID: a\n   Content: missing the score line\n"
    res = cut_top_items(broken)
    assert res.text == broken and not res.parsed  # looks like a list but is not parseable
    res = cut_top_items("Error: tool crashed")
    assert res.text == "Error: tool crashed" and res.parsed


def test_lever_applies_to_kb_search_and_grep_only_and_only_when_configured():
    assert CUT_TOOLS == {"KB_search", "grep"}
    raw = result(10)
    assert cut_for_tool("KB_search", raw, None) is None
    assert cut_for_tool("get_current_time", raw, 3) is None
    assert cut_for_tool("KB_search_bm25", raw, 3) is None  # not in this variant, not in the plan
    assert cut_for_tool("grep", raw, 3).omitted == 7
    assert TOP_K == 3


def test_configurations():
    assert list(BANKING_CONFIGS) == ["default", "E:low", "C:top3", "EC"]
    assert BANKING_CONFIGS["default"] == Policy()
    assert BANKING_CONFIGS["E:low"] == Policy(effort="low")
    assert BANKING_CONFIGS["C:top3"] == Policy(cut_top=3)
    assert BANKING_CONFIGS["EC"] == Policy(effort="low", cut_top=3)
    assert BANKING_CONFIGS["E:low"] == CONFIGS["E:low"]
    assert list(CONFIGS) == ["default", "E:medium", "E:low", "T:bt"]  # test 1 iterates over this
    assert set(ALL_CONFIGS) == set(CONFIGS) | set(BANKING_CONFIGS)
    assert CONFIGS["E:low"].llm_args() == {"effort": "low", "thinking": None}  # retail unchanged
    assert BANKING_CONFIGS["EC"].llm_args() == {"effort": "low", "thinking": None, "cut_top": 3}


# -- inside the agent ---------------------------------------------------------------------------


@pytest.fixture
def banking_tools():
    return registry.get_env_constructor("banking_knowledge")(
        retrieval_variant="bm25_grep"
    ).get_tools()


def make(sdk, ledger, tools, policy):
    ctx = RunContext("C:top3", "t1", 1)
    set_context(ctx)
    agent = AnthropicAgent(tools, "POLICY", SONNET, policy, AnthropicClient(ledger, sdk))
    return agent, agent.get_init_state([]), ctx


def step(agent, state, call, content, error=False):
    return agent.generate_next_message(
        ToolMessage(id=call.id, role="tool", content=content, error=error), state
    )


def test_agent_cuts_at_first_insertion_and_never_touches_earlier_messages(ledger, banking_tools):
    big = result(10)
    sdk = FakeSDK(
        [
            [thinking(), tool_use("KB_search", query="q1")],
            [thinking(), tool_use("grep", pattern="p")],
            [thinking(), tool_use("get_current_time")],
            [text("done")],
        ]
    )
    agent, state, ctx = make(sdk, ledger, banking_tools, BANKING_CONFIGS["C:top3"])
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    msg, state = step(agent, state, msg.tool_calls[0], big)
    msg, state = step(agent, state, msg.tool_calls[0], result(7, trailer=False))
    long_time = result(10)  # item-like text from a non-retrieval tool: must stay as it is
    msg, state = step(agent, state, msg.tool_calls[0], long_time)
    reqs = [r["messages"] for r in sdk.requests]
    assert len(reqs) == 4
    # cache prefix: every request starts with the exact messages of the previous one
    for a, b in zip(reqs, reqs[1:], strict=False):
        assert b[: len(a)] == a
    res1 = reqs[1][-1]["content"][0]["content"]
    res2 = reqs[2][-1]["content"][0]["content"]
    res3 = reqs[3][-1]["content"][0]["content"]
    assert res1 == cut_top_items(big).text and res1.endswith(line(7))
    assert res2.endswith(line(4)) and independent_count(res2) == 3
    assert res3 == long_time  # get_current_time is not a retrieval tool
    assert reqs[3][2]["content"][0]["content"] == res1  # still the same string, still cut
    assert (ctx.retrieval_results, ctx.cut_results, ctx.cut_items_omitted) == (2, 2, 11)
    assert ctx.retrieval_chars_raw == len(big) + len(result(7, trailer=False))
    assert ctx.retrieval_chars_in < ctx.retrieval_chars_raw and ctx.cut_unparsed == 0


def test_default_and_effort_only_configurations_never_cut(ledger, banking_tools):
    for name in ("default", "E:low"):
        sdk = FakeSDK([[thinking(), tool_use("KB_search", query="q")], [text("ok")]])
        agent, state, ctx = make(sdk, ledger, banking_tools, BANKING_CONFIGS[name])
        msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
        step(agent, state, msg.tool_calls[0], result(10))
        assert sdk.requests[1]["messages"][-1]["content"][0]["content"] == result(10), name
        assert ctx.cut_results == 0 and ctx.retrieval_chars_in == ctx.retrieval_chars_raw


def test_ec_cuts_and_sets_low_effort(ledger, banking_tools):
    sdk = FakeSDK([[thinking(), tool_use("grep", pattern="p")], [text("ok")]])
    agent, state, _ = make(sdk, ledger, banking_tools, BANKING_CONFIGS["EC"])
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    step(agent, state, msg.tool_calls[0], result(5, trailer=False))
    assert sdk.requests[0]["output_config"] == {"effort": "low"}
    assert sdk.requests[1]["messages"][-1]["content"][0]["content"].endswith(line(2))


def test_parallel_calls_each_result_is_cut_in_one_user_turn(ledger, banking_tools):
    sdk = FakeSDK(
        [
            [
                thinking(),
                tool_use("KB_search", query="a"),
                tool_use("get_current_time"),
                tool_use("grep", pattern="b"),
            ],
            [text("ok")],
        ]
    )
    agent, state, _ = make(sdk, ledger, banking_tools, BANKING_CONFIGS["C:top3"])
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    c = msg.tool_calls
    multi = MultiToolMessage(
        role="tool",
        tool_messages=[
            ToolMessage(id=c[0].id, role="tool", content=result(10)),
            ToolMessage(id=c[1].id, role="tool", content="2026-10-23"),
            ToolMessage(id=c[2].id, role="tool", content=result(6, trailer=False)),
        ],
    )
    agent.generate_next_message(multi, state)
    blocks = sdk.requests[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in blocks] == [x.id for x in c]
    assert blocks[0]["content"].endswith(line(7)) and blocks[1]["content"] == "2026-10-23"
    assert blocks[2]["content"].endswith(line(3))


def test_error_results_and_unparsable_results_are_not_cut_and_unparsable_is_counted(
    ledger, banking_tools
):
    sdk = FakeSDK(
        [
            [thinking(), tool_use("grep", pattern="(")],
            [thinking(), tool_use("grep", pattern="p")],
            [text("ok")],
        ]
    )
    agent, state, ctx = make(sdk, ledger, banking_tools, BANKING_CONFIGS["C:top3"])
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    msg, state = step(
        agent, state, msg.tool_calls[0], "Error: bad pattern " + result(9), error=True
    )
    broken = "1. T\n   ID: a\n   Content: no score\n"
    step(agent, state, msg.tool_calls[0], broken)
    assert sdk.requests[1]["messages"][-1]["content"][0]["content"].startswith("Error: bad")
    assert sdk.requests[2]["messages"][-1]["content"][0]["content"] == broken
    assert ctx.cut_unparsed == 1 and ctx.cut_results == 0
