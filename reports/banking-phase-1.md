# Test 2, phase 1 (b1) report: banking_knowledge smoke test

Benchmark: `sierra-research/tau2-bench` at `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`, domain `banking_knowledge`, retrieval variant `bm25_grep`, tasks in `banking_order(seed=42, exclude=ACTION)` order (88 tasks; the 9 ACTION tasks are a free pass under ENV), `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`. Models: agent `claude-haiku-5-5` (checks) and `claude-sonnet-5-5`; user simulator `claude-haiku-5-5` in every configuration. Cost per run is agent plus user simulator unless stated.

## Verdict of the smoke test

- PASS: no API error. all runs finished without error
- PASS: cache reads above zero from the second call. task_047: call 2 cache_read=5473; task_073: call 2 cache_read=5511; task_077: call 2 cache_read=5526
- PASS: no 400 from thinking blocks. no 400 / thinking error text in run errors
- PASS: no call to any OpenAI model. OPENAI_API_KEY is unset; no OpenAI error text in any run
- PASS: BM25 index builds; KB_search and grep return; the item parser agrees with the tool. KB_search: {"returns": true, "items": 10, "parsed_items": 10, "cut_ok": true}; grep: {"returns": true, "items": 10, "parsed_items": 10, "cut_ok": true}
- PASS: user simulator with user tools works through LiteLLM, thinking disabled. user calls per run: [9, 6, 12]; user tool calls in traces: 1; agent retrieval calls in traces: {'KB_search': 20, 'grep': 0} (information only)
- MANUAL: ledger total vs Console. Ledger total for this dispatch (2026-10-23 UTC): $2.5309. Founder: enter the Console figure for the day here.

First tasks of the order: `task_073`, `task_077`, `task_047`

## Runs

| model | config | task | status | reward | LLM calls | agent $ | user $ | input | cache write | cache read | output | context at last call | retrieval results (raw / in context, est. tokens) | cut results | omitted items | termination |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| haiku | default | task_047 | ok | 0 | 16 | 0.0113 | 0.0052 | 46 | 39672 | 287409 | 6872 | 39674 | 5 (19743 / 19743) | 0 | 0 | user_stop |
| haiku | default | task_073 | ok | 0 | 19 | 0.0229 | 0.0023 | 48 | 77995 | 769474 | 10908 | 77997 | 6 (36898 / 36898) | 0 | 0 | user_stop |
| haiku | default | task_077 | ok | 0 | 39 | 0.1588 | 0.0060 | 100 | 123306 | 2828033 | 18688 | 123310 | 9 (68012 / 68012) | 0 | 0 | user_stop |
| sonnet | C:top3 | task_073 | ok | 1 | 27 | 0.3451 | 0.0031 | 68 | 63787 | 947903 | 9075 | 69300 | 19 (94373 / 30028) | 16 | 107 | user_stop |
| sonnet | E:low | task_073 | ok | 0 | 8 | 0.0931 | 0.0016 | 24 | 24337 | 120886 | 2017 | 29850 | 3 (15755 / 15755) | 0 | 0 | user_stop |
| sonnet | EC | task_047 | ok | 0 | 19 | 0.1317 | 0.0028 | 48 | 28935 | 278763 | 3141 | 28937 | 8 (42375 / 14217) | 8 | 56 | user_stop |
| sonnet | EC | task_073 | ok | 0 | 15 | 0.1068 | 0.0020 | 40 | 22254 | 242658 | 2685 | 27767 | 10 (44155 / 13650) | 8 | 56 | user_stop |
| sonnet | EC | task_077 | ok | 0 | 37 | 0.2822 | 0.0046 | 94 | 45366 | 984323 | 7017 | 45368 | 11 (70761 / 22024) | 10 | 64 | user_stop |
| sonnet | default | task_047 | ok | 0 | 22 | 0.3514 | 0.0034 | 56 | 87253 | 916874 | 4149 | 87255 | 13 (57456 / 57456) | 0 | 0 | user_stop |
| sonnet | default | task_073 | ok | 1 | 18 | 0.4683 | 0.0016 | 44 | 119093 | 1048419 | 6559 | 119097 | 15 (69397 / 69397) | 0 | 0 | user_stop |
| sonnet | default | task_077 | ok | 0 | 33 | 0.5225 | 0.0041 | 84 | 96062 | 2013788 | 8076 | 96064 | 8 (55698 / 55698) | 0 | 0 | user_stop |

### Runs that hit `max_steps` or the abort rail

- none
- C:top3 could not parse a retrieval result in: no run

## Cost anatomy, Sonnet 5.5

| config | runs | mean $/run | LLM calls/run | cache read | cache write | uncached input | output | context at last call (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 3 | 0.4504 | 24.3 | 30% | 56% | 0% | 14% | 100805 |
| E:low | 1 | 0.0947 | 8.0 | 13% | 65% | 0% | 22% | 29850 |
| C:top3 | 1 | 0.3483 | 27.0 | 27% | 46% | 0% | 26% | 69300 |
| EC | 3 | 0.1767 | 23.7 | 29% | 46% | 0% | 25% | 34024 |

Shares are of agent cost (user simulator excluded). Configurations have different numbers of runs on different tasks: compare them on the first task only (below).

### `C:top3` and the number of calls (first task `task_073`, one run each)

| config | LLM calls | retrieval results | cut results | omitted items | result tokens in context (est.) | mean $ |
|---|---:|---:|---:|---:|---:|---:|
| default | 18 | 15 | 0 | 0 | 69397 | 0.4698 |
| E:low | 8 | 3 | 0 | 0 | 15755 | 0.0947 |
| C:top3 | 27 | 19 | 16 | 107 | 30028 | 0.3483 |
| EC | 15 | 10 | 8 | 56 | 13650 | 0.1089 |

`C:top3` vs `default` on this task: LLM calls 27 vs 18, retrieval calls 19 vs 15 (re-queries show up as extra retrieval calls). One run each: a first look, not a result.

## Budget request (computed, TEST2.md formula)

Measured `c(x)` = mean cost per run, user simulator included: `default` $0.4504, `E:low` $0.0947, `C:top3` $0.3483, `EC` $0.1767.

- `Sigma` = **$1.0702**; stand-ins `c(global*)` = $0.0947 (cheapest of the four), `c(layer)` = $0.4504 (= `c(default)`).
- `T_needed = 1.1 x (4.0 + 20 x Sigma / 0.8 + (10 x c(default) + 30 x (c(default) + c(global*) + c(layer))) / 0.8)` = **$81.09**.
- At that total: `N_train` = 20, `N_test` = 30 (the minimums; sum 50 <= 88: OK).
- Without `EC` (three configurations): `T_needed` = $76.23.

If the total is lower, what each option buys (`N_train` with `N_test` held at 30; `N_test` with `N_train` held at 20; a 0 means that side cannot be afforded):

| configurations | total | N_train (at N_test=30) | N_test (at N_train=20) | both minimums |
|---|---:|---:|---:|---|
| all four | $81.09 (T_needed) | 20 | 30 | yes |
| all four | $64.87 (80% of T_needed) | 8 | 18 | no |
| all four | $48.65 (60% of T_needed) | 0 | 6 | no |
| without EC | $76.23 (T_needed) | 20 | 30 | yes |
| without EC | $60.98 (80% of T_needed) | 7 | 18 | no |
| without EC | $45.74 (60% of T_needed) | 0 | 7 | no |

Caveat: the means rest on a few runs and, for `E:low` and `C:top3`, on the first task only, while `default` and `EC` may include tasks 2 and 3. Task cost varies a lot. On the first task alone the same formula gives: `T_needed` = $81.62.

**Stop. The coordinator sets the budget.**

## OpenAI (and other non-ours) defaults in tau2

| default | value | refs outside config.py | status |
|---|---|---:|---|
| `DEFAULT_LLM_AGENT` | "gpt-4.1-2025-04-14" | 7 | agent model (overridden: we use our own agent, model per config) |
| `DEFAULT_LLM_USER` | "gpt-4.1-2025-04-14" | 6 | user simulator (overridden: anthropic/claude-haiku-5-5) |
| `DEFAULT_LLM_NL_ASSERTIONS` | "gpt-5.4-mini-2026-03-17" | 1 | LLM judge for NL assertions (not reached with ENV evaluation) |
| `DEFAULT_LLM_ENV_INTERFACE` | "gpt-4.1-2025-04-14" | 1 | LLM environment interface (not used by retail) |
| `DEFAULT_SEMANTIC_COMMUNICATE_MODEL` | "gpt-5.5" | 2 | semantic communicate judge (not reached with ENV) |
| `DEFAULT_ELICITATION_LLM_USER` | "gpt-5.5" | 5 | intake domain user (not our domain) |
| `DEFAULT_LLM_EVAL_USER_SIMULATOR` | "claude-opus-4-5" | 27 | user-simulator eval (not used) |

## Budget

- Phase b1 spent (ledger): **$2.5309** of $4.00
- Total spent so far (test 2): $2.5309 of $4.00
- API calls in the ledger: 338; by purpose: user_sim $0.0367, agent $2.4942
- Console reconciliation: the ledger total for the UTC day beside the Console figure must agree within 5%. Ledger total for this dispatch (2026-10-23 UTC): $2.5309. Founder: enter the Console figure for the day here.

## What this cannot show

- Whether the agent says the right things (ENV checks the database state only; `task_102`'s language check is dropped).
- That quality is unchanged: with 30 to 40 test tasks a layer that truly loses 10 to 15 points can still pass. A KEEP earns a larger quality test, never a launch.
- Anything about other agents, other retrieval variants or other models: this is one benchmark, one variant, Sonnet 5.5, one simulated user.
- Anything about success rates or savings: this phase has 1 to 3 tasks. It checks the pipeline and measures costs for the budget request. Haiku numbers say nothing about Sonnet.
