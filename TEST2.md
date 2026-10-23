# Test 2: the long-context case (banking_knowledge)

Read `CLAUDE.md` first. **Everything in it still applies** (verified facts about tau2 and the Anthropic API, ledger and cap rules, infrastructure-error handling, protocol rules for the agent, statistics, reports, conventions) unless this file says otherwise. This file only states what is different for test 2. Do not change a threshold after seeing data.

## Why this test exists

Test 1 (retail, Sonnet 5.5, 21 train tasks) found: `E:low` costs 0.90 of `default` and lost 2 of 21 tasks; per-type choice did not beat a single setting and the type classifier was at 47.6%; dependent read-only chains would remove at most 12.6% of agent cost. Even with everything stacked the best case is about 0.79 of `default`, short of the K2 bar (0.70). Retail is short (about 11 LLM calls, about 7K tokens of context). So test 1 is the unfavorable case for context and step levers.

**Test 2 is the last test of the thesis "a layer around a fixed model cuts the cost of an agent by about a third".** It runs the favorable case: long conversations where each tool result is 4,400 to 8,700 tokens. Outcome:

- **KEEP** (K1 and K2 both hold on the held-out tasks): worth a larger quality test.
- **Anything else:** the thesis is closed as a standalone savings product. State that plainly in the report.

## Verified facts (from `reports/banking-recon.md`, read from the pinned code and data)

- Domain `banking_knowledge`: 97 tasks, 698 documents (about 817K characters), no official split. Reward bases: `DB` 87, `ACTION` 9, `DB`+`NL_ASSERTION` 1 (`task_102`).
- **The 9 `ACTION` tasks are a free pass under `EvaluationType.ENV`** (an agent that does nothing scores 1.0 on them and 0.0 on the others). They are excluded: `banking_split(..., exclude=reward_basis_groups()["ACTION"])`, 88 tasks left. `task_102` stays and is scored on its `DB` part only (the language check is dropped, as in retail). Decide the exclusion once; never change it.
- **Variant: `bm25_grep`** (agent tools: the 14 base tools plus `KB_search` over local BM25 and `grep`). It needs no key and no gold. The fixed prefix (system prompt plus tool schemas) is about 4.2K tokens (estimate; real counts run 10 to 25% higher). Everything that needs a key (`*embeddings*`, `*reranker*`, `alltools*`), `golden_retrieval` (oracle), `terminal_use*` (not runnable here) and `full_kb` (250K-token prefix, about $0.62 to write per run) is out. The runner already refuses missing, key-needing and oracle variants.
- A `KB_search` result is about 4,400 to 6,000 tokens and a `grep` result 5,600 to 8,700 (measured locally, estimated tokens). Tool results, not the fixed prefix, dominate the context.
- `call_discoverable_agent_tool` is a WRITE tool in `tool_kinds.json`, so a chain miner that goes by kind never chains through it. Keep that.
- The user simulator has user tools and runs through LiteLLM with `thinking: disabled`; this is untested live. Phase 1 settles it.
- Order and split: `banking_order(seed=42, exclude=...)`; train is a prefix of that order. Test tasks are the first `N_test` ids of the remaining list, in order.

## Cost model (measure)

In a cached agent the cost is cache writes (every new token is written once at $2.50 per million), cache reads (every later call re-reads it at $0.10 per million) and output at $10 per million. A tool result of 5K tokens inserted at call i of N costs about $0.0125 to write plus $0.0005 per later call. With long results, **the size of the results that enter the context is the main lever**, then the number of calls, then reasoning effort. Everything below is allocated from the numbers measured in phase 1, not from this estimate.

## Configurations (all with automatic caching, `max_tokens=16000`, same agent class)

| Name | What differs |
|---|---|
| `default` | nothing: default effort, default adaptive thinking |
| `E:low` | `output_config.effort = low` |
| `C:top3` | every `KB_search` and `grep` result is cut to its first 3 items (documents for `KB_search`, matches for `grep`, in the tool's own order) and one line is appended: `[N more results omitted; refine the query to see them]`. Applied deterministically at the moment the result is first inserted, never later. |
| `EC` | `E:low` and `C:top3` together |
| `global*` | the cheapest acceptable of the four, chosen on train tasks only (selection rule as in test 1) |
| `layer` | the learned policy book plus mined composites (phase 3) |

`E:medium` and `T:bt` are dropped: in test 1 they were no cheaper than `E:low` and lost more tasks. The value 3 and the cut rule are fixed now, before any banking data. If the result format cannot be cut deterministically (items not separable), report it in phase 0 and propose a character-based cut with a fixed length; do not tune it later. `C:top3` is a gateway lever (it only edits what enters the request) and a customer who owns its retrieval can set a top-k alone; that is why `global*` includes it and K1 judges the layer against `global*`.

## Decision rules (unchanged from test 1)

- `margin = max(10 points, the A/A flip rate)`.
- **K2:** layer cost ≤ 70% of `default`, and layer success ≥ `default` success − margin.
- **K1:** layer cost ≤ 85% of `global*`, and layer success ≥ `global*` success − margin.
- KEEP if both hold. K2 holds and K1 fails: FEATURE ONLY. K2 fails: KILL. Cost = agent cost plus the classifier's; the user simulator is excluded from ratios and included in the ledger.

## Budget

- **Test 2 has its own ledger and total.** Use a separate branch `spend2` and a file `budget.json` committed by the coordinator (fields: `total`, and `phase_caps` for `b1`, `b2`, `b3`). The runner reads it and refuses to start a phase without it. The hard-cap mechanics of `CLAUDE.md` apply unchanged (reserve before each call, start files, `if: always()` ledger commit, 80% allocation, per-run abort rail at `max($1.00, 5 x c(default))`, reconciliation against the Console). Test 1's `spend` branch and $27 cap are not touched.
- **Phase 1 (`b1`) cap: $4.00.** It is in `budget.json` from the start. The totals for `b2` and `b3` are set by the coordinator after phase 1, from your numbers.
- The Console's monthly limit is the outer wall. Never plan to reach it.

## Phases

### Phase 0: build (free)

1. Budget mechanism: create `budget.json` with `total` 4.00 and only the `b1` cap (4.00), and the branch `spend2`; the coordinator edits the file later to add the `b2` and `b3` caps and the new total. The runner refuses `b2` and `b3` while their caps are missing.
2. The `C:top3` lever and the four configurations in the agent (`default`, `E:low`, `C:top3`, `EC`), with offline tests: the cut is deterministic and idempotent, the appended line is exact, items are cut in the tool's own order, results with 3 or fewer items pass unchanged, non-retrieval tool results are untouched, the cache prefix of earlier messages never changes.
3. A banking phase-1 flow in the runner (`domain="banking_knowledge"`, `retrieval_variant="bm25_grep"`, tasks in `banking_order(seed=42, exclude=ACTION)` order) with the same report structure as `reports/phase-1.md`.
4. A workflow input to choose the test and phase (`b1`, `b2`, `b3`), each of `b2` and `b3` refusing to run without `go` equal to `b2` or `b3`. `workflow_dispatch` only, as before.
5. Open the PR and **stop. The coordinator merges it and dispatches phase 1.**

### Phase 1: smoke test and cost anatomy (`b1`, cap $4.00)

1. **Haiku 5.5, first 3 tasks of the order, `default`.** Checks: no API error; BM25 index builds and `KB_search` and `grep` return; the user simulator with its user tools works through LiteLLM with `thinking: disabled`; cache reads above zero from the second call; no OpenAI call; ledger total matches the Console.
2. **Sonnet 5.5**, in this order: the first task on all four configurations; then tasks 2 and 3 on `default` and `EC`, only while the cap allows.
3. Write `reports/banking-phase-1.md`: cost per run and per configuration; LLM calls per run; context size at the last call; the cost split (cache read, cache write, uncached input, output; the result tokens that entered the context); whether `C:top3` changes the number of calls (re-queries); any run that hit `max_steps` or the abort rail.
4. **Budget request.** Compute, with the measured `c(x)` (user simulator included): `Sigma = c(default)+c(E:low)+c(C:top3)+c(EC)` and `T_needed = 1.1 x (4.0 + 20 x Sigma / 0.8 + (10 x c(default) + 30 x (c(default)+c(global*)+c(layer))) / 0.8)`, with `c(global*) = min of the four` and `c(layer) = c(default)` as stand-ins. These are the minimum sizes: **`N_train` at least 20 and `N_test` at least 30**, with `N_train + N_test` at most 88. Report `T_needed`, the resulting `N_train` and `N_test` at that total, and, if the founder's total is lower, what each option gives (for example `Sigma` without `EC`). **Stop. The coordinator sets the budget.**

If any check fails, stop and report. Do not change the benchmark, the models, the variant or a threshold.

### Phase 2: sweep on train tasks (`b2`)

Run the four configurations on the first `N_train` train tasks of the order, task by task. Reuse a phase 1 run only if model, configuration and task match, the run finished and its per-call costs add up to its summary. Then, from the sanitized traces only (the optimizer package must not import `tau2`), exactly as in test 1: anatomy, paired success, selection rule (`failures <= default + floor(0.10 x n)`), `global*`, task types (most common first write tool among the four traces), leave-one-out headroom with the 200-shuffle null, mining of dependent read-only chains on `global*` traces (calls the model issued in parallel are excluded; report the share of calls and of cost), classifier accuracy (below 70% switches per-type off).

**Gate (stricter than test 1, because test 1's arithmetic showed that a weak gate wastes money):** build the **optimistic projection** of the best layer the data allows: `projected = cost(global*) x (1 - chain_cost_share)`, or `cost(global*) x (1 - out_of_sample_headroom)` if the headroom beats the null (take the lower), with the classifier overhead counted as zero.
- Projected ≤ 0.70 of `default` **and** ≤ 0.85 of `global*`: go to phase 3.
- Projected ≤ 0.70 of `default` but above 0.85 of `global*`: **FEATURE ONLY**; stop.
- Projected above 0.70 of `default`: **KILL**; stop.

Write `reports/banking-phase-2.md` with the gate result and **stop.** The coordinator decides the next step from the rule above; no phase 3 without a gate pass.

### Phase 3: held-out verdict (`b3`)

As in test 1: freeze the policy book before any test run (commit and record the hash); build only the levers the gate approved (composites from the mined chains, per-type choices only if allowed); run `default`, `global*`, `layer` on the first `N_test` test tasks in blocks of 10, the A/A run of `default` on the first 10 after block 1, the stop rules (cost stop on the 80% interval of the `layer`/`global*` ratio, quality stop at 15 points and 3 more losses than wins), the same verdict table, and `reports/banking-phase-3.md` with both ratios and their intervals, discordant tasks, A/A result, savings split by lever and by gateway-only versus needs-SDK, fallback rate, and ledger beside the Console.

## What this cannot show

- Whether the agent says the right things (ENV checks the database state only; `task_102`'s language check is dropped).
- That quality is unchanged: with 30 to 40 test tasks a layer that truly loses 10 to 15 points can still pass. A KEEP earns a larger quality test, never a launch.
- Anything about other agents, other retrieval variants or other models: this is one benchmark, one variant, Sonnet 5.5, one simulated user.

Say all three in every report.
