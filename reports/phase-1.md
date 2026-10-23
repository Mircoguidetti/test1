# Phase 1 report: smoke test

> **Correction added in the phase 2 PR.** The token counts of every `default` Sonnet run in the tables below were inflated by the Haiku smoke runs on the same three tasks: the runner summed ledger rows by (config, task, attempt) and ignored the model. This affects the `default` row of "Cost anatomy", the `default` line of "Thinking tokens", the sentence "E:low vs default", and the token columns of the `default` runs. It does **not** affect any cost in dollars, `LLM calls`, `c(x)`, the allocation table (N_train 21, N_test 40) or the budget, because those come from per-run spend, not from the token sums. Corrected numbers, straight from the ledger (`spend` branch, dispatch 39998961720-1), Sonnet only:
>
> | config | calls | cache read | cache write | uncached input | output | output/call | reported thinking/call | non-thinking output/call |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|
> | default | 41 | 18% | 49% | 0% | 33% | 192 | 24.3 | 168 |
> | E:medium | 17 | 21% | 45% | 0% | 34% | 174 | 14.9 | 159 |
> | E:low | 41 | 18% | 53% | 0% | 29% | 149 | 6.9 | 142 |
> | T:bt | 19 | 22% | 43% | 0% | 35% | 171 | 0.0 | 171 |
>
> Cache-write tokens per run, Sonnet: `default` [15688, 15660, 15599] (mean 15649); `E:low` [15022, 14895, 15259] (mean 15059).
>
> Corrected reading: cache writes per run are the same for `default` and `E:low` (about 15.6K vs 15.1K tokens, not 2x). Output per call is 192 for `default` and 149 for `E:low` (-22%, not -64%). The API's thinking-token field is small in every configuration (24 and 7 per call) and explains only about 17 of the 43 tokens per call that differ; the rest is output the field does not call thinking. With 1 to 3 tasks per configuration this is a first look, not a result. The per-call split is investigated in the phase 2 PR.

Benchmark: `sierra-research/tau2-bench` at `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2` (retail, official train split, seed-42 order, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`). User simulator: `claude-haiku-5-5`, same in every configuration. Cost per run below is agent plus user simulator unless stated.

## Verdict of the smoke test

- PASS: no API error. all runs finished without error
- PASS: cache reads above zero from the second call. 23: call 2 cache_read=6755; 103: call 2 cache_read=6728; 54: call 2 cache_read=6691
- PASS: no 400 from thinking blocks. no 400 / thinking error text in run errors
- PASS: no call to any OpenAI model. OPENAI_API_KEY is unset; no OpenAI error text in any run
- MANUAL: ledger total vs Console. Ledger total for this dispatch (2026-10-23 UTC): $0.6671. Founder: enter the Console figure for the day here.

## Runs

| model | config | task | status | reward | LLM calls | agent $ | user $ | input | cache write | cache read | output | termination |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sonnet | E:low | 103 | ok | 1 | 17 | 0.0797 | 0.0024 | 52 | 15022 | 175594 | 2452 | user_stop |
| sonnet | E:low | 23 | ok | 1 | 10 | 0.0640 | 0.0007 | 26 | 14895 | 90737 | 1759 | user_stop |
| sonnet | E:low | 54 | ok | 1 | 14 | 0.0702 | 0.0015 | 40 | 15259 | 128987 | 1910 | user_stop |
| sonnet | E:medium | 103 | ok | 1 | 17 | 0.0859 | 0.0026 | 52 | 15380 | 177868 | 2959 | user_stop |
| sonnet | T:bt | 103 | ok | 1 | 19 | 0.0932 | 0.0034 | 60 | 15902 | 208963 | 3243 | user_stop |
| sonnet | default | 103 | ok | 1 | 17 | 0.0881 | 0.0025 | 96 | 31269 | 329322 | 6254 | user_stop |
| sonnet | default | 23 | ok | 0 | 11 | 0.0756 | 0.0014 | 66 | 31410 | 231043 | 5221 | user_stop |
| sonnet | default | 54 | ok | 1 | 13 | 0.0750 | 0.0014 | 80 | 32438 | 283642 | 5729 | user_stop |

## Cost anatomy, Sonnet 5.5 (question 1)

| config | runs | mean $/run (agent+user) | LLM calls/run | cache read | cache write | uncached input | output |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 3 | 0.0813 | 13.7 | 17% | 48% | 0% | 35% |
| E:medium | 1 | 0.0885 | 17.0 | 21% | 45% | 0% | 34% |
| E:low | 3 | 0.0728 | 13.7 | 18% | 53% | 0% | 29% |
| T:bt | 1 | 0.0966 | 19.0 | 22% | 43% | 0% | 35% |

Shares are of agent cost (user simulator excluded).

### Thinking tokens (Sonnet)

- default: output 420 tokens/call; thinking 24/call (reported by the API)
- E:medium: output 174 tokens/call; thinking 15/call (reported by the API)
- E:low: output 149 tokens/call; thinking 7/call (reported by the API)
- T:bt: output 171 tokens/call; thinking 0/call (reported by the API)

**E:low vs default:** output per call 149 vs 420 (-64%); cache-write tokens 45176 vs 95117. Few runs: treat as a first look, not a result.
- `E:medium` accepted by the API on Sonnet 5.5: yes
- `T:bt` accepted by the API on Sonnet 5.5: yes

## Allocation table (computed, phases 2 and 3)

Measured `c(x)` = mean cost per run, user simulator included: `default` $0.0813, `E:medium` $0.0885, `E:low` $0.0728, `T:bt` $0.0966.

- Phase 2: `N_train = min(74, floor(0.8 x 9.00 / sum c))` = **21** (minimum 12): OK.
- Phase 3 (stand-ins `c(global*)` = $0.0728 = min(default, E:low, T:bt), `c(layer)` = $0.0854 = default +5%): `N_test = min(40, floor((0.8 x 13.00 - 10 x c(default)) / (sum)))` = **40** (minimum 20): OK.

Caveat: `c(E:medium)` and `c(T:bt)` rest on one run each, and `c(default)`/`c(E:low)` on up to three. Task cost varies a lot, so these numbers are rough.

## OpenAI (and other non-ours) defaults in tau2 (phase 0 audit)

Every default model name in `tau2/config.py`, where else it is referenced, and what we do about it. The workflow has no OpenAI key, so a stray call fails loudly.

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

- Phase 1 spent (ledger): **$0.6671** of $2.50
- Total spent so far: $0.6671 of $27.00 hard cap
- API calls in the ledger: 247; by purpose: user_sim $0.0205, agent $0.6466
- Console reconciliation: ledger total for the UTC day beside the Console figure must agree within 5%. Ledger total for this dispatch (2026-10-23 UTC): $0.6671. Founder: enter the Console figure for the day here.

## What this cannot show

- Whether the agent says the right things: `EvaluationType.ENV` checks only the final database state and environment assertions.
- Anything about success rates or savings: this phase has 1 to 3 tasks per configuration. It only checks that the pipeline, the ledger and the settings work.
- Haiku numbers say nothing about Sonnet.
