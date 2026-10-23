# Phase 2 report: sweep on train tasks

Benchmark: `sierra-research/tau2-bench` at `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2` (retail, official train split, seed-42 order, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`). Agent: `claude-sonnet-5-5`; user simulator and classifier: `claude-haiku-5-5`. Cost in the ratios is agent cost only (user simulator excluded).
Tasks planned: 21; complete in all four configurations: 21; runs reused from phase 1: 0; runs executed now: 84.

**Honest expectation (every report).** Per-type learning is statistically thin here (7 task types, 12 to 24 train tasks): most types have 1 to 4 tasks and the per-type effort choice usually collapses to `global*`. K1 can realistically pass only through fewer agent steps (S1 composites, S2 prefetch), which are structural and measurable. If only gateway levers help and they do not beat `global*`, the verdict is FEATURE ONLY.

## Phase 1 follow-up: the cache-write doubling and the thinking field

The doubling in the phase 1 report was a bug in my token bookkeeping, not an effect of `default` or `E:low`. A run's tokens were summed over all ledger rows with the same (config, task, attempt), ignoring the model; Haiku and Sonnet had both run `default` on the same three tasks, so each Sonnet `default` run carried the Haiku run's tokens (about 15.6K cache-write tokens each: 15.6K + 15.6K = 31K) and the Haiku files were overwritten. Dollar costs, LLM-call counts, `c(x)` and the allocation were not affected (they come from per-run spend). Fixed: the model is now part of the key and of the file names, the phase 1 files were rebuilt from the ledger, and the correction is at the top of `reports/phase-1.md`. Corrected: cache writes per run are about 15.6K (`default`) and 15.1K (`E:low`).

Why the API's thinking field does not explain the output difference: output per call was 192 (`default`) vs 149 (`E:low`) in phase 1, but the reported thinking tokens were 24 vs 7 per call, so the field covers about 17 of the 43 tokens per call. Most output is not reported as thinking; it is the visible reply and the tool-call JSON. The table below splits every call of the sweep into reported thinking, estimated visible output (text plus tool-call JSON) and the remainder, which is what the field does not explain.

## Cost anatomy (question 1)

| config | runs | agent $/run | calls/run | cache read | cache write | uncached input | output | output/call | reported thinking/call | visible est./call | remainder/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 21 | 0.0592 | 11.0 | 16.0% | 52.5% | 0.1% | 31.4% | 168 | 19 | 92 | 58 |
| E:medium | 21 | 0.0561 | 10.2 | 15.3% | 54.1% | 0.1% | 30.5% | 167 | 10 | 95 | 62 |
| E:low | 21 | 0.0532 | 10.2 | 16.1% | 54.1% | 0.1% | 29.7% | 154 | 8 | 88 | 58 |
| T:bt | 21 | 0.0574 | 10.4 | 15.4% | 53.5% | 0.1% | 31.0% | 171 | 1 | 102 | 68 |

Shares are of agent cost. Visible output is estimated at about 3.5 characters per token; the remainder is output that is neither reported thinking nor visible text or tool-call JSON (tokenizer error included).

## Best single setting (question 2)

| config | successes | failures | agent $ total | cost / default (80% interval) |
|---|---:|---:|---:|---|
| default | 20/21 | 1 | 1.2436 | 1.00 |
| E:medium | 16/21 | 5 | 1.1788 | 0.95 (0.93 to 0.97) |
| E:low | 18/21 | 3 | 1.1174 | 0.90 (0.87 to 0.92) |
| T:bt | 17/21 | 4 | 1.2062 | 0.97 (0.94 to 1.00) |

Acceptable (failures <= default + floor(0.10 x 21)): default, E:low. **`global*` = `E:low`.**
- `E:medium` vs `default`: lost ['109', '69', '82', '83'], won none
- `E:low` vs `default`: lost ['43', '82'], won none
- `T:bt` vs `default`: lost ['20', '82', '83'], won none

## Beyond a setting: per-type choice (question 3, gateway levers G1/G2)

| task type (first write tool) | tasks | per-type choice |
|---|---:|---|
| return_delivered_order_items | 6 | default |
| cancel_pending_order | 4 | E:low |
| none | 3 | E:low |
| exchange_delivered_order_items | 3 | E:low |
| modify_pending_order_address | 2 | default |
| modify_pending_order_items | 2 | default |
| modify_user_address | 1 | default |

Out-of-sample headroom over `global*` (leave one task out): **-8.5%** (chosen cost 1.2120 vs `global*` 1.1174; 2 failures). Null (type labels shuffled 200 times): mean -7.9%, 90th percentile -5.3%; does NOT beat the null. Oracle upper bound (biased low, not used by the gate): 1.6%.

## Mining on `E:low` traces (SDK levers S1/S2 and G3)

**(a) Dependent read-only chains.** Composite tools would remove 48 of 236 agent LLM calls (**20.3%**) and about **12.6%** of agent cost (measured cost of those calls, $0.1403 of $1.1174).
- 4x `find_user_id_by_name_zip -> get_user_details -> get_order_details+get_order_details+get_order_details+get_order_details`
- 2x `find_user_id_by_email -> get_user_details -> get_order_details+get_order_details`
- 2x `find_user_id_by_email -> get_user_details -> get_order_details+get_order_details+get_order_details+get_order_details+get_order_details`
- 2x `find_user_id_by_name_zip -> get_user_details -> get_order_details+get_order_details+get_order_details`
- 1x `find_user_id_by_name_zip -> get_user_details -> get_order_details+get_order_details+get_order_details -> get_product_details+get_product_details+get_product_details+get_product_details`

**(b) Tool-result fields used later** (values reappear in a later tool call or reply). Share of result characters in fields used in under 15% of cases:

| tool | observations | unused char share |
|---|---:|---:|
| get_order_details | 69 | 0.0% |
| get_product_details | 20 | 0.0% |
| get_user_details | 19 | 0.0% |

Retail tool results are small (a few hundred tokens), so G3 has little to gain.

**(c) First-message type classifier** (Haiku, few-shot, leave one out): accuracy **47.6%** on 21 tasks. Below 70%: per-type choices are switched off for the layer.

## Gate

- (a) headroom >= 10.0% and beats the null: no (-8.5%)
- (b) chains remove >= 15.0% of calls AND >= 10.0% of cost: YES (20.3% of calls, 12.6% of cost)

**Gate passed: phase 3 may go ahead, but only on the founder's go.**

Gateway-only savings (G1/G2): `global*` cost 0.90 of `default`, per-type headroom on top -8.5%. Needs-SDK savings (S1): up to 12.6% of agent cost.
## Budget

- Phase 2 spent (ledger, this dispatch): **$4.8632** of $9.00
- Total spent so far: $5.5302 of $27.00 hard cap
- API calls in the ledger: 1414; by purpose: user_sim $0.1122, agent $4.7461, classifier $0.0049
- Console reconciliation (must agree within 5%): Ledger total for this dispatch (2026-10-23 UTC): $4.8632. Founder: enter the Console figure for the day here.

## What this cannot show

- Whether the agent says the right things: `EvaluationType.ENV` checks only the final database state and environment assertions.
- That quality is unchanged: with this few tasks a configuration that truly loses several points of success can still pass the selection rule.
- A real cost saving in production: it is measured on one benchmark with a simulated user, on a handful of tasks, with the cost ratio's sampling noise shown above.
