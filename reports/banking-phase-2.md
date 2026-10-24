# Test 2, phase 2 (b2) report: banking_knowledge sweep and gate

Benchmark: `sierra-research/tau2-bench` at `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`, domain `banking_knowledge`, retrieval variant `bm25_grep`, tasks in `banking_order(seed=42, exclude=ACTION)` order, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`. Agent `claude-sonnet-5-5`; user simulator and classifier `claude-haiku-5-5`. Cost in the ratios is agent cost only (user simulator excluded).
Train tasks planned: 12 (`task_073, task_077, task_047, task_044, task_027, task_039, task_060, task_082, task_023, task_080, task_056, task_002`); complete in all four configurations: 12; runs reused from phase 1: 8; runs executed now: 40.

**What to expect (every report).** With 12 train tasks per-type learning is not expected to be reachable: most task types have one or two tasks and a type with fewer than 3 gets `default`. The gate is computed as written anyway.

## Gate (optimistic projection)

```
VERDICT: FEATURE ONLY
projected cost: $2.7562 (agent cost over the paired train tasks; basis: chains)
projected / default: 0.574   (K2 bar 0.70)
projected / global*: 0.990   (K1 bar 0.85)
```

Rule (TEST2.md, fixed): `projected = cost(global*) x (1 - chain_cost_share)`, or `cost(global*) x (1 - out_of_sample_headroom)` if the headroom beats the null (the lower is taken); classifier overhead counted as zero. Projected <= 0.70 of `default` and <= 0.85 of `global*`: go to phase 3. <= 0.70 of `default` but above 0.85 of `global*`: FEATURE ONLY. Above 0.70 of `default`: KILL.

Inputs: `global*` = `EC`; agent cost over the 12 paired tasks: `default` $4.8018, `global*` $2.7855; chain cost share 1.0%; out-of-sample headroom -12.0% (does not beat the null).

## Cost per configuration, with successes and LLM calls

| config | runs | successes | failures | agent $/run | cost / default (80% interval) | LLM calls/run | LLM calls/run, failed runs |
|---|---:|---:|---:|---:|---|---:|---:|
| default | 12 | 2 | 10 | 0.4002 | 1.00 | 21.2 | 21.3 |
| E:low | 12 | 1 | 11 | 0.2515 | 0.63 (0.54 to 0.71) | 18.0 | 19.2 |
| C:top3 | 12 | 2 | 10 | 0.2986 | 0.75 (0.66 to 0.83) | 25.6 | 24.0 |
| EC | 12 | 1 | 11 | 0.2321 | 0.58 (0.49 to 0.67) | 25.2 | 25.0 |

Intervals: bootstrap over tasks, 10,000 resamples, 80%. Paired tasks: 12.

**Savings that coincide with fewer calls AND fewer successes:** `E:low`. For these, part or all of the saving is runs ending early, not cheaper work.
- `E:low`: cheaper than `default`: yes; fewer calls per run: yes; fewer successes: yes
- `C:top3`: cheaper than `default`: yes; fewer calls per run: no; fewer successes: no
- `EC`: cheaper than `default`: yes; fewer calls per run: no; fewer successes: yes

## Cost anatomy (question 1)

| config | agent $/run | calls/run | cache read | cache write | uncached input | output | retrieval results/run | result tokens in context/run (est.) | raw tokens/run (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.4002 | 21.2 | 27.8% | 54.6% | 0.0% | 17.6% | 10.2 | 52,913 | 52,913 |
| E:low | 0.2515 | 18.0 | 26.4% | 54.2% | 0.0% | 19.4% | 6.0 | 30,971 | 30,971 |
| C:top3 | 0.2986 | 25.6 | 30.9% | 44.0% | 0.0% | 25.1% | 17.9 | 28,652 | 91,397 |
| EC | 0.2321 | 25.2 | 30.2% | 44.3% | 0.1% | 25.4% | 12.8 | 20,616 | 67,075 |

Shares are of agent cost. Result sizes come from the run summaries (characters recorded at the time, about 3.5 characters per token), never from trace text.

## Paired success and the selection rule (question 2)

- `E:low` vs `default`: lost ['task_056', 'task_073'], won ['task_002']
- `C:top3` vs `default`: lost ['task_056'], won ['task_080']
- `EC` vs `default`: lost ['task_056', 'task_073'], won ['task_002']

Acceptable (failures <= default + floor(0.10 x 12) = 10 + 1): default, E:low, C:top3, EC. **`global*` = `EC`.**

## Task types and per-type headroom (question 3, gateway)

| task type (first write tool) | tasks | per-type choice |
|---|---:|---|
| log_verification | 11 | C:top3 |
| none | 1 | default |

Out-of-sample headroom over `global*` (leave one task out): **-12.0%** (chosen cost 3.1187 vs `global*` 2.7855; 11 failures). Null (type labels shuffled 200 times): mean -8.4%, 90th percentile -0.4%; does NOT beat the null. Oracle upper bound (biased low, not used by the gate): -8.4%.

## Mining on `EC` traces (dependent read-only chains)

Composite tools would remove 11 of 302 agent LLM calls (**3.6%**) and about **1.0%** of agent cost (measured cost of those calls, $0.0292 of $2.7855). Calls issued in parallel in one turn are not chained; `call_discoverable_agent_tool` is a WRITE tool and never joins a chain; a call joins only if every argument value appears in the user's messages or in an earlier result of the chain, **as the agent saw it** (the cut text under `C:top3`/`EC`, so a value that only an omitted item contained earns no credit).
- 4x `KB_search -> get_user_information_by_name+get_current_time`
- 1x `KB_search -> get_user_information_by_email+get_current_time`
- 1x `KB_search -> grep`
- 1x `KB_search -> get_user_information_by_name -> get_current_time+get_credit_card_accounts_by_user`
- 1x `KB_search+KB_search -> KB_search`

Trace sources used for mining: reconstructed: 3, seen: 9. `seen` = saved by the runner after the cut fix; `uncut` = a configuration that never cuts; `reconstructed` = a phase 1 trace (raw tool results; phase 1 files are not rewritten) with the deterministic cut re-applied in memory and accepted only because the summed raw and in-context lengths equal the sizes recorded in the run summary; `unverified` runs are left out of the mining. **Limit:** for reused phase 1 runs the trace text is a reconstruction checked by sizes, not a record; costs and result sizes of reused runs always come from the run summary and the ledger.

**Classifier** (Haiku, few-shot, leave one out, first message only): accuracy **75.0%** on 12 tasks. Per-type choices stay on.

## Budget

- Phase b2 spent (ledger, this dispatch): **$12.0630** of $16.10
- Test 2 total so far: $14.5939 of $22.00
- API calls in the ledger: 1232; by purpose: user_sim $0.1736, agent $11.8875, classifier $0.0019
- Console reconciliation (must agree within 5%): Ledger total for this dispatch (2026-10-24 UTC): $12.0630. Founder: enter the Console figure for the day here.

## What this cannot show

- Whether the agent says the right things: ENV checks the database state only, and `task_102`'s language check is dropped.
- That quality is unchanged: with 30 to 40 test tasks a layer that truly loses 10 to 15 points can still pass. A KEEP earns a larger quality test, never a launch.
- Anything about other agents, other retrieval variants or other models: this is one benchmark, one variant, Sonnet 5.5, one simulated user.
