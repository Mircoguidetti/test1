# Failure anatomy (free analysis, no API call)

Pinned tau2-bench commit: `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`. Models: Sonnet 5.5 (agent), Haiku 5.5 (user simulator, excluded from every number below). Reward: database state (`EvaluationType.ENV`). Cost spent on this analysis: $0.00 (no API call, no ledger, `budget.json` untouched).

## Artifacts and sources

Run summaries come from `results/phase-2/runs` (test 1, retail train sweep: 21 tasks x 4 configurations) and `results/banking-phase-2/runs` (test 2, banking_knowledge: 12 tasks x 4 configurations, ACTION tasks excluded). Per-call costs, tool calls and tool results come from the sanitized trace artifacts:

| artifact | state |
|---|---|
| traces-phase-1-39998961720 (test 1 phase 1; host productionresultssa13.blob.core.windows.net, HTTP 403) | download blocked |
| traces-phase-2-40003551319 (test 1 phase 2; host productionresultssa0.blob.core.windows.net, HTTP 403) | download blocked |
| traces-banking-b1-40019775854 (test 2 b1; host productionresultssa13.blob.core.windows.net, HTTP 403) | download blocked |
| traces-banking-b2-40076249021 (test 2 b2; host productionresultssa10.blob.core.windows.net, HTTP 403) | download blocked |

Runs with a usable sanitized trace: Test 1: retail: 0 of 84; Test 2: banking_knowledge (bm25_grep): 0 of 48.

## Definitions and bars (fixed before looking at any data)

Definitions (fixed before looking at any data; the report states them):

Q1 cost of failed runs, per test and configuration.
  * A run is a success if its status is `ok` and its reward is 1; every other finished run
    (`ok` with reward 0, `agent_error`, `censored`) is a failure. `infra_error` runs are not
    finished runs and are left out (counted in the report).
  * Rule R(m): abort a run as soon as its cumulative AGENT cost (after an LLM call) exceeds
    T_m = m x median(total agent cost of the successful `default` runs of that test).
    The call that crosses T_m has been paid. For a run stopped at cumulative cost s with total
    cost C: saved = C - s. A failed run that is stopped saves money; a successful run that is
    stopped is "wrongly stopped".
  * net saving = (sum of `saved` over stopped FAILED runs) - (sum of the cost spent up to the stop,
    s, over WRONGLY STOPPED successful runs: money that buys nothing under the rule). It is shown
    as a share of the total agent cost of all finished runs of the test. The components are
    reported too. Rule R(m) is applied to the runs of all configurations of the test.
  * Without traces only the summaries exist, so per-call costs are unknown. Then `saved` is
    bounded above by C - T_m (the run is assumed to cross T_m exactly at the stop) and the sunk
    cost of a wrongly stopped success is bounded below by T_m: an UPPER bound on the net saving.
  * Signals (traces only): (a) the same tool name with identical arguments called again in the
    same run (position = the repeating call); (b) a tool call of the agent returning an error.
    Position = index of the agent LLM call that issued the call, counted from 1; reported as that
    index, as a fraction of the run's LLM calls and as the fraction of the run's cumulative cost
    spent through that call.

Q2 repetition across tasks, per test and configuration. Read-only = tool kind READ in
  tool_kinds.json (WRITE and GENERIC are never counted). A read-only call is a cross-task repeat
  if it is the FIRST occurrence of its (tool name, canonical arguments) inside its own task and the
  same key was called in an EARLIER task of the sweep order, same configuration. Repetition inside
  one task is excluded (that is caching). Share of calls = repeats / all read-only calls. Share of
  cost = sum of the cost of the repeating calls / total agent cost, where the cost of a call is
  the cost of the LLM call that issued it divided by the number of tool calls in that LLM call.
  Banking also: the share of KB_search and grep results (with at least one document) whose SET of
  returned document ids was already returned, for an earlier task, in the same configuration.
  (Under the cutting configurations the saved result holds the items the agent saw.)

Q3 early signal (informational): the first three tool-call NAMES of a run. Statistic = leave-one-
  task-out accuracy of a lookup table (signature -> majority label of the other tasks' runs, unseen
  signature -> overall majority) at predicting failed/successful over all runs of the test; null =
  200 shuffles (seed 42) of the labels. Rank = (1 + #null >= real) / (1 + 200).

Fixed bars: Q1 passes if for at least one m the net saving is >= 10% of total cost AND at most 1
successful run per 10 is wrongly stopped. Q2 passes if repeated read-only calls are >= 15% of
read-only calls AND >= 5% of agent cost. Both bars are checked per test on the pooled
configurations; the overall answer needs both tests.


# Results

## Test 1: retail, 21 train tasks, Sonnet 5.5

Finished runs: 84 (0 with a usable sanitized trace); left out as infrastructure errors: 0. Trace sources: {'none': 84}.

### Q1. Cost of failed runs

| config | runs | successes | failed runs | total agent cost | cost of failed | share |
|---|---|---|---|---|---|---|
| default | 21 | 20 | 1 | $1.24 | $0.04 | 3.5% |
| E:medium | 21 | 16 | 5 | $1.18 | $0.24 | 20.5% |
| E:low | 21 | 18 | 3 | $1.12 | $0.14 | 12.9% |
| T:bt | 21 | 17 | 4 | $1.21 | $0.22 | 18.0% |
| all | 84 | 71 | 13 | $4.75 | $0.65 | 13.6% |

Stop rule R(m), mode: UPPER BOUND from summaries (per-call costs unknown): each stopped run is assumed to cross the threshold exactly at the stop. Base = median agent cost of successful `default` runs = $0.06.

| m | threshold | failed stopped | saved on failed | successes wrongly stopped | sunk on those | net saving | net / total cost |
|---|---|---|---|---|---|---|---|
| 1.5 | $0.08 | 0 | $0.00 | 5 of 71 | $0.42 | $-0.42 | -8.8% |
| 2 | $0.11 | 0 | $0.00 | 0 of 71 | $0.00 | $0.00 | 0.0% |
| 3 | $0.17 | 0 | $0.00 | 0 of 71 | $0.00 | $0.00 | 0.0% |

- m=1.5: net >= 10.0%: False; wrongly stopped <= 1 per 10 successes: True
- m=2: net >= 10.0%: False; wrongly stopped <= 1 per 10 successes: True
- m=3: net >= 10.0%: False; wrongly stopped <= 1 per 10 successes: True
- **Q1 for this test: FAIL (holds even with the upper bound)**

Per configuration (same threshold):

| m | config | failed stopped | wrongly stopped | net / cost |
|---|---|---|---|---|
| 1.5 | default | 0 | 2 of 20 | -13.5% |
| 1.5 | E:medium | 0 | 1 of 16 | -7.1% |
| 1.5 | E:low | 0 | 1 of 18 | -7.5% |
| 1.5 | T:bt | 0 | 1 of 17 | -6.9% |
| 2 | default | 0 | 0 of 20 | 0.0% |
| 2 | E:medium | 0 | 0 of 16 | 0.0% |
| 2 | E:low | 0 | 0 of 18 | 0.0% |
| 2 | T:bt | 0 | 0 of 17 | 0.0% |
| 3 | default | 0 | 0 of 20 | 0.0% |
| 3 | E:medium | 0 | 0 of 16 | 0.0% |
| 3 | E:low | 0 | 0 of 18 | 0.0% |
| 3 | T:bt | 0 | 0 of 17 | 0.0% |

Signals (a) repeated identical call, (b) tool error: NOT COMPUTED: the sanitized traces of this test were not available.

### Q2. Repetition of read-only calls across tasks

NOT COMPUTED: the sanitized traces of this test were not available.

### Q3. Early signal from the first three tool-call names (informational)

NOT COMPUTED: the sanitized traces of this test were not available.

## Test 2: banking_knowledge (bm25_grep), 12 train tasks, Sonnet 5.5

Finished runs: 48 (0 with a usable sanitized trace); left out as infrastructure errors: 0. Trace sources: {'none': 48}.

### Q1. Cost of failed runs

| config | runs | successes | failed runs | total agent cost | cost of failed | share |
|---|---|---|---|---|---|---|
| default | 12 | 2 | 10 | $4.80 | $3.65 | 76.1% |
| E:low | 12 | 1 | 11 | $3.02 | $2.95 | 97.7% |
| C:top3 | 12 | 2 | 10 | $3.58 | $2.72 | 76.0% |
| EC | 12 | 1 | 11 | $2.79 | $2.57 | 92.4% |
| all | 48 | 6 | 42 | $14.19 | $11.90 | 83.8% |

Stop rule R(m), mode: UPPER BOUND from summaries (per-call costs unknown): each stopped run is assumed to cross the threshold exactly at the stop. Base = median agent cost of successful `default` runs = $0.57.

| m | threshold | failed stopped | saved on failed | successes wrongly stopped | sunk on those | net saving | net / total cost |
|---|---|---|---|---|---|---|---|
| 1.5 | $0.86 | 0 | $0.00 | 0 of 6 | $0.00 | $0.00 | 0.0% |
| 2 | $1.15 | 0 | $0.00 | 0 of 6 | $0.00 | $0.00 | 0.0% |
| 3 | $1.72 | 0 | $0.00 | 0 of 6 | $0.00 | $0.00 | 0.0% |

- m=1.5: net >= 10.0%: False; wrongly stopped <= 1 per 10 successes: True
- m=2: net >= 10.0%: False; wrongly stopped <= 1 per 10 successes: True
- m=3: net >= 10.0%: False; wrongly stopped <= 1 per 10 successes: True
- **Q1 for this test: FAIL (holds even with the upper bound)**

Per configuration (same threshold):

| m | config | failed stopped | wrongly stopped | net / cost |
|---|---|---|---|---|
| 1.5 | default | 0 | 0 of 2 | 0.0% |
| 1.5 | E:low | 0 | 0 of 1 | 0.0% |
| 1.5 | C:top3 | 0 | 0 of 2 | 0.0% |
| 1.5 | EC | 0 | 0 of 1 | 0.0% |
| 2 | default | 0 | 0 of 2 | 0.0% |
| 2 | E:low | 0 | 0 of 1 | 0.0% |
| 2 | C:top3 | 0 | 0 of 2 | 0.0% |
| 2 | EC | 0 | 0 of 1 | 0.0% |
| 3 | default | 0 | 0 of 2 | 0.0% |
| 3 | E:low | 0 | 0 of 1 | 0.0% |
| 3 | C:top3 | 0 | 0 of 2 | 0.0% |
| 3 | EC | 0 | 0 of 1 | 0.0% |

Signals (a) repeated identical call, (b) tool error: NOT COMPUTED: the sanitized traces of this test were not available.

### Q2. Repetition of read-only calls across tasks

NOT COMPUTED: the sanitized traces of this test were not available.

### Q3. Early signal from the first three tool-call names (informational)

NOT COMPUTED: the sanitized traces of this test were not available.

## Q1 and Q2 against the fixed bars

| test | Q1 | Q2 |
|---|---|---|
| Test 1: retail, 21 train tasks, Sonnet 5.5 | FAIL (holds even with the upper bound) | NOT COMPUTED (traces unavailable) |
| Test 2: banking_knowledge (bm25_grep), 12 train tasks, Sonnet 5.5 | FAIL (holds even with the upper bound) | NOT COMPUTED (traces unavailable) |

Q3 is informational and has no bar. No verdict is drawn here beyond the pass / fail above.

## What this cannot show

- Banking `default` solved only 2 of 12 tasks. "Failed" is the normal outcome there, so the failed-run share is high by construction and says little about a well-tuned production agent.
- Sample sizes are small: 21 retail and 12 banking tasks, 4 configurations each. A threshold based on the median successful `default` run rests on a handful of runs (see the counts above); one run moves it.
- Benchmark agents are cleaner than production agents: no stale caches, no flaky tools, no user who changes the request, one fixed policy. Repetition and failure patterns in production can be very different, in either direction.
- The four configurations are different policies of the same model, not independent customers. Pooling them makes the stop rule look more stable than it is.
- A success here is the database-state reward only; what the agent said to the user is not checked.
- Nothing here says a company would pay for any of this.

