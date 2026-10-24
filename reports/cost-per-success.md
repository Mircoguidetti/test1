# Cost per completed task (free recomputation of tests 1 and 2)

Pinned tau2-bench commit: `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`. Agent: Sonnet 5.5. Cost spent on this analysis: $0.00 (no Anthropic or OpenAI call, no paid workflow, `spend`/`spend2`/`budget.json` untouched). Only run summaries and the two existing `analysis.json` files are read; the blocked trace artifacts are not used.

## Answer

- Retail: `E:low` costs 0.899 of `default` per run but 0.998 per completed task (80% interval 0.90 to 1.12): the saving per run is gone once the 3 lost tasks are counted. `E:medium` (1.18) and `T:bt` (1.14) are dearer than `default` per completed task.
- Banking: `default` $2.40 per success on 2 successes; `C:top3` $1.79 (0.746); `E:low` $3.02 and `EC` $2.79 are DEARER per success than `default` although cheaper per run. The 80% intervals are very wide: the share of resamples with no success at all is 12% (`default`), 35% (`E:low`), 11% (`C:top3`), 35% (`EC`).
- `global*` by CPS: retail `E:low` (old: `E:low`), banking `C:top3` (old: `EC`).
- Banking gate recomputed with CPS: **KILL** (projected / default 0.738, bar 0.70); the old per-run gate gave **FEATURE ONLY** (0.574).
- Details, the verification and the list of what changes are below.

## Data and definitions (fixed before any number was looked at)

Data: `results/phase-2/runs` (retail: 21 tasks x 4 configurations, 84 runs) and `results/banking-phase-2/runs` (banking_knowledge: 12 tasks x 4 configurations, 48 runs; it already contains the 8 phase 1 runs the sweep reused). Phase 1 runs are not added again. The retail sweep reused none; the 8 retail phase 1 Sonnet runs are independent repeats and are used only as a noise remark in the weaknesses table.

- **Cost of a run** = agent (Sonnet) cost only. **The user simulator cost and any classifier cost are excluded**, as in the old reports (the simulator share is in the weaknesses table).
- **Success** = status `ok` and reward 1. Every other finished run (reward 0, `agent_error`, `censored`) is a failure and its cost stays in the numerator. `infra_error` runs would be excluded from numerator and denominator and listed: retail none, banking none; dropped tasks: retail none, banking none.
- **CPS** of a configuration = sum of the cost of all its finished runs / its number of successes. Zero successes: undefined. **CPS ratio** = CPS(config) / CPS(`default`) on the same tasks.
- **Intervals:** bootstrap over tasks, 10,000 resamples, seed 42, the same resampled tasks for all configurations, every statistic recomputed from the resampled sums; 80% interval = 10th to 90th percentile of the defined resamples; the number of undefined resamples is printed (never dropped silently).
- **`global*`** = lowest CPS among configurations with failures <= `default` failures + floor(0.10 x n) (ties: first in configuration order).

## Test 1: retail, 21 train tasks

| config | runs | successes | failures | agent $ total | $/run (80%) | success rate (80%) | cost per success, CPS (80%) | CPS ratio vs default (80%) | old cost ratio per run (80%) | calls/run | calls/run, failed runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 21 | 20 | 1 | $1.2436 | $0.0592 ($0.0546 to $0.0639) | 95.2% (90% to 100%) | $0.0622 ($0.0567 to $0.0680) | 1.00 | 1.00 | 11.0 | 7.0 |
| `E:medium` | 21 | 16 | 5 | $1.1788 | $0.0561 ($0.0520 to $0.0603) | 76.2% (62% to 86%) | $0.0737 ($0.0638 to $0.0870) | 1.185 (1.05 to 1.38) | 0.948 (0.92 to 0.97) | 10.2 | 8.8 |
| `E:low` | 21 | 18 | 3 | $1.1174 | $0.0532 ($0.0489 to $0.0575) | 85.7% (76% to 95%) | $0.0621 ($0.0546 to $0.0711) | 0.998 (0.90 to 1.12) | 0.899 (0.87 to 0.92) | 10.2 | 9.0 |
| `T:bt` | 21 | 17 | 4 | $1.2062 | $0.0574 ($0.0533 to $0.0616) | 81.0% (71% to 90%) | $0.0710 ($0.0615 to $0.0829) | 1.141 (1.02 to 1.30) | 0.970 (0.94 to 1.00) | 10.4 | 9.0 |

Saving per run vs saving per success (cheaper / fewer calls / fewer successes):

- `E:medium`: cheaper per run True; fewer calls per run True; fewer successes True; cheaper per success False
- `E:low`: cheaper per run True; fewer calls per run True; fewer successes True; cheaper per success True
- `T:bt`: cheaper per run True; fewer calls per run True; fewer successes True; cheaper per success False

A saving coincides with fewer calls AND fewer successes in: `E:medium`, `E:low`, `T:bt`.

Acceptable by the selection rule: `default`, `E:low`. `global*` by cost per run (old): `E:low`; by CPS (new): `E:low`. Same choice.

`E:low` beats `default` by CPS only by $0.00010 per success (0.16%), well inside the interval: the choice of `E:low` is a near-tie decided by a rounding-sized difference, not a finding.

## Test 2: banking_knowledge (bm25_grep), 12 train tasks

| config | runs | successes | failures | agent $ total | $/run (80%) | success rate (80%) | cost per success, CPS (80%) | CPS ratio vs default (80%) | old cost ratio per run (80%) | calls/run | calls/run, failed runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 12 | 2 | 10 | $4.8018 | $0.4002 ($0.3477 to $0.4569) | 16.7% (0% to 33%) | $2.4009 ($1.3211 to $4.7711 (1151 undef.)) | 1.00 | 1.00 | 21.2 | 21.3 |
| `E:low` | 12 | 1 | 11 | $3.0183 | $0.2515 ($0.1985 to $0.3080) | 8.3% (0% to 17%) | $3.0183 ($0.9945 to $3.4766 (3486 undef.)) | 1.257 (0.31 to 2.02 (4344 undef.)) | 0.629 (0.54 to 0.71) | 18.0 | 19.2 |
| `C:top3` | 12 | 2 | 10 | $3.5831 | $0.2986 ($0.2376 to $0.3643) | 16.7% (0% to 33%) | $1.7915 ($0.9767 to $3.6669 (1136 undef.)) | 0.746 (0.34 to 1.64 (1926 undef.)) | 0.746 (0.66 to 0.83) | 25.6 | 24.0 |
| `EC` | 12 | 1 | 11 | $2.7855 | $0.2321 ($0.1859 to $0.2835) | 8.3% (0% to 17%) | $2.7855 ($1.0204 to $3.2292 (3486 undef.)) | 1.160 (0.31 to 1.83 (4344 undef.)) | 0.580 (0.50 to 0.67) | 25.2 | 25.0 |

Saving per run vs saving per success (cheaper / fewer calls / fewer successes):

- `E:low`: cheaper per run True; fewer calls per run True; fewer successes True; cheaper per success False
- `C:top3`: cheaper per run True; fewer calls per run False; fewer successes False; cheaper per success True
- `EC`: cheaper per run True; fewer calls per run False; fewer successes True; cheaper per success False

A saving coincides with fewer calls AND fewer successes in: `E:low`.

Acceptable by the selection rule: `default`, `E:low`, `C:top3`, `EC`. `global*` by cost per run (old): `EC`; by CPS (new): `C:top3`. **Different choice.**

## Stacked best case (upper bound)

| test | global* (CPS) | CPS ratio | chain cost share | stacked CPS ratio | old claim |
|---|---|---|---|---|---|
| retail | `E:low` | 0.998 | 12.6% | 0.873 | 0.79 (= 0.90 x (1 - 0.126)) |
| banking | `C:top3` | 0.746 | 1.0% (mined on `EC`, assumed for `C:top3`) | 0.738 | 0.574 / default (gate projection on per-run cost) |

Assumption (stated, it makes the number an UPPER bound on the saving): successes do not change when chain composites are added, and the chain share of cost applies unchanged to the CPS. Retail: 0.998 x (1 - 0.126) = 0.873, further from the 0.70 bar than the old 0.79. The chain shares come from the existing mining reports (`analysis.json`); mining was not redone.

## Banking gate recomputed with CPS

| variant | default | global* | chain share | projected | projected / default (bar 0.70) | projected / global* (bar 0.85) | outcome |
|---|---|---|---|---|---|---|---|
| old: cost per run, `EC` | $4.8018 | $2.7855 | 1.0% | $2.7562 | 0.574 | 0.990 | FEATURE ONLY |
| CPS, re-selected `C:top3` | $2.4009 | $1.7915 | 1.0% | $1.7727 | 0.738 | 0.990 | KILL |
| CPS, old choice `EC` | $2.4009 | $2.7855 | 1.0% | $2.7562 | 1.148 | 0.990 | KILL |

The first row reproduces the old report (projected $2.7562, 0.574, 0.990, FEATURE ONLY) with the repository's own `gate.project`, which is a check of the inputs. The per-type headroom is -12.0% by cost (null 90th percentile -0.4%) and -111.6% by CPS (null 90th percentile -73.9%, 149 of 200 shuffles undefined): it does not beat the null in either metric, so the chain term decides. To pass the 0.70 bar with the re-selected `global*` the chain cost share would have to be at least 6.2%, against 1.0% measured (on `EC` traces). Note the data cannot separate 0.74 from 0.70: the CPS ratio of `C:top3` has an 80% interval of 0.34 to 1.64 (1926 undef.).

Retail (test 1 gate): (a) per-type headroom is a cost comparison; recomputed: -8.5% by cost, -2.8% by CPS (null 90th percentile 4.9%): does not beat the null either way. (b) chains (20.3% of calls and 12.6% of cost, mined on `E:low` traces) do not use the metric; `global*` is still `E:low`, so (b) is unchanged. The test 1 gate outcome (passed through (b)) does not change.

## Where the old metric drove a decision, and what changes

| decision | old (cost per run) | CPS | verdict |
|---|---|---|---|
| retail `global*` | `E:low` | `E:low` (near-tie) | unchanged |
| retail: does `E:low` save about 10%? | yes, 0.899 | no, 0.998 (0.90 to 1.12) | **changes** |
| retail stacked best case vs 0.70 | 0.79, fails | 0.873, fails | unchanged (fails) |
| retail test 1 gate (phase 2) | passed through chains (b) | unchanged: chains do not use the metric | unchanged |
| banking `global*` | `EC` | `C:top3` | **changes** |
| banking gate | FEATURE ONLY | KILL | **changes** |
| banking: does `EC`/`E:low` save money? | yes, 0.58 / 0.63 of default per run | no, 1.16 / 1.26 per success | **changes** |
| thesis outcome of TEST2.md (KEEP or closed as a standalone savings product) | closed (FEATURE ONLY) | closed (KILL) | unchanged: not KEEP |

Other places where per-run cost is the metric but nothing was decided on it yet: the K1/K2 definitions in `CLAUDE.md` and `TEST2.md` (cost ratio with a separate success margin of at least 10 points: a layer at 0.70 of the cost and 10 points fewer successes passes K2 by cost and sits at 0.70/0.9 = 0.78 per success), the per-run abort rail, the allocation formulas (these are budgeting, where cost per run is the right quantity), and the selection rule (failures only).

**Does any verdict change? Yes, two labels and no outcome.** (1) Banking gate: FEATURE ONLY becomes KILL under the fixed gate formula, because `EC` and `E:low` finish fewer tasks and the cheapest per-success configuration (`C:top3`, 0.746 of `default`) projects to 0.738 of `default` with 1.0% chains, above the 0.70 bar. (2) Retail: the 10% saving of `E:low` disappears per completed task. Not changed: retail `global*`, the retail gate result, and the fact that neither test reaches KEEP.

## Known weaknesses audit

| weakness | evidence (from the repo) | effect on the verdicts | fixable by recomputation (free) | needs a new paid run |
|---|---|---|---|---|
| Cost per run as the headline | retail `E:low` 0.90 per run but 0.998 per success; banking `EC` 0.58 per run but 1.16 per success | Changes the saving claims and the banking gate label (above) | Fixed here (free) | No |
| Banking `default` solves only 2 of 12 | `default` CPS rests on 2 successes; 11.5% of resamples have no `default` success | Ratios against `default` are very unstable (intervals above) | Cannot be fixed by recomputation | Yes: easier tasks, more tasks, or a stronger baseline configuration |
| Sample sizes (21 and 12 tasks) | ratio intervals above; selection rule allows floor(0.10 n) = 2 and 1 extra failures | A true loss of 10 points passes the rule easily; retail `E:low` lost 2 tasks and was `acceptable` | Intervals reported (free) | Yes, for any narrower claim |
| `task_102` scored on its DB part only | in a sweep: False (checked against `task_orders.json`) | None on these results; would matter for a test split | n/a | No |
| ACTION tasks excluded (9 of 97) | free pass under ENV; fixed once | None on these results; the result does not speak about those tasks | n/a | No |
| User simulator excluded from cost | retail 2.4%, banking 1.4% of agent cost (all runs) | Small; but the simulator reacts to the agent, so extra turns are not free in production | Share computed here (free); a CPS including it is a one-line change | No |
| Censored runs and the abort rail | statuses seen: retail ['ok'], banking ['ok']; hit max_steps: 0 / 0; max_tokens stops 0 / 0; empty replies 0 / 0; API errors 0 / 0 | No effect: none occurred | n/a | No |
| Reused phase 1 runs (banking) | 8 files on tasks task_047, task_073, task_077; identical cost and reward to the sweep copy: True; different dispatch, traces only reconstructed | Sensitivity below | Sensitivity computed (free) | No |
| Type classifier | retail 47.6% (below 70%: off); banking 75.0% on 12 tasks, but 11 of 12 tasks have one type, so always guessing it scores 91.7% | Per-type choice is not supported in either test; banking 'per-type' is one type | n/a (inputs fixed) | Yes, for a real per-type test |
| No A/A noise measured in tests 1 and 2 | retail phase 1 repeats: 1 flip(s) in 8 paired reruns (default/23), mean absolute cost difference $0.0041 | Success differences of 1 to 2 tasks cannot be told from run-to-run noise | No (needs repeats) | Yes: repeat `default` on the same tasks |
| Mixed text and tool calls: the text is dropped | dropped text blocks per configuration: retail: default=1, E:medium=4, E:low=4, T:bt=31; banking: default=2, E:low=6, C:top3=6, EC=12 | Configurations differ in how often the model writes text next to a tool call (`T:bt` 31 vs `default` 1 in retail); the text is not in the context the orchestrator sees, which could change behaviour | Counts shown (free) | Yes, to test the effect |
| Success is ENV only | what the agent says is not checked | A cheaper agent could complete the database state but answer worse | n/a | Yes (judge), or accept |
| Selection rule is lenient | `acceptable` = failures <= default + floor(0.10 n); retail `E:low` has 3 failures vs 1 | Lets a configuration give up tasks and still be 'global*'; CPS exposes it | Replaced here by the same rule + lowest CPS (free) | No |
| Gate is optimistic by construction | chain share applied to `global*` with successes unchanged, classifier overhead zero, chain share mined on `EC` not `C:top3` | Biased toward passing; a KILL under it is robust, a pass would not be | n/a | Mining on `C:top3` traces would need the blocked traces (free if the artifact is readable)  |
| Token and cost accounting | cache TTL 5 min, price table fixed; concurrency 3 may change cache hits | Not quantified here | n/a | No |
| One benchmark, one model, one simulated user | - | Nothing generalises beyond them | n/a | Yes |

Banking sensitivity, without the tasks that have a reused phase 1 run (task_047, task_073, task_077; 9 tasks left; A and B agree: True):

| config | successes | failures | agent $ total | CPS | CPS ratio |
|---|---|---|---|---|---|
| `default` | 1 | 8 | $3.4597 | $3.4597 | 1.000 |
| `E:low` | 1 | 8 | $2.3681 | $2.3681 | 0.684 |
| `C:top3` | 1 | 8 | $2.6661 | $2.6661 | 0.771 |
| `EC` | 1 | 8 | $2.2647 | $2.2647 | 0.655 |

On these 9 tasks the successes per configuration are [1]: every configuration has the same number of successes, so the CPS ranking is the cost-per-run ranking.
Dropping three tasks therefore reverses the picture of the main table: the banking result depends on which one or two tasks `default` happens to solve, which is the point of the 'tiny success counts' limit.

## Verification

1. **Two implementations.** A (`cps.py`): plain loops over the raw JSON files. B (`cps_check.py`): reads through `load_runs`, bootstraps with multiplicity counts, percentiles from `statistics.quantiles`, leave-one-out choice by `min` over keys. Only the random draws are shared (`cps_rng.py`). Every number in the tables (point estimates, intervals, undefined counts, `global*`, stacked ratio, headroom and nulls) is compared with a tolerance of 1e-9: retail differences 0, banking differences 0 (banking sensitivity: 0). Any difference stops the report.

2. **Cross-check with the numbers printed in the old reports** (compared at the printed precision):

   40 of 40 comparisons agree; mismatches: none.

   Your hand arithmetic (CPS = total / successes, rounded as you wrote it):

| test | config | hint | computed | agree |
|---|---|---|---|---|
| retail | `default` | 0.062 | $0.06218 | True |
| retail | `E:medium` | 0.074 | $0.07368 | True |
| retail | `E:low` | 0.062 | $0.06208 | True |
| retail | `T:bt` | 0.071 | $0.07095 | True |
| banking | `default` | 2.40 | $2.40090 | True |
| banking | `E:low` | 3.02 | $3.01828 | True |
| banking | `C:top3` | 1.79 | $1.79155 | True |
| banking | `EC` | 2.79 | $2.78547 | True |

3. **Independent reviewer.** A subagent saw only the definitions and the raw run-summary files (not the code, not these numbers) and recomputed the headline table. Reviewer and implementation A:

| test | config | runs | successes | total $ (rev. / mine) | CPS (rev. / mine) | CPS ratio (rev. / mine) | $/run (rev. / mine) | agree |
|---|---|---|---|---|---|---|---|---|
| retail | `default` | 21 / 21 | 20 / 20 | 1.2436 / 1.2436 | 0.0622 / 0.0622 | 1.0000 / 1.0000 | 0.0592 / 0.0592 | True |
| retail | `E:medium` | 21 / 21 | 16 / 16 | 1.1788 / 1.1788 | 0.0737 / 0.0737 | 1.1849 / 1.1849 | 0.0561 / 0.0561 | True |
| retail | `E:low` | 21 / 21 | 18 / 18 | 1.1174 / 1.1174 | 0.0621 / 0.0621 | 0.9984 / 0.9984 | 0.0532 / 0.0532 | True |
| retail | `T:bt` | 21 / 21 | 17 / 17 | 1.2062 / 1.2062 | 0.0710 / 0.0710 | 1.1411 / 1.1411 | 0.0574 / 0.0574 | True |
| banking | `default` | 12 / 12 | 2 / 2 | 4.8018 / 4.8018 | 2.4009 / 2.4009 | 1.0000 / 1.0000 | 0.4002 / 0.4002 | True |
| banking | `E:low` | 12 / 12 | 1 / 1 | 3.0183 / 3.0183 | 3.0183 / 3.0183 | 1.2571 / 1.2571 | 0.2515 / 0.2515 | True |
| banking | `C:top3` | 12 / 12 | 2 / 2 | 3.5831 / 3.5831 | 1.7915 / 1.7915 | 0.7462 / 0.7462 | 0.2986 / 0.2986 | True |
| banking | `EC` | 12 / 12 | 1 / 1 | 2.7855 / 2.7855 | 2.7855 / 2.7855 | 1.1602 / 1.1602 | 0.2321 / 0.2321 | True |

   Result: all 8 rows agree (runs, successes exactly; dollars and ratios to 4 decimals) and the reviewer picks the same `global*` in both tests; it found no `infra_error` run.

4. **Unit tests** (`tests/test_cps.py`): the definitions on constructed fixtures, zero-success handling, `infra_error` exclusion, bootstrap paired by task, a regression test that fails if cost is counted by another rule, and that A and B agree on a constructed data set; the optimizer package still has no `tau2` import.

## What this cannot show

- Banking's success counts are tiny (2, 1, 2, 1 of 12): every CPS and ratio there rests on one or two tasks and the intervals are wide or undefined.
- Sample sizes of 21 and 12 tasks: nothing here is a measurement of a population.
- Benchmark agents are cleaner than production agents (no stale caches, no flaky tools, one policy, a cooperative simulated user).
- One benchmark, one model (Sonnet 5.5), one simulated user (Haiku 5.5).
- The user simulator cost is excluded; ENV checks the database state only.
- Nothing here says a company would pay for any of this.

## What remains to try

- Repeat `default` and the cheapest configuration on the same tasks to measure run-to-run noise (A/A), before reading any success difference.
- A baseline that is not near the floor on banking (easier tasks, or a task subset where `default` succeeds about half the time), so that cost per success has a denominator.
- Mine chains on `C:top3` traces (free if the trace artifact can be read) instead of assuming the `EC` share.
- Replace the K1/K2 cost-plus-margin rule by a rule on cost per completed task for any future paid test.
- Include the user-simulator cost in a CPS variant to see whether longer conversations change the ranking.
