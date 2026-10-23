# Agent cost layer: test 1

## What we are testing

A layer between a company's agents and the model. **The model never changes:** if the customer uses Sonnet, every call still goes to Sonnet. The layer only changes what surrounds the call: steps, how much the model thinks, context, cache use, tool results and output length.

**Hypothesis:** on production-style customer-service agents, the layer cuts cost per task by at least 30% with the same task success rate.

**Kill:** savings below 20%, or success clearly drops.

This repository is the experiment, not the product. Keep it small and honest.

## Benchmark

- **τ-bench** (sierra-research/tau-bench): customer-service agents with real tools and automatic success checking (reward).
- **Domains:** retail (115 test tasks) and airline (50).
- **Before using it:** check the license and pin the commit.
- **User simulator:** always `claude-haiku-5-5`. It is the customer, not the product, and it must be the same in every configuration.

## Models

- **Phase 1 (build):** agent `claude-haiku-5-5`. It is used only to check that the code and levers work. Draw no conclusions from Haiku results.
- **Phases 2–3 (results):** agent `claude-sonnet-5-5`. Opus comes later, only if Sonnet passes, and only with the founder's go.
- **Sonnet 5.5 notes:**
  - Setting `temperature`, `top_p` or `top_k` returns a 400 error, so results vary run to run. Always compare in pairs and over many tasks.
  - Forced tool use returns an error.
  - Text between tool calls comes back in `thinking` blocks.
- **Prices per million tokens:**

  | Model | Input | Output | Cache write (5 min) | Cache read |
  |---|---|---|---|---|
  | Sonnet 5.5 | $2 | $10 | $2.50 | $0.10 |
  | Haiku 5.5 (≤100K-token prompts) | $0.10 | $0.50 | $0.125 | $0.01 |

  Check them against platform.claude.com/docs before the first paid run.

## Baseline (must be honest)

The baseline is τ-bench's standard tool-calling agent:
- with basic prompt caching (cache breakpoints on system prompt and tools);
- with default thinking settings.

This is what a reasonable company would run. Savings must not come from things the customer could switch on alone in five minutes.

## Levers

### Group A: cannot change the answer

These are estimated from traces, with no quality test:

1. Cache-friendly prompt order (stable prefix).
2. Cache duration (5 min vs 1 h) and duplicate cache writes across parallel runs.
3. Duplicate task detection (reuse the result of identical work).
4. Batch API for non-urgent work.
5. Budget per task and loop stop.

τ-bench has no real traffic, duplicates or nightly jobs. Report 2–5 as estimates, separate from the verdict.

### Group B: can change the answer

These are tested on Sonnet:

6. **Fewer steps:** suggested path per task type, independent tool calls in one step, stop when the task is done.
7. **Thinking effort per task type.**
8. **Context per task type:** only the relevant policy sections and tools, and a shorter system prompt.
9. **Tool results trimmed** to the fields the model actually uses for that task type.
10. **History compaction** at the right moment for the task type.
11. **Output length limit** where long output adds nothing.

### How the layer works

- **Task type:** a small classifier (Haiku) labels each task from the first user message. Its cost counts as layer cost.
- **Learning:** for each task type the layer learns from training traces only: which steps, policy sections, tools, tool-result fields and effort level are needed.
- **Fallback:** if a run leaves the learned path (for example, the model needs a tool we removed), every later request in that run goes back to the full baseline request. Count how often this happens.

## Train / test split

- Split tasks by task id with a fixed seed: half train, half test.
- The layer learns only from phase 2 traces on train tasks.
- The verdict is measured only on test tasks it has never seen.

## Budget: hard rules

- **Total hard cap: $27** across all phases. The organization's console limit is $35, and the cap must trip before it does.
- **Ledger:** keep `ledger.jsonl` with model, token counts by type and cost for every API call, the user simulator and the classifier included. Before each call, abort if spent plus a conservative estimate for the call exceeds the cap.
- **Phase caps:**
  - Phase 1: $3
  - Phase 2: $7
  - Phase 3: $15
- **Before any paid run:** do a dry run that estimates its cost from a few tasks, and print the estimate.
- **Phase 3** runs in blocks of 10 test tasks, each run as a pair (baseline and layer).
- **Stop rule:** after the first 20 test tasks, stop phase 3 if savings are below 15% or the success rate is more than 15 points lower than baseline. Then write the report and wait.

## Phases

0. **Setup (free):** τ-bench, the cost ledger, per-step logging, the layer with all 11 levers, a dry-run mode and tests.
1. **Build (Haiku, ≤ $3):**
   - check that every lever fires and nothing breaks;
   - report the cache hit rate, the tokens removed per lever and any errors.
2. **Learn (Sonnet, ≤ $7):** baseline on train tasks. Traces become the layer's training data.
3. **Verdict (Sonnet, ≤ $15):** baseline vs layer on test tasks, in blocks, with the stop rule.

**Stop and wait for the founder's go before phase 2 and before phase 3.** Post the phase report and the planned cost.

## Metrics per run

- Cost in USD.
- Tokens: fresh input, cache write, cache read, output (thinking included).
- Number of steps.
- τ-bench reward (success).
- Whether fallback happened.

## Reports

- Write one report per phase to `reports/phase-N.md`:
  - cost per task, baseline vs layer, and savings %;
  - success rate for each, with a confidence interval;
  - attribution of savings per lever, from per-step token logs (no separate runs per lever);
  - fallback rate;
  - total spent so far against the cap.
- Write results as small JSON or Markdown files committed to the repo. No large trace dumps.

## Running

- **Paid runs** happen in GitHub Actions, `workflow_dispatch` only, with secret `ANTHROPIC_API_KEY`.
  - Never print the key, and never commit it.
  - No `pull_request` or `pull_request_target` triggers on workflows that use the secret.
- **The workflow** commits ledger summaries and reports back to the repo when it finishes. Large traces go to Actions artifacts.

## Conventions

- Python 3.12+, `uv`, typed, `ruff` and `pytest`.
- Everything in English: code, comments, reports, commit messages.
- Small, focused commits.
- No secrets and no large files in git.
