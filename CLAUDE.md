# Agent cost layer: test 1

Read this whole file before writing code. It is the plan, the budget rules and the decision rules. The founder approved the budget and the stop points. Do not change a threshold after seeing data. Anything in this file marked **(verified)** was checked against the pinned code or the official docs; anything marked **(measure)** must be measured in phase 1, not assumed.

## The question

A company runs customer-service agents in production on Claude Sonnet 5.5. The model never changes. We add a layer around the model calls that makes each task cost less.

**The risk we are testing is that this is just a feature.** If one global setting (lower reasoning effort, say) gets most of the savings, a company builds it in an afternoon and nobody buys it. So the test asks three questions, in order:

1. **Cost anatomy.** In a well-configured, cached Sonnet 5.5 agent, where does the money go: cached input, cache writes, uncached input, or output (thinking included)?
2. **Best single setting.** What is the cheapest global setting that keeps task success? This is what an in-house engineer finds with a sweep.
3. **Beyond a setting.** Does a layer that learns from the company's own traces beat that best single setting?

### What this test can and cannot show

- It **can** count the cost of every run exactly (token counts are exact) and compare it paired by task, and it can **reject** a layer that saves little or loses a lot of quality. The cost *ratio* still has sampling noise from the 20–40 tasks; report its interval.
- It **cannot** confirm "same quality". With 20–40 test tasks, a layer that truly loses 15 points of success still passes a −10 point check about one time in four. Passing this test earns a larger quality test, never a launch.
- **Per-type learning is statistically thin here.** There are 7 task types in the train split and only 12–24 train tasks will be run, so most types have 1–4 tasks and only the 2–3 largest types can ever get their own configuration. Expect the per-type effort choice to collapse to `global*` most of the time.
- **So the honest expectation is this:** K1 can realistically pass only through fewer agent steps (S1 composites, S2 prefetch), which are structural and measurable, not through per-type effort. If K1 passes that way, the product is a gateway plus a small SDK. If only gateway levers help and they do not beat `global*`, the verdict is FEATURE ONLY. Say this in every report.

### Decision rules (fixed now, judged on point estimates; always report 80% intervals)

Cost = agent model cost per task summed over the held-out test tasks, plus the layer's own overhead (the classifier). The user simulator's cost is excluded from the ratios (it is identical across configurations) but is in the ledger. Success = the τ-bench database-state reward.

- **Margin:** `margin = max(10 points, the A/A flip rate in points)`. The A/A run (phase 3) measures how often `default` flips success against itself; the success checks cannot be stricter than the benchmark's own noise.
- **K2 (headline):** layer cost ≤ 70% of `default` cost, and layer success ≥ `default` success − margin.
- **K1 (not just a setting):** layer cost ≤ 85% of `global*` cost, and layer success ≥ `global*` success − margin.

| K1 | K2 | Verdict |
|---|---|---|
| holds | holds | **KEEP**: something beyond a setting exists |
| fails | holds | **FEATURE ONLY**: a global setting does it |
| any | fails | **KILL** |

## Verified facts: the benchmark

- **Benchmark:** `sierra-research/tau2-bench` (τ³-bench v1.0.0), MIT license. **Pin commit `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`.** The older `tau-bench` repo says its own tasks are outdated; do not use it. Set `TAU2_DATA_DIR` to the pinned checkout's `data/` (a git-installed package does not find it by itself).
- **Domain: retail only.** Official split: 74 train, 40 test (`data/tau2/domains/retail/split_tasks.json`). Never make your own split.
- **Context is small:** policy about 1,700 tokens; tool results a few hundred tokens (product details about 550 median, orders about 260, users about 125). Context-trimming levers have almost nothing to gain here. That is a measured scoping decision (see "Levers left out").
- **Reward:** `EvaluationType.ENV` (final database state plus environment assertions). It runs with no LLM call and no API key. Do NOT use the default evaluation: retail tasks also use an LLM judge (`NL_ASSERTION`) whose default model is an OpenAI model. We give up checking what the agent says; state this in every report.
- **OpenAI defaults exist in `config.py`** (user simulator `gpt-4.1`, judges `gpt-5.x`). Override every one. The workflow has no OpenAI key, so a stray call must fail loudly. In phase 0, grep for every use of those defaults and list them in the report.
- **Retries multiply spend:** `RunConfig.max_retries` defaults to 3 and re-runs the whole simulation on any exception. **Set `max_retries=0`.** Handle failures ourselves, and count every attempt in the ledger.
- **Gold actions are in the task data** (`evaluation_criteria.actions`), and `Results.tasks` embeds the full tasks in every saved result. **The agent and the optimizer must never see them.** The agent factory receives `task=`: ignore it. The runner (phase 0 item 3) owns one sanitizer that strips `tasks`, `evaluation_criteria` and anything derived from them, keeping messages, tool calls, tool results, token usage and the final reward. The runner also writes `tool_kinds.json` (tool name to READ, WRITE or GENERIC, taken from the tau2 tool decorators), because the optimizer needs it and **must not import `tau2` at all**. Test: sanitized files contain no `evaluation_criteria`, and the optimizer package has no `tau2` import.
- **The conversation starts with an assistant greeting** (`"Hi! How can I help you today?"`, `orchestrator.py:48`). We could not confirm from the docs whether the Messages API accepts a first assistant message, so do not depend on it: always put the greeting in the system prompt ("You already greeted the user with: ..."), identically for every configuration, and start the messages with the user's first message.
- **Mixed text and tool calls:** `enforce_communication_protocol` defaults to False, so the orchestrator does not reject them. Rule for every configuration: send only the tool calls to the orchestrator (the text would be ignored anyway) and keep the full raw content for echoing to the API. Count how many times text was dropped, per configuration.
- **Several `tool_use` blocks** arrive as a `MultiToolMessage`. Reply with one user turn holding all the `tool_result` blocks, in order.
- **Handle explicitly:** an empty or thinking-only reply, and `stop_reason == "max_tokens"` (thinking counts toward `max_tokens`, so a tool call can be truncated). Treat both as an agent error that ends the run as a failure, and count them per configuration. An uncaught exception triggers nothing else, because retries are off.
- **Termination:** a run that ends for any reason other than agent stop or user stop scores reward 0. Use `max_steps=100`, identical for every configuration, and report how many runs hit it, per configuration.
- **tau2's own agent is not used.** It goes through LiteLLM with `temperature=0.0` (a 400 on Sonnet 5.5), sets no `cache_control`, and may drop thinking blocks. Write our own agent on the official `anthropic` SDK and register it with `registry.register_agent_factory`. Set `cost` and `usage` on every `AssistantMessage`.
- **Replay:** evaluation replays the recorded trajectory on a fresh environment and skips non-mutating tools. The `Tool` objects given to the agent are bound to the live environment, so read-only tool calls made outside the trajectory are safe. Write tools must always go through the recorded trajectory.
- **User simulator:** `claude-haiku-5-5`, the same in every configuration, through LiteLLM with `llm_args_user={"thinking": {"type": "disabled"}, "max_tokens": 1024}` and no `temperature`. Haiku 5.5 has adaptive thinking on by default and accepts `disabled` at effort `high` or below. To capture its cost, patch **`tau2.user.user_simulator.generate`** (the name is imported there; patching `llm_utils.generate` does nothing). If the smoke test shows it misbehaving, replace LiteLLM for the user with a small class on the Anthropic SDK.

## Verified facts: the Anthropic API (Sonnet 5.5)

Source: platform.claude.com docs (thinking, thinking-tool-workflows, effort, prompt-caching).

- **Prices per million tokens (use this table, not multipliers):** Sonnet 5.5 input $2, output $10, 5-minute cache write $2.50, cache read $0.10. Haiku 5.5 (prompts ≤ 100K) input $0.10, output $0.50, cache write $0.125, cache read $0.01. Thinking tokens are billed as output. Check the raw `usage` for a thinking-token field (the docs mention `output_tokens_details.thinking_tokens` in one place and not in another): use it if present. Otherwise report `output_tokens` and estimate visible output (text plus tool-call JSON) separately; the difference is the thinking estimate, labelled approximate.
- **Thinking is adaptive and on by default.** Omitting `thinking` means adaptive. In a tool loop, echo the assistant content array back unmodified, thinking blocks included, or the API returns a 400. One thinking configuration per assistant turn.
- **Sonnet 5.5 rejects `thinking: {"type": "disabled"}` with a 400 at every effort level.** The lowest setting is **`thinking: {"type": "between_tools"}`**: it removes up-front thinking, the model still writes progress updates between tool calls, it is accepted at effort `high` or below, and it takes no other thinking fields. Thinking blocks still come back and must still be echoed.
- **Effort:** `output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}`, no beta header, default `high` on Sonnet 5.5. Lower effort tends to combine operations into fewer tool calls and skip preamble.
- **Forced tool use returns a 400** on Sonnet 5.5. Never set `tool_choice` to a tool.
- **Caching:** one top-level `cache_control={"type":"ephemeral"}` is automatic caching. Order is tools, system, messages. Minimum 512 tokens, TTL 5 minutes. Usage reports `input_tokens` (uncached, after the last breakpoint), `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`. Store the raw `usage` JSON of every call.
- **Changing `output_config.effort` or the thinking configuration invalidates the cached message prefix.** So effort and thinking mode are chosen once per task and never change inside a task.
- **Any change to earlier messages invalidates the cache from that point.** A transformation of tool results must be deterministic and applied when the result is first inserted, never retroactively.
- **`max_tokens`: use 16000 for every call**, as the effort docs advise for `high`. It is a hard limit on thinking plus text.

## Cost model (measure)

Retail contexts are small, so with caching I expect most of the bill to be output tokens (thinking, text, tool-call arguments), around $0.08–0.20 per Sonnet run. That is a guess. Everything below is allocated from the number measured in phase 1.

## Configurations

All configurations use the same agent class, the same tau2 system prompt, `max_tokens=16000`, the same tools, and **automatic caching on**. A baseline without caching would credit us with savings any customer gets by flipping a flag.

| Name | What differs |
|---|---|
| `default` | nothing: default effort, default adaptive thinking |
| `E:medium`, `E:low` | `output_config.effort` set |
| `T:bt` | `thinking={"type":"between_tools"}`, default effort |
| `global*` | the winner of the sweep, chosen on train tasks only |
| `layer` | the learned policy book (phase 3) |

## Levers

**Gateway-compatible** (change only the request, so a proxy can do them):

- **G1 effort per task.** Constant inside a task.
- **G2 thinking mode per task** (`between_tools` or default). Constant inside a task.
- **G3 tool-result trimming.** Only if phase 2 shows tool results matter. Deterministic, applied at first insertion.

**SDK-side** (execute tools, so they need a small library in the customer's own runtime; a gateway cannot execute the customer's tools):

- **S1 composite read-only tools.** Learned from traces by value provenance: a chain of read-only calls where each later call's arguments come from the user message or the previous result becomes one tool. Write tools always go through the model.
- **S2 prefetch.** Run a very predictable first read-only call before the model asks and put the result in context.

Every report splits savings into gateway-only and needs-SDK. That decides what the product is.

### Levers left out, and why

- **History compaction, policy slimming, tool-subset pruning:** retail histories are short, re-reading cached context is cheap, and compaction invalidates the cache. Expected net negative here. They matter for long-context agents: test them in a second test on `banking_knowledge` (97 tasks, 698 documents), only if test 1 passes.
- **Cache ordering, cache duration, batch API, duplicate detection, budget caps:** the baseline already caches and the benchmark has no traffic. These are the easy savings a customer can switch on alone; we claim no credit for them.
- **Semantic response caching and routing to a cheaper model:** they change the answer or the model. Out of scope.

## Budget: hard rules

- **Total cap $27.** The organization limit in the Console is $35 and about $4.27 was already spent this month, so the cap must trip first. Phase caps: phase 1 $2.50, phase 2 $9, phase 3 $13, reserve $2.50.
- **Cumulative ledger across dispatches, on its own branch.** Ledgers live on a dedicated branch `spend` (never on a PR branch, because phase 2 is dispatched while the phase 1 PR is still unmerged). At start, a dispatch commits `spend/start-<run id>.json` recording its phase cap as reserved. A paid workflow ends with an `if: always()` step that commits `spend/ledger-<run id>.jsonl` and uploads it as an Actions artifact. At start, the runner reads the `spend` branch and refuses to start if prior spend plus the planned phase cap would pass $27. **A start file with no matching ledger counts as spending its whole reserved cap** (the runner died before it could commit). The workflow sets `timeout-minutes` and `concurrency: {group: paid-runs, cancel-in-progress: false}` so two dispatches never overlap.
- **One ledger function.** Every API call goes through it. It reserves the worst-case cost of a call before sending (so three concurrent runs cannot overshoot), calls the API, then records model, the four token counts, raw `usage`, cost from the price table, configuration, task id and attempt number. Calls with no response (429, timeout) are recorded with cost 0.
- **Infrastructure errors are not task failures.** A 429, 5xx or timeout that survives the SDK's own retries, or a network error, must be caught per run (the custom loop does not catch it for you), ledgered, and marked as an infrastructure error. Such a run is excluded from the success statistics, rerun once in the same block at a counted cost, and if the rerun also fails the task is dropped from the paired analysis for all configurations and listed in the report. Agent errors (empty reply, `max_tokens`, malformed tool call) are different: they are failures, scored 0.
- **Per-run abort is a safety rail, not a measurement.** Abort a run only when its actual spend passes `max($1.00, 5 × c(default))` (after phase 1; $1.00 before). Count the aborted run's cost, mark it failed, and report censored runs per configuration. Never use the reservation to decide an abort.
- **Allocate against 80% of each phase cap.** Reuse phase 1 runs in the phase 2 sweep when the task and configuration match.
- **Before every paid run:** a dry run prints the estimated cost from the measured per-run cost and the number of runs.
- **Reconciliation:** each phase report prints the ledger total beside the Console figure for that UTC day, so the founder can compare in ten seconds. A difference above 5% stops the work.

## Phases and stop points

Paid runs happen only in GitHub Actions through `workflow_dispatch` with the secret `ANTHROPIC_API_KEY`. The workflow needs `permissions: contents: write` to commit the ledger. Never print or commit the key. No `pull_request` or `pull_request_target` trigger on a workflow that can read the secret. If you cannot dispatch the workflow yourself, ask the founder to press Run workflow.

### Phase 0: build (free)

Build in this order. Stop after item 4 and run phase 1.

1. **Ledger and caps** as above.
2. **`AnthropicAgent`:** own agent on the `anthropic` SDK with a policy object holding effort and thinking mode; all the protocol rules above.
3. **Runner.** Pinned tau2, explicit task objects in the shuffled order (`filter_tasks` returns dataset order; do not rely on it), a custom loop over `run_single_task` so that the configurations of one task run together and a cap hit leaves complete pairs, `EvaluationType.ENV`, `max_steps=100`, `max_retries=0`, concurrency 3. Save one JSON summary per run (cost, tokens by type, LLM calls, steps, reward, termination reason, dropped-text count, errors); full traces only as Actions artifacts.
4. **Tests with a fake Anthropic client (no network):** ledger arithmetic; the cap trips before the limit, also with concurrent calls; raw content is echoed unchanged; text is dropped when tool calls exist; multiple tool calls produce one user turn; empty, thinking-only and `max_tokens` replies end the run as failures; the sanitizer removes gold and nothing else it should keep.
5. **After the phase 2 go:** analysis scripts, the optimizer package (no `tau2` import, tests for that) and, only if phase 2 says they matter, the layer (G3, S1, S2, classifier, policy book).

### Phase 1: smoke test (cap $2.50)

1. **Haiku 5.5, 3 retail train tasks, `default` (about $0.30).** Check: no API error; cache reads above zero from the second call; no 400 from thinking blocks; no call to any OpenAI model; ledger total matches the Console within 5%.
2. **Sonnet 5.5 (about $1.50), in this order:** first the first train task (seed-42 order) on all four of `default`, `E:low`, `E:medium`, `T:bt`; then tasks 2 and 3 on `default` and `E:low`, only while the phase cap allows. Measure: cost per run for each configuration; LLM calls per run; the share of cost in cache reads, cache writes, uncached input and output; estimated thinking tokens per call; whether `E:low` lowers output tokens or only breaks the cache; whether `T:bt` and `E:medium` are accepted at all.
3. Write `reports/phase-1.md` with those numbers and the **allocation table for both phase 2 and phase 3** (see below), open a PR, and **stop. Wait for the founder's go.**

If any check fails, stop and report. Do not work around a failure by changing the benchmark, the models or the thresholds.

**Allocation rule (compute it, do not guess):** let `c(x)` be the measured mean cost per run of configuration `x`, user simulator included, and let the budget be 80% of the cap.
- Phase 2: `N_train = min(74, floor(0.8 × 9.00 / (c(default)+c(E:medium)+c(E:low)+c(T:bt))))`, at least 12.
- Phase 3: `N_test = min(40, floor((0.8 × 13.00 − 10 × c(default)) / (c(default)+c(global*)+c(layer))))`, at least 20. The `10 × c(default)` term is the A/A run below. **In phase 1 compute this too**, with the stand-ins `c(global*) = min(c(default), c(E:low), c(T:bt))` and `c(layer) = c(default)` plus 5% for the classifier. Recompute it at the start of phase 3 with the measured `c(global*)`; only `c(layer)` stays a stand-in.
- **Both minima are checked in phase 1, before any phase 2 spending.** If either cannot be met, stop and report it with three options (for example drop `E:medium` from the sweep, lower a minimum, or move budget between phases) and the resulting numbers for each. Do not choose; the founder does.
- Task order: shuffle the official split with seed 42 and take the first N.

### Phase 2: sweep on train tasks (cap $9)

Run `default`, `E:medium`, `E:low`, `T:bt` on the same `N_train` train tasks, task by task. Then, from those sanitized traces only:

- **Anatomy** per configuration (question 1).
- **Paired success** per configuration versus `default`.
- **Selection rule, the same everywhere:** a configuration is acceptable for a set of `n` tasks if its failures ≤ `default`'s failures + floor(0.10 × n). `global*` is the cheapest acceptable configuration over all train tasks. A per-type choice uses the same rule on that type's tasks, and a type with fewer than 3 train tasks gets `default`.
- **Task type** = the most common first write tool among the task's four traces (or `none`); ties go to the `default` trace's choice.
- **Out-of-sample headroom (this is what the gate uses):** for each train task, leave that task out, choose the cheapest acceptable configuration for its type from the other tasks, and take that configuration's measured cost on the held-out task. Compare the total with `global*`. Also compute a null: shuffle the type labels 200 times and recompute; the headroom counts only if it beats the 90th percentile of the null. The plain "oracle" (cheapest configuration per task) is reported only as an upper bound: it is biased low because it takes a minimum over noisy costs.
- **Mining (traces of `global*` only, because K1 compares against it):** (a) **dependent** read-only call chains (each later call's arguments come, by value, from the user message or the previous result), excluding calls the model already issued in parallel in one turn, because those need no composite; report the share of agent LLM calls such chains would remove, and the share of cost, estimated from the measured cost of those calls; (b) which tool-result fields the model uses later; (c) leave-one-out accuracy of a first-message type classifier (few-shot, Haiku). If that accuracy is below 70%, per-type choices are switched off for the layer.
- Write `reports/phase-2.md` and **stop for the founder's go.**

**Gate:** go to phase 3 only if (a) the out-of-sample headroom over `global*` is at least 10% of cost and beats the null, or (b) dependent chains would remove at least 15% of agent LLM calls **and** at least 10% of cost. Note that (a) is reachable only through the few types with at least 4 tasks, so at this sample size expect the decision to rest on (b). If neither holds the verdict is **FEATURE ONLY**: write it up and spend nothing more.

### Phase 3: held-out verdict (cap $13)

1. **Freeze the policy book** before any test run: commit it and record the hash in the report. The layer applies the per-type configuration from the sweep (when allowed by the 70% rule) plus any lever the gate approved. A cheap classifier (`claude-haiku-5-5`, few-shot from train traces, first user message only) predicts the type; low confidence means `default`. If a run leaves the learned path, every later request in that run goes back to the baseline request; count how often.
2. Run `default`, `global*`, `layer` on the test tasks, in blocks of 10, the three configurations of a task together.
3. **A/A noise run, mandatory:** after block 1 (the first 10 test tasks), run `default` a second time on those same 10 tasks. The flip rate is the share of those tasks whose `default` success differs between the two runs; also report the spread of cost. It sets the `margin` in the decision rules.
4. **Stop rule, checked after each block from the second block on, and only if `N_test` is above 20:**
   - **Cost stop:** stop if the **lower end** of the 80% interval of the `layer`/`global*` cost ratio is above 0.85 (even an optimistic reading cannot pass K1).
   - **Quality stop:** stop if `layer` success is at least 15 points below `default` and at least 3 more tasks went wrong than right for `layer`.
   - **Verdict after a stop:** a cost stop is **FEATURE ONLY** if the `layer`/`default` cost ratio so far is at most 0.70, otherwise **KILL**. A quality stop is **KILL**. Write the report with what was run.
5. Write `reports/phase-3.md`: the verdict from the table, both cost ratios with 80% intervals, success differences with the discordant tasks listed, the A/A result, savings split by lever and by gateway-only versus needs-SDK, fallback rate, censored and dropped-text counts per configuration, total spent against $27, and the ledger total beside the Console figure.

Opus comes later, only if K1 and K2 both hold on Sonnet, and only with the founder's go: it has a different default effort (`medium`), so it needs its own sweep.

## Statistics

- Pair by task. The cost ratio is the sum of layer cost over the sum of baseline cost; get its interval by bootstrapping tasks (10,000 resamples). The success difference is paired; list the tasks that flipped each way.
- The configurations of a task run in the same block, close in time, to limit drift.
- The user simulator is stochastic and cannot be seeded. Do not claim determinism.

## Reports

One file per phase in `reports/`. Always include: pinned commit, models, spent versus cap for the phase and in total, the numbers above, and a section "What this cannot show". Small JSON or Markdown only; no trace dumps in git.

## Conventions

Python 3.12+, `uv`, typed, `ruff` and `pytest`. Everything in English: code, comments, reports, commit messages. Small, focused commits. No secrets and no large files in git.
