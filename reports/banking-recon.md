# banking_knowledge reconnaissance (free work, no API call)

Pinned `sierra-research/tau2-bench` at `cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2`. Everything below was read from that code and data, or computed locally with the repository's own functions (`python -m costlayer.banking` regenerates the variant table). No paid call was made, no threshold is set here, and `CLAUDE.md` is untouched.

## Summary

- The runner now takes `domain` (`retail` or `banking_knowledge`) and, for banking, an explicit `retrieval_variant`. Retail code paths, file names and summary shape are unchanged (all 63 earlier tests still pass).
- **Trap:** if a banking run does not name a variant, tau2 sets `retrieval_config = "alltools"` itself, and `alltools` builds an OpenAI embeddings index of all 698 documents. The runner refuses a missing variant, any variant that needs a key, and the oracle `golden_retrieval`.
- 7 of the 20 variants need no key and no gold: `no_knowledge`, `full_kb`, `bm25`, `bm25_grep`, `grep_only`, `terminal_use`, `terminal_use_write`. Of these, `terminal_use*` needs four extra binaries that are not installed here, so I could not run them. `golden_retrieval` needs no key but puts the task's gold documents in the prompt, so it is an oracle, not a usable condition.
- **Big difference from retail:** one `KB_search` result is about 4,400 to 6,000 tokens and one `grep` result 5,600 to 8,700 tokens (measured locally, tokens estimated), against a few hundred tokens for a retail tool result. Tool results, not the fixed prompt, will dominate the context here.
- **The 9 `ACTION` tasks are a free pass under `EvaluationType.ENV`:** an agent that does nothing scores 1.0 on all 9 and 0.0 on every other task. They must be excluded or scored another way (section 5).
- A deterministic seed-42 order and split exist; no sizes are chosen. The sanitizer covers the banking fields and now has tests.

## 1. Runner: `domain` parameter

- `Runner(..., domain="retail" | "banking_knowledge", retrieval_variant=None)`, `make_config(model, config, domain, retrieval_variant)`, `tool_kinds(domain, retrieval_variant)`, `check_data_dir(domain)`, `tasks_by_id(domain, ids)` in `costlayer/runner.py`. Defaults reproduce retail exactly; banking summaries carry two extra keys (`domain`, `retrieval_variant`), retail summaries do not.
- Guard `check_domain`: banking requires a variant that `costlayer.banking.assert_usable` accepts (no external key, no gold in the prompt); retail refuses a variant.
- The dependency is now `tau2[knowledge]` (adds `rank-bm25` and the `openai` package; tau2 imports `openai` for any retrieval pipeline, even BM25, but that import needs no key).
- Banking specifics seen in the code, not exercised live: the agent's environment is built with a `read_log_allowlist` derived from the gold actions (it only changes which read calls are logged for the database hash; our agent never sees it); evaluation rebuilds the environment twice per task (replay and gold), so for `bm25*` each task indexes the 698 documents about three times (about 0.7 s each here); the user simulator has user tools (`give_discoverable_user_tool` and per-task ones), so it makes tool calls through LiteLLM with `thinking: disabled`, which is untested live.
- Also relevant for later mining: `call_discoverable_agent_tool` is a WRITE tool in `tool_kinds.json` even when the tool it calls is read-only, so a chain miner that goes by kind will never chain through it.

## 2. Retrieval variants

All 20 names in `RETRIEVAL_VARIANTS` (alias `AllTools` = `alltools`). Every variant gives the agent the same 14 base tools:

`call_discoverable_agent_tool` (WRITE), `change_user_email` (WRITE), `log_verification` (WRITE); `get_credit_card_accounts_by_user`, `get_credit_card_transactions_by_user`, `get_current_time`, `get_referrals_by_user`, `get_user_information_by_email`, `get_user_information_by_id`, `get_user_information_by_name`, `list_discoverable_agent_tools` (READ); `give_discoverable_user_tool`, `transfer_to_human_agents`, `unlock_discoverable_agent_tool` (GENERIC).

The table adds what each variant gives on top, whether it needs a key or binaries, and an estimated size (about 3.5 characters per token). The same estimator gave 5,408 for the retail system prompt plus tools, where the real first-call cache write was 6,711 to 6,755 tokens including the first user message, so **real counts are probably 10 to 25% above these estimates**; `count_tokens` would settle it but needs a key.

| variant | agent tools beyond the base set | external key | extra binaries | prompt tokens (est.) | tool-schema tokens (est.) | fixed prefix (est.) |
|---|---|---|---|---:|---:|---:|
| `no_knowledge` | none | none | - | 1,600 | 2,369 | 3,969 |
| `full_kb` | none | none | - | 247,616 | 2,369 | 249,985 |
| `golden_retrieval` | none | oracle: reads task gold | - | 6,396 (1,899 to 15,576) | 2,369 | 8,765 |
| `qwen_embeddings_grep` | KB_search, grep | OPENROUTER_API_KEY | - | 1,648 | 2,588 | 4,236 |
| `openai_embeddings_grep` | KB_search, grep | OPENAI_API_KEY | - | 1,649 | 2,588 | 4,237 |
| `qwen_embeddings_reranker_grep` | KB_search, grep | OPENAI_API_KEY, OPENROUTER_API_KEY | - | 1,648 | 2,588 | 4,236 |
| `openai_embeddings_reranker_grep` | KB_search, grep | OPENAI_API_KEY | - | 1,649 | 2,588 | 4,237 |
| `bm25_grep` | KB_search, grep | none | - | 1,642 | 2,588 | 4,230 |
| `bm25_reranker_grep` | KB_search, grep | OPENAI_API_KEY | - | 1,642 | 2,588 | 4,230 |
| `qwen_embeddings` | KB_search | OPENROUTER_API_KEY | - | 1,643 | 2,464 | 4,107 |
| `openai_embeddings` | KB_search | OPENAI_API_KEY | - | 1,646 | 2,464 | 4,110 |
| `qwen_embeddings_reranker` | KB_search | OPENAI_API_KEY, OPENROUTER_API_KEY | - | 1,643 | 2,464 | 4,107 |
| `openai_embeddings_reranker` | KB_search | OPENAI_API_KEY | - | 1,646 | 2,464 | 4,110 |
| `bm25` | KB_search | none | - | 1,639 | 2,464 | 4,103 |
| `bm25_reranker` | KB_search | OPENAI_API_KEY | - | 1,639 | 2,464 | 4,103 |
| `grep_only` | grep | none | - | 1,637 | 2,494 | 4,131 |
| `terminal_use` | shell | none | srt, rg, bwrap, socat | 2,020 | 2,524 | 4,544 |
| `terminal_use_write` | shell | none | srt, rg, bwrap, socat | 2,418 | 2,524 | 4,942 |
| `alltools` | KB_search_bm25, KB_search_dense, shell | OPENAI_API_KEY | srt, rg, bwrap, socat | 2,155 | 2,791 | 4,946 |
| `alltools-qwen` | KB_search_bm25, KB_search_dense, shell | OPENROUTER_API_KEY | srt, rg, bwrap, socat | 2,154 | 2,791 | 4,945 |

Per variant:

| variant | runs locally? | notes |
|---|---|---|
| `no_knowledge` | yes | no retrieval tool; the agent cannot read the policy documents. A floor, not a realistic agent |
| `full_kb` | yes | all 698 documents in the system prompt: 866,655 characters, about 248K to 310K tokens. Above Haiku's 100K-token price tier. On Sonnet 5.5 the first call of every run writes about 250K tokens to cache (about $0.62 at $2.50/M) and each later call reads it (about $0.025 at $0.10/M); the cache lasts 5 minutes |
| `bm25` | yes | `KB_search` over BM25 (`rank-bm25`), top 10 documents; built and queried here |
| `bm25_grep` | yes | `KB_search` plus `grep`; built and queried here |
| `grep_only` | yes | `grep` only; built and queried here |
| `terminal_use`, `terminal_use_write` | **not tested** | a `shell` tool in a sandbox that needs `srt` (`npm i -g @anthropic-ai/sandbox-runtime@0.0.23`), `rg`, `bwrap`, `socat`. Here only `rg` exists. On a GitHub runner this needs an install step, and `bwrap` needs unprivileged user namespaces, which Ubuntu 24.04 restricts by default; unverified. The write variant lets the agent modify files in its private copy of the documents |
| `golden_retrieval` | needs no key | the prompt contains the task's `required_documents` (about 1.9K to 15.6K tokens): an oracle that reads gold fields. Not usable as a condition |
| `*_embeddings*`, `alltools`, `alltools-qwen` | no | embeddings through OpenAI (`text-embedding-3-large`, `OPENAI_API_KEY`) or OpenRouter (`qwen3-embedding-8b`, `OPENROUTER_API_KEY`); `alltools*` also need the shell binaries |
| `*_reranker*` | no | a pointwise LLM reranker that builds an OpenAI client (`gpt-5.2`, `OPENAI_API_KEY`), also for the BM25 rerankers |

Measured tool-result sizes (estimated tokens, three queries each): `KB_search` 4,352 / 5,666 / 6,007; `grep` 5,583 / 8,043 / 8,720. Documents: 698, median 929 characters, mean 1,171, max 7,814, about 817K characters in total. Gold actions per task: median 8, max 37, so `max_steps=100` is enough, but long conversations with 5K-token results will grow the context fast.

## 3. Deterministic split

`costlayer/banking.py`:

- `banking_order(seed=42, exclude=())`: sorted task ids, then `random.Random(seed).shuffle`. Independent of file-system order and of `PYTHONHASHSEED` (tested in separate interpreters).
- `banking_split(n_train, seed=42, exclude=())`: train is the first `n_train` ids of that order, test is the rest. **`n_train` has no default** and is validated (1 to len-1).
- Because train is a prefix of one fixed order, choosing a different size later only moves tasks across the boundary; it never reshuffles. `exclude` is applied before the shuffle, so decide the exclusions once and keep them.
- Tests (`tests/test_banking.py`): permutation of all 97 ids; same seed same order, other seed different; hash-seed independence; disjoint and complete; prefix stability across sizes; no default size; exclusions removed (88 left when the 9 `ACTION` tasks are excluded); `tasks_by_id` keeps the order.

Task ids run from `task_001` to `task_102` (97 files, with gaps). First ten of the seed-42 order (all 97 tasks): `task_091`, `task_065`, `task_068`, `task_012`, `task_038`, `task_031`, `task_002`, `task_074`, `task_019`, `task_066`.

## 4. Sanitizer coverage

`sanitize_simulation` is a whitelist (task id, termination reason, reward, messages with tool calls, tool results, usage, cost), so nothing from a task object can pass through. I also widened the detector `find_banned` / `BANNED_KEYS`, which the runner asserts on every trace, with `required_documents`, `annotations`, `issues`, `reward_basis` and `user_scenario` (it already had `evaluation_criteria` and the derived check fields).

Findings and tests:

- In the 97 raw task files every task has `evaluation_criteria`, `required_documents` (1 to 30 document ids, median 10) and an `annotations` key (null everywhere in this data). tau2's `Task` model has `required_documents` but **drops `annotations`**, so the test checks the raw JSON files for that key and the parsed `Task` objects for the other two.
- A `SimulationRun` that embeds the whole task in `info` and in `reward_info.info` sanitizes to exactly four keys; none of the task's required-document ids, none of its gold `action_id`s, and none of the banned key names appear in the output.
- `required_documents` is also what `golden_retrieval` prints in the agent's prompt, one more reason that variant is out.

## 5. The 9 `ACTION` tasks under `EvaluationType.ENV`

Tasks by `reward_basis`: `DB` 87, `ACTION` 9, `DB`+`NL_ASSERTION` 1 (`task_102`). The nine: `task_004`, `008`, `012`, `014`, `032`, `033`, `034`, `035`, `083`. Their gold actions are mostly `transfer_to_human_agents` (the agent should refuse and transfer), `unlock_discoverable_agent_tool`, `call_discoverable_agent_tool`, `request_human_agent_transfer` and `log_verification`.

**How ENV scores them.** `EnvironmentEvaluator` multiplies only the `DB` and `ENV_ASSERTION` components, and only if they are in the task's `reward_basis`. For a task whose basis is just `ACTION`, nothing is multiplied and the reward stays at its initial 1.0. I ran the real evaluator on an empty trajectory (an agent that did nothing) for all 97 tasks: **exactly these 9 score 1.0; the other 88 score 0.0.** Under ENV the nine are therefore a constant pass for any run that ends with a user or agent stop (a run that ends any other way still scores 0). Covered by `test_env_evaluation_gives_a_free_pass_to_action_tasks_and_only_to_them`.

**Options** (the coordinator decides; I would take A):

- **A. Exclude the 9** (`banking_split(..., exclude=reward_basis_groups()["ACTION"])`, 88 tasks left). Under ENV they carry no information about quality, they add the same constant to every configuration's success, and running them costs money for nothing. This keeps the rule "success = database-state reward".
- **B. Score them with `EvaluationType.ACTION`.** `ActionEvaluator` needs no LLM and no key (checked: it imports none). It requires every gold action to be matched by an agent call, read-only ones included, so it treats the gold list as the one correct trajectory: stricter and noisier than the database check, and it needs a per-task evaluation type and a second kind of success in the same table.
- **C. Keep them under ENV.** Not recommended: success is inflated equally everywhere and a degenerate agent gets 9 free points.

`task_102` (`DB` + `NL_ASSERTION`): under ENV only the `DB` part is scored, so the language check is dropped, as the plan already says for retail ("we give up checking what the agent says"). It stays unless the coordinator says otherwise.

## 6. Not done and not verified

- No API call and no `count_tokens` (needs a key), so every token figure above is an estimate.
- `terminal_use` and `terminal_use_write` were not run (missing `srt`, `bwrap`, `socat`).
- The user simulator with user tools through LiteLLM and Haiku has not been tried live; the offline tests use a fake user.
- No variant, split size or exclusion is chosen, and no threshold is written.

## Decisions needed from the coordinator

1. Which local variant(s) to run (`bm25`, `bm25_grep` and `grep_only` ran here; `terminal_use*` need a verified CI install; `full_kb` has a very different cost shape).
2. Option A, B or C for the 9 `ACTION` tasks, and whether `task_102` stays.
3. `n_train` (and so the test size) for `banking_split`.
