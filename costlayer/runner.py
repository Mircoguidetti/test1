"""Runner: one tau2 retail task under one configuration, with ledger, rails and a summary."""

from __future__ import annotations

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.environment.toolkit import get_tool_types
from tau2.evaluator.evaluator import EvaluationType
from tau2.registry import registry
from tau2.runner.batch import run_single_task

from costlayer import agent as agent_mod
from costlayer.agent import AGENT_NAME, CONFIGS, AgentRunFailure, RuntimeHandles
from costlayer.client import AnthropicClient
from costlayer.context import (
    ApiRequestError,
    InfraError,
    RunAborted,
    RunContext,
    set_context,
)
from costlayer.ledger import BudgetExceeded, Ledger
from costlayer.pricing import est_tokens
from costlayer.sanitize import find_banned, sanitize_simulation
from costlayer.usersim import USER_LLM, USER_LLM_ARGS

DOMAIN = "retail"
MAX_STEPS = 100
ORDER_SEED = 42
FIRST_RAIL_USD = 1.00


def check_data_dir() -> None:
    d = os.environ.get("TAU2_DATA_DIR")
    if not d or not (Path(d) / "tau2" / "domains" / "retail" / "tasks.json").exists():
        raise RuntimeError("Set TAU2_DATA_DIR to the pinned tau2-bench checkout's data/ folder")


def split_tasks(split: str) -> list[Task]:
    """Official split (74 train / 40 test), shuffled with seed 42 (never our own split)."""
    tasks = registry.get_tasks_loader(DOMAIN)(split)
    ids = [t.id for t in tasks]
    order = list(ids)
    random.Random(ORDER_SEED).shuffle(order)
    by_id = {t.id: t for t in tasks}
    return [by_id[i] for i in order]


def tool_kinds() -> dict[str, str]:
    env = registry.get_env_constructor(DOMAIN)()
    return {name: t.value.upper() for name, t in get_tool_types(env.tools).items()}


def make_config(model: str, config: str) -> TextRunConfig:
    assert model.startswith("claude-") and USER_LLM.split("/")[-1].startswith("claude-")
    return TextRunConfig(
        domain=DOMAIN,
        agent=AGENT_NAME,
        llm_agent=model,
        llm_args_agent=CONFIGS[config].llm_args(),
        llm_user=USER_LLM,
        llm_args_user=dict(USER_LLM_ARGS),
        max_steps=MAX_STEPS,
        max_retries=0,
        num_trials=1,
        max_concurrency=1,
        enforce_communication_protocol=False,
        auto_review=False,
    )


class Runner:
    def __init__(
        self, ledger: Ledger, client: AnthropicClient, out_dir: Path, trace_dir: Path | None = None
    ) -> None:
        self.ledger = ledger
        self.out_dir = Path(out_dir)
        self.trace_dir = Path(trace_dir) if trace_dir else None
        RuntimeHandles.client = client
        agent_mod.register()
        (self.out_dir / "runs").mkdir(parents=True, exist_ok=True)
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.rail_usd = FIRST_RAIL_USD

    # -- one attempt -----------------------------------------------------------------------

    def run_once(self, model: str, config: str, task: Task, attempt: int = 1) -> dict[str, Any]:
        ctx = RunContext(
            config=config, task_id=task.id, attempt=attempt, abort_above_usd=self.rail_usd
        )
        set_context(ctx)
        status, reason, reward, sim_info = "ok", None, 0.0, {}
        try:
            sim = run_single_task(
                make_config(model, config), task, evaluation_type=EvaluationType.ENV
            )
            clean = sanitize_simulation(sim)
            assert not find_banned(clean), find_banned(clean)
            reward = float(clean["reward"] or 0.0)
            reason = clean["termination_reason"]
            sim_info = {"n_messages": len(clean["messages"])}
            if self.trace_dir:
                (
                    self.trace_dir / f"{config.replace(':', '-')}__{task.id}__a{attempt}.json"
                ).write_text(json.dumps(clean))
        except BudgetExceeded:
            raise
        except InfraError as e:
            status, reason = "infra_error", str(e)[:200]
        except RunAborted as e:
            status, reason = "censored", str(e)
        except AgentRunFailure as e:
            status, reason = "agent_error", e.kind
        except ApiRequestError as e:
            status, reason = "api_error", str(e)[:200]
        except Exception as e:  # a bug or an unexpected tau2 failure: counted, never hidden
            status, reason = "unexpected_error", f"{type(e).__name__}: {e}"[:300]
        finally:
            set_context(None)
        summary = {
            "model": model,
            "config": config,
            "task": task.id,
            "attempt": attempt,
            "status": status,
            "termination": reason,
            "reward": reward if status == "ok" else 0.0,
            "llm_calls": ctx.llm_calls,
            "user_calls": ctx.user_calls,
            "cost_agent": round(ctx.spent_agent, 6),
            "cost_user": round(ctx.spent_user, 6),
            "dropped_text": ctx.dropped_text,
            "max_tokens_stops": ctx.max_tokens_stops,
            "empty_replies": ctx.empty_replies,
            "api_errors": ctx.api_errors,
            "hit_max_steps": reason == "max_steps",
            **sim_info,
            "thinking_tokens_reported": ctx.thinking_tokens_reported
            if ctx.thinking_reported_calls
            else None,
        }
        summary.update(self._tokens(model, config, task.id, attempt))
        name = f"{config.replace(':', '-')}__{task.id}__a{attempt}.json"
        (self.out_dir / "runs" / name).write_text(json.dumps(summary, indent=1) + "\n")
        return summary

    def _tokens(self, model: str, config: str, task: str, attempt: int) -> dict[str, Any]:
        t = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
        for line in self.ledger.path.read_text().splitlines():
            r = json.loads(line)
            if (r.get("config"), r.get("task"), r.get("attempt"), r["purpose"]) == (
                config,
                task,
                attempt,
                "agent",
            ):
                for k in t:
                    t[k] += r[k]
        return {"tokens": t}

    # -- a task under several configurations ---------------------------------------------

    def run_task(self, model: str, configs: list[str], task: Task) -> list[dict[str, Any]]:
        """All configurations of one task together; an infra error is rerun once."""
        out = []
        for c in configs:
            s = self.run_once(model, c, task, attempt=1)
            if s["status"] == "infra_error":
                s = self.run_once(model, c, task, attempt=2)
            out.append(s)
        return out

    def run_block(
        self, model: str, configs: list[str], tasks: list[Task], concurrency: int = 3
    ) -> list[dict[str, Any]]:
        """Tasks in order; a cap hit leaves earlier tasks complete."""
        results: list[dict[str, Any]] = []
        lock = threading.Lock()
        stop = threading.Event()

        def work(t: Task) -> None:
            if stop.is_set():
                return
            try:
                r = self.run_task(model, configs, t)
            except BudgetExceeded:
                stop.set()
                raise
            with lock:
                results.extend(r)

        errors: list[BudgetExceeded] = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for f in futs:
                try:
                    f.result()
                except BudgetExceeded as e:
                    errors.append(e)
        if errors:
            raise errors[0]
        return results


def estimate_prompt_tokens(text: str) -> int:
    return est_tokens(text)
