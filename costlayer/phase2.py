"""Phase 2: sweep on the first N_train train tasks (seed-42 order), reusing phase 1 runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task

from costlayer import allocation
from costlayer.agent import CONFIGS
from costlayer.ledger import BudgetExceeded
from costlayer.optimizer.data import COMPLETE
from costlayer.pricing import SONNET
from costlayer.runner import FIRST_RAIL_USD, Runner, run_name, split_tasks

CONFIG_NAMES = tuple(CONFIGS)


@dataclass
class Prior:
    """Earlier runs that may be reused: summaries plus the directories holding their traces."""

    runs_dir: Path
    trace_dirs: list[Path]


def measured_costs(runs_dir: Path, model: str = SONNET) -> dict[str, float]:
    """c(x): mean cost per run (agent plus user simulator) of each configuration."""
    by: dict[str, list[float]] = {}
    for f in Path(runs_dir).glob("*.json"):
        s = json.loads(f.read_text())
        if s.get("model") == model and s["status"] == "ok":
            by.setdefault(s["config"], []).append(s["cost_agent"] + s["cost_user"])
    return {c: sum(v) / len(v) for c, v in by.items()}


def rail_for(c_default: float) -> float:
    return max(FIRST_RAIL_USD, 5 * c_default)


def _trace_file(prior: Prior, s: dict[str, Any]) -> Path | None:
    for d in prior.trace_dirs:
        for name in (
            run_name(s["model"], s["config"], s["task"], s["attempt"]),
            f"{s['config'].replace(':', '-')}__{s['task']}__a{s['attempt']}.json",
        ):
            if (Path(d) / name).exists():
                return Path(d) / name
    return None


def reusable(prior: Prior, model: str, config: str, task: str) -> tuple[dict, Path] | None:
    """A finished run of the same model, configuration and task, with a trace whose per-call
    costs add up to the summary (guards against the file mix-up found in phase 1)."""
    for f in sorted(Path(prior.runs_dir).glob("*.json")):
        s = json.loads(f.read_text())
        if (s.get("model"), s["config"], s["task"]) != (model, config, task):
            continue
        if s["status"] not in COMPLETE:
            continue
        tf = _trace_file(prior, s)
        if tf is None:
            continue
        trace = json.loads(tf.read_text())
        traced = sum(m.get("cost") or 0.0 for m in trace["messages"] if m["role"] == "assistant")
        if abs(traced - s["cost_agent"]) > 0.02 * max(s["cost_agent"], 1e-9):
            continue
        return s, tf
    return None


def plan_sweep(
    tasks: list[Task],
    priors: list[Prior],
    out_dir: Path,
    trace_dir: Path,
    model: str = SONNET,
    configs: tuple[str, ...] = CONFIG_NAMES,
) -> tuple[list[tuple[Task, list[str]]], list[str]]:
    """Copy reusable runs into the phase 2 folders; return what is still to run."""
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    plan: list[tuple[Task, list[str]]] = []
    reused: list[str] = []
    for t in tasks:
        todo = []
        for c in configs:
            hit = next((h for p in priors if (h := reusable(p, model, c, t.id))), None)
            if hit is None:
                todo.append(c)
                continue
            s, tf = hit
            name = run_name(model, c, t.id, s["attempt"])
            if not (out_dir / "runs" / name).exists():
                (out_dir / "runs" / name).write_text(
                    json.dumps({**s, "reused_from": "earlier dispatch"}, indent=1) + "\n"
                )
                shutil.copy(tf, trace_dir / name)
            reused.append(f"{c}/{t.id}")
        if todo:
            plan.append((t, todo))
    return plan, reused


def n_train_from(runs_dir: Path) -> tuple[int, dict[str, Any]]:
    c = measured_costs(runs_dir)
    a = allocation.allocate(c)
    return a["n_train"], a


def estimate(plan: list[tuple[Task, list[str]]], c: dict[str, float]) -> float:
    return sum(c[cfg] for _, cfgs in plan for cfg in cfgs)


def run_phase2(
    runner: Runner,
    out_dir: Path,
    trace_dir: Path,
    priors: list[Prior],
    n_train: int,
    rail: float,
) -> dict[str, Any]:
    tasks = split_tasks("train")[:n_train]
    plan, reused = plan_sweep(tasks, priors, out_dir, trace_dir)
    runner.rail_usd = rail
    result: dict[str, Any] = {
        "n_train": n_train,
        "reused": reused,
        "to_run": sum(len(c) for _, c in plan),
        "stopped": None,
    }
    try:
        runner.run_plan(SONNET, plan, concurrency=3)
    except BudgetExceeded as e:
        result["stopped"] = f"budget: {e}"
    return result
