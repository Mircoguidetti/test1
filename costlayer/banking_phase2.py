"""Test 2, phase 2 (`b2`): fail-fast sweep on the first 12 train tasks (TEST2.md, Amendment 1).

The four configurations (default, E:low, C:top3, EC) on the first `N_TRAIN` ids of
`banking_order(seed=42, exclude=ACTION)`, task by task, the four runs of a task together,
concurrency 3. Phase 1 Sonnet runs are reused when model, configuration and task match, the run
finished and its per-call costs add up to its summary. A task is only started if the cap left
covers its missing runs; an infrastructure error is rerun once; each run is aborted by the rail
`max($1.00, 5 x c(default))` with c(default) measured in phase 1.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task

from costlayer import banking_phase1 as bank1
from costlayer import phase2 as p2
from costlayer.agent import BANKING_CONFIGS
from costlayer.ledger import BudgetExceeded, Ledger
from costlayer.pricing import SONNET
from costlayer.runner import BANKING_DOMAIN, Runner, tasks_by_id

N_TRAIN = 12  # Amendment 1: the minimum of 20 is waived for phase 2 only
CONFIG_NAMES = tuple(BANKING_CONFIGS)  # default, E:low, C:top3, EC
CLASSIFIER_USD = 0.05  # 12 few-shot Haiku calls, generous
COVER_MARGIN = 1.5  # cover per run = max(highest observed, 1.5 x mean)


def train_ids() -> list[str]:
    return bank1.phase1_order()[:N_TRAIN]


@dataclass(frozen=True)
class CostBasis:
    """Per-run cost (agent + user simulator) of each configuration, measured in phase 1."""

    mean: dict[str, float]
    high: dict[str, float]

    def cover(self, config: str) -> float:
        """What the planner reserves for one run of `config`: conservative on purpose."""
        return max(self.high[config], COVER_MARGIN * self.mean[config])


def cost_basis(prior_runs_dir: Path, model: str = SONNET) -> CostBasis:
    by: dict[str, list[float]] = {}
    for f in Path(prior_runs_dir).glob("*.json"):
        s = json.loads(f.read_text())
        if s.get("model") == model and s["status"] == "ok":
            by.setdefault(s["config"], []).append(s["cost_agent"] + s["cost_user"])
    missing = [c for c in CONFIG_NAMES if c not in by]
    if missing:
        raise RuntimeError(f"phase 1 has no finished Sonnet run for {missing}: cannot size phase 2")
    return CostBasis(
        mean={c: sum(v) / len(v) for c, v in by.items()}, high={c: max(v) for c, v in by.items()}
    )


def rail_for(basis: CostBasis) -> float:
    return p2.rail_for(basis.mean["default"])


@dataclass
class SweepPlan:
    ids: list[str]
    todo: list[tuple[Task, list[str]]]  # tasks with at least one run to execute, in order
    reused: list[str]  # "config/task"

    @property
    def n_total(self) -> int:
        return len(self.ids) * len(CONFIG_NAMES)

    @property
    def n_todo(self) -> int:
        return sum(len(c) for _, c in self.todo)


def plan_b2(priors: list[p2.Prior], out_dir: Path, trace_dir: Path) -> SweepPlan:
    ids = train_ids()
    todo, reused = p2.plan_sweep(
        tasks_by_id(BANKING_DOMAIN, ids), priors, out_dir, trace_dir, SONNET, CONFIG_NAMES
    )
    return SweepPlan(ids=ids, todo=todo, reused=reused)


def estimate(plan: SweepPlan, basis: CostBasis) -> dict[str, Any]:
    """`certain`: every run still to execute at its measured mean, plus the classifier check.
    `cover`: the same runs at the planner's conservative cost. Infrastructure reruns are the
    conditional part and are not counted."""
    certain = sum(basis.mean[c] for _, cs in plan.todo for c in cs) + CLASSIFIER_USD
    cover = sum(basis.cover(c) for _, cs in plan.todo for c in cs) + CLASSIFIER_USD
    return {"certain": certain, "cover": cover, "runs": plan.n_todo}


def tasks_startable(plan: SweepPlan, basis: CostBasis, remaining: float) -> int:
    """How many tasks the planner would start if every run cost its cover (no concurrency)."""
    left, n = remaining - CLASSIFIER_USD, 0
    for _, cs in plan.todo:
        need = sum(basis.cover(c) for c in cs)
        if need > left:
            break
        left -= need
        n += 1
    return n


def preflight(
    plan: SweepPlan, basis: CostBasis, cap: float, prior_phase: float, prior_total: float,
    budget_total: float,
) -> dict[str, Any]:  # fmt: skip
    """Numbers the dry run prints, and the refusal rule of the paid step: it does not start if
    the certain part of the estimate is above the cap left for this phase, if the total would be
    passed, or if not even the first task fits."""
    remaining = cap - prior_phase
    est = estimate(plan, basis)
    first_cover = sum(basis.cover(c) for c in plan.todo[0][1]) if plan.todo else 0.0
    reasons = []
    if remaining <= 0:
        reasons.append("no cap left for this phase")
    if est["certain"] > remaining:
        reasons.append(f"certain part ${est['certain']:.2f} > cap left ${remaining:.2f}")
    if prior_total + remaining > budget_total:
        reasons.append(f"prior ${prior_total:.2f} + cap left ${remaining:.2f} > total")
    if plan.todo and first_cover > remaining:
        reasons.append("the first task's runs do not fit in the cap left")
    return {
        **est,
        "remaining": remaining,
        "eighty_percent": 0.8 * remaining,
        "above_eighty_percent": est["certain"] > 0.8 * remaining,
        "startable_at_cover": tasks_startable(plan, basis, remaining),
        "tasks_to_run": len(plan.todo),
        "refuse": reasons,
    }


@dataclass
class _Cover:
    """Cap bookkeeping for the planner: what running tasks may still cost."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    outstanding: dict[str, float] = field(default_factory=dict)
    stopped: str | None = None


def run_sweep(
    runner: Runner, ledger: Ledger, plan: SweepPlan, basis: CostBasis, concurrency: int = 3
) -> dict[str, Any]:
    """Run the plan task by task. A task is started only if the cap left, after what the tasks
    already running may still cost, covers its missing runs; the first task that does not fit
    stops the sweep (earlier tasks stay complete). Reuses `runner.run_task` (one rerun on an
    infrastructure error)."""
    cover = _Cover()
    started: list[str] = []
    skipped: list[str] = []

    def work(item: tuple[Task, list[str]]) -> None:
        task, cfgs = item
        need = sum(basis.cover(c) for c in cfgs)
        with cover.lock:
            if cover.stopped:
                skipped.append(task.id)
                return
            # the classifier check after the sweep keeps its small share of the cap
            room = (
                ledger.phase_cap - ledger.spent - sum(cover.outstanding.values()) - CLASSIFIER_USD
            )
            if room < need:
                cover.stopped = (
                    f"cap: ${room:.2f} left, task {task.id} needs ${need:.2f} for its runs"
                )
                skipped.append(task.id)
                return
            cover.outstanding[task.id] = need
            started.append(task.id)
        try:
            runner.run_task(SONNET, cfgs, task)
        finally:
            with cover.lock:
                cover.outstanding.pop(task.id, None)

    budget_error: BudgetExceeded | None = None
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, it) for it in plan.todo]
        for f in futs:
            try:
                f.result()
            except BudgetExceeded as e:
                budget_error = budget_error or e
                cover.stopped = cover.stopped or f"budget: {e}"
    return {
        "started": started,
        "not_started": skipped,
        "stopped": cover.stopped,
        "budget_error": str(budget_error) if budget_error else None,
    }
