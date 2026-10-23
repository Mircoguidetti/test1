"""Run many tasks in parallel and persist compact results (full traces go to an artifact dir)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from costlayer.domains import TaskRef
from costlayer.learn import trace_dict
from costlayer.ledger import BudgetExceeded
from costlayer.llm import LLM
from costlayer.profile import Profile
from costlayer.runner import RunResult, run_task


def load_runs(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def run_batch(
    refs: list[TaskRef],
    modes: list[str],
    llm: LLM,
    model: str,
    runs_path: Path,
    trace_dir: Path | None = None,
    profile: Profile | None = None,
    workers: int = 4,
    compact_override: int | None = None,
    label_baseline: bool = False,
) -> list[RunResult]:
    """Skips (task, mode) pairs already in runs_path, so an interrupted job can resume."""
    done = {r["run_id"] for r in load_runs(runs_path)}
    jobs = [(r, m) for r in refs for m in modes if f"{r.key}-{m}" not in done]
    lock = threading.Lock()
    results: list[RunResult] = []
    stop = threading.Event()
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    def work(job: tuple[TaskRef, str]) -> None:
        if stop.is_set():
            return
        ref, mode = job
        try:
            res = run_task(
                ref,
                mode,
                llm,
                model,
                profile=profile,
                compact_override=compact_override,
                label_baseline=label_baseline,
            )
        except BudgetExceeded:
            stop.set()
            raise
        with lock:
            results.append(res)
            row = {**res.summary(), "steps": [vars(s) for s in res.steps]}
            with runs_path.open("a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if trace_dir:
                (trace_dir / f"{res.run_id}.json").write_text(json.dumps(trace_dict(res)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        errors = []
        for f in futures:
            try:
                f.result()
            except BudgetExceeded as e:
                errors.append(e)
        if errors:
            raise errors[0]
    return results
