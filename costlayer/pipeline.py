"""Phase pipelines: phase 1 (build check), phase 2 (learn), phase 3 (verdict with stop rule)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from costlayer.batch import load_runs, run_batch
from costlayer.domains import DOMAINS, TaskRef, make_split
from costlayer.learn import learn_profile
from costlayer.llm import LLM
from costlayer.profile import Profile
from costlayer.stats import wilson

STOP_AFTER_TASKS = 20
STOP_MIN_SAVINGS = 0.15
STOP_MAX_SUCCESS_DROP = 0.15
BLOCK_SIZE = 10


def pick(split: str, per_domain: int) -> list[TaskRef]:
    refs = make_split()[split]
    out: list[TaskRef] = []
    for d in DOMAINS:
        out += [r for r in refs if r.domain == d][:per_domain]
    return out


def load_traces(trace_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(trace_dir).glob("*-baseline.json"))]


@dataclass
class Phase1Config:
    model: str
    per_domain: int = 12
    workers: int = 4
    min_support: int = 2
    force_compaction_tokens: int = 2500  # build check only: make lever 10 fire on short runs


def phase1(llm: LLM, cfg: Phase1Config, out_dir: Path, trace_dir: Path) -> Profile:
    """Baseline on train tasks, learn a profile from them, run the layer on the same tasks."""
    runs_path = out_dir / "runs.jsonl"
    refs = pick("train", cfg.per_domain)
    # Smoke test first: one cheap baseline task; fail fast if the API shape is wrong.
    smoke = run_batch(
        refs[:1], ["baseline"], llm, cfg.model, runs_path, trace_dir, workers=1, label_baseline=True
    )
    for r in smoke:
        if r.error or not r.steps:
            raise RuntimeError(f"smoke test failed: {r.error}")
    run_batch(
        refs,
        ["baseline"],
        llm,
        cfg.model,
        runs_path,
        trace_dir,
        workers=cfg.workers,
        label_baseline=True,
    )
    profile = learn_profile(load_traces(trace_dir), min_support=cfg.min_support)
    for ts in profile.types.values():
        for tp in ts.values():
            cur = tp.compact_after_tokens or cfg.force_compaction_tokens
            tp.compact_after_tokens = min(cur, cfg.force_compaction_tokens)
    profile.meta["phase1_forced_compaction_tokens"] = cfg.force_compaction_tokens
    profile.save(out_dir / "profile.json")
    run_batch(
        refs, ["layer"], llm, cfg.model, runs_path, trace_dir, profile=profile, workers=cfg.workers
    )
    return profile


def phase2(
    llm: LLM, model: str, out_dir: Path, trace_dir: Path, workers: int = 4, min_support: int = 3
) -> Profile:
    runs_path = out_dir / "runs.jsonl"
    refs = make_split()["train"]
    run_batch(
        refs, ["baseline"], llm, model, runs_path, trace_dir, workers=workers, label_baseline=True
    )
    profile = learn_profile(load_traces(trace_dir), min_support=min_support)
    profile.save(out_dir / "profile.json")
    return profile


def stop_check(pairs: list[tuple[dict, dict]]) -> tuple[bool, str]:
    """Phase 3 stop rule, evaluated once at least STOP_AFTER_TASKS test tasks are done."""
    if len(pairs) < STOP_AFTER_TASKS:
        return False, "fewer than 20 tasks"
    b = sum(x["cost_agent"] + x["cost_classifier"] for x, _ in pairs)
    y = sum(x["cost_agent"] + x["cost_classifier"] for _, x in pairs)
    savings = 1 - y / b if b else 0.0
    sb = sum(x["reward"] == 1.0 for x, _ in pairs) / len(pairs)
    sl = sum(x["reward"] == 1.0 for _, x in pairs) / len(pairs)
    if savings < STOP_MIN_SAVINGS:
        return True, f"savings {savings:.1%} < {STOP_MIN_SAVINGS:.0%}"
    if sb - sl > STOP_MAX_SUCCESS_DROP:
        return True, f"success drop {100 * (sb - sl):.1f} points > 15"
    return False, f"savings {savings:.1%}, success {sl:.1%} vs {sb:.1%}"


def phase3(
    llm: LLM, model: str, profile: Profile, out_dir: Path, trace_dir: Path, workers: int = 4
) -> str:
    """Blocks of 10 test tasks, each run as a (baseline, layer) pair; stop rule after 20."""
    runs_path = out_dir / "runs.jsonl"
    refs = make_split()["test"]
    # interleave domains so every block is mixed
    by = {d: [r for r in refs if r.domain == d] for d in DOMAINS}
    order: list[TaskRef] = []
    while any(by.values()):
        for d in DOMAINS:
            if by[d]:
                order.append(by[d].pop(0))
    for i in range(0, len(order), BLOCK_SIZE):
        run_batch(
            order[i : i + BLOCK_SIZE],
            ["baseline", "layer"],
            llm,
            model,
            runs_path,
            trace_dir,
            profile=profile,
            workers=workers,
        )
        runs = load_runs(runs_path)
        lb = {(r["domain"], r["idx"]): r for r in runs if r["mode"] == "layer"}
        pairs = [
            (r, lb[(r["domain"], r["idx"])])
            for r in runs
            if r["mode"] == "baseline" and (r["domain"], r["idx"]) in lb
        ]
        stop, why = stop_check(pairs)
        if stop:
            return f"stopped after {len(pairs)} tasks: {why}"
    return "completed all test tasks"


__all__ = ["Phase1Config", "phase1", "phase2", "phase3", "stop_check", "wilson"]
