"""Command line entry point. Paid runs happen only inside GitHub Actions (see LiveLLM)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from costlayer import pipeline
from costlayer.batch import load_runs
from costlayer.domains import make_split
from costlayer.ledger import PHASE_CAPS_USD, Ledger, summarize
from costlayer.pricing import HAIKU, SONNET
from costlayer.profile import Profile
from costlayer.report import build_report
from costlayer.sim import ScriptedLLM

MODELS = {"haiku": HAIKU, "sonnet": SONNET}
SAFETY = 3.0  # the scripted model under-counts real thinking, retries and longer replies


def _estimate(phase: int, model: str, per_domain: int) -> dict:
    """Free estimate: run the whole phase with the scripted LLM and price the synthetic usage."""
    llm = ScriptedLLM()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        if phase == 1:
            pipeline.phase1(
                llm,
                pipeline.Phase1Config(model=model, per_domain=per_domain, workers=1),
                d,
                d / "t",
            )
        else:
            raise SystemExit("dry run for phases 2 and 3 is part of their own go/no-go report")
        runs = load_runs(d / "runs.jsonl")
    n = len(runs)
    exp = sum(
        r["cost_agent"] + r["cost_classifier"] + r["cost_user"] + r["cost_label"] for r in runs
    )
    return {
        "runs": n,
        "expected_usd": exp,
        "with_safety_usd": exp * SAFETY,
        "per_run_usd": exp / max(1, n),
    }


def cmd_dry_run(a: argparse.Namespace) -> int:
    model = MODELS[a.model]
    est = _estimate(a.phase, model, a.per_domain)
    cap = PHASE_CAPS_USD[a.phase]
    print(f"DRY RUN phase {a.phase}, agent model {model}, {a.per_domain} tasks per domain")
    print(f"  runs (baseline + layer): {est['runs']}")
    print(f"  expected cost: ${est['expected_usd']:.3f} (${est['per_run_usd']:.4f} per run)")
    print(f"  with x{SAFETY} safety: ${est['with_safety_usd']:.3f}   phase cap: ${cap:.2f}")
    ok = est["with_safety_usd"] <= cap
    print("  verdict:", "within cap" if ok else "EXCEEDS cap, reduce --per-domain")
    return 0 if ok else 1


def cmd_phase1(a: argparse.Namespace) -> int:
    model = MODELS[a.model]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(a.trace_dir)
    ledger = Ledger(Path(a.ledger), phase=1)
    from costlayer.llm import LiveLLM

    llm = LiveLLM(ledger)
    cfg = pipeline.Phase1Config(model=model, per_domain=a.per_domain, workers=a.workers)
    try:
        pipeline.phase1(llm, cfg, out, trace_dir)
    finally:  # always leave a summary and a report, even after an abort
        _write_report(1, out, Path(a.ledger), Path(a.report))
    return 0


def _write_report(phase: int, out: Path, ledger_path: Path, report_path: Path) -> None:
    runs = load_runs(out / "runs.jsonl")
    summ = summarize(ledger_path)
    (out / "ledger-summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(phase, runs, summ))


def cmd_report(a: argparse.Namespace) -> int:
    _write_report(a.phase, Path(a.out), Path(a.ledger), Path(a.report))
    return 0


def _guard_go(a: argparse.Namespace) -> None:
    if a.founder_go != f"phase{a.cmd[-1]}":
        raise SystemExit(f"{a.cmd} needs the founder's go: pass --founder-go phase{a.cmd[-1]}")


def cmd_phase2(a: argparse.Namespace) -> int:
    _guard_go(a)
    from costlayer.llm import LiveLLM

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LiveLLM(Ledger(Path(a.ledger), phase=2))
    try:
        pipeline.phase2(llm, SONNET, out, Path(a.trace_dir), workers=a.workers)
    finally:
        _write_report(2, out, Path(a.ledger), Path(a.report))
    return 0


def cmd_phase3(a: argparse.Namespace) -> int:
    _guard_go(a)
    from costlayer.llm import LiveLLM

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LiveLLM(Ledger(Path(a.ledger), phase=3))
    profile = Profile.load(Path(a.profile))
    try:
        status = pipeline.phase3(llm, SONNET, profile, out, Path(a.trace_dir), workers=a.workers)
        print(status)
    finally:
        _write_report(3, out, Path(a.ledger), Path(a.report))
    return 0


def cmd_split(a: argparse.Namespace) -> int:
    s = make_split()
    Path(a.out).write_text(
        json.dumps({k: [r.key for r in v] for k, v in s.items()}, indent=1) + "\n"
    )
    print({k: len(v) for k, v in s.items()})
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="costlayer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, phase: int) -> None:
        sp.add_argument("--out", default=f"results/phase-{phase}")
        sp.add_argument("--trace-dir", default=f"traces/phase-{phase}")
        sp.add_argument("--ledger", default="ledger.jsonl")
        sp.add_argument("--report", default=f"reports/phase-{phase}.md")
        sp.add_argument("--workers", type=int, default=4)

    d = sub.add_parser("dry-run", help="free cost estimate (scripted model, no API)")
    d.add_argument("--phase", type=int, default=1)
    d.add_argument("--model", choices=MODELS, default="haiku")
    d.add_argument("--per-domain", type=int, default=12)
    d.set_defaults(fn=cmd_dry_run)

    p1 = sub.add_parser("phase1", help="paid: only inside GitHub Actions")
    common(p1, 1)
    p1.add_argument("--model", choices=MODELS, default="haiku")
    p1.add_argument("--per-domain", type=int, default=12)
    p1.set_defaults(fn=cmd_phase1)

    p2 = sub.add_parser("phase2", help="paid; needs the founder's go")
    common(p2, 2)
    p2.add_argument("--founder-go", default="")
    p2.set_defaults(fn=cmd_phase2)

    p3 = sub.add_parser("phase3", help="paid; needs the founder's go")
    common(p3, 3)
    p3.add_argument("--founder-go", default="")
    p3.add_argument("--profile", default="results/phase-2/profile.json")
    p3.set_defaults(fn=cmd_phase3)

    r = sub.add_parser("report")
    r.add_argument("--phase", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ledger", default="ledger.jsonl")
    r.add_argument("--report", required=True)
    r.set_defaults(fn=cmd_report)

    s = sub.add_parser("split")
    s.add_argument("--out", default="splits.json")
    s.set_defaults(fn=cmd_split)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
