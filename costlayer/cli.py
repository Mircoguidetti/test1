"""Command line entry point. Paid runs happen only inside GitHub Actions (see AnthropicClient)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from costlayer import ledger as ledger_mod
from costlayer import phase1 as phase1_mod
from costlayer import phase2 as phase2_mod
from costlayer import report, report2
from costlayer.client import AnthropicClient
from costlayer.ledger import BudgetExceeded, Ledger, plan_dispatch, read_prior, write_start
from costlayer.optimizer import analysis
from costlayer.optimizer.classify_eval import loo_accuracy
from costlayer.optimizer.data import load_runs, load_tool_kinds
from costlayer.runner import Runner, check_data_dir, split_tasks, tool_kinds
from costlayer.usersim import install

# Planned cost of phase 1 (CLAUDE.md): there is no measured per-run cost yet.
PLAN_HAIKU_USD = 0.30
PLAN_SONNET_USD = 1.50
HAIKU_RUNS = 3
SONNET_RUNS_MAX = 4 + 2 * 2
CLASSIFIER_USD = 0.05  # 21 Haiku calls with a few-shot prompt, generous


def _console_note(total: float) -> str:
    return (
        f"Ledger total for this dispatch ({dt.datetime.now(dt.UTC):%Y-%m-%d} UTC): "
        f"${total:.4f}. Founder: enter the Console figure for the day here."
    )


def cmd_dry_run(a: argparse.Namespace) -> int:
    cap = ledger_mod.PHASE_CAPS_USD[a.phase]
    prior = read_prior(Path(a.spend_dir)) if a.spend_dir else ledger_mod.Prior()
    print(f"DRY RUN phase {a.phase} (no API call, no cost)")
    if a.phase == 1:
        est = PLAN_HAIKU_USD + PLAN_SONNET_USD
        print(f"  runs: {HAIKU_RUNS} Haiku (default) + up to {SONNET_RUNS_MAX} Sonnet")
        print(f"  estimated cost: ~${est:.2f} (plan figures; nothing is measured yet)")
    else:
        from tau2.registry import registry  # noqa: F401  (loads the tau2 data)

        c = phase2_mod.measured_costs(Path(a.phase1_runs))
        n, alloc = phase2_mod.n_train_from(Path(a.phase1_runs))
        out = Path(a.out)
        plan, reused = phase2_mod.plan_sweep(
            split_tasks("train")[:n],
            [phase2_mod.Prior(Path(a.phase1_runs), [Path(d) for d in a.prior_traces])],
            out,
            Path(a.trace_dir),
        )
        to_run = sum(len(cfgs) for _, cfgs in plan)
        est = phase2_mod.estimate(plan, c) + CLASSIFIER_USD
        print(
            f"  N_train = {n} (allocation rule); runs: {4 * n} total, {len(reused)} reused, "
            f"{to_run} to run"
        )
        print("  measured c(x): " + ", ".join(f"{k} ${v:.4f}" for k, v in c.items()))
        print(f"  estimated cost: ~${est:.2f} (measured per-run cost x runs to run + classifier)")
        print(f"  rail per run: ${phase2_mod.rail_for(c['default']):.2f}")
        cap_ok = alloc["train_ok"]
        print(f"  N_train minimum (12) met: {cap_ok}")
    print(
        f"  phase cap ${cap:.2f} (80% = ${0.8 * cap:.2f}); spent before this dispatch "
        f"${prior.total:.2f} of ${ledger_mod.TOTAL_CAP_USD:.2f}"
    )
    ok = est <= 0.8 * cap and prior.total + cap <= ledger_mod.TOTAL_CAP_USD
    print("  verdict:", "within 80% of the phase cap" if ok else "TOO HIGH, do not run")
    return 0 if ok else 1


def _setup(a: argparse.Namespace, phase: int) -> tuple[Ledger, AnthropicClient, float]:
    check_data_dir()
    spend = Path(a.spend_dir)
    cap = plan_dispatch(spend, a.dispatch_id, phase)
    write_start(spend, a.dispatch_id, phase, cap)
    prior = read_prior(spend, exclude=a.dispatch_id)
    led = Ledger(
        spend / f"ledger-{a.dispatch_id}.jsonl", a.dispatch_id, phase, cap, prior_total=prior.total
    )
    led.path.touch()
    client = AnthropicClient(led)
    install(led)
    return led, client, prior.total


def cmd_phase1(a: argparse.Namespace) -> int:
    led, client, prior_total = _setup(a, 1)
    out = Path(a.out)
    runner = Runner(led, client, out, Path(a.trace_dir))
    (out / "tool_kinds.json").write_text(json.dumps(tool_kinds(), indent=1) + "\n")
    result: dict = {}
    try:
        result = phase1_mod.run_phase1(runner, led, out)
    finally:
        runs = report.load_runs(out)
        summ = ledger_mod.summarize([led.path])
        (out / "ledger-summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(
            report.build(
                result, runs, summ, prior_total, Path(a.trace_dir), _console_note(summ["total"])
            )
        )
    return 0


def _classifier_examples(
    runs: list, kinds: dict, res: dict[str, Any]
) -> list[tuple[str, str, str]]:
    ex = []
    by = {r.task: r for r in runs if r.config == "default" and r.trace}
    for t in res["tasks"]:
        tr = by[t].trace if t in by else None
        first = (
            next((m["content"] for m in tr["messages"] if m["role"] == "user"), None)
            if tr
            else None
        )
        if first:
            ex.append((t, first, res["types"][t]))
    return ex


def cmd_phase2(a: argparse.Namespace) -> int:
    if a.founder_go != "phase2":
        raise SystemExit("phase2 needs the founder's go: pass --founder-go phase2")
    led, client, prior_total = _setup(a, 2)
    out, trace_dir = Path(a.out), Path(a.trace_dir)
    runner = Runner(led, client, out, trace_dir)
    kinds_path = out / "tool_kinds.json"
    kinds_path.write_text(json.dumps(tool_kinds(), indent=1) + "\n")
    p1 = Path(a.phase1_runs)
    n, _ = phase2_mod.n_train_from(p1)
    c = phase2_mod.measured_costs(p1)
    priors = [phase2_mod.Prior(p1, [Path(d) for d in a.prior_traces])]
    phase: dict[str, Any] = {"n_train": n, "reused": [], "to_run": 0, "stopped": None}
    classifier = None
    res: dict[str, Any] = {"error": "sweep did not run"}
    try:
        phase = phase2_mod.run_phase2(
            runner, out, trace_dir, priors, n, phase2_mod.rail_for(c["default"])
        )
    finally:
        runs = load_runs(out / "runs", [trace_dir])
        kinds = load_tool_kinds(kinds_path)
        res = analysis.analyze(runs, kinds)
        if "error" not in res:
            try:
                classifier = loo_accuracy(
                    client, _classifier_examples(runs, kinds, res), a.dispatch_id
                )
            except BudgetExceeded as e:
                phase["stopped"] = f"{phase.get('stopped') or ''} classifier: {e}".strip()
        summ = ledger_mod.summarize([led.path])
        (out / "ledger-summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        res_small = {k: v for k, v in res.items() if k not in ("fields",)}
        (out / "analysis.json").write_text(report2.dumps({**res_small, "classifier": classifier}))
        dropped = sorted({r.task for r in runs} - set(res.get("tasks", [])))
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(
            report2.build(
                res, phase, classifier, summ, prior_total, _console_note(summ["total"]), dropped
            )
        )
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    """Offline: rebuild the analysis and report from committed runs and local traces."""
    runs = load_runs(Path(a.out) / "runs", [Path(d) for d in a.trace_dir])
    res = analysis.analyze(runs, load_tool_kinds(Path(a.out) / "tool_kinds.json"))
    ledger = {"total": 0.0, "calls": 0, "by_purpose": {}}
    Path(a.report).write_text(
        report2.build(
            res,
            {"n_train": res.get("n_tasks")},
            None,
            ledger,
            0.0,
            "offline analysis, no ledger",
            [],
        )
    )
    return 0


def cmd_audit(_: argparse.Namespace) -> int:
    from costlayer.audit import audit

    for r in audit():
        print(f"{r['name']} = {r['value']}  ({len(r['uses'])} refs) - {r['note']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="costlayer")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="free cost estimate before a paid run")
    d.add_argument("--phase", type=int, default=1, choices=[1, 2])
    d.add_argument("--spend-dir", default="")
    d.add_argument("--phase1-runs", default="results/phase-1/runs")
    d.add_argument("--prior-traces", nargs="*", default=["traces/phase-1-prior"])
    d.add_argument("--out", default="/tmp/costlayer-dryrun/phase-2")
    d.add_argument("--trace-dir", default="/tmp/costlayer-dryrun/traces")
    d.set_defaults(fn=cmd_dry_run)

    def paid(name: str, phase: int, fn: Any) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help="paid: only inside GitHub Actions")
        sp.add_argument("--spend-dir", required=True)
        sp.add_argument("--dispatch-id", required=True)
        sp.add_argument("--out", default=f"results/phase-{phase}")
        sp.add_argument("--trace-dir", default=f"traces/phase-{phase}")
        sp.add_argument("--report", default=f"reports/phase-{phase}.md")
        sp.set_defaults(fn=fn)
        return sp

    paid("phase1", 1, cmd_phase1)
    p2 = paid("phase2", 2, cmd_phase2)
    p2.add_argument("--founder-go", default="")
    p2.add_argument("--phase1-runs", default="results/phase-1/runs")
    p2.add_argument("--prior-traces", nargs="*", default=["traces/phase-1-prior"])
    an = sub.add_parser("analyze", help="offline: analysis and report from runs and traces")
    an.add_argument("--out", default="results/phase-2")
    an.add_argument("--trace-dir", nargs="*", default=["traces/phase-2"])
    an.add_argument("--report", default="reports/phase-2.md")
    an.set_defaults(fn=cmd_analyze)
    sub.add_parser("audit", help="list tau2 OpenAI defaults").set_defaults(fn=cmd_audit)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
