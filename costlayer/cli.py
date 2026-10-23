"""Command line entry point. Paid runs happen only inside GitHub Actions (see AnthropicClient)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from costlayer import ledger as ledger_mod
from costlayer import phase1 as phase1_mod
from costlayer import report
from costlayer.client import AnthropicClient
from costlayer.ledger import Ledger, plan_dispatch, read_prior, write_start
from costlayer.runner import Runner, check_data_dir, tool_kinds
from costlayer.usersim import install

# Planned cost of phase 1 (CLAUDE.md): there is no measured per-run cost yet.
PLAN_HAIKU_USD = 0.30
PLAN_SONNET_USD = 1.50
HAIKU_RUNS = 3
SONNET_RUNS_MAX = 4 + 2 * 2


def cmd_dry_run(a: argparse.Namespace) -> int:
    cap = ledger_mod.PHASE_CAPS_USD[a.phase]
    prior = read_prior(Path(a.spend_dir)) if a.spend_dir else ledger_mod.Prior()
    est = PLAN_HAIKU_USD + PLAN_SONNET_USD
    print(f"DRY RUN phase {a.phase} (no API call, no cost)")
    print(
        f"  runs: {HAIKU_RUNS} Haiku (default) + up to {SONNET_RUNS_MAX} Sonnet "
        f"(task 1 x 4 configurations, tasks 2-3 x default and E:low)"
    )
    print(
        f"  estimated cost: Haiku ~${PLAN_HAIKU_USD:.2f} + Sonnet ~${PLAN_SONNET_USD:.2f} = "
        f"~${est:.2f}  (plan figures; nothing is measured yet)"
    )
    print(
        f"  phase cap ${cap:.2f}; spent before this dispatch ${prior.total:.2f} of "
        f"${ledger_mod.TOTAL_CAP_USD:.2f}"
    )
    ok = est <= 0.8 * cap and prior.total + cap <= ledger_mod.TOTAL_CAP_USD
    print("  verdict:", "within 80% of the phase cap" if ok else "TOO HIGH, do not run")
    return 0 if ok else 1


def cmd_phase1(a: argparse.Namespace) -> int:
    check_data_dir()
    spend = Path(a.spend_dir)
    cap = plan_dispatch(spend, a.dispatch_id, 1)
    write_start(spend, a.dispatch_id, 1, cap)
    prior = read_prior(spend, exclude=a.dispatch_id)
    led = Ledger(
        spend / f"ledger-{a.dispatch_id}.jsonl", a.dispatch_id, 1, cap, prior_total=prior.total
    )
    led.path.touch()
    client = AnthropicClient(led)
    install(led)
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
        note = (
            f"Ledger total for this dispatch ({dt.datetime.now(dt.UTC):%Y-%m-%d} UTC): "
            f"${summ['total']:.4f}. Founder: enter the Console figure for the day here."
        )
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(
            report.build(result, runs, summ, prior.total, Path(a.trace_dir), note)
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
    d.add_argument("--phase", type=int, default=1, choices=[1])
    d.add_argument("--spend-dir", default="")
    d.set_defaults(fn=cmd_dry_run)
    p1 = sub.add_parser("phase1", help="paid: only inside GitHub Actions")
    p1.add_argument("--spend-dir", required=True)
    p1.add_argument("--dispatch-id", required=True)
    p1.add_argument("--out", default="results/phase-1")
    p1.add_argument("--trace-dir", default="traces/phase-1")
    p1.add_argument("--report", default="reports/phase-1.md")
    p1.set_defaults(fn=cmd_phase1)
    sub.add_parser("audit", help="list tau2 OpenAI defaults").set_defaults(fn=cmd_audit)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
