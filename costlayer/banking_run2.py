"""Test 2, phase 2 (`b2`): the dry run, the refusal rules and the paid flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from costlayer import banking_phase2 as b2
from costlayer import ledger as ledger_mod
from costlayer import phase2 as p2
from costlayer import report_banking2
from costlayer.banking_phase1 import VARIANT
from costlayer.budget import Budget
from costlayer.client import AnthropicClient
from costlayer.context import ApiRequestError, InfraError
from costlayer.ledger import BudgetExceeded, Ledger, plan_dispatch, read_prior, write_start
from costlayer.optimizer import analysis
from costlayer.optimizer.classify_eval import BANKING_SYSTEM, examples_from_runs, loo_accuracy
from costlayer.optimizer.data import BANKING_CONFIGS, load_runs, load_tool_kinds
from costlayer.optimizer.gate import verdict_lines
from costlayer.runner import BANKING_DOMAIN, Runner, check_data_dir, tool_kinds
from costlayer.usersim import install


def _priors(a: Any) -> list[p2.Prior]:
    return [p2.Prior(Path(a.prior_runs), [Path(d) for d in a.prior_traces])]


def reuse_refusal(a: Any) -> list[str]:
    """A missing phase 1 trace folder means the artifact download failed and nothing would be
    reused: the plan (8 runs reused, about $2.3 saved) would not hold. Stop, unless the
    coordinator accepts the extra cost with `--allow-no-reuse`."""
    if getattr(a, "allow_no_reuse", False) or any(Path(d).is_dir() for d in a.prior_traces):
        return []
    return [
        f"phase 1 traces not found at {', '.join(a.prior_traces)} (artifact download failed?): "
        "no run would be reused; pass --allow-no-reuse to accept the extra cost"
    ]


def dry_run_b2(a: Any, budget: Budget, cap: float) -> int:
    """Free: plans the sweep in a scratch folder, prints runs to execute, runs reused and the
    estimated cost, and returns 1 if the paid step would refuse to start."""
    import tempfile

    spend = Path(a.spend_dir) if a.spend_dir else None
    prior = read_prior(spend) if spend else ledger_mod.Prior()
    basis = b2.cost_basis(Path(a.prior_runs))
    with tempfile.TemporaryDirectory() as td:
        plan = b2.plan_b2(_priors(a), Path(td) / "out", Path(td) / "traces")
    pf = b2.preflight(plan, basis, cap, prior.by_phase.get(a.phase, 0.0), prior.total, budget.total)
    print(f"DRY RUN test 2, phase {a.phase} (no API call, no cost)")
    print(
        f"  train tasks ({b2.N_TRAIN}, banking_order seed 42, ACTION excluded): "
        f"{', '.join(plan.ids)}"
    )
    print(f"  configurations: {', '.join(b2.CONFIG_NAMES)} (Sonnet 5.5), variant {VARIANT}")
    print(
        f"  runs: {plan.n_total} in total, {len(plan.reused)} reused from phase 1, "
        f"{plan.n_todo} to execute"
    )
    print(
        "  reused: "
        + (", ".join(sorted(plan.reused)) or "none (is the phase 1 artifact downloaded?)")
    )
    print(
        "  measured c(x) (agent + user, phase 1): "
        + ", ".join(f"{c} ${basis.mean[c]:.4f}" for c in b2.CONFIG_NAMES)
    )
    print(f"  per-run abort rail: ${b2.rail_for(basis):.2f}")
    print(
        f"  estimated cost, certain part: ${pf['certain']:.2f} (runs to execute at their "
        f"measured mean + ${b2.CLASSIFIER_USD:.2f} classifier check)"
    )
    print(
        f"  at the planner's conservative cost per run: ${pf['cover']:.2f}; it would start "
        f"{pf['startable_at_cover']} of {pf['tasks_to_run']} tasks if every run cost that"
    )
    print(
        f"  phase cap ${cap:.2f}; left for this phase ${pf['remaining']:.2f} "
        f"(80% = ${pf['eighty_percent']:.2f}); spent so far in test 2 ${prior.total:.2f} of "
        f"${budget.total:.2f}"
    )
    if pf["above_eighty_percent"]:
        print(
            "  WARNING: the certain part is above 80% of the cap left (CLAUDE.md sizes against 80%)"
        )
    if pf["refuse"]:
        print("  verdict: REFUSED -", "; ".join(pf["refuse"]))
        return 1
    print("  verdict: the certain part fits in the cap left")
    return 0


def run_b2(a: Any, budget: Budget, cap: float, console_note_fn: Any) -> int:
    check_data_dir(BANKING_DOMAIN)
    out, trace_dir = Path(a.out), Path(a.trace_dir)
    spend = Path(a.spend_dir)
    prior = read_prior(spend, exclude=a.dispatch_id)
    basis = b2.cost_basis(Path(a.prior_runs))
    # pre-flight on a scratch copy: a refused start must leave nothing in the results folders
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        scratch = b2.plan_b2(_priors(a), Path(td) / "out", Path(td) / "traces")
    pf = b2.preflight(
        scratch, basis, cap, prior.by_phase.get(a.phase, 0.0), prior.total, budget.total
    )
    pf["refuse"] = reuse_refusal(a) + pf["refuse"]
    if pf["refuse"]:
        raise SystemExit("REFUSED: " + "; ".join(pf["refuse"]))
    plan = b2.plan_b2(_priors(a), out, trace_dir)
    phase_cap = plan_dispatch(
        spend, a.dispatch_id, a.phase, phase_caps=budget.phase_caps, total_cap=budget.total
    )
    write_start(spend, a.dispatch_id, a.phase, phase_cap)
    led = Ledger(
        spend / f"ledger-{a.dispatch_id}.jsonl",
        a.dispatch_id,
        a.phase,
        phase_cap,
        prior_total=prior.total,
        total_cap=budget.total,
    )
    led.path.touch()
    client = AnthropicClient(led)
    install(led)
    runner = Runner(led, client, out, trace_dir, domain=BANKING_DOMAIN, retrieval_variant=VARIANT)
    runner.rail_usd = b2.rail_for(basis)
    kinds_path = out / "tool_kinds.json"
    kinds_path.write_text(json.dumps(tool_kinds(BANKING_DOMAIN, VARIANT), indent=1) + "\n")
    info: dict[str, Any] = {
        "ids": plan.ids,
        "reused": plan.reused,
        "n_todo": plan.n_todo,
        "stopped": None,
    }
    res: dict[str, Any] = {"error": "sweep did not run"}
    classifier = None
    try:
        sweep = b2.run_sweep(runner, led, plan, basis)
        info["stopped"] = sweep["stopped"]
    finally:
        runs = load_runs(out / "runs", [trace_dir])
        kinds = load_tool_kinds(kinds_path)
        res = analysis.analyze(runs, kinds, BANKING_CONFIGS)
        if "error" not in res:
            try:
                classifier = loo_accuracy(
                    client, examples_from_runs(runs, res), a.dispatch_id, BANKING_SYSTEM
                )
            except (BudgetExceeded, InfraError, ApiRequestError) as e:
                info["classifier_note"] = f"{type(e).__name__}: {e}"[:200]
        summ = ledger_mod.summarize([led.path])
        (out / "ledger-summary.json").write_text(json.dumps(summ, indent=1) + "\n")
        proj = report_banking2.projections(res, classifier) if "error" not in res else {}
        small = {k: v for k, v in res.items() if k != "fields"}
        small["projection"] = {k: (v.__dict__ if v is not None else None) for k, v in proj.items()}
        (out / "analysis.json").write_text(
            json.dumps({**small, "classifier": classifier}, indent=1, default=str)
        )
        dropped = sorted(set(plan.ids) - set(res.get("tasks", [])))
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(
            report_banking2.build(
                res,
                info,
                classifier,
                summ,
                prior.total,
                budget,
                console_note_fn(summ["total"]),
                dropped,
                a.phase,
            )
        )
        if proj:
            print("\n".join(verdict_lines(proj["literal"])))
    return 0
