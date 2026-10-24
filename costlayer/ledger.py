"""Spend ledger with hard caps, cumulative across workflow dispatches.

Layout (the `spend` branch, checked out into a directory):
  start-<dispatch id>.json   written when a dispatch begins; records the reserved phase cap
  ledger-<dispatch id>.jsonl one row per API call (committed by an `if: always()` step)
A start file without a matching ledger counts as spending its whole reserved cap.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from costlayer.context import RunContext
from costlayer.pricing import Usage, cost_usd

TOTAL_CAP_USD = 27.0
PHASE_CAPS_USD = {1: 2.50, 2: 9.00, 3: 13.00}  # plus a $2.50 reserve that needs a decision


class BudgetExceeded(RuntimeError):
    pass


Phase = int | str  # test 1 uses 1, 2, 3; test 2 uses "b1", "b2", "b3" (its own `spend2` branch)


@dataclass
class Prior:
    total: float = 0.0
    by_phase: dict[Phase, float] = field(default_factory=dict)


def read_prior(spend_dir: Path, exclude: str | None = None) -> Prior:
    """Spend of all earlier dispatches. Unfinished dispatches count at their reserved cap."""
    prior = Prior()
    spend_dir = Path(spend_dir)
    if not spend_dir.exists():
        return prior
    starts = {
        p.stem.removeprefix("start-"): json.loads(p.read_text())
        for p in spend_dir.glob("start-*.json")
    }
    ledgers = {p.stem.removeprefix("ledger-"): p for p in spend_dir.glob("ledger-*.jsonl")}
    for run_id in set(starts) | set(ledgers):
        if run_id == exclude:
            continue
        if run_id in ledgers:
            cost = 0.0
            phase = starts.get(run_id, {}).get("phase", 0)
            for line in ledgers[run_id].read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    cost += row["cost"]
                    phase = row.get("phase", phase)
        else:
            cost, phase = starts[run_id]["cap"], starts[run_id]["phase"]
        prior.total += cost
        prior.by_phase[phase] = prior.by_phase.get(phase, 0.0) + cost
    return prior


def write_start(spend_dir: Path, dispatch_id: str, phase: Phase, cap: float) -> None:
    Path(spend_dir).mkdir(parents=True, exist_ok=True)
    (Path(spend_dir) / f"start-{dispatch_id}.json").write_text(
        json.dumps({"phase": phase, "cap": cap, "ts": round(time.time())}) + "\n"
    )


def plan_dispatch(
    spend_dir: Path,
    dispatch_id: str,
    phase: Phase,
    *,
    phase_caps: Mapping[Phase, float] | None = None,
    total_cap: float | None = None,
) -> float:
    """Return the cap this dispatch may spend, or raise if the total cap would be passed.

    Defaults are test 1's ($27 and its phase caps). Test 2 passes the caps from `budget.json`.
    """
    caps = PHASE_CAPS_USD if phase_caps is None else phase_caps
    total = TOTAL_CAP_USD if total_cap is None else total_cap
    if phase not in caps:
        raise BudgetExceeded(f"phase {phase!r} has no cap")
    prior = read_prior(spend_dir, exclude=dispatch_id)
    remaining_phase = caps[phase] - prior.by_phase.get(phase, 0.0)
    if remaining_phase <= 0:
        raise BudgetExceeded(f"phase {phase} cap already used: {prior.by_phase.get(phase)}")
    if prior.total + remaining_phase > total:
        raise BudgetExceeded(
            f"prior spend ${prior.total:.2f} + phase budget ${remaining_phase:.2f} "
            f"would pass the ${total} cap"
        )
    return remaining_phase


class Ledger:
    """Every API call goes through `reserve` (before) and `record` (after)."""

    def __init__(
        self,
        path: Path,
        dispatch_id: str,
        phase: Phase,
        phase_cap: float,
        prior_total: float = 0.0,
        total_cap: float = TOTAL_CAP_USD,
    ) -> None:
        self.path = Path(path)
        self.dispatch_id = dispatch_id
        self.phase = phase
        self.phase_cap = phase_cap
        self.prior_total = prior_total
        self.total_cap = total_cap
        self._lock = threading.Lock()
        self._inflight = 0.0
        self.spent = 0.0  # this dispatch
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(self, estimate: float) -> float:
        with self._lock:
            projected = self.spent + self._inflight + estimate
            if projected > self.phase_cap:
                raise BudgetExceeded(
                    f"phase {self.phase} cap: spent {self.spent:.4f} + in flight "
                    f"{self._inflight:.4f} + worst case {estimate:.4f} > {self.phase_cap:.2f}"
                )
            if self.prior_total + projected > self.total_cap:
                raise BudgetExceeded(
                    f"total cap: prior {self.prior_total:.4f} + {projected:.4f} "
                    f"> {self.total_cap:.2f}"
                )
            self._inflight += estimate
            return estimate

    def release(self, estimate: float) -> None:
        with self._lock:
            self._inflight = max(0.0, self._inflight - estimate)

    def record(
        self,
        *,
        estimate: float,
        ctx: RunContext | None,
        purpose: str,
        model: str,
        usage: Usage | None,
        raw_usage: dict[str, Any] | None = None,
        status: str = "ok",
        error: str | None = None,
        role: str | None = None,
    ) -> float:
        """Write one row. A call with no response (usage None) costs 0."""
        cost = cost_usd(model, usage) if usage else 0.0
        row = {
            "ts": round(time.time(), 2),
            "dispatch": self.dispatch_id,
            "phase": self.phase,
            **(ctx.tag() if ctx else {}),
            "purpose": purpose,
            "model": model,
            **(usage.as_dict() if usage else Usage().as_dict()),
            "raw_usage": raw_usage,
            "cost": round(cost, 8),
            "status": status,
            "error": error[:300] if error else None,
        }
        with self._lock:
            self._inflight = max(0.0, self._inflight - estimate)
            self.spent += cost
            with self.path.open("a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        if ctx is not None:
            ctx.spent += cost
            if purpose == "agent":
                ctx.spent_agent += cost
            else:
                ctx.spent_user += cost
        return cost


def summarize(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total": 0.0,
        "calls": 0,
        "by_purpose": {},
        "by_model": {},
        "by_config": {},
    }
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out["total"] += r["cost"]
            out["calls"] += 1
            for key, val in (
                ("by_purpose", r["purpose"]),
                ("by_model", r["model"]),
                ("by_config", r.get("config", "-")),
            ):
                out[key][val] = out[key].get(val, 0.0) + r["cost"]
    out["total"] = round(out["total"], 6)
    for key in ("by_purpose", "by_model", "by_config"):
        out[key] = {k: round(v, 6) for k, v in out[key].items()}
    return out
