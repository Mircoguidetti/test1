"""Append-only cost ledger with hard budget caps."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from costlayer.pricing import Usage, cost_usd

TOTAL_CAP_USD = 27.0
PHASE_CAPS_USD = {0: 0.0, 1: 3.0, 2: 7.0, 3: 15.0}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    """Every API call (agent, user simulator, classifier) is written here.

    `reserve` must be called before each call: it aborts when spent + in-flight reservations
    + a conservative estimate for the call would exceed the phase cap or the total cap.
    """

    path: Path
    phase: int
    total_cap: float = TOTAL_CAP_USD
    phase_cap: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _inflight: float = 0.0
    _spent_total: float = 0.0
    _spent_phase: float = 0.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.phase_cap is None:
            self.phase_cap = PHASE_CAPS_USD[self.phase]
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._spent_total += row["cost"]
                if row["phase"] == self.phase:
                    self._spent_phase += row["cost"]

    @property
    def spent_total(self) -> float:
        return self._spent_total

    @property
    def spent_phase(self) -> float:
        return self._spent_phase

    def reserve(self, estimate: float) -> float:
        assert self.phase_cap is not None
        with self._lock:
            projected_total = self._spent_total + self._inflight + estimate
            projected_phase = self._spent_phase + self._inflight + estimate
            if projected_total > self.total_cap:
                raise BudgetExceeded(
                    f"total cap: spent {self._spent_total:.4f} + est {estimate:.4f} "
                    f"> {self.total_cap}"
                )
            if projected_phase > self.phase_cap:
                raise BudgetExceeded(
                    f"phase {self.phase} cap: spent {self._spent_phase:.4f} + est "
                    f"{estimate:.4f} > {self.phase_cap}"
                )
            self._inflight += estimate
            return estimate

    def release(self, estimate: float) -> None:
        with self._lock:
            self._inflight = max(0.0, self._inflight - estimate)

    def record(
        self, *, estimate: float, run_id: str, purpose: str, model: str, usage: Usage
    ) -> float:
        cost = cost_usd(model, usage)
        row = {
            "ts": round(time.time(), 2),
            "phase": self.phase,
            "run": run_id,
            "purpose": purpose,
            "model": model,
            **usage.as_dict(),
            "cost": round(cost, 8),
        }
        with self._lock:
            self._inflight = max(0.0, self._inflight - estimate)
            self._spent_total += cost
            self._spent_phase += cost
            with self.path.open("a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        return cost


def summarize(path: Path) -> dict:
    """Totals by phase, purpose and model (committed as a small JSON summary)."""
    out: dict = {"total": 0.0, "calls": 0, "by_phase": {}, "by_purpose": {}, "by_model": {}}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out["total"] += r["cost"]
        out["calls"] += 1
        for key, val in (
            ("by_phase", str(r["phase"])),
            ("by_purpose", r["purpose"]),
            ("by_model", r["model"]),
        ):
            out[key][val] = out[key].get(val, 0.0) + r["cost"]
    out["total"] = round(out["total"], 6)
    for key in ("by_phase", "by_purpose", "by_model"):
        out[key] = {k: round(v, 6) for k, v in out[key].items()}
    return out
