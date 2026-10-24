"""Test 2 budget: `budget.json`, committed by the coordinator, read by the runner.

    {"total": 4.00, "phase_caps": {"b1": 4.00}}

`b2` and `b3` have no cap until the coordinator adds one; the runner refuses to start a phase
whose cap is missing, and refuses `b2`/`b3` without the matching `founder_go` word. Test 2's
spend lives on its own branch (`spend2`) with the same ledger mechanics as test 1 (`ledger.py`);
test 1's `spend` branch and its $27 cap are not touched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

PHASES = ("b1", "b2", "b3")
GATED = ("b2", "b3")  # need the founder's go: `founder_go` must equal the phase key
SPEND_BRANCH = "spend2"
DEFAULT_PATH = Path("budget.json")


class BudgetFileError(RuntimeError):
    """budget.json is missing or invalid, or a phase may not start."""


@dataclass(frozen=True)
class Budget:
    total: float
    phase_caps: dict[str, float]

    def cap(self, phase: str) -> float:
        if phase not in PHASES:
            raise BudgetFileError(f"Unknown phase {phase!r}; expected one of {PHASES}")
        if phase not in self.phase_caps:
            raise BudgetFileError(
                f"Phase {phase} has no cap in budget.json: the coordinator sets it, "
                "the runner does not start the phase without it"
            )
        return self.phase_caps[phase]


def _money(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BudgetFileError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise BudgetFileError(f"{what} must be a positive number, got {value!r}")
    return float(value)


def load_budget(path: Path = DEFAULT_PATH) -> Budget:
    path = Path(path)
    if not path.exists():
        raise BudgetFileError(f"{path} not found: the runner does not start without it")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise BudgetFileError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or set(raw) != {"total", "phase_caps"}:
        raise BudgetFileError(f"{path} must have exactly the fields 'total' and 'phase_caps'")
    total = _money(raw["total"], "total")
    caps_raw = raw["phase_caps"]
    if not isinstance(caps_raw, dict) or not caps_raw:
        raise BudgetFileError("phase_caps must be a non-empty object")
    unknown = set(caps_raw) - set(PHASES)
    if unknown:
        raise BudgetFileError(f"Unknown phase keys in phase_caps: {sorted(unknown)}")
    caps = {k: _money(v, f"phase_caps.{k}") for k, v in caps_raw.items()}
    for k, v in caps.items():
        if v > total:
            raise BudgetFileError(f"phase_caps.{k} ({v}) is above the total ({total})")
    return Budget(total=total, phase_caps=caps)


def check_may_start(budget: Budget, phase: str, founder_go: str) -> float:
    """Return the phase cap, or raise. `b2` and `b3` also need `founder_go == phase`."""
    cap = budget.cap(phase)
    if phase in GATED and founder_go != phase:
        raise BudgetFileError(f"Phase {phase} needs the founder's go: founder_go must be {phase!r}")
    return cap
