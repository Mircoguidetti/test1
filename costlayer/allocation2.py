"""Test 2 budget request (TEST2.md, phase 1 step 4). Computed from measured costs, never guessed.

    Sigma    = c(default) + c(E:low) + c(C:top3) + c(EC)
    T_needed = 1.1 x (4.0 + 20 x Sigma / 0.8
                      + (10 x c(default) + 30 x (c(default) + c(global*) + c(layer))) / 0.8)

with c(global*) = the cheapest of the four and c(layer) = c(default) as stand-ins. `c(x)` is the
mean cost per run, user simulator included. The 4.0 is phase 1's cap; 1.1 is the reserve factor;
20 and 30 are the minimum N_train and N_test; N_train + N_test is at most 88.
"""

from __future__ import annotations

import math
from typing import Any

PHASE1_CAP = 4.0
RESERVE = 1.1
SHARE = 0.8
AA_RUNS = 10
MIN_TRAIN, MIN_TEST, MAX_TASKS = 20, 30, 88
EPS = 1e-9  # floating-point guard: exactly T_needed must buy exactly the minimums
CONFIGS = ("default", "E:low", "C:top3", "EC")


def sigma(c: dict[str, float], configs: tuple[str, ...] = CONFIGS) -> float:
    return sum(c[k] for k in configs)


def _sum3(c: dict[str, float], configs: tuple[str, ...]) -> float:
    """c(default) + c(global*) + c(layer), with the stand-ins of the plan."""
    return c["default"] + min(c[k] for k in configs) + c["default"]


def total_for(
    c: dict[str, float],
    n_train: int,
    n_test: int,
    configs: tuple[str, ...] = CONFIGS,
) -> float:
    p2 = n_train * sigma(c, configs) / SHARE
    p3 = (AA_RUNS * c["default"] + n_test * _sum3(c, configs)) / SHARE
    return RESERVE * (PHASE1_CAP + p2 + p3)


def t_needed(c: dict[str, float], configs: tuple[str, ...] = CONFIGS) -> float:
    return total_for(c, MIN_TRAIN, MIN_TEST, configs)


def affordable(
    c: dict[str, float], total: float, configs: tuple[str, ...] = CONFIGS
) -> dict[str, Any]:
    """What a given total buys: N_train with N_test held at its minimum, and the other way round."""
    usable = total / RESERVE - PHASE1_CAP  # left for phases 2 and 3 before the 80% share
    p3_min = (AA_RUNS * c["default"] + MIN_TEST * _sum3(c, configs)) / SHARE
    p2_min = MIN_TRAIN * sigma(c, configs) / SHARE
    n_train = (
        math.floor((usable - p3_min) * SHARE / sigma(c, configs) + EPS) if usable > p3_min else 0
    )
    n_test = (
        math.floor(((usable - p2_min) * SHARE - AA_RUNS * c["default"]) / _sum3(c, configs) + EPS)
        if usable > p2_min
        else 0
    )
    n_train = max(0, min(n_train, MAX_TASKS - MIN_TEST))
    n_test = max(0, min(n_test, MAX_TASKS - MIN_TRAIN))
    return {
        "total": round(total, 2),
        "n_train_at_min_test": n_train,
        "n_test_at_min_train": n_test,
        "meets_both_minimums": total >= t_needed(c, configs) - 1e-9,
    }


def request(c: dict[str, float]) -> dict[str, Any]:
    """The full budget request for a dict of measured c(x) (all four configurations)."""
    missing = [k for k in CONFIGS if k not in c]
    if missing:
        return {"error": f"no successful Sonnet run for {missing}"}
    need = t_needed(c)
    sets = {"all four": CONFIGS, "without EC": ("default", "E:low", "C:top3")}
    options = []
    for name, cfgs in sets.items():
        t_set = t_needed(c, cfgs)
        for label, total in (
            ("T_needed", t_set),
            ("80% of T_needed", 0.8 * t_set),
            ("60% of T_needed", 0.6 * t_set),
        ):
            options.append({"configs": name, "at": label, **affordable(c, total, cfgs)})
    return {
        "c": c,
        "sigma": sigma(c),
        "c_global_standin": min(c[k] for k in CONFIGS),
        "c_layer_standin": c["default"],
        "t_needed": need,
        "n_train": MIN_TRAIN,
        "n_test": MIN_TEST,
        "tasks_ok": MIN_TRAIN + MIN_TEST <= MAX_TASKS,
        "t_needed_without_ec": t_needed(c, sets["without EC"]),
        "options": options,
    }
