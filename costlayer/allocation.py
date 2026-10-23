"""Allocation rule for phases 2 and 3 (computed from measured costs, never guessed)."""

from __future__ import annotations

import math
from typing import Any

PHASE2_CAP, PHASE3_CAP = 9.00, 13.00
SHARE = 0.8
MIN_TRAIN, MIN_TEST = 12, 20
MAX_TRAIN, MAX_TEST = 74, 40
AA_RUNS = 10


def n_train(
    c: dict[str, float], configs: tuple[str, ...] = ("default", "E:medium", "E:low", "T:bt")
) -> int:
    per_task = sum(c[k] for k in configs)
    return min(MAX_TRAIN, math.floor(SHARE * PHASE2_CAP / per_task))


def n_test(c_default: float, c_global: float, c_layer: float) -> int:
    budget = SHARE * PHASE3_CAP - AA_RUNS * c_default
    return min(MAX_TEST, math.floor(budget / (c_default + c_global + c_layer)))


def allocate(c: dict[str, float]) -> dict[str, Any]:
    """`c` maps configuration -> measured mean cost per run (user simulator included)."""
    c_def = c["default"]
    c_glob = min(c_def, c["E:low"], c["T:bt"])  # stand-in for the sweep winner
    c_layer = c_def * 1.05  # stand-in: default plus 5% for the classifier
    res: dict[str, Any] = {
        "c": c,
        "c_global_standin": c_glob,
        "c_layer_standin": c_layer,
        "n_train_raw": n_train(c),
        "n_test_raw": n_test(c_def, c_glob, c_layer),
    }
    res["n_train"] = min(MAX_TRAIN, res["n_train_raw"])
    res["n_test"] = min(MAX_TEST, res["n_test_raw"])
    res["train_ok"] = res["n_train_raw"] >= MIN_TRAIN
    res["test_ok"] = res["n_test_raw"] >= MIN_TEST
    options: dict[str, Any] = {}
    if not res["train_ok"]:
        no_med = n_train(c, ("default", "E:low", "T:bt"))
        need = MIN_TRAIN * sum(c[k] for k in ("default", "E:medium", "E:low", "T:bt")) / SHARE
        options["phase2"] = {
            "A drop E:medium from the sweep": {"n_train": no_med, "meets_min": no_med >= MIN_TRAIN},
            "B lower the minimum": {"n_train": res["n_train_raw"]},
            "C move budget to phase 2": {
                "phase2_cap_needed": round(need, 2),
                "extra_vs_cap": round(need - PHASE2_CAP, 2),
            },
        }
    if not res["test_ok"]:
        per = c_def + c_glob + c_layer
        need = (MIN_TEST * per + AA_RUNS * c_def) / SHARE
        options["phase3"] = {
            "A lower the minimum": {"n_test": res["n_test_raw"]},
            "B A/A run on 5 tasks instead of 10 (needs a plan change)": {
                "n_test": math.floor((SHARE * PHASE3_CAP - 5 * c_def) / per)
            },
            "C move budget to phase 3": {
                "phase3_cap_needed": round(need, 2),
                "extra_vs_cap": round(need - PHASE3_CAP, 2),
            },
        }
    res["options"] = options
    return res
