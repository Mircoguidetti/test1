"""Small statistics helpers: Wilson interval and a paired bootstrap for the savings ratio."""

from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def savings_ci(
    base: list[float], layer: list[float], iters: int = 2000, seed: int = 7
) -> tuple[float, float, float]:
    """Savings = 1 - sum(layer)/sum(base) over paired tasks, with a 95% bootstrap interval."""
    assert len(base) == len(layer) and base
    n = len(base)

    def ratio(idx: list[int]) -> float:
        b = sum(base[i] for i in idx)
        return 1 - sum(layer[i] for i in idx) / b if b else 0.0

    point = ratio(list(range(n)))
    rng = random.Random(seed)
    boots = sorted(ratio([rng.randrange(n) for _ in range(n)]) for _ in range(iters))
    return point, boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]


def ratio_ci(
    base: list[float], other: list[float], level: float = 0.8, iters: int = 10_000, seed: int = 7
) -> tuple[float, float, float]:
    """Cost ratio sum(other)/sum(base) over paired tasks, with a bootstrap interval over tasks."""
    assert len(base) == len(other) and base
    n = len(base)

    def ratio(idx: list[int]) -> float:
        b = sum(base[i] for i in idx)
        return sum(other[i] for i in idx) / b if b else float("nan")

    point = ratio(list(range(n)))
    rng = random.Random(seed)
    boots = sorted(ratio([rng.randrange(n) for _ in range(n)]) for _ in range(iters))
    lo = boots[int((1 - level) / 2 * iters)]
    hi = boots[min(iters - 1, int((1 + level) / 2 * iters) - 1)]
    return point, lo, hi
