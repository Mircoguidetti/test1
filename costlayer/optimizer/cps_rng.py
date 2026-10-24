"""The random draws shared by both implementations of the cost-per-success analysis.

Both implementations must see the same resamples, otherwise their intervals could never agree to
the cent. Only the draws are shared; every sum, ratio and percentile is computed separately.
"""

from __future__ import annotations

import random

RESAMPLES = 10_000
SHUFFLES = 200
SEED = 42


def resamples(n: int, b: int = RESAMPLES, seed: int = SEED) -> list[list[int]]:
    """`b` bootstrap samples of `n` task positions, drawn with replacement."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(b)]


def shuffles(n: int, k: int = SHUFFLES, seed: int = SEED) -> list[list[int]]:
    """`k` permutations of `range(n)` (used to shuffle task-type labels)."""
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        p = list(range(n))
        rng.shuffle(p)
        out.append(p)
    return out
