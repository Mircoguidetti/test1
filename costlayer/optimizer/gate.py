"""The gate of TEST2.md phase 2: an optimistic projection of the best layer the data allows.

    projected = cost(global*) x (1 - chain_cost_share)
    or, if the out-of-sample headroom beats the null,
    projected = cost(global*) x (1 - headroom)            (the lower of the two is taken)

Costs are agent costs summed over the paired train tasks (user simulator excluded) and the
classifier's overhead counts as zero. The verdict is then:

    projected <= 0.70 x default and <= 0.85 x global*   -> GO TO PHASE 3
    projected <= 0.70 x default, above 0.85 x global*   -> FEATURE ONLY
    projected above 0.70 x default                      -> KILL

The thresholds are fixed in TEST2.md; do not tune them.
"""

from __future__ import annotations

from dataclasses import dataclass

EPS = 1e-9
K2_RATIO = 0.70  # vs default
K1_RATIO = 0.85  # vs global*
GO = "GO TO PHASE 3"
FEATURE_ONLY = "FEATURE ONLY"
KILL = "KILL"


@dataclass(frozen=True)
class Projection:
    cost_default: float
    cost_global: float
    chain_cost_share: float
    headroom: float
    headroom_beats_null: bool
    projected: float
    basis: str  # which term gave the lower projection
    ratio_default: float
    ratio_global: float
    verdict: str


def project(
    cost_default: float,
    cost_global: float,
    chain_cost_share: float,
    headroom: float,
    headroom_beats_null: bool,
) -> Projection:
    if cost_default <= 0 or cost_global <= 0:
        raise ValueError("costs must be positive")
    share = min(max(chain_cost_share, 0.0), 1.0)
    candidates = {"chains": cost_global * (1 - share)}
    if headroom_beats_null:
        candidates["per-type headroom"] = cost_global * (1 - min(headroom, 1.0))
    basis = min(candidates, key=lambda k: candidates[k])
    projected = candidates[basis]
    r_def, r_glob = projected / cost_default, projected / cost_global
    # the bars are inclusive ("<="); a tiny tolerance keeps float noise from deciding a boundary
    if r_def > K2_RATIO + EPS:
        verdict = KILL
    elif r_glob > K1_RATIO + EPS:
        verdict = FEATURE_ONLY
    else:
        verdict = GO
    return Projection(
        cost_default, cost_global, share, headroom, headroom_beats_null, projected, basis,
        r_def, r_glob, verdict,
    )  # fmt: skip


def verdict_lines(p: Projection) -> list[str]:
    """The literal verdict and the three numbers."""
    return [
        f"VERDICT: {p.verdict}",
        f"projected cost: ${p.projected:.4f} (agent cost over the paired train tasks; "
        f"basis: {p.basis})",
        f"projected / default: {p.ratio_default:.3f}   (K2 bar {K2_RATIO:.2f})",
        f"projected / global*: {p.ratio_global:.3f}   (K1 bar {K1_RATIO:.2f})",
    ]
