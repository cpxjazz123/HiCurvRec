"""The per-example assignment audit: does the mechanism's term change the DECISION?

The JOIN R review of issue #115 said the two candidate mechanisms were UNMEASURABLE: the
repository could report partition health and task metrics, but not, per example, how much the
mechanism's own term contributed to the winner/runner-up gap, nor whether removing it would have
chosen a different code. A mechanism whose term never flips a decision is a no-op dressed as
mathematics, and the only honest way to say so is to measure the flips.

Two matrices of the SAME shape, on the SAME examples, with every other term identical:

    full      the score the mechanism actually uses (lower is better, argmin chooses)
    ablated   the same score with the mechanism's term zeroed

The audit reports, per example and in aggregate: the chosen code under both, whether it changed,
the full-score winner/runner-up gap, and the gap the mechanism's term contributes. The
pre-registered decision criterion is the FLIP RATE, not a variance or a synthetic witness.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

__all__ = ["audit", "GATE"]

#: A mechanism whose term flips fewer than this fraction of decisions, in any stage, has not been
#: shown to do anything. Fixed before any run.
GATE = {"min_flip_rate": 0.05, "max_tie_share": 0.10}


def audit(full: np.ndarray, ablated: np.ndarray, *,
          min_flip_rate: float = GATE["min_flip_rate"],
          max_tie_share: float = GATE["max_tie_share"]) -> Dict[str, Any]:
    """Compare the decisions of ``full`` and ``ablated`` per example.

    ``full`` and ``ablated`` are (n_examples, n_codes) score matrices; the lowest score wins.
    Returns the flip rate, the tie share (a zero winner/runner-up gap makes a flip meaningless),
    the distribution of the mechanism's own contribution to the gap, and the verdict.
    """
    full = np.asarray(full, dtype=np.float64)
    ablated = np.asarray(ablated, dtype=np.float64)
    if full.shape != ablated.shape or full.ndim != 2:
        raise ValueError(f"the two score matrices must be (n, codes) and equal, got "
                         f"{full.shape} and {ablated.shape}")
    chosen_full = full.argmin(axis=1)
    chosen_ablated = ablated.argmin(axis=1)
    flipped = chosen_full != chosen_ablated

    ordered = np.sort(full, axis=1)
    gap_full = ordered[:, 1] - ordered[:, 0] if full.shape[1] > 1 else np.zeros(len(full))
    ordered_ab = np.sort(ablated, axis=1)
    gap_ab = ordered_ab[:, 1] - ordered_ab[:, 0] if ablated.shape[1] > 1 else np.zeros(len(ablated))
    ties = gap_full <= 1e-12
    scale = float(np.median(np.abs(full))) or 1.0
    contribution = (gap_full - gap_ab) / scale

    flip_rate = float(flipped.mean()) if len(full) else 0.0
    tie_share = float(ties.mean()) if len(full) else 0.0
    failures = []
    if flip_rate < float(min_flip_rate):
        failures.append(f"the mechanism's term flips {flip_rate:.4f} of the decisions, below the "
                        f"pre-registered {min_flip_rate}: it is a no-op")
    if tie_share > float(max_tie_share):
        failures.append(f"{tie_share:.4f} of the examples are ties, above the allowed "
                        f"{max_tie_share}: the comparison cannot decide them")
    return {
        "examples": int(len(full)),
        "codes": int(full.shape[1]),
        "flip_rate": round(flip_rate, 6),
        "tie_share": round(tie_share, 6),
        "changed_codes_per_stage": int(flipped.sum()),
        "gap_full_median": float(np.median(gap_full)) if len(full) else 0.0,
        "gap_ablated_median": float(np.median(gap_ab)) if len(ablated) else 0.0,
        "contribution_median_scaled": round(float(np.median(contribution)), 6) if len(full) else 0.0,
        "flip_rate_by_decile": [round(float(flipped[i::10].mean()), 4) for i in range(10)],
        "min_flip_rate": float(min_flip_rate),
        "max_tie_share": float(max_tie_share),
        "ok": not failures,
        "failures": failures,
        "verdict": ("the mechanism's term changes the decision" if not failures
                    else "the mechanism's term is not shown to change any decision: "
                         + "; ".join(failures)),
    }
