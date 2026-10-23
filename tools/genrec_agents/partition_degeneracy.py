"""The partition-degeneracy gate: does a level's assignment actually partition the catalogue?

Why this exists (issue #112, the approved candidate's cold run of 2026-10-23): the candidate
`hyp_product_l1_quantiser` passed the CGV's six gates, the H gates, the join and the
implementation-parity check, and then produced a level-0 partition with **one** distinct code for
all 24587 items. Its ER collapsed to 1.0 against the control's 16.3345.

The cause was measurable: the level-0 distance is dominated by the codeword's radius in the
Poincare ball, so the codeword with the smallest row norm (0.29712 against a mean of 2.907) is the
argmin for every item. The assignment was a norm comparison, not a geometry comparison. Equalising
the codebook rows' norms recovered only 9 distinct argmins out of 256, so the degeneracy is the
mechanism's, not one unlucky codeword.

None of the existing gates tested this property: they tested that the forward pass is finite, that
the gradients reach the declared parameters, that the update matches the declared optimiser, that
the DDP aggregation agrees, and that the real entry point moves parameters. None of them asked
whether the level's assignment *distinguishes* the catalogue.

The gate below is that missing question, stated as a refusal:

* a level must use more than ``min_distinct`` codes (a partition, not a constant);
* no single cell may hold more than ``max_cell_share`` of the catalogue.

Both are reported with the numbers behind them, so a refusal says which one failed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["assess", "GATE"]

#: The thresholds. A level that uses fewer than 16 of its 256 codes, or whose largest cell holds
#: more than a quarter of the catalogue, is not a partition.
GATE = {"min_distinct": 16, "max_cell_share": 0.25}


def assess(sids: np.ndarray, *, level: int = 0, codebook_size: int = 256,
           min_distinct: int = GATE["min_distinct"],
           max_cell_share: float = GATE["max_cell_share"]) -> dict[str, Any]:
    """Assess one level's partition. ``sids`` is an (n_items, n_levels+) integer array."""
    matrix = np.asarray(sids)
    if matrix.ndim != 2:
        raise ValueError(f"the SID matrix must be two-dimensional, got {matrix.shape}")
    if not 0 <= level < matrix.shape[1]:
        raise ValueError(f"level {level} is outside the matrix with {matrix.shape[1]} columns")
    column = matrix[:, level]
    distinct = int(np.unique(column).size)
    counts = np.bincount(column.astype(np.int64), minlength=int(codebook_size))
    largest_cell = int(counts.max()) if counts.size else 0
    share = float(largest_cell / len(column)) if len(column) else 0.0
    occupied = int((counts > 0).sum())
    failures = []
    if distinct < min_distinct:
        failures.append(f"the level uses {distinct} distinct codes, fewer than the required {min_distinct}")
    if share > max_cell_share:
        failures.append(
            f"the largest cell holds {share:.4f} of the catalogue, more than the allowed {max_cell_share}")
    return {
        "level": int(level),
        "items": int(len(column)),
        "distinct_codes": distinct,
        "occupied_codes": occupied,
        "codebook_size": int(codebook_size),
        "largest_cell": largest_cell,
        "largest_cell_share": round(share, 6),
        "min_distinct": int(min_distinct),
        "max_cell_share": float(max_cell_share),
        "ok": not failures,
        "failures": failures,
        "verdict": ("the level partitions the catalogue" if not failures
                    else "the level does not partition the catalogue: " + "; ".join(failures)),
    }
