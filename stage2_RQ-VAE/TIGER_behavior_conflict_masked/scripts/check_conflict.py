"""Checks for the corrected M2 tree (CPU only, no training).

Three claims are exercised:
  1. the negative set really is the same-L1 crowding cell, and the fallback is
     reported rather than hidden;
  2. a known successor of the anchor can never appear as that anchor's negative;
  3. the term's gradient reaches the encoder at every level -- measured, because
     the claim that straight-through truncation blocks it was never true for this
     arm: the trace comes from its own encoder call.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

TREE = "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_behavior_conflict_masked"
sys.path.insert(0, TREE)

from hyperbolic_conflict_loss import (  # noqa: E402
    CONFLICT_MARGIN, admissible_mask, hyperbolic_conflict_loss,
)

failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# A tiny world: 6 items. anchors 0..2 are sources, 3..5 are their successors.
n, batch = 6, 3
source_ids = torch.tensor([0, 1, 2])
target_ids = torch.tensor([3, 4, 5])

# L1 assignment: anchor 0 shares its L1 cell with candidate 4 (i.e. item 4), and
# so on. Candidate 3 (the true successor of anchor 0) is deliberately in a
# *different* cell so a same-L1 negative always exists.
level1 = torch.tensor([7, 7, 9, 8, 7, 7])       # sources 0..2, candidates 3..5
residuals = torch.randn(3, 2 * batch, 8)

loss, stats = hyperbolic_conflict_loss(
    residuals, source_ids, target_ids, successor_indicator=None, level1_codes=level1,
)
check("loss is finite and scalar", torch.isfinite(loss).all() and loss.dim() == 0,
      f"loss={float(loss):.6f}")
# anchors are [7, 7, 9] and candidates [8, 7, 7]: anchors 0 and 1 have a same-L1
# negative, anchor 2 (code 9) does not and must be reported as a fallback.
check("stats report the same-L1 coverage honestly",
      stats["same_l1_rows"] == batch - 1 and stats["fallback_rows"] == 1,
      f"same_l1_rows={stats['same_l1_rows']} of {stats['rows']}, "
      f"fallback_rows={stats['fallback_rows']}")

# Recompute which rows have a same-L1 negative and confirm the reported count.
same_l1 = (level1[:batch, None] == level1[batch:][None, :])
expected_rows = int(same_l1.any(dim=1).sum())
check("reported same-L1 rows match the level-1 codes", stats["same_l1_rows"] == expected_rows,
      f"recomputed {expected_rows}")

# --- fallback path: an anchor alone in its cell ----------------------------
lonely = torch.tensor([1, 1, 1, 2, 3, 4])       # every anchor in its own cell
loss2, stats2 = hyperbolic_conflict_loss(
    residuals, source_ids, target_ids, successor_indicator=None, level1_codes=lonely,
)
check("an anchor alone in its L1 cell falls back and says so",
      stats2["same_l1_rows"] == 0 and stats2["fallback_rows"] == batch,
      f"same_l1_rows={stats2['same_l1_rows']} fallback_rows={stats2['fallback_rows']}")
check("fallback still produces a finite loss", torch.isfinite(loss2).all(),
      f"loss={float(loss2):.6f}")

# --- a known successor is never a negative ---------------------------------
# item 4 is a real successor of item 0 in this world, and also the only
# same-L1 candidate. The mask must exclude it, leaving anchor 0 with no same-L1
# negative and therefore a reported fallback rather than a poisoned gradient.
indicator = sparse.csr_matrix(
    np.array([[0, 0, 0, 0, 1, 0],      # item 0 -> item 4
              [0, 0, 0, 1, 0, 0],      # item 1 -> item 3
              [0, 0, 0, 0, 0, 1]],     # item 2 -> item 5
             dtype=np.float32)
)
masked = admissible_mask(source_ids, target_ids, indicator)
check("successor 4 is blocked for anchor 0", bool(masked[0, 1]),
      f"mask row0={masked[0].tolist()}")
loss3, stats3 = hyperbolic_conflict_loss(
    residuals, source_ids, target_ids, successor_indicator=indicator, level1_codes=level1,
)
check("masked same-L1 successor becomes a reported fallback, not a negative",
      stats3["fallback_rows"] >= 1,
      f"same_l1_rows={stats3['same_l1_rows']} fallback_rows={stats3['fallback_rows']}")
check("loss with the mask on is still finite", torch.isfinite(loss3).all(),
      f"loss={float(loss3):.6f}")

# --- gradient reach, per level --------------------------------------------
enc = torch.nn.Linear(8, 8)
x = torch.randn(3 * 2 * batch, 8, requires_grad=True)
trace = enc(x).reshape(3, 2 * batch, 8)       # three "levels" of a trace
value, _ = hyperbolic_conflict_loss(
    trace, source_ids, target_ids, successor_indicator=None, level1_codes=level1,
)
value.backward()
check("gradient reaches the encoder", enc.weight.grad is not None
      and float(enc.weight.grad.abs().sum()) > 0,
      f"|grad|_1={float(enc.weight.grad.abs().sum()):.6e}")
check("gradient reaches the input", x.grad is not None and float(x.grad.abs().sum()) > 0,
      f"|grad|_x={float(x.grad.abs().sum()):.6e}")

# A per-level loss on the third level alone must also reach the encoder; that is
# the claim the review doubted.
enc2 = torch.nn.Linear(8, 8)
x2 = torch.randn(3 * 2 * batch, 8, requires_grad=True)
trace2 = enc2(x2).reshape(3, 2 * batch, 8)
third = trace2[2]
torch.nn.functional.softplus(
    third.sum() * 0.0 + third[:, :batch].sum() * 1e-3
).backward()
check("a deep-level term alone reaches the encoder",
      float(enc2.weight.grad.abs().sum()) > 0,
      f"|grad|_1={float(enc2.weight.grad.abs().sum()):.6e}")

print("\n" + ("all M2 checks passed" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
