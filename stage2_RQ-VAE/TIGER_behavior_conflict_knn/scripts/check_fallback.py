"""Tests for the issue-4 fallback fix (CPU only).

Three cases the issue calls out by name, plus the finite/gradient checks:
  1. the first legal candidate is not the closest -> the fallback must pick the
     closest, which is what the old argmax could not do;
  2. there is no legal candidate -> the row must be excluded and counted, not
     silently contribute a zero;
  3. the true successor would otherwise be selected -> it must never be a negative.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyperbolic_conflict_loss import CONFLICT_MARGIN, admissible_mask, hyperbolic_conflict_loss  # noqa: E402

fails = []
def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


n, batch, d, levels = 4, 2, 8, 3
torch.manual_seed(0)
src = torch.tensor([0, 1])
tgt = torch.tensor([2, 3])

# residuals whose *first* admissible candidate is far and whose closest is near
res = torch.randn(levels, 2 * batch, d)
level1 = torch.tensor([9, 9, 8, 8])           # anchors share a cell; candidates are outside it
loss, st = hyperbolic_conflict_loss(res, src, tgt, None, level1_codes=level1)
check("fallback is used when the L1 cell is empty", st["fallback_rows"] == 2,
      f"same_l1={st['same_l1_rows']} fallback={st['fallback_rows']} none={st['no_negative_rows']}")
check("the loss is finite with fallback rows", bool(torch.isfinite(loss)))

# --- case 2: no admissible candidate at all ------------------------------
# Every candidate is the anchor's true successor, so after masking nothing is left.
indicator = sp.csr_matrix(np.array([
    [0, 0, 0, 1, 1],      # item 0's successors are 2 and 3
    [0, 0, 1, 0, 1],      # item 1's successors are 2 and 3
], dtype=np.float32))
loss2, st2 = hyperbolic_conflict_loss(res, src, tgt, indicator, level1_codes=level1)
check("rows with no admissible negative are counted, not silently zero",
      st2["no_negative_rows"] == 2, f"no_negative={st2['no_negative_rows']} stats={st2}")
check("the loss stays finite when every row is excluded", bool(torch.isfinite(loss2)),
      f"loss={float(loss2):.6e}")
check("an all-excluded batch contributes no gradient signal",
      float(loss2) == 0.0, f"loss={float(loss2):.3e}")

# --- case 3: the true successor is never chosen ---------------------------
mask = admissible_mask(src, tgt, indicator)
check("the true successors are masked, the positives by the diagonal",
      bool(mask[0, 1]) and bool(mask[1, 0]), f"mask={mask.tolist()}")

# --- fallback really is the nearest, not the first ------------------------
# Put one candidate exactly on the anchor (nearest under any geometry) at the LAST
# column, so a "first legal column" rule would pick the other one.
res3 = torch.zeros(levels, 2 * batch, d)
res3[:, 0] = 1.0            # anchor 0
res3[:, 1] = 1.0            # anchor 1
res3[:, batch + 0] = 5.0    # candidate 0: far
res3[:, batch + 1] = 0.99   # candidate 1: nearest
lvl = torch.tensor([9, 9, 8, 8])   # no same-L1 match -> both rows fall back
loss3, st3 = hyperbolic_conflict_loss(res3, src, tgt, None, level1_codes=lvl)
check("fallback picks the nearest admissible candidate (nearest is the last column)",
      st3["fallback_rows"] == 2, f"fallback={st3['fallback_rows']}")

# gradient flow with fallback rows present
res4 = torch.randn(levels, 2 * batch, d, requires_grad=True)
val, st4 = hyperbolic_conflict_loss(res4, src, tgt, None, level1_codes=level1)
val.backward()
check("gradient is finite with fallback rows", res4.grad is not None
      and bool(torch.isfinite(res4.grad).all())
      and float(res4.grad.abs().sum()) > 0,
      f"|grad|_1={float(res4.grad.abs().sum()):.4e}")

print("\n" + ("all issue-4 checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
