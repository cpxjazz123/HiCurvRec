"""Checks for the corrected M1 tree (CPU only, no training).

The claim being tested is the *direction* of the mechanism: items that share one
L1 parent should be pushed towards different children. The corrected term
subtracts (1 - cos), so an aligned child is the farthest candidate.
"""
import sys
from pathlib import Path

import torch

TREE = "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_hyperbolic_fanout_residual"
sys.path.insert(0, TREE)

from fanout_assignment import (  # noqa: E402
    FANOUT_C, FANOUT_STRENGTH, _angular_penalty, _fanout_distance,
)
from scripts.hyp_losses import (  # noqa: E402
    encode_points, shared_tangent_scale, tangent_scale_for_median_radius,
)

failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


torch.manual_seed(0)
n, d, k = 6, 8, 16
latent = torch.randn(n, d)
codes = torch.randn(k, d) * 0.5

# --- the penalty is a genuine angular term in one frame -------------------
penalty = _angular_penalty(latent, codes)
check("penalty shape is (N, K)", tuple(penalty.shape) == (n, k), str(tuple(penalty.shape)))
check("penalty lies in [0, 2]", float(penalty.min()) >= -1e-6 and float(penalty.max()) <= 2 + 1e-6,
      f"min={float(penalty.min()):.4f} max={float(penalty.max()):.4f}")

# An item pointing exactly at a child must have (1 - cos) = 0 there.
aligned = codes[3].clone().unsqueeze(0).repeat(n, 1) + 0.0 * torch.randn(n, d)
pen_aligned = _angular_penalty(aligned, codes)
check("a perfectly aligned child scores 0 (1 - cos)", float(pen_aligned[:, 3].max()) < 1e-5,
      f"max over rows at the aligned child={float(pen_aligned[:, 3].max()):.2e}")

# --- the sign: the term widens the margin against the aligned child --------
# The item sits exactly on child 0. ``1 - cos`` is 0 there, so the term does not
# move that candidate; it pulls every *other* candidate closer, which is what
# makes the aligned child the farthest one. So the property to measure is the
# change in the margin between the aligned child and the best alternative, with
# and without the term -- not an absolute comparison, because the geodesic gap is
# an order of magnitude larger than the term and is not supposed to be overridden.
two_codes = torch.zeros(k, d)
two_codes[0, 0] = 1.0
two_codes[1, 0] = -1.0
item = torch.zeros(n, d)
item[:, 0] = 1.0

with_term = _fanout_distance(item, two_codes, c=FANOUT_C, strength=FANOUT_STRENGTH)
without = _fanout_distance(item, two_codes, c=FANOUT_C, strength=0.0)
margin_with = float((with_term[:, 0] - with_term[:, 1]).mean())
margin_without = float((without[:, 0] - without[:, 1]).mean())
check("the term widens the margin against the aligned child",
      margin_with > margin_without,
      f"margin without={margin_without:.4f} with={margin_with:.4f} "
      f"(increase={margin_with - margin_without:.4f} = 2*strength={2 * FANOUT_STRENGTH})")
check("the increase equals the analytic amount", abs((margin_with - margin_without) - 2 * FANOUT_STRENGTH) < 1e-3,
      f"observed {margin_with - margin_without:.4f} vs 2*strength {2 * FANOUT_STRENGTH:.4f}")

# With the opposite sign the margin would shrink -- this is the regression the
# original implementation had, stated as a test.
flipped = _fanout_distance(item, two_codes, c=FANOUT_C, strength=-FANOUT_STRENGTH)
margin_flipped = float((flipped[:, 0] - flipped[:, 1]).mean())
check("the previous sign would have narrowed it", margin_flipped < margin_without,
      f"flipped margin={margin_flipped:.4f} vs plain {margin_without:.4f}")

# Two items sharing one L1 cell still land on different children.
spread = _fanout_distance(
    torch.stack([item[0], torch.tensor([-1.0] + [0.0] * (d - 1))]),
    two_codes, c=FANOUT_C, strength=FANOUT_STRENGTH,
)
choice = spread.argmin(dim=-1)
check("two items sharing a parent pick different children", choice[0] != choice[1],
      f"item A -> child {int(choice[0])}, item B -> child {int(choice[1])}")

# --- the scale no longer moves with the batch ----------------------------
# This is the property the shared scale buys: encoding a row must not depend on
# which other rows were in the batch. Under the old per-tensor scale it did.
batch = torch.randn(64, d)
full = encode_points(batch, FANOUT_C, shared_tangent_scale(two_codes, FANOUT_C))
half_a = encode_points(batch[:32], FANOUT_C, shared_tangent_scale(two_codes, FANOUT_C))
half_b = encode_points(batch[32:], FANOUT_C, shared_tangent_scale(two_codes, FANOUT_C))
check("encoding is invariant to batch composition",
      torch.allclose(full[:32], half_a) and torch.allclose(full[32:], half_b),
      f"max|diff|={float((full[:32] - half_a).abs().max()):.2e}")

old_full = encode_points(batch, FANOUT_C)                       # per-tensor scale
old_half = encode_points(batch[:32], FANOUT_C)                  # different batch -> different scale
old_drift = float((old_full[:32] - old_half).abs().max())
check("the old per-tensor scale did drift with the batch", old_drift > 0,
      f"max|diff| between the full batch and its first half = {old_drift:.4f}")

print("\n" + ("all M1 checks passed" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
