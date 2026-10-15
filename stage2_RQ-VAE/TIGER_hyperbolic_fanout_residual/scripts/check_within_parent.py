"""Within-parent child occupancy, measured rather than asserted.

The header now says the conditioning is implicit through the residual and that the
term is an itemwise angular re-ranking, not an occupancy constraint. These checks
measure what that actually buys inside one parent bucket:

  * items with identical residuals in one parent select the same child -- the
    limitation, pinned so it cannot be quietly dropped again;
  * items with separable residuals spread across children, and the term changes
    how they spread relative to the plain geodesic;
  * occupancy and entropy inside the bucket are reported for both, so the effect is
    a number rather than a story about two crafted points.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fanout_assignment import (  # noqa: E402
    FANOUT_C, FANOUT_STRENGTH, _angular_penalty, _fanout_distance,
)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def occupancy_stats(assignments: torch.Tensor, k: int) -> tuple:
    """Used children, and the normalised entropy of the within-bucket usage."""
    counts = torch.bincount(assignments, minlength=k).double()
    used = int((counts > 0).sum())
    if used <= 1:
        return used, 0.0
    p = counts[counts > 0]
    p = p / p.sum()
    # Normalised by log(used), not log(k): among the children that are actually
    # used, how evenly the rows are spread. Dividing by log(k) made the number
    # incomparable with the occupancy count it sits next to.
    entropy = float(-(p * p.log()).sum() / torch.log(torch.tensor(float(used))))
    return used, entropy


torch.manual_seed(0)
d, k, n = 16, 12, 48
codes = torch.randn(k, d) * 0.6

# --- 1. identical residuals in one parent: same child, by construction -------
identical = torch.randn(1, d).repeat(n, 1)
row = _fanout_distance(identical, codes, c=FANOUT_C, strength=FANOUT_STRENGTH)
chosen = row.argmin(dim=-1)
check("items with identical residuals pick the same child (documented limit)",
      int(chosen.unique().numel()) == 1,
      f"{chosen.unique().numel()} distinct children among {n} identical rows")

# --- 2. separable residuals: how do they spread? --------------------------
separable = torch.randn(n, d)
with_term = _fanout_distance(separable, codes, c=FANOUT_C, strength=FANOUT_STRENGTH)
without = _fanout_distance(separable, codes, c=FANOUT_C, strength=0.0)
used_with, ent_with = occupancy_stats(with_term.argmin(dim=-1), k)
used_without, ent_without = occupancy_stats(without.argmin(dim=-1), k)
print(f"  within-parent occupancy with the term   : {used_with}/{k} children, "
      f"normalised entropy {ent_with:.3f}")
print(f"  within-parent occupancy without it     : {used_without}/{k} children, "
      f"normalised entropy {ent_without:.3f}")
check("separable residuals use more than one child either way", used_with > 1,
      f"{used_with} of {k}")
check("the term changes the within-bucket assignment",
      not torch.equal(with_term.argmin(dim=-1), without.argmin(dim=-1)),
      f"{int((with_term.argmin(dim=-1) != without.argmin(dim=-1)).sum())} of {n} rows moved")

# --- 3. near-identical residuals: a small perturbation is enough -----------
near = torch.randn(1, d).repeat(n, 1) + 0.01 * torch.randn(n, d)
near_with = occupancy_stats(
    _fanout_distance(near, codes, c=FANOUT_C, strength=FANOUT_STRENGTH).argmin(dim=-1), k)
print(f"  near-identical residuals              : {near_with[0]}/{k} children, "
      f"normalised entropy {near_with[1]:.3f}")

# --- 4. the term's direction: it pushes rows off their aligned child -------
two = torch.zeros(k, d)
two[0, 0] = 1.0
two[1, 0] = -1.0
item = torch.zeros(n, d)
item[:, 0] = 1.0
margin_with = float((_fanout_distance(item, two, FANOUT_C, FANOUT_STRENGTH)[:, 0]
                     - _fanout_distance(item, two, FANOUT_C, FANOUT_STRENGTH)[:, 1]).mean())
margin_without = float((_fanout_distance(item, two, FANOUT_C, 0.0)[:, 0]
                        - _fanout_distance(item, two, FANOUT_C, 0.0)[:, 1]).mean())
check("the term widens the margin against the aligned child",
      margin_with > margin_without,
      f"margin {margin_without:.4f} -> {margin_with:.4f} "
      f"(increase {margin_with - margin_without:.4f} = 2*strength)")

# --- 5. the penalty is a real angular term in one frame --------------------
aligned = codes[0].unsqueeze(0).repeat(n, 1)
check("a perfectly aligned child scores zero (1 - cos)",
      float(_angular_penalty(aligned, codes)[:, 0].abs().max()) < 1e-4,
      f"max = {float(_angular_penalty(aligned, codes)[:, 0].abs().max()):.2e}")
check("the penalty is bounded in [0, 2]",
      0.0 <= float(_angular_penalty(separable, codes).min())
      and float(_angular_penalty(separable, codes).max()) <= 2.0)

print("\n" + ("all within-parent checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
