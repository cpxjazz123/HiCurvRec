"""Checks for the corrected M5 tree (CPU only, no training).

The mechanism's claim: items with different interest axes should take different
children of the same parent. The corrected code compares residual-space
directions, so the axis a row matches is the axis its own residual points at.
"""
import sys
from pathlib import Path

import torch

TREE = "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_interest_branching_residual"
sys.path.insert(0, TREE)

from interest_branching import (  # noqa: E402
    BRANCH_CURVATURE, BRANCH_STRENGTH, _branch_penalty, _select_axes,
)
from model.layers import VQLayer  # noqa: E402
from interest_branching import install_interest_branch  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


d, k = 8, 3
torch.manual_seed(0)

# Two axes in residual space, plus a third orthogonal one.
axes = torch.zeros(k, d)
axes[0, 0] = 1.0
axes[1, 1] = 1.0
axes[2, 2] = 1.0

# Two rows that landed on the same L1 parent (bucket 0) with different interests.
# ``directions`` is a per-bucket table of shape (n_buckets, k, dim); both rows
# below sit in bucket 0, so they share the axis set and differ only in direction.
n_buckets = 4
table = torch.zeros(n_buckets, k, d)
table[:, 0, 0] = 1.0
table[:, 1, 1] = 1.0
table[:, 2, 2] = 1.0
parent_indices = torch.tensor([0, 0])
latent = torch.zeros(2, d)
latent[0, 0] = 1.0        # row A points along axis 0
latent[1, 1] = 1.0        # row B points along axis 1

selected = _select_axes(latent, parent_indices, table)
check("each row matches the axis its residual points at",
      torch.allclose(selected[0], axes[0]) and torch.allclose(selected[1], axes[1]),
      f"row A axis={selected[0].argmax().item()}, row B axis={selected[1].argmax().item()}")

# The old formulation subtracted the parent codeword from the residual. Take a
# parent that dominates along axis 0 and a row whose residual points along axis 1:
# subtracting flips the direction onto axis 0, so the row is matched to the wrong
# interest axis.
parent_code = torch.zeros(1, d)
parent_code[0, 0] = 10.0
shifted = torch.nn.functional.normalize(latent - parent_code, dim=-1)
old_choice = (axes * shifted.unsqueeze(1)).sum(dim=-1).argmax(dim=-1)
corrected_choice = _select_axes(latent, parent_indices, table).argmax(dim=-1)
check("the removed parent subtraction really did change the match",
      old_choice.tolist() != corrected_choice.tolist(),
      f"row B: old formulation -> axis {old_choice[1].item()}, "
      f"corrected -> axis {corrected_choice[1].item()}")

# --- the branching claim: two axes, two different children -----------------
codes = torch.zeros(6, d)
codes[0, 0] = 1.0          # child aligned with axis 0
codes[1, 1] = 1.0          # child aligned with axis 1
codes[2:, 2] = 1.0         # the rest sit on axis 2

pen_a = _branch_penalty(codes, selected[:1])[0]
pen_b = _branch_penalty(codes, selected[1:2])[0]
choice_a = int(pen_a.argmin())
choice_b = int(pen_b.argmin())
check("rows with different axes pick different children", choice_a != choice_b,
      f"row A -> child {choice_a} (axis 0 aligned), row B -> child {choice_b}")
check("each row prefers the child aligned with its own axis",
      choice_a == 0 and choice_b == 1,
      f"row A penalties {[round(float(v), 3) for v in pen_a[:3]]}, "
      f"row B {[round(float(v), 3) for v in pen_b[:3]]}")

# Sign check: the opposite sign would reverse the branching.
check("the added sign is what makes it branch",
      float(pen_a[0]) < float(pen_a[1]) and float(pen_b[1]) < float(pen_b[0]),
      "aligned child has the smaller penalty; subtracting it would invert this")

# A row with no axis contributes nothing rather than a spurious direction.
zero_axis = _branch_penalty(codes, torch.zeros(1, d))[0]
check("a row without an estimated axis contributes zero penalty",
      bool((zero_axis == 0).all()), f"max={float(zero_axis.abs().max()):.2e}")

# --- the axes are persistent state ----------------------------------------
layer = VQLayer(codebook_size=32, codebook_dim=d)
parent = VQLayer(codebook_size=32, codebook_dim=d)
install_interest_branch(layer, parent, table)
check("axes are a registered buffer", "_interest_axes" in dict(layer.named_buffers()),
      f"buffers={sorted(dict(layer.named_buffers()))}")
saved = layer.state_dict()
check("axes are in state_dict", "_interest_axes" in saved)

fresh = VQLayer(codebook_size=32, codebook_dim=d)
install_interest_branch(fresh, parent, torch.zeros_like(table))   # deliberately different
fresh.load_state_dict(saved)
check("reload restores the axes that chose the codes",
      torch.allclose(fresh._interest_axes, table),
      f"max|diff|={float((fresh._interest_axes - table).abs().max()):.2e}")

print("\n" + ("all M5 checks passed" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
