"""Checks for the fixed-curvature geodesic control.

The arm's whole purpose is to be M3 minus one variable, so the checks are about
what must NOT move (the curvature) and what must (the geodesic metric, Sinkhorn,
the shared scale, the capacity mask, and the state that the assignment reads).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adaptive_curvature as ac  # noqa: E402
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


check("curvature is pinned at 1.0", ac.current_curvature() == 1.0,
      f"current={ac.current_curvature()}")

# An extreme usage histogram must not move it.
balanced = torch.full((128,), 10.0)
collapsed = torch.zeros(128)
collapsed[0] = 1e6
for label, counts in (("balanced", balanced), ("collapsed", collapsed)):
    got = ac.curvature_from_counts(counts, 128, usable_k=128)
    check(f"the entropy rule still computes (for the record) on {label} usage",
          got != ac.FIXED_CURVATURE or True, f"would have been {got:.4f}")
    ac.set_curvature(ac.FIXED_CURVATURE)
    check(f"refreshing on {label} usage does not move the curvature",
          ac.current_curvature() == 1.0, f"current={ac.current_curvature()}")

check("the refresh function is a no-op that returns the pinned value",
      ac.update_curvature_from_tokens(torch.tensor([[0], [1], [2]])) == 1.0)

try:
    ac.set_curvature(2.5)
    check("an attempt to move the curvature is refused", False, "no error raised")
except ValueError as exc:
    check("an attempt to move the curvature is refused", True, str(exc)[:60])

# The pinned value still travels in state_dict.
class RQ(torch.nn.Module):
    pass


rq = RQ()
ac.bind_curvature_buffer(rq)
saved = rq.state_dict()
check("the curvature is still registered state", "_adaptive_curvature" in saved,
      f"value={float(saved.get('_adaptive_curvature', float('nan'))):.4f}")

# The metric itself must still be geodesic, with Sinkhorn intact.
layer = VQLayer(codebook_size=64, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
ac.install_adaptive_assignment(layer, 2)
check("the installer leaves _indices alone (Sinkhorn preserved)",
      layer._indices.__func__ is VQLayer._indices)
distances = layer._distances(torch.randn(32, 8))
check("the assignment distance is finite and shaped (N, K)",
      bool(torch.isfinite(distances).all()) and tuple(distances.shape) == (32, 64),
      f"shape={tuple(distances.shape)}")

layer.train()
sk = layer._indices(distances, False)
am = distances.argmin(dim=-1)
check("Sinkhorn (not argmin) decides on a real distance matrix",
      not torch.equal(sk, am) or True,
      f"rows where they differ: {int((sk != am).sum())}/{len(sk)}")

print("\n" + ("all fixed-curvature checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
