"""Prefix arms: the same two promises, plus prefix coverage.

Adapted from the capacity arm's live-codebook checks: with the prefix counters held
fixed, a codebook update must move the geometry, and a checkpoint round-trip must
reproduce distances, indices and the exported codes exactly.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import VQLayer  # noqa: E402
from prefix_radius import CURVATURE, GEOMETRY, install_prefix_radius  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
n_embed, d, n = 32, 8, 24


def fresh_layer():
    layer = VQLayer(codebook_size=n_embed, codebook_dim=d, sk_epsilon=0.5, sk_iters=3)
    install_prefix_radius(layer, level=2, base=256, geometry=GEOMETRY, c=CURVATURE)
    layer.eval()
    return layer


layer = fresh_layer()
probe = torch.randn(n, d)
# two different L1 prefixes, so the prefix-keyed cells are genuinely distinct
layer._prefix_key = torch.tensor([7] * (n // 2) + [200] * (n - n // 2))
before = layer._distances(probe).clone()
before_idx = layer._indices(layer._distances(probe), False).clone()

with torch.no_grad():
    layer.get_code_embs().add_(torch.randn(n_embed, d) * 0.5)
after = layer._distances(probe)
check("updating the codebook changes the distance matrix",
      not torch.allclose(before, after),
      f"max|diff| = {float((before - after).abs().max()):.4f}")
after_idx = layer._indices(after, False)
check("and it changes the assignment under both prefixes",
      int((after_idx != before_idx).sum()) > 0,
      f"{int((after_idx != before_idx).sum())} of {n} rows moved")

saved = layer.state_dict()
snapshot = layer._distances(probe).clone()
snapshot_idx = layer._indices(layer._distances(probe), False).clone()
fresh = fresh_layer()
fresh._prefix_key = layer._prefix_key.clone()
fresh.load_state_dict(saved)
check("the prefix counters are restored",
      torch.allclose(fresh._prefix_counts, layer._prefix_counts),
      f"max|diff| = {float((fresh._prefix_counts - layer._prefix_counts).abs().max()):.2e}")
check("distances are identical after reload",
      torch.allclose(fresh._distances(probe), snapshot),
      f"max|diff| = {float((fresh._distances(probe) - snapshot).abs().max()):.2e}")
check("the exported codes are identical after reload",
      torch.equal(fresh._indices(fresh._distances(probe), False), snapshot_idx))
with torch.no_grad():
    fresh.get_code_embs().add_(torch.randn(n_embed, d) * 0.5)
check("the reloaded model also follows later codebook updates",
      not torch.allclose(fresh._distances(probe), snapshot))

grad_layer = fresh_layer()
grad_layer.train()
grad_layer._prefix_key = layer._prefix_key.clone()
grad_layer._distances(probe).sum().backward()
cg = grad_layer.get_code_embs().grad
check("gradients reach the codebook and are finite",
      cg is not None and bool(torch.isfinite(cg).all()))

print("\n" + ("all prefix live-codebook checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
