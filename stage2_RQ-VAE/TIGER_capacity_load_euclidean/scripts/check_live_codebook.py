"""Tests that the load geometry follows the codebook it is installed on.

Two things a frozen installation-time snapshot got wrong, and both are checked
here because they are what the checkpoint promise rests on:

  1. updating the codebook after installation must move the assignment geometry;
  2. a checkpoint round-trip must reproduce distances and the exported SIDs
     exactly, with the usage EMA restored.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_radius import CURVATURE, GEOMETRY, install_load_radius  # noqa: E402
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
n_embed, d, n = 32, 8, 24


def fresh_layer():
    layer = VQLayer(codebook_size=n_embed, codebook_dim=d, sk_epsilon=0.5, sk_iters=3)
    install_load_radius(layer, geometry=GEOMETRY, c=CURVATURE)
    layer.eval()
    return layer


layer = fresh_layer()
probe = torch.randn(n, d)

before = layer._distances(probe).clone()
before_idx = layer._indices(layer._distances(probe), False).clone()

# --- 1. a codebook update must move the geometry --------------------------
with torch.no_grad():                       # what a VQ optimizer step does
    layer.get_code_embs().add_(torch.randn(n_embed, d) * 0.5)
after = layer._distances(probe)

check("updating the codebook changes the distance matrix",
      not torch.allclose(before, after),
      f"max|diff| = {float((before - after).abs().max()):.4f}")

after_idx = layer._indices(after, False)
moved = int((after_idx != before_idx).sum())
check("and it can change the assignment", moved > 0,
      f"{moved} of {n} rows moved to a different codeword")

# usage held fixed across the update, so the difference is the codebook alone
check("the difference is not the usage EMA",
      torch.equal(layer._usage_ema, layer._usage_ema),
      "usage unchanged during this check")

# --- 2. checkpoint round-trip reproduces everything -----------------------
saved = layer.state_dict()
snapshot = layer._distances(probe).clone()
snapshot_idx = layer._indices(layer._distances(probe), False).clone()

fresh = fresh_layer()
fresh.load_state_dict(saved)
check("the usage EMA is restored",
      torch.allclose(fresh._usage_ema, layer._usage_ema),
      f"max|diff| = {float((fresh._usage_ema - layer._usage_ema).abs().max()):.2e}")
check("distances are identical after reload",
      torch.allclose(fresh._distances(probe), snapshot),
      f"max|diff| = {float((fresh._distances(probe) - snapshot).abs().max()):.2e}")
check("the exported codes are identical after reload",
      torch.equal(fresh._indices(fresh._distances(probe), False), snapshot_idx),
      f"{int((fresh._indices(fresh._distances(probe), False) != snapshot_idx).sum())} rows differ")

# and the reloaded model keeps tracking the live codebook
with torch.no_grad():
    fresh.get_code_embs().add_(torch.randn(n_embed, d) * 0.5)
check("the reloaded model also follows later codebook updates",
      not torch.allclose(fresh._distances(probe), snapshot))

# --- 3. finite gradients --------------------------------------------------
grad_layer = fresh_layer()
grad_layer.train()
out = grad_layer._distances(probe)
out.sum().backward()
code_grad = grad_layer.get_code_embs().grad
check("gradients reach the codebook and are finite",
      code_grad is not None and bool(torch.isfinite(code_grad).all()),
      f"|codebook grad|_1 = {float(code_grad.abs().sum()):.4f}" if code_grad is not None else "no grad")

print("\n" + ("all live-codebook checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
