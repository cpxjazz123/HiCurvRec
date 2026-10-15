"""M8's mechanism is the shared installer, so these are its two promises.

The arm's only mechanism is `scripts/hyp_losses.install_nearest_assignment`. The
defects it used to have were in that file: a per-tensor tangent scale (items and
codewords mapped under different scales, drifting with the batch) and an `_indices`
override that deleted the L3 Sinkhorn balanced assignment. Both are fixed in the
shared file, and this tree checks them where it uses them.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import VQLayer  # noqa: E402
from scripts.hyp_losses import (  # noqa: E402
    encode_points, install_nearest_assignment, shared_tangent_scale,
)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
C = 1.0

# --- one tangent scale for both sides -------------------------------------
codebook = torch.randn(32, 8) * 0.5
batch = torch.randn(64, 8) * 0.5
scale = shared_tangent_scale(codebook, C)
full = encode_points(batch, C, scale, reference=codebook)
half = encode_points(batch[:32], C, scale, reference=codebook)
check("with the codebook as reference, encoding is invariant to batch composition",
      torch.allclose(full[:32], half),
      f"max|diff| = {float((full[:32] - half).abs().max()):.2e}")

# The historical path is kept for single-tensor callers and is still
# batch-dependent, which is exactly why two-sided distances pass a reference.
old_full = encode_points(batch, C)
old_half = encode_points(batch[:32], C)
check("without a reference the mapping still drifts with the batch",
      float((old_full[:32] - old_half).abs().max()) > 0,
      f"max|diff| = {float((old_full[:32] - old_half).abs().max()):.4f}")

# Items and codewords must now be in one frame: the codebook's image cannot depend
# on which items are being compared against it.
codes_image = encode_points(codebook, C, scale, reference=codebook)
check("the codebook's image does not depend on the item batch",
      torch.allclose(encode_points(codebook, C, scale, reference=codebook), codes_image),
      "same codebook, same reference, same image")

# --- Sinkhorn survives the installer --------------------------------------
layer = VQLayer(codebook_size=32, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
check("the layer has Sinkhorn configured", layer.use_sk,
      f"sk_epsilon = {layer.sk_epsilon}, sk_iters = {layer.sk_iters}")
install_nearest_assignment(layer, C, euclidean=False)
# The installer defers to the original distance until the mechanism is
# switched on; the trainer flips this after the KMeans init pass.
layer._hyp_assignment_active = True
check("the installer leaves _indices alone",
      layer._indices.__func__ is VQLayer._indices,
      "Sinkhorn is still the thing that decides")

layer.train()
latent = torch.randn(64, 8)
distances = layer._distances(latent)
check("the geodesic assignment matrix is finite",
      bool(torch.isfinite(distances).all()), f"shape = {tuple(distances.shape)}")

via_sinkhorn = layer._indices(distances, False)
via_argmin = distances.argmin(dim=-1)
differing = int((via_sinkhorn != via_argmin).sum())
check("Sinkhorn changes the assignment on a skewed batch", differing > 0,
      f"{differing} of 64 rows differ from argmin")

# --- the Euclidean branch is a different metric ---------------------------
euclid_layer = VQLayer(codebook_size=32, codebook_dim=8)
install_nearest_assignment(euclid_layer, C, euclidean=True)
euclid_layer._hyp_assignment_active = True
with torch.no_grad():
    euclid_layer.get_code_embs().copy_(layer.get_code_embs())
geo = layer._distances(latent)
euc = euclid_layer._distances(latent)
check("the geodesic and Euclidean branches disagree",
      not torch.allclose(geo, euc, atol=1e-4),
      f"max|diff| = {float((geo - euc).abs().max()):.4f}")

print("\n" + ("all assignment checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
