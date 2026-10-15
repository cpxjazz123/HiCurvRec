"""Curvature behaviour of the adaptive arm, on the instance-local API.

Four things the arm claims: the entropy normaliser reaches c_min on a balanced
*assignable* usage, the curvature survives a checkpoint, the installer leaves
VQLayer._indices alone so Sinkhorn still decides, and Sinkhorn actually changes
the answer on a skewed batch.
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


def build(curvature: float):
    """Three patched layers, each carrying its own curvature buffer."""
    layers = torch.nn.ModuleList([VQLayer(codebook_size=64, codebook_dim=8) for _ in range(3)])
    rq = torch.nn.Module()
    rq.vq_layers = layers
    buffers = ac.ensure_curvature_buffers(rq)
    for level, layer in enumerate(layers):
        ac.install_adaptive_assignment(layer, level)
    ac.set_curvature(curvature, buffers)
    return rq, buffers, layers


# --- 1. the entropy normaliser ---------------------------------------------
uniform = torch.full((128,), 10.0)
c_with = ac.curvature_from_counts(uniform, 256, usable_k=128)
c_without = ac.curvature_from_counts(uniform, 256)
check("a balanced *assignable* usage reaches c_min",
      abs(c_with - ac.C_MIN) < 1e-9, f"usable_k=128 -> {c_with:.4f} (c_min={ac.C_MIN})")
check("normalising by the full codebook cannot", c_without > ac.C_MIN,
      f"full 256 -> {c_without:.4f}, the old ceiling")
collapsed = torch.zeros(128)
collapsed[0] = 1000.0
check("a collapsed usage reaches c_max",
      abs(ac.curvature_from_counts(collapsed, 256, usable_k=128) - ac.C_MAX) < 1e-9)

rq, buffers, layers = build(1.0)
tokens = torch.tensor([[0], [1], [200]])       # 200 is outside the assignable prefix
ac.update_curvature_from_tokens(tokens, buffers, usable_k=128)
check("the refresh is instance-local and clamped",
      ac.C_MIN <= ac.current_curvature(layer=layers[0]) <= ac.C_MAX,
      f"current = {ac.current_curvature(layer=layers[0]):.6f}")

# --- 2. checkpoint round-trip ----------------------------------------------
ac.set_curvature(2.5, buffers)
saved = rq.state_dict()
check("the curvature is persistent state",
      "vq_layers.0._adaptive_curvature" in saved,
      f"value = {float(saved['vq_layers.0._adaptive_curvature']):.4f}")

fresh, fresh_buffers, fresh_layers = build(1.0)
fresh.load_state_dict(saved)
check("reload restores the curvature",
      abs(ac.current_curvature(layer=fresh_layers[0]) - 2.5) < 1e-12,
      f"after reload = {ac.current_curvature(layer=fresh_layers[0]):.4f}")

# --- 3. Sinkhorn is left in charge -----------------------------------------
# A dedicated layer with Sinkhorn actually configured: VQLayer defaults to
# sk_epsilon = -1, which disables it and makes _indices a plain argmin.
sk_layer = VQLayer(codebook_size=64, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
ac.install_adaptive_assignment(sk_layer, 2)
layer = sk_layer
check("the layer has Sinkhorn enabled", layer.use_sk,
      f"sk_epsilon = {layer.sk_epsilon}, sk_iters = {layer.sk_iters}")
check("the installer leaves VQLayer._indices alone",
      layer._indices.__func__ is VQLayer._indices)

torch.manual_seed(0)
distances = torch.rand(64, layer.n_embed)
layer.train()
via_sinkhorn = layer._indices(distances, False)
via_argmin = distances.argmin(dim=-1)
differing = int((via_sinkhorn != via_argmin).sum())
check("Sinkhorn changes the assignment on a skewed batch", differing > 0,
      f"{differing} of 64 rows differ from argmin")

# --- 4. distances follow this layer's own curvature -------------------------
probe = torch.randn(16, 8)
ac.set_curvature(0.5, buffers)
low = layers[0]._distances(probe).clone()
ac.set_curvature(3.5, buffers)
high = layers[0]._distances(probe)
check("changing this model's curvature changes its distances",
      not torch.allclose(low, high),
      f"max|diff| = {float((low - high).abs().max()):.4f}")

print("\n" + ("all curvature checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
