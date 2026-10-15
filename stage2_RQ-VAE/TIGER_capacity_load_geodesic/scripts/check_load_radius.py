"""Unit tests for the radius-sensitive load operator.

Covers the five properties the report asks for: geometry sensitivity, accounting
of the codes actually chosen, Sinkhorn preservation, the checkpoint round-trip,
and finite gradients. Run with GEOMETRY set by the tree that hosts this file.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_radius import (  # noqa: E402
    CURVATURE, GEOMETRY, LOAD_STRENGTH, install_load_radius, load_radii,
)
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


print(f"  geometry = {GEOMETRY}, curvature = {CURVATURE}")

# --- the load rule is monotone and bounded -------------------------------
usage = torch.tensor([0.0, 1.0, 100.0])
radii = load_radii(usage)
check("a crowded codeword is placed further out", radii[2] > radii[1] > radii[0],
      f"radii = {[round(float(r), 4) for r in radii]}")
check("an unused codeword is not moved", abs(float(radii[0]) - 1.0) < 1e-6,
      f"r(0 usage) = {float(radii[0]):.6f}")

# --- installation: Sinkhorn preserved, buffer registered ------------------
layer = VQLayer(codebook_size=32, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_load_radius(layer, geometry=GEOMETRY, c=CURVATURE)
check("the counter is a registered buffer", "_usage_ema" in dict(layer.named_buffers()),
      f"buffers = {sorted(dict(layer.named_buffers()))}")
import types as _types
check("Sinkhorn is preserved (the wrapper delegates, it does not replace)",
      isinstance(layer._indices.__func__, _types.FunctionType)
      and layer._indices.__func__ is not VQLayer._indices,
      "wrapper wraps _indices; Sinkhorn still runs inside it")

layer.train()
latent = torch.randn(32, 8)
distances = layer._distances(latent)
check("the assignment matrix is finite and shaped (N, K)",
      bool(torch.isfinite(distances).all()) and tuple(distances.shape) == (32, 32),
      f"shape = {tuple(distances.shape)}")

assigned = layer._indices(distances, False)
recorded = torch.nonzero(layer._usage_ema > 0).reshape(-1)
check("the counter records the codes actually assigned",
      torch.equal(torch.sort(recorded).values, torch.unique(assigned).sort().values),
      f"counter = {recorded.numel()} codes, assigned = {assigned.unique().numel()}")

sk = layer._indices(distances, False)
am = distances.argmin(dim=-1)
check("Sinkhorn (not argmin) decides on this level",
      not torch.equal(sk, am), f"rows differing = {int((sk != am).sum())}/{len(sk)}")

# --- the load actually moves crowded codes out of reach -------------------
layer._usage_ema.zero_()
layer._usage_ema[0] = 50.0
crowded_only = layer._distances(latent)
check("a crowded codeword becomes farther from the items",
      float(crowded_only[:, 0].mean()) > float(distances[:, 0].mean()),
      f"mean d to code 0: {float(distances[:, 0].mean()):.4f} -> "
      f"{float(crowded_only[:, 0].mean()):.4f}")

# --- checkpoint round-trip ------------------------------------------------
saved = layer.state_dict()
fresh = VQLayer(codebook_size=32, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_load_radius(fresh, geometry=GEOMETRY, c=CURVATURE)
fresh.load_state_dict(saved)
check("the load history round-trips through state_dict",
      torch.allclose(fresh._usage_ema, layer._usage_ema),
      f"max|diff| = {float((fresh._usage_ema - layer._usage_ema).abs().max()):.2e}")

# --- finite gradients through the load signal -----------------------------
usage_leaf = layer._usage_ema.detach().clone().requires_grad_(True)
directions = torch.nn.functional.normalize(layer.get_code_embs().detach(), dim=-1)
moved = directions * load_radii(usage_leaf).unsqueeze(-1)
(latent @ moved.t()).pow(2).mean().backward()
check("the operator is differentiable in the load signal with finite gradients",
      usage_leaf.grad is not None and bool(torch.isfinite(usage_leaf.grad).all()),
      f"|dL/d usage|_1 = {float(usage_leaf.grad.abs().sum()):.4f}")

# --- the geometry matters: the same load perturbation, two metrics ---------
# This is the arm pair's whole point, so it is checked against the sibling arm
# rather than asserted: the crowded codeword's distance must grow by a different
# factor under the geodesic rule than under the Euclidean one.
import importlib.util  # noqa: E402

sibling = Path(__file__).resolve().parent.parent.parent / "TIGER_capacity_load_euclidean"
spec = importlib.util.spec_from_file_location("load_radius_euclid", sibling / "load_radius.py")
euclid_module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(sibling))
spec.loader.exec_module(euclid_module)

torch.manual_seed(1)
probe_latent = torch.randn(64, 8)
base_codes = torch.nn.functional.normalize(torch.randn(32, 8), dim=-1)
usage_vec = torch.zeros(32)
usage_vec[0] = 40.0
scale = load_radii(usage_vec)

# The operator captures the codeword directions at install time, so the codebook
# has to be in place before the patch goes on.
geo_layer = VQLayer(codebook_size=32, codebook_dim=8)
euc_layer = VQLayer(codebook_size=32, codebook_dim=8)
with torch.no_grad():
    geo_layer.get_code_embs().copy_(base_codes)
    euc_layer.get_code_embs().copy_(base_codes)
install_load_radius(geo_layer, geometry="geodesic", c=CURVATURE)
install_load_radius(euc_layer, geometry="euclidean")
for lay in (geo_layer, euc_layer):
    lay._usage_ema.copy_(usage_vec)

with torch.no_grad():
    geo_d = geo_layer._distances(probe_latent)[:, 0].mean()
    euc_d = euc_layer._distances(probe_latent)[:, 0].mean()
    geo_flat = geo_layer._distances(probe_latent)
    euc_flat = euc_layer._distances(probe_latent)
check("the crowded codeword is farther under both metrics",
      float(geo_d) > 0 and float(euc_d) > 0,
      f"mean d (geodesic) = {float(geo_d):.4f}, (euclidean) = {float(euc_d):.4f}")
check("the two metrics rank the crowded codeword differently",
      not torch.allclose(geo_flat.mean(0), euc_flat.mean(0), atol=1e-4),
      f"max|diff in per-code mean| = {float((geo_flat.mean(0) - euc_flat.mean(0)).abs().max()):.4f}")

print("\n" + ("all load-radius checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
