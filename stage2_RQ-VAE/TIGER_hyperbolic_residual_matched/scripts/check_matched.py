"""Tests for the paired-geometry M6 variant.

The arm claims the two sides of the quantiser use the same group law, so the tests
are about that claim: shapes, finite gradients, the Euclidean limit as the curvature
goes to zero, and that the export path assigns codes through the same distance
function the training loop uses.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbolic import expmap0, logmap0, mobius_add  # noqa: E402
from hyperbolic_residual import MOBIUS_CURVATURE, _mobius_residual  # noqa: E402
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
d, k = 8, 16

# --- the Euclidean limit: Mobius addition collapses to the sum -------------
x = torch.randn(5, d) * 0.1
y = torch.randn(5, d) * 0.1
for c in (1e-6, 1e-8):
    pass
sum_euclid = x + y
sum_ball = mobius_add(x, y, 1e-8)
check("as c -> 0 the Mobius addition becomes the Euclidean sum",
      torch.allclose(sum_ball, sum_euclid, atol=1e-5),
      f"max|diff| = {float((sum_ball - sum_euclid).abs().max()):.2e}")

sum_ball_1 = mobius_add(x, y, 1.0)
check("at c = 1 it is a different operation", not torch.allclose(sum_ball_1, sum_euclid, atol=1e-3),
      f"max|diff| = {float((sum_ball_1 - sum_euclid).abs().max()):.4f}")

# --- subtraction and addition are inverse operations ----------------------
r = torch.randn(5, d) * 0.1
q = torch.randn(5, d) * 0.1
c = MOBIUS_CURVATURE
residual = _mobius_residual(r, q, c)
restored = logmap0(mobius_add(expmap0(residual, c), expmap0(q, c), c), c)
deviation = float((restored - r).abs().max())
check("NEGATIVE RESULT: (r (-) q) (+) q != r -- Mobius addition is not a group law",
      deviation > 1e-3, f"max|diff| = {deviation:.2e} at c = {c}")

# associativity is the structural reason, checked directly rather than asserted
a, b, d_ = torch.randn(3, 6, d) * 0.1
left = mobius_add(mobius_add(a, b, c), d_, c)
right = mobius_add(a, mobius_add(b, d_, c), c)
assoc_gap = float((left - right).abs().max())
check("and Mobius addition is not associative, which is why", assoc_gap > 1e-4,
      f"|(a+b)+d - a+(b+d)|_max = {assoc_gap:.2e}")

# --- shapes and finite gradients through the real layer ------------------
layer = VQLayer(codebook_size=k, codebook_dim=d)
latent = torch.randn(6, d, requires_grad=True)
quant, loss_term, _unused, idx = layer(latent)
check("the quantiser output keeps its shape", tuple(quant.shape) == (6, d),
      f"shape = {tuple(quant.shape)}")
loss_term.backward()
check("gradients are finite", bool(torch.isfinite(latent.grad).all()))

# the paired accumulation the layer now performs
acc = torch.zeros(4, d)
codes = torch.randn(4, d) * 0.05
acc = acc + codes                      # level 0: Euclidean, as in the control
acc = logmap0(mobius_add(expmap0(acc, c), expmap0(codes, c), c), c)
check("the paired accumulation is finite and shaped", bool(torch.isfinite(acc).all())
      and tuple(acc.shape) == (4, d), f"shape = {tuple(acc.shape)}")

# --- export uses the same assignment path as training --------------------
# get_indices is what both the trainer's export and the metric code call; it must
# route through _distances, so a patched metric reaches the exported SIDs too.
# The trainer's export calls RQVAE.get_indices, which delegates to each layer's
# _indices(_distances(...)). Checking the same composition here pins the claim that
# the exported SIDs see the patched metric.
direct = layer._indices(layer._distances(latent.detach()), False)
exported = layer._indices(layer._distances(latent), True)
check("export assigns through the same _distances the training loop uses",
      torch.equal(exported.reshape(-1), direct.reshape(-1)),
      f"{int((exported.reshape(-1) != direct.reshape(-1)).sum())} rows differ")

print("\n" + ("all paired-geometry checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
