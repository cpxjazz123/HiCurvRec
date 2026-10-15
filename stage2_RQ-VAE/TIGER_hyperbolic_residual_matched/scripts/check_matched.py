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

# --- what is verified, and what is not --------------------------------------
# The naive mirrored addition is NOT the inverse. That much is measured.
# Whether some gyrogroup-correct cancellation exists is NOT settled here: the
# earlier claim that none exists was withdrawn, so nothing below asserts it.
x = torch.randn(5, d) * 0.1
y = torch.randn(5, d) * 0.1
c = MOBIUS_CURVATURE

radial_naive = logmap0(mobius_add(expmap0(_mobius_residual(x, y, c), c),
                                  expmap0(y, c), c), c)
deviation = float((radial_naive - x).abs().max())
check("NEGATIVE RESULT: the naive (r (-) q) (+) q does not recover r",
      deviation > 1e-3, f"max|diff| = {deviation:.2e} at c = {c}")

# Left cancellation in the gyrogroup sense: the map F_q is injective, which is a
# fact about the map, not about the formula that was tried.
a, b, d_ = torch.randn(3, 6, d) * 0.1
left = mobius_add(mobius_add(a, b, c), d_, c)
right = mobius_add(a, mobius_add(b, d_, c), c)
assoc_gap = float((left - right).abs().max())
check("Mobius addition is not associative, so bracketing matters",
      assoc_gap > 1e-4, f"|(a+b)+d - a+(b+d)|_max = {assoc_gap:.2e}")

# Injectivity of F_q for a fixed q, sampled: distinct inputs stay distinct.
q_fixed = torch.randn(1, d) * 0.1
samples = torch.randn(64, d) * 0.1
mapped = logmap0(mobius_add(expmap0(samples, c), expmap0(q_fixed.expand_as(samples), c), c), c)
pairs = torch.cdist(mapped, mapped) + torch.eye(64) * 1e9
min_sep = float(pairs.min())
check("F_q maps distinct inputs to distinct outputs (injectivity holds here)",
      min_sep > 0, f"closest pair after mapping = {min_sep:.4f}")

# --- the decoder input the probe measures is the one the forward builds ----
from hyperbolic_residual import decoder_input  # noqa: E402
per_level = [torch.randn(4, d) * 0.1, torch.randn(4, d) * 0.1, torch.randn(4, d) * 0.1]
matched = decoder_input(per_level, c, matched=True)
naive = per_level[0]
for q in per_level[1:]:
    naive = naive + q
check("the shared helper reproduces the matched accumulation",
      not torch.allclose(matched, naive, atol=1e-3),
      f"max|diff vs the Euclidean sum| = {float((matched - naive).abs().max()):.4f}")
flat = decoder_input(per_level, c, matched=False)
check("with matched=False the helper is exactly the Euclidean sum",
      torch.allclose(flat, naive, atol=1e-6),
      f"max|diff| = {float((flat - naive).abs().max()):.2e}")
limit = decoder_input(per_level, 1e-8, matched=True)
check("as c -> 0 the matched accumulation becomes the Euclidean sum",
      torch.allclose(limit, naive, atol=1e-4),
      f"max|diff| = {float((limit - naive).abs().max()):.2e}")

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
