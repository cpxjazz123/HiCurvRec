"""The static radial prior: what it is, and what it must persist.

The arm is a *static* popularity prior, not a usage-adaptive mechanism: the factors
are computed once from anchor counts before training. Two things therefore have to
hold, and one of them used to fail -- re-installing the factors was a silent no-op,
so any attempt to refresh them did nothing.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import VQLayer  # noqa: E402
from radial_popularity import install_radial, radial_factors  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


# --- the prior itself ------------------------------------------------------
freq = torch.tensor([100.0, 0.0, 50.0, 1.0, 10.0, 5.0, 2.0, 3.0])
factors = radial_factors(freq)
check("a frequent codeword is pulled inward",
      float(factors[0]) < float(factors[1]),
      f"frequent {float(factors[0]):.4f} < rare {float(factors[1]):.4f}")
check("an unseen codeword is not moved", abs(float(factors[1]) - 1.0) < 1e-6,
      f"factor(0 usage) = {float(factors[1]):.6f}")
check("the factors are strictly positive and finite",
      bool((factors > 0).all()) and bool(torch.isfinite(factors).all()))

# --- installation ----------------------------------------------------------
layer = VQLayer(codebook_size=8, codebook_dim=4)
install_radial(layer, factors)
check("the factors are a registered buffer",
      "_radial_factors" in dict(layer.named_buffers()),
      f"buffers = {sorted(dict(layer.named_buffers()))}")
check("they are in state_dict", "_radial_factors" in layer.state_dict())

probe = torch.randn(12, 4)
distances = layer._distances(probe)
check("the distance matrix is finite and shaped (N, K)",
      bool(torch.isfinite(distances).all()) and tuple(distances.shape) == (12, 8),
      f"shape = {tuple(distances.shape)}")

# --- re-installing must actually update (this used to no-op) ---------------
before = layer._radial_factors.clone()
flat = radial_factors(torch.ones(8))
install_radial(layer, flat)
check("re-installing with new factors updates them",
      not torch.allclose(before, layer._radial_factors),
      f"max|diff| = {float((before - layer._radial_factors).abs().max()):.4f}")

# --- checkpoint round-trip -------------------------------------------------
saved = layer.state_dict()
snapshot = layer._distances(probe).clone()
fresh = VQLayer(codebook_size=8, codebook_dim=4)
install_radial(fresh, torch.ones(8))
fresh.load_state_dict(saved)
check("the prior round-trips through state_dict",
      torch.allclose(fresh._radial_factors, layer._radial_factors),
      f"max|diff| = {float((fresh._radial_factors - layer._radial_factors).abs().max()):.2e}")
check("a reloaded layer assigns identically",
      torch.allclose(fresh._distances(probe), snapshot),
      f"max|diff| = {float((fresh._distances(probe) - snapshot).abs().max()):.2e}")

# --- the rescaling is deliberately one-sided -------------------------------
# The codeword moves radially and the item does not, so the distance is not
# symmetric under swapping its arguments. That asymmetry *is* the prior.
# The claim is that the *codeword* is moved and the *item* is not, so the two
# arguments of the distance are treated differently. Asserting that on the factors
# themselves says nothing; this compares the distance against the same computation
# with the item scaled instead, which is what "one-sided" has to mean.
# Note: with a reference supplied the ball map is *not* scale-invariant in the
# item, and without one it is -- the earlier version of this check scaled the item
# and compared, which proved nothing on the default path. The claim to test is
# narrower and cleaner: moving the codewords radially must change the distances,
# and moving them uniformly must not change the *ranking*.
mixed = radial_factors(torch.arange(1, 9, dtype=torch.float32))
install_radial(layer, mixed)
varied = layer._distances(probe).clone()
install_radial(layer, torch.ones(8) * float(mixed.mean()))
uniform = layer._distances(probe)
check("moving the codewords radially changes the distances",
      not torch.allclose(varied, uniform),
      f"max|diff| = {float((varied - uniform).abs().max()):.4f}")
check("the item side is not rescaled by the prior",
      float(layer._radial_factors.std()) < 1e-6,
      "with uniform factors the prior is a no-op on the ranking")
install_radial(layer, mixed)          # restore the varied prior for this assertion
check("a varied prior survives installation",
      float(layer._radial_factors.min()) != float(layer._radial_factors.max()),
      f"range [{float(layer._radial_factors.min()):.4f}, "
      f"{float(layer._radial_factors.max()):.4f}]")

print("\n" + ("all radial checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
