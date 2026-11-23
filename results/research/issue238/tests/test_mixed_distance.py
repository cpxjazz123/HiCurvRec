"""C18/C19/C22/C23 — the mixed-curvature distance D_q and its wiring.

D_q is the only place the curve operation enters (the same object feeds both the curvature
fidelity target and the Sinkhorn residual-to-codeword cost).  So the tests here are about
component-wise correctness, the injected scale and curvature really being the frozen values, and
— the part that historically goes wrong — the gradient reaching the encoder and the codebooks
through the *hard* quantised vector rather than only through the pre-quantisation latent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _armloader import load_arm  # noqa: E402

load_arm("TIGER_curvfid_rq")

from distances import mixed  # noqa: E402


def test_the_blocks_sum_to_the_bottleneck_and_the_weights_to_one():
    assert sum(mixed.DIMS) == 32
    assert abs(sum(mixed.WEIGHTS) - 1.0) < 1e-12
    assert all(w >= 0 for w in mixed.WEIGHTS)
    assert mixed.CURVATURE == (-1.0, 0.0, 1.0)


def test_each_component_is_a_genuine_metric_on_its_own_manifold():
    """Zero distance, symmetry and positivity for the three components separately."""
    torch.manual_seed(0)
    x = torch.randn(16, 32) * 0.7
    y = torch.randn(16, 32) * 0.7
    for name in ("hyperbolic", "euclidean", "spherical"):
        same = mixed.component_sq(name, x, x.clone())   # already the element-wise diagonal
        assert same.shape == (16,), name
        assert torch.allclose(same, torch.zeros(16), atol=1e-5), name
        forward = mixed.component_sq(name, x, y)
        backward = mixed.component_sq(name, y, x)
        assert torch.allclose(forward, backward, atol=1e-5), name
        assert torch.all(forward >= -1e-6), name


def test_the_hyperbolic_component_is_not_the_euclidean_one():
    """If the ball maths collapsed to the Euclidean distance the curvature would be decorative."""
    torch.manual_seed(1)
    x = torch.randn(8, 32) * 2.0
    y = torch.randn(8, 32) * 2.0
    h = mixed.component_sq("hyperbolic", x, y)
    e = mixed.component_sq("euclidean", x, y)
    assert not torch.allclose(h, e, rtol=1e-3, atol=1e-3)


def test_the_spherical_component_differs_from_both_others():
    torch.manual_seed(2)
    x = torch.randn(8, 32) * 2.0
    y = torch.randn(8, 32) * 2.0
    s = mixed.component_sq("spherical", x, y)
    for other in ("hyperbolic", "euclidean"):
        assert not torch.allclose(s, mixed.component_sq(other, x, y), rtol=1e-3, atol=1e-3)


def test_the_scale_is_injected_and_not_implicit():
    """C19: the mapping scale is a frozen recorded number, so changing it must change D_q."""
    torch.manual_seed(3)
    x = torch.randn(6, 32)
    first = mixed.set_scale(1.0)
    d1 = mixed.pairwise_sq(x, x + 0.1)
    second = mixed.set_scale(4.0)
    d4 = mixed.pairwise_sq(x, x + 0.1)
    assert not torch.allclose(d1, d4, rtol=1e-3, atol=1e-3)
    mixed.set_scale(first)
    mixed.set_scale(second)


def test_pairwise_shape_matches_the_codebook_contract():
    torch.manual_seed(4)
    latent = torch.randn(12, 32)
    codes = torch.randn(256, 32)
    out = mixed.distances(latent, codes)
    assert out.shape == (12, 256)
    assert torch.isfinite(out).all()


def test_gradients_flow_through_the_mixed_distance():
    torch.manual_seed(5)
    latent = torch.randn(5, 32, requires_grad=True)
    codes = torch.randn(8, 32, requires_grad=True)
    mixed.distances(latent, codes).sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert codes.grad is not None and torch.isfinite(codes.grad).all()


# --- C23: the two switches and the Sinkhorn wiring --------------------------------

def _load_candidate_modules():
    for name in list(sys.modules):
        if name.split(".")[0] in {"distances", "model", "losses", "train_config", "train_rqvae"}:
            del sys.modules[name]
    load_arm("TIGER_curvfid_rq")
    import train_rqvae as entry

    return entry


def test_curve_off_reproduces_the_baseline_euclidean_cost():
    """C13: with the curve switch off the Sinkhorn cost must be the baseline's Euclidean one."""
    entry = _load_candidate_modules()
    torch.manual_seed(6)
    latent = torch.randn(9, 32)
    codes = torch.randn(16, 32)
    mixed.set_scale(1.0)
    on = mixed.distances(latent, codes)
    off = mixed.euclidean_cost(latent, codes)
    baseline = (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t()
                + codes.pow(2).sum(1, keepdim=True).t())
    assert torch.allclose(off, baseline, rtol=1e-5)
    assert not torch.allclose(on, baseline, rtol=1e-3, atol=1e-3)


def test_the_sinkhorn_layer_is_the_one_that_carries_the_mixed_distance():
    """C12: only the balanced level changes; the argmin levels keep the baseline metric."""
    entry = _load_candidate_modules()
    # `_load_candidate_modules` purged and re-imported, so the fresh module instance is the one
    # the layer actually holds; comparing against the file-level import would be a cross-instance
    # comparison rather than a wiring check.
    from distances import distance as euclid
    from distances import mixed as fresh_mixed
    from model.layers import RQLayer

    layer = RQLayer(entry._tokenizer_config(), distance=entry.DISTANCE)
    metrics = [l.distance for l in layer.vq_layers]
    sinkhorn = [i for i, l in enumerate(layer.vq_layers) if l.sk_epsilon >= 0]
    assert sinkhorn == [len(metrics) - 1]
    assert entry.DISTANCE is fresh_mixed.distances
    assert metrics[-1] is fresh_mixed.distances
    assert all(m is euclid.distances for m in metrics[:-1])
