"""The mixed-curvature arm must curve exactly the Sinkhorn level.

CLAUDE.md 19h allows the mechanism only inside the balanced-assignment cost matrix, and this
stack runs Sinkhorn on the last level alone (`sk_epsilon >= 0`).  If the wiring curved a lower
level instead, the arm would be testing argmin assignment and the claim would be empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
ARM = ROOT / "stage2_RQ-VAE" / "TIGER_kappa_sk_only_rq"


@pytest.fixture(scope="module")
def stack():
    sys.path.insert(0, str(ARM))
    import train_rqvae as entry
    from model.layers import RQLayer
    import distances

    layer = RQLayer(entry._tokenizer_config(), distance=entry.DISTANCE)
    yield entry, layer, distances


def test_the_entry_selects_the_curved_metric(stack):
    entry, _, distances = stack
    assert entry.DISTANCE is distances.curved.distances


def test_only_the_sinkhorn_level_is_curved(stack):
    entry, layer, distances = stack
    epsilon = [l.sk_epsilon for l in layer.vq_layers]
    sinkhorn_levels = [i for i, e in enumerate(epsilon) if e >= 0]
    assert sinkhorn_levels == [len(epsilon) - 1], epsilon
    metrics = [l.distance for l in layer.vq_layers]
    assert metrics[-1] is distances.curved.distances
    assert all(m is distances.distance.distances for m in metrics[:-1])


def test_the_lower_levels_really_are_euclidean(stack):
    """Guard against a shared object making the lower levels curved by accident."""
    entry, layer, distances = stack
    import torch

    latent = torch.randn(4, 8)
    codes = torch.randn(16, 8)
    baseline = (latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ codes.t()
                + codes.pow(2).sum(1, keepdim=True).t())
    assert torch.allclose(layer.vq_layers[0].distance(latent, codes), baseline)
    assert not torch.allclose(layer.vq_layers[-1].distance(latent, codes), baseline,
                              rtol=1e-3, atol=1e-3)
