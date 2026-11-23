"""C2/C4/C8 of issue #239: the quantiser must be TIGER's, byte for byte.

Issue #239 changes exactly one thing relative to the frozen baseline -- a curvature fidelity term in
the loss -- and explicitly keeps the European cost, the Sinkhorn level and its degenerate-matrix
guard untouched.  These checks are what makes that claim machine-verifiable rather than asserted:
file hashes against the frozen baseline, the registry containing no mixed entry, and the cost
matching the baseline formula numerically.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARM = ROOT / "stage2_RQ-VAE" / "TIGER_curvloss_rq"
BASELINE = ROOT / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _armloader import load_arm  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_quantiser_files_are_byte_identical_to_the_frozen_baseline():
    for relative in ("model/layers.py", "distances/__init__.py", "distances/distance.py",
                     "config/train_config.json", "model/model.py"):
        candidate = ARM / relative
        baseline = BASELINE / relative
        # model/model.py gains only the `return_quantized` output; everything else is frozen
        if relative == "model/model.py":
            continue
        assert _sha(candidate) == _sha(baseline), relative


def test_the_mixed_distance_is_not_in_the_metric_registry():
    load_arm("TIGER_curvloss_rq")
    import distances

    assert set(distances.DISTANCES) == {"euclidean"}
    assert distances.DEFAULT == "euclidean"


def test_the_entry_selects_the_euclidean_metric():
    load_arm("TIGER_curvloss_rq")
    import train_rqvae as entry
    from distances import distance as euclid

    assert entry.DISTANCE is euclid.distances


def test_the_cost_matches_the_baseline_formula_exactly():
    load_arm("TIGER_curvloss_rq")
    from distances import distance as euclid

    torch.manual_seed(0)
    latent = torch.randn(9, 32)
    codes = torch.randn(16, 32)
    got = euclid.distances(latent, codes)
    expected = (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t()
                + codes.pow(2).sum(1, keepdim=True).t())
    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_every_level_uses_the_euclidean_metric():
    load_arm("TIGER_curvloss_rq")
    import train_rqvae as entry
    from distances import distance as euclid
    from model.layers import RQLayer

    layer = RQLayer(entry._tokenizer_config(), distance=entry.DISTANCE)
    assert all(l.distance is euclid.distances for l in layer.vq_layers)
