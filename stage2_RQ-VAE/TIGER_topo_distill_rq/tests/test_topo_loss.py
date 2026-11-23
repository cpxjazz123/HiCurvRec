"""Tests for the encoder-only topology distillation (issue #244 §5, C19/C20)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import topo_loss  # noqa: E402


def make_bank(tmp: Path, pairs, scores) -> Path:
    path = tmp / "bank.npz"
    np.savez_compressed(path, pairs=np.array(pairs, dtype=np.int64),
                        scores=np.array(scores, dtype=np.float32))
    return path


class ToyEncoder(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.proj = nn.Linear(8, dim, bias=False)
        self.decoder = nn.Linear(dim, 1)

    def forward(self, x):
        return self.proj(x)


def build(tmp, pairs, scores, lambda_max=0.2, epochs=3000):
    embeddings = torch.randn(6, 8)
    bank_path = make_bank(tmp, pairs, scores)
    torch.manual_seed(0)
    distill = topo_loss.TopologyDistill(embeddings, ROOT, torch.device("cpu"), epochs,
                                        bank_path=bank_path)
    distill.lambda_max = lambda_max
    topo_loss.LAMBDA_MAX = lambda_max
    return distill, embeddings


def test_ramp_is_flat_then_linear(tmp):
    distill, _ = build(tmp, [[0, 1]], [0.5])
    assert distill.weight(1) == 0.0
    assert distill.weight(1500) == 0.0
    mid = distill.weight(1500 + 750)
    assert abs(mid - 0.5 * topo_loss.LAMBDA_MAX) < 1e-9, mid
    assert abs(distill.weight(3000) - topo_loss.LAMBDA_MAX) < 1e-9


def test_zero_norm_pairs_do_not_produce_nan(tmp):
    distill, _ = build(tmp, [[0, 1]], [0.5])
    encoder = ToyEncoder()
    with torch.no_grad():
        encoder.proj.weight.zero_()  # every pair has zero norm
    value = distill.loss(encoder)
    assert torch.isfinite(value), value
    assert distill.stats["zero_norm_pairs"] > 0


def test_gradient_reaches_the_encoder_only(tmp):
    distill, _ = build(tmp, [[0, 1], [2, 3]], [0.2, 0.8])
    encoder = ToyEncoder()
    value = distill.loss(encoder)
    value.backward()
    assert encoder.proj.weight.grad is not None
    assert float(encoder.proj.weight.grad.abs().sum()) > 0
    assert encoder.decoder.weight.grad is None, "the loss must not touch the decoder"


def test_huber_matches_a_hand_computed_value(tmp):
    distill, embeddings = build(tmp, [[0, 1]], [0.5])
    encoder = ToyEncoder()
    with torch.no_grad():
        encoder.proj.weight.copy_(torch.eye(4, 8))
    # the cosine is computed on the ENCODER OUTPUT, not on the raw embedding
    left, right = encoder(embeddings[0]), encoder(embeddings[1])
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    prediction = 0.5 * (1 + cosine)
    residual = abs(prediction - 0.5)
    expected = 0.5 * residual ** 2 if residual <= topo_loss.DELTA else topo_loss.DELTA * (residual - 0.5 * topo_loss.DELTA)
    value = float(distill.loss(encoder))
    assert abs(value - expected) < 1e-4, (value, expected)


def test_pairs_and_scores_are_aligned(tmp):
    distill, _ = build(tmp, [[0, 1], [2, 3]], [0.1, 0.9])
    assert distill.pairs.shape == (2, 2)
    assert np.allclose(distill.scores.numpy(), [0.1, 0.9], atol=1e-6)


def main() -> int:
    import tempfile
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            try:
                function(Path(tmp))
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                failures.append((name, str(error)))
                print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
