"""Does the structure term actually reach the encoder and the codebook?"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_hooks as H
import structure as S
from model import RQVAE


def tiny_model(geometry=True, sk=True):
    cfg = SimpleNamespace(hidden_sizes=[32], codebook_num=3, codebook_size=(8, 8, 8),
                          codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq",
                          ema_decay=0.99, fix_code_embs=False,
                          sk_epsilon=0.003 if sk else -1.0, sk_iters=5 if sk else -1)
    torch.manual_seed(0)
    return RQVAE(cfg, in_dim=32)


def synthetic_graph():
    nodes = np.arange(6)
    neighbours = {0: {1, 2}, 1: {0}, 2: {0}, 3: {4}, 4: {3}, 5: {}}
    return S.Graph(nodes=nodes, neighbours=neighbours,
                   triangle_pairs={0: [(1, 2)]}, wedge_pairs={0: [(1, 2)]})


def test_forward_with_codes_returns_every_level():
    model = tiny_model()
    codes = model.forward_with_codes(torch.randn(4, 32))
    assert len(codes) == 5
    levels = codes[-1]
    assert len(levels) == 3
    assert levels[-1].shape == (4, 32)


def test_behaviour_off_makes_the_term_zero():
    model = tiny_model()
    embeddings = torch.randn(6, 32)
    term = H.structure_term(model, embeddings, synthetic_graph(), random.Random(0),
                            scale=1.0, size=16, behaviour_on=False)
    assert float(term) == 0.0


def test_structure_term_is_finite_and_moves_parameters():
    model = tiny_model()
    embeddings = torch.randn(6, 32)
    term = H.structure_term(model, embeddings, synthetic_graph(), random.Random(0),
                            scale=1.0, size=16, behaviour_on=True)
    assert torch.isfinite(term) and term.item() > 0
    before = [p.detach().clone() for p in model.encoder.parameters()]
    term.backward()
    encoder_grad = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    codebook_grad = sum(float(p.grad.abs().sum()) for p in model.rq.parameters() if p.grad is not None)
    assert encoder_grad > 0 or codebook_grad > 0


def test_geometry_switch_changes_the_value():
    model = tiny_model()
    embeddings = torch.randn(6, 32)
    mixed = H.structure_term(model, embeddings, synthetic_graph(), random.Random(1),
                             scale=1.0, size=32, behaviour_on=True, geometry_on=True)
    control = H.structure_term(model, embeddings, synthetic_graph(), random.Random(1),
                               scale=1.0, size=32, behaviour_on=True, geometry_on=False)
    assert mixed.item() != control.item()


def test_calibrated_scale_is_positive_and_reproducible():
    model = tiny_model()
    embeddings = torch.randn(6, 32)
    first = H.calibrate_structure_scale(model, embeddings, synthetic_graph(), random.Random(2), size=32)
    second = H.calibrate_structure_scale(model, embeddings, synthetic_graph(), random.Random(2), size=32)
    assert first > 0 and first == second


def test_gradient_weight_calibration_hits_the_ratio():
    model = tiny_model()
    embeddings = torch.randn(6, 32, requires_grad=False)
    base = sum((p ** 2).sum() for p in model.parameters())
    lam, detail = H.calibrate_lambda_by_gradient(
        model, embeddings, synthetic_graph(), random.Random(4), scale=1.0, size=16,
        base_loss=base, target_ratio=0.1, geometry_on=True)
    assert detail["structure_grad_norm"] > 0 and detail["base_grad_norm"] > 0
    assert abs(lam * detail["structure_grad_norm"] / detail["base_grad_norm"] - 0.1) < 1e-9


def test_block_budget_is_the_minimum_over_blocks():
    model = tiny_model()
    embeddings = torch.randn(6, 32)
    base = sum((p ** 2).sum() for p in model.parameters())
    lam, detail = H.calibrate_lambda_block_budget(
        model, embeddings, synthetic_graph(), random.Random(5), scale=1.0, size=16,
        base_loss=base, geometry_on=True)
    ratios = [b["base_norm"] / b["structure_norm"] for b in detail["blocks"].values()]
    assert abs(lam - min(ratios)) < 1e-9
    assert set(detail["blocks"]) == {"encoder", "codebooks"}
    for block in detail["blocks"].values():
        assert lam * block["structure_norm"] <= block["base_norm"] + 1e-9


def test_block_budget_reports_a_degenerate_block_instead_of_guessing():
    model = tiny_model()
    embeddings = torch.randn(6, 32)
    # a base loss with no path to the codebooks makes that block degenerate
    base = sum((p ** 2).sum() for p in model.encoder.parameters())
    try:
        H.calibrate_lambda_block_budget(
            model, embeddings, synthetic_graph(), random.Random(6), scale=1.0, size=16,
            base_loss=base, geometry_on=True)
    except H.CalibrationDegenerate as failure:
        assert failure.status == "CALIBRATION_DEGENERATE"
        assert failure.detail["degenerate_block"] == "codebooks"
        assert failure.detail["lambda"] is None
    else:
        raise AssertionError("a degenerate block must raise, not return a usable lambda")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
