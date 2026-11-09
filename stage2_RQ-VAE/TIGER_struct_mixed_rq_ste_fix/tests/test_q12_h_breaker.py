"""The H-break counterexample must run through the same real Jacobian comparator."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed_geometry
import native_product
from test_q10_native_and_rqlayer import build_layer, ref_reconstruct

E = 16


def full_row_jac(layer, z, k, j):
    level = layer(z, return_levels=True)[4][k][0]
    embs = layer.vq_layers[j].get_code_embs()
    index = j + 1
    jac = []
    for i in range(32):
        g = torch.autograd.grad(level[i], embs, retain_graph=True, allow_unused=True)[0]
        jac.append(torch.zeros_like(embs) if g is None else g.clone())
    full = torch.stack(jac)                       # (32 outputs, 256 rows, 32 coords)
    return full[:, index, :], full


def ref_jac(layer, k, j):
    rows = [layer.vq_layers[i].get_code_embs()[i + 1].detach().requires_grad_(True)
            for i in range(k + 1)]
    jac = torch.autograd.functional.jacobian(lambda *cs: ref_reconstruct(list(cs)),
                                             tuple(rows), vectorize=True)
    return jac[j].reshape(32, 32)


def assert_matches(layer, z, k, j):
    got, full = full_row_jac(layer, z, k, j)
    want = ref_jac(layer, k, j)
    diff = float((got - want).abs().max())
    assert diff < 1e-8, (k, j, diff)
    return got, want, full


def test_positive_control_matches_and_unselected_rows_are_zero():
    layer = build_layer(seed=5)
    torch.manual_seed(11)
    z = torch.empty(1, 32, dtype=torch.float64).normal_(0, 0.3).requires_grad_(True)
    for k in range(3):
        for j in range(3):
            if j > k:
                _, full = full_row_jac(layer, z, k, j)
                assert float(full.abs().sum()) == 0.0, (k, j)
            else:
                _, _, full = assert_matches(layer, z, k, j)
                others = torch.cat([full[:, r, :] for r in range(256) if r != j + 1], dim=0)
                assert float(others.abs().sum()) == 0.0, ("unselected rows", k, j)


def test_the_h_break_is_rejected_by_the_same_comparator():
    layer = build_layer(seed=5)
    torch.manual_seed(11)
    z = torch.empty(1, 32, dtype=torch.float64).normal_(0, 0.3).requires_grad_(True)
    original = mixed_geometry.code_ball

    def broken(code):
        # code_ball returns only the 16 H coordinates; detach that block
        return original(code).detach()

    mixed_geometry.code_ball = broken
    try:
        for k in range(3):
            for j in range(k + 1):
                got, want = full_row_jac(layer, z, k, j)[0], ref_jac(layer, k, j)
                h_diff = float((got[:, E:] - want[:, E:]).abs().max())
                assert h_diff > 1e-6, ("the comparator must reject the H break", k, j, h_diff)
    finally:
        mixed_geometry.code_ball = original


if __name__ == "__main__":
    test_positive_control_matches_and_unselected_rows_are_zero()
    test_the_h_break_is_rejected_by_the_same_comparator()
    print("ok")
