"""E/H factor attribution must follow the geometry, not the column layout."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_factors as F


def test_a_parameter_block_is_not_split_into_e_and_h():
    # only output row 20 receives gradient, and only from input column 0: a column
    # split would call this an E gradient, which is the axis confusion
    grad = torch.zeros(32, 48)
    grad[20, 0] = 1.0
    out = F.factor_norms("encoder", grad, kind="parameter")
    assert out["split"] == "NOT_APPLICABLE"
    assert abs(out["norm"] - 1.0) < 1e-9


def test_a_codebook_matrix_splits_along_its_last_dimension():
    grad = torch.zeros(256, 32)
    grad[0, :16] = 2.0
    grad[0, 16:] = 3.0
    out = F.factor_norms("C1", grad, kind="code")
    assert abs(out["E_norm"] - (16 * 4.0) ** 0.5) < 1e-9
    assert abs(out["H_norm"] - (16 * 9.0) ** 0.5) < 1e-9


def test_a_latent_vector_splits_along_its_last_dimension():
    grad = torch.zeros(4, 32)
    grad[:, :16] = 1.0
    out = F.factor_norms("latent", grad, kind="latent")
    assert abs(out["E_norm"] - (4 * 16) ** 0.5) < 1e-9
    assert abs(out["H_norm"]) < 1e-9


if __name__ == "__main__":
    test_a_parameter_block_is_not_split_into_e_and_h()
    test_a_codebook_matrix_splits_along_its_last_dimension()
    test_a_latent_vector_splits_along_its_last_dimension()
    print("ok")
