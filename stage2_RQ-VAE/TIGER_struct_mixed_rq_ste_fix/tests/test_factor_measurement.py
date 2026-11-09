"""The shared split rule is last-dimension only; parameter blocks are not split."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_q9 as M


def test_a_parameter_block_is_not_split_into_e_and_h():
    grad = torch.zeros(32, 48)
    grad[20, 0] = 1.0
    out = M.report_blocks({"encoder": [torch.nn.Parameter(torch.zeros(32, 48))]},
                          {"encoder": [grad]})["encoder"]
    assert "E_norm" not in out and "H_norm" not in out
    assert abs(out["norm"] - 1.0) < 1e-9


def test_a_code_or_latent_matrix_splits_along_its_last_dimension():
    grad = torch.zeros(256, 32)
    grad[0, :16] = 2.0
    grad[0, 16:] = 3.0
    out = M.split_latent(grad)
    assert abs(out["E_norm"] - (16 * 4.0) ** 0.5) < 1e-9
    assert abs(out["H_norm"] - (16 * 9.0) ** 0.5) < 1e-9


if __name__ == "__main__":
    test_a_parameter_block_is_not_split_into_e_and_h()
    test_a_code_or_latent_matrix_splits_along_its_last_dimension()
    print("ok")
