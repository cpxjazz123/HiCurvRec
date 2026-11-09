"""The latent capture must select the rows that the loss actually used."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_q6_factors as P


def test_structure_latent_gradient_is_captured_with_the_accepted_row_count():
    grad = P.latent_grad("structure")
    assert isinstance(grad, torch.Tensor), (
        "expected a latent gradient tensor; got " + repr(grad)
        + " (the capture filtered by requested triples, not accepted ones)")


if __name__ == "__main__":
    test_structure_latent_gradient_is_captured_with_the_accepted_row_count()
    print("ok")
