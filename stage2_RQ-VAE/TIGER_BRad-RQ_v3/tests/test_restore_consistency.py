"""The export must not be able to forget the mechanism, and a restored model must
export exactly what the training run exported.

Two paths used to break that: the radius table was a non-persistent buffer, so a
model rebuilt from the checkpoint had no radii at all, and get_indices() then
exported a SID computed without the behaviour radius - silently, and differently
from the training run.
"""
import unittest
from types import SimpleNamespace

import torch

from model.model import RQVAE


def _config():
    return SimpleNamespace(
        hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16), codebook_dim=8,
        dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
        sk_epsilon=0.003, sk_iters=5, frechet_codebook=False, use_behaviour=True,
        use_curvature=True, frechet_eta=1.0, radial=True)


class ExportGuardTests(unittest.TestCase):

    def test_an_export_without_the_radius_table_is_refused(self):
        torch.manual_seed(0)
        model = RQVAE(_config(), in_dim=16)
        embeddings = torch.randn(24, 16)
        ids = torch.arange(24)
        model.set_radii(torch.linspace(0.1, 1.2, 24))
        with torch.no_grad():
            model.init_codebook(embeddings, item_ids=ids)
            model.radius_table = torch.zeros(0)
            with self.assertRaises(RuntimeError):
                model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)

    def test_the_radius_table_travels_with_the_checkpoint(self):
        torch.manual_seed(0)
        model = RQVAE(_config(), in_dim=16)
        radius = torch.linspace(0.1, 1.3, 24)
        model.set_radii(radius)
        state = model.state_dict()
        self.assertIn("radius_table", state)

        restored = RQVAE.from_checkpoint(_config(), state, in_dim=16)
        self.assertTrue(torch.allclose(restored.radius_table, radius, atol=1e-6))

    def test_a_restored_model_exports_the_same_sid(self):
        torch.manual_seed(0)
        model = RQVAE(_config(), in_dim=16)
        embeddings = torch.randn(24, 16)
        ids = torch.arange(24)
        radius = torch.linspace(0.1, 1.3, 24)
        with torch.no_grad():
            model.set_radii(radius)
            model.init_codebook(embeddings, item_ids=torch.arange(embeddings.shape[0]))
            first = model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
            state = model.state_dict()
        restored = RQVAE.from_checkpoint(_config(), state, in_dim=16)
        with torch.no_grad():
            # the checkpoint carries the codebook: a restored model must NOT
            # re-initialise it, or the two exports cannot be compared
            second = restored.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
