"""The exported SID must be the code the trained quantizer assigns.

The radius is part of the assignment, so an export that encodes without it — or a
tree that still builds the inherited Frechet codebook — produces a SID that does
not match the training path.
"""
import unittest
from types import SimpleNamespace

import torch

from model.model import RQVAE


def config(**extra):
    base = dict(hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16),
                codebook_dim=8, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
                fix_code_embs=False, sk_epsilon=0.003, sk_iters=5,
                frechet_codebook=False, use_behaviour=True, use_curvature=True,
                frechet_eta=1.0, radial=True)
    base.update(extra)
    return SimpleNamespace(**base)


class ExportTests(unittest.TestCase):

    def test_the_export_uses_the_same_tangent_target_as_training(self):
        torch.manual_seed(0)
        model = RQVAE(config(), in_dim=16)
        embeddings = torch.randn(24, 16)
        ids = torch.arange(24)
        radii = torch.linspace(0.05, 1.4, 24)
        model.set_radii(radii)
        with torch.no_grad():
            model.init_codebook(embeddings)
            training_tokens = model(embeddings, ids, radius=radii[ids])[3]
            exported = model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
        self.assertTrue(torch.equal(training_tokens, exported))

    def test_an_export_without_stored_radii_is_refused(self):
        """An export that ignores the behaviour radius is a different mechanism.

        The permissive version of this test WAS the defect: a model rebuilt
        without the radius table exported a SID computed without the behaviour
        radius, silently and differently from the training run.
        """
        torch.manual_seed(0)
        model = RQVAE(config(), in_dim=16)
        embeddings = torch.randn(24, 16)      # at least one sample per code for KMeans
        ids = torch.arange(24)
        # the radii are bound so the codebook can initialise; the export guard is
        # what this test is about, so the empty table is what must be refused
        model.set_radii(torch.linspace(0.1, 1.2, 24))
        with torch.no_grad():
            model.init_codebook(embeddings, item_ids=ids)
            model.radius_table = torch.zeros(0)
            with self.assertRaises(RuntimeError):
                model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)


class NoInheritedCodebookTests(unittest.TestCase):

    def test_the_candidate_does_not_build_the_inherited_frechet_codebook(self):
        """The radial path replaces it; a leftover would freeze the codebook."""
        source = (__import__("pathlib").Path(__file__).resolve().parents[1] /
                  "train_rqvae.py").read_text(encoding="utf-8")
        self.assertIn("frechet_codebook=False", source)


if __name__ == "__main__":
    unittest.main()
