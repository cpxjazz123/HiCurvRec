"""The four rules the frozen design contract fixes, checked on the real functions.

Each test drives the candidate's own model and quantizer -- not a re-implementation
of the rule -- because the defects this file exists for were all of the form
"the code runs, the tests pass, and the training computes something else":

* the assignment rule differed between training (Sinkhorn) and export (argmin),
* the reconstruction target was the raw encoder output while the quantizer scored
  the behaviour-scaled tangent,
* the geometry-off arm kept the exponential map and only changed the distance,
* the radius rank was taken over the whole catalogue instead of the items with a
  training statistic.

The contract these tests bind is results/research/pipeline_v2/design/BRad-RQ.json
(version 2).
"""
import unittest
from types import SimpleNamespace

import torch

from model.behaviour_radius import rank_radii
from model.model import RQVAE
from model.poincare_contract import expmap0


def config(use_curvature=True, radial=True):
    return SimpleNamespace(hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16),
                           codebook_dim=8, dropout=0.0, beta=0.25, vq_type="vq",
                           ema_decay=0.99, fix_code_embs=False,
                           # Sinkhorn is configured exactly as the trainer configures
                           # it: the radial path must not use it anyway.
                           sk_epsilon=0.003, sk_iters=50, frechet_codebook=False,
                           use_behaviour=True, use_curvature=use_curvature,
                           frechet_eta=1.0, radial=radial)


def build(use_curvature=True, items=32, dim=16, seed=0):
    torch.manual_seed(seed)
    model = RQVAE(config(use_curvature), in_dim=dim)
    embeddings = torch.randn(items, dim)
    with torch.no_grad():
        model.init_codebook(embeddings)
    radius = torch.linspace(0.05, 1.2, items)
    model.set_radii(radius)
    return model, embeddings, torch.arange(items), radius


class AssignmentRuleTests(unittest.TestCase):
    """Rule: the nearest codeword at every level, in training and at export."""

    def test_training_uses_the_nearest_codeword_not_sinkhorn(self):
        model, embeddings, ids, radius = build()
        model.train()
        with torch.no_grad():
            tokens = model(embeddings, ids, radius=radius[ids])[3]
            # the rule, computed from the same model state
            layer = model.rq.vq_layers[0]
            target = model.scale_by_radius(model.encoder(embeddings), radius[ids])
            expected = torch.argmin(
                layer.mapped_cost(layer.point_of(target.view(-1, layer.dim))), dim=-1)
        self.assertTrue(torch.equal(tokens[:, 0], expected))

    def test_the_export_writes_the_code_the_training_chose(self):
        model, embeddings, ids, radius = build()
        model.train()
        with torch.no_grad():
            trained = model(embeddings, ids, radius=radius[ids])[3]
        model.eval()
        with torch.no_grad():
            exported = model.get_indices(embeddings, item_ids=ids)
        self.assertTrue(torch.equal(trained, exported))


class LossTargetTests(unittest.TestCase):
    """Rule: L = MSE(z_rec, t) + the quantizer's commitment terms."""

    def test_the_reconstruction_is_the_code_sum_of_the_scored_target(self):
        model, embeddings, ids, radius = build()
        with torch.no_grad():
            recon, quant_loss, _, tokens, reference = model(embeddings, ids, radius=radius[ids])
            target = model.scale_by_radius(model.encoder(embeddings), radius[ids])
            code_sum = model.code_sum(tokens)
        self.assertTrue(torch.allclose(reference, target))
        self.assertTrue(torch.allclose(recon, code_sum))
        loss, recon_loss = model.compute_loss(reference, recon, quant_loss)
        self.assertAlmostEqual(float(recon_loss), float(torch.nn.functional.mse_loss(code_sum, target)),
                               places=6)

    def test_the_raw_embedding_is_not_the_target(self):
        """The defect this guards: reconstructing z while quantising t.

        The reference is the tangent the quantizer scored, and its norm is the
        behaviour radius rather than the encoder's own norm.
        """
        model, embeddings, ids, radius = build()
        with torch.no_grad():
            _, _, _, _, reference = model(embeddings, ids, radius=radius[ids])
            encoded = model.encoder(embeddings)
        self.assertFalse(torch.allclose(reference, encoded))
        self.assertTrue(torch.allclose(reference.norm(dim=-1), radius[ids], atol=1e-5))


class GeometryOffTests(unittest.TestCase):
    """Rule: geometry-off removes the map and scores the tangent vectors."""

    def test_geometry_off_scores_the_tangents_with_the_euclidean_distance(self):
        model, embeddings, ids, radius = build(use_curvature=False)
        with torch.no_grad():
            tokens = model(embeddings, ids, radius=radius[ids])[3]
            layer = model.rq.vq_layers[0]
            target = model.scale_by_radius(model.encoder(embeddings), radius[ids])
            expected = torch.argmin(
                (target.view(-1, layer.dim).unsqueeze(1)
                 - layer.get_code_embs().unsqueeze(0)).pow(2).sum(-1), dim=-1)
        self.assertTrue(torch.equal(tokens[:, 0], expected))

    def test_geometry_off_and_on_choose_different_codes(self):
        on, embeddings, ids, radius = build(use_curvature=True)
        off, _, _, _ = build(use_curvature=False)
        with torch.no_grad():
            a = on(embeddings, ids, radius=radius[ids])[3]
            b = off(embeddings, ids, radius=radius[ids])[3]
        self.assertFalse(torch.equal(a, b),
                         "geometry-off and geometry-on agree on every level: the switch is inert")


class RadiusRankTests(unittest.TestCase):
    """Rule: rank the measured items; unmeasured items take the median radius."""

    def test_only_the_items_with_a_statistic_are_ranked(self):
        scale = torch.tensor([1.0, 2.0, 3.0, 4.0])
        counts = torch.tensor([5.0, 5.0, 0.0, 5.0])
        radius = rank_radii(scale, counts)
        measured = [0, 1, 3]
        self.assertEqual(sorted(radius[i].item() for i in measured),
                         sorted({radius[i].item() for i in measured}))
        median = sorted(radius[i].item() for i in measured)[1]
        self.assertAlmostEqual(radius[2].item(), median, places=6)

    def test_an_unmeasured_catalogue_item_does_not_move_the_measured_radii(self):
        scale = torch.tensor([1.0, 2.0])
        counts = torch.tensor([5.0, 5.0])
        base = rank_radii(scale, counts)
        extended = rank_radii(torch.tensor([1.0, 2.0, 9.9]), torch.tensor([5.0, 5.0, 0.0]))
        self.assertTrue(torch.allclose(base, extended[:2], atol=1e-6))

    def test_without_any_measured_item_the_radius_is_undefined_and_refused(self):
        with self.assertRaises(ValueError):
            rank_radii(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))


class MissingRadiusTests(unittest.TestCase):
    """Rule: a radial model without a radius is an error, not a silent Euclidean step."""

    def test_a_radial_model_refuses_to_run_without_a_radius(self):
        model, embeddings, ids, _ = build()
        with self.assertRaises(ValueError):
            model(embeddings, ids)


if __name__ == "__main__":
    unittest.main()
