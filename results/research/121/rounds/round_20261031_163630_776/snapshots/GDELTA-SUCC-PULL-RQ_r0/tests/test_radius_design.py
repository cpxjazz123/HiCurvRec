"""The item's ball radius must be the DESIGNED one: 0.9 * rank quantile.

The proposal puts an item at radius ``0.9 q`` in the Poincare ball. A tangent
norm is not a radius: mapping the tangent through ``exp_0`` gives the radius
``tanh(||t||)``, and squashing the tangent instead caps the norm at 0.9, which
saturates the radius at ``tanh(0.9) = 0.7163`` for every item above the
seventy-ninth percentile. The test compares the radius the mechanism actually
produces with the radius the proposal specifies.
"""
import unittest

import torch

from model.poincare_contract import expmap0
from model.behaviour_radius import rank_radii


class DesignRadiusTests(unittest.TestCase):

    def test_the_mapped_radius_is_the_designed_one(self):
        """The radius the mechanism produces is 0.9 * the rank quantile.

        The quantile comes from the RANK of the statistic, so it is the rank the
        mechanism uses, not the value of the statistic.
        """
        scores = torch.linspace(0.0, 1.0, 201)[1:]
        radius = rank_radii(scores)
        count = scores.numel()
        ranks = torch.empty(count, dtype=torch.long)
        order = torch.argsort(scores)
        ranks[order] = torch.arange(1, count + 1)
        quantiles = ranks.float() / (count + 1)
        designed = 0.9 * quantiles
        direction = torch.ones(count, 1)
        mapped = expmap0(direction * radius.unsqueeze(-1), 1.0).norm(dim=-1)
        self.assertTrue(torch.allclose(mapped, designed, atol=1e-4),
                        "the mapped radius differs from the design: %r vs %r"
                        % (mapped[:5].tolist(), designed[:5].tolist()))

    def test_a_tangent_capped_at_the_ball_radius_saturates(self):
        """The defect this test exists for: squashing the tangent loses the radius."""
        quantiles = torch.linspace(0.0, 1.0, 201)[1:]
        radius = rank_radii(quantiles)
        squashed_radius = torch.tanh(radius.clamp(max=0.9))
        designed = 0.9 * quantiles
        saturated = squashed_radius - designed
        self.assertGreater(float(saturated.abs().max()), 0.1,
                           "the squashed radius no longer saturates: the check is stale")
        self.assertGreater(int((saturated.abs() > 1e-3).sum()), 0,
                           "some items lose their radius entirely")


if __name__ == "__main__":
    unittest.main()


class RealPathTests(unittest.TestCase):
    """The radius the CANDIDATE actually uses, not the formula in isolation."""

    def test_the_point_the_candidate_scores_sits_at_the_designed_radius(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from types import SimpleNamespace
        from model.model import RQVAE
        from model.behaviour_radius import rank_radii

        torch.manual_seed(0)
        config = SimpleNamespace(
            hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16), codebook_dim=8,
            dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
            sk_epsilon=0.003, sk_iters=5, frechet_codebook=False, use_behaviour=True,
            use_curvature=True, frechet_eta=1.0, radial=True)
        model = RQVAE(config, in_dim=16)
        embeddings = torch.randn(64, 16)
        with torch.no_grad():
            model.init_codebook(embeddings)
        scores = torch.linspace(0.0, 1.0, 64)
        radius = rank_radii(scores)
        count = scores.numel()
        order = torch.argsort(scores)
        ranks = torch.empty(count, dtype=torch.long)
        ranks[order] = torch.arange(1, count + 1)
        designed = 0.9 * (ranks.float() / (count + 1))
        with torch.no_grad():
            encoded = model.encoder(embeddings)
            tangent = model.scale_by_radius(encoded, radius)
            layer = model.rq.vq_layers[0]
            point = layer.point_of(tangent)
        achieved = point.norm(dim=-1)
        self.assertTrue(torch.allclose(achieved, designed, atol=1e-3),
                        "the scored point is not at the designed radius: max gap %.4f"
                        % float((achieved - designed).abs().max()))
