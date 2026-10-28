"""The mechanism inside the real model: wiring, gradient, and the two switches.

These run the candidate's own quantizer with a real batch, not a synthetic
tensor, because every defect this file guards against looked fine on a synthetic
input: a codebook that quietly receives a gradient, a switch that does not change
the computation, a commitment that saturates.
"""
import unittest
from types import SimpleNamespace

import torch

from model.frechet_poincare import BALL_EPS
from model.model import RQVAE


def config(use_behaviour=True, use_curvature=True):
    return SimpleNamespace(hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16),
                           codebook_dim=8, dropout=0.0, beta=0.25, vq_type="vq",
                           ema_decay=0.99, fix_code_embs=False, sk_epsilon=0.003,
                           sk_iters=5, frechet_codebook=True,
                           use_behaviour=use_behaviour, use_curvature=use_curvature,
                           frechet_eta=1.0)


def build(use_behaviour=True, use_curvature=True, items=48, dim=16, seed=0):
    torch.manual_seed(seed)
    model = RQVAE(config(use_behaviour, use_curvature), in_dim=dim)
    weights = torch.rand(items) * 3 + 0.25
    for layer in model.rq.vq_layers:
        layer.frechet.set_weights(weights)
    embeddings = torch.randn(items, dim)
    with torch.no_grad():
        model.init_codebook(embeddings)
    return model, embeddings, torch.arange(items), weights


def forward_backward(model, embeddings, ids):
    recon, quant_loss, _, tokens = model(embeddings, ids)
    loss = torch.nn.functional.mse_loss(recon, embeddings) + quant_loss
    loss.backward()
    return float(loss.detach()), tokens.detach(), loss


class WiringTests(unittest.TestCase):

    def test_the_codebook_takes_no_gradient(self):
        """The Frechet step is the codebook's only estimator.

        A parameter that requires grad but never receives one fails the DDP
        reducer on the second step with find_unused_parameters=False.
        """
        model, embeddings, ids, _ = build()
        forward_backward(model, embeddings, ids)
        for layer in model.rq.vq_layers:
            self.assertFalse(layer.embed.weight.requires_grad)
            self.assertIsNone(layer.embed.weight.grad)

    def test_the_encoder_keeps_its_gradient(self):
        model, embeddings, ids, _ = build()
        forward_backward(model, embeddings, ids)
        total = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()
                    if p.grad is not None)
        self.assertGreater(total, 0.0)

    def test_the_loss_is_finite_and_small(self):
        model, embeddings, ids, _ = build()
        loss, _, _ = forward_backward(model, embeddings, ids)
        self.assertTrue(loss == loss and abs(loss) < 1e6)

    def test_the_encoder_point_stays_inside_the_ball(self):
        model, embeddings, ids, _ = build()
        layer = model.rq.vq_layers[0]
        with torch.no_grad():
            recon, quant_loss, _, _ = model(embeddings, ids)
            points = layer.frechet._last["point"]
        self.assertLess(float(points.norm(dim=-1).max()), 1.0 - BALL_EPS / 2)
        self.assertLess(float(layer.get_code_embs().norm(dim=-1).max()), 1.0)

    def test_the_step_moves_the_codebook(self):
        model, embeddings, ids, _ = build()
        forward_backward(model, embeddings, ids)
        before = model.rq.vq_layers[0].get_code_embs().detach().clone()
        model.apply_frechet_codebook_step(model.frechet_step_sums())
        after = model.rq.vq_layers[0].get_code_embs().detach()
        self.assertGreater(float((after - before).abs().sum()), 1e-4)
        self.assertTrue(bool(torch.isfinite(after).all()))


class SwitchTests(unittest.TestCase):
    """Both switches must change the computation; neither may delete it."""

    def _tokens(self, use_behaviour, use_curvature):
        model, embeddings, ids, _ = build(use_behaviour, use_curvature)
        _, tokens, _ = forward_backward(model, embeddings, ids)
        return tokens, model

    def test_the_three_arms_are_three_computations(self):
        both, _ = self._tokens(True, True)
        behaviour_off, _ = self._tokens(False, True)
        geometry_off, _ = self._tokens(True, False)
        self.assertFalse(torch.equal(both, behaviour_off))
        self.assertFalse(torch.equal(both, geometry_off))
        self.assertFalse(torch.equal(behaviour_off, geometry_off))

    def test_the_mechanism_exists_with_the_geometry_off(self):
        """geometry-off must be the same mechanism with Euclidean distances."""
        _, model = self._tokens(True, False)
        for layer in model.rq.vq_layers:
            self.assertIsNotNone(layer.frechet)
            self.assertFalse(layer.frechet.use_curvature)
            self.assertTrue(layer.frechet.use_behaviour)

    def test_the_mechanism_exists_with_the_behaviour_off(self):
        _, model = self._tokens(False, True)
        for layer in model.rq.vq_layers:
            self.assertIsNotNone(layer.frechet)
            self.assertTrue(layer.frechet.use_curvature)
            self.assertFalse(layer.frechet.use_behaviour)
            self.assertTrue(torch.allclose(layer.frechet.weight_of(torch.tensor([0, 1])),
                                           torch.ones(2)))

    def test_the_mechanism_carries_no_per_item_behaviour_weight(self):
        """CURVCTRL-SID's statistic is one scalar per level, not a weight per item.

        The inherited Frechet candidate weighted the commitment by the successor
        attraction of every item; this proposal deliberately does not, so the
        remembered weights are uniform and the curvature is what carries the
        behaviour signal.
        """
        model, embeddings, ids, weights = build()
        layer = model.rq.vq_layers[0]
        with torch.no_grad():
            model(embeddings, ids)
            seen = layer.frechet._last["weights"]
        self.assertTrue(torch.allclose(seen, torch.ones_like(seen)))


class InitTests(unittest.TestCase):

    def test_the_start_assigns_a_tangent_of_any_size(self):
        """Levels above the first receive a residual, which is not a ball point.

        Feeding that tangent to the curved distance without mapping it into the
        ball first saturates the atanh: every distance becomes the same value, so
        the assignment degenerates to one code for the whole batch. A real start
        spreads the batch over the codebook.
        """
        from model.frechet_poincare import clamp_ball, exp_0, poincare_distance
        model, embeddings, ids, _ = build()
        layer = model.rq.vq_layers[1]
        with torch.no_grad():
            torch.manual_seed(0)
            tangent = torch.randn(64, layer.dim) * 3.0
            # What the old code did: the tangent used directly as a ball point.
            saturated = torch.argmin(
                poincare_distance(tangent.unsqueeze(1),
                                  layer.get_code_embs().unsqueeze(0)), dim=-1)
            residual = layer.init_codebook(tangent, torch.device("cpu"))
            mapped = torch.argmin(
                poincare_distance(clamp_ball(exp_0(tangent)).unsqueeze(1),
                                  layer.get_code_embs().unsqueeze(0)), dim=-1)
        self.assertTrue(bool(torch.isfinite(residual).all()))
        self.assertEqual(int(torch.unique(saturated).numel()), 1,
                         "the saturated assignment should collapse to one code")
        self.assertGreater(int(torch.unique(mapped).numel()), 1,
                           "the mapped assignment must spread over the codebook")

    def test_the_start_is_inside_the_ball(self):
        model, embeddings, ids, _ = build()
        for layer in model.rq.vq_layers:
            self.assertLess(float(layer.get_code_embs().norm(dim=-1).max()), 1.0)

    def test_the_start_uses_the_curved_cost(self):
        """A Euclidean start would put levels 2 and 3 in another geometry.

        The residual handed to the next level must be a tangent, ``x - log_0(code)``
        of a point inside the ball, so it stays bounded; a Euclidean start would
        assign against ball points with a chord distance and hand on a chord
        residual instead.
        """
        model, embeddings, ids, _ = build()
        with torch.no_grad():
            points = model.encoder(embeddings)
            residual = model.rq.vq_layers[0].init_codebook(points, embeddings.device)
        self.assertTrue(bool(torch.isfinite(residual).all()))
        # log_0 of a ball point has norm at most 2*atanh(1-eps) ~ 20 at c=1, and
        # the encoder outputs here are O(1), so the residual must stay far below
        # the 1/(1-|x|) blow-up a Euclidean start would keep.
        self.assertLess(float(residual.norm(dim=-1).max()), 25.0)


if __name__ == "__main__":
    unittest.main()
