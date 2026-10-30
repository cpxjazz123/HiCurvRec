"""The radius must reach the code: ranking, score and switches inside the model."""
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


def build(**extra):
    torch.manual_seed(0)
    model = RQVAE(config(**extra), in_dim=16)
    embeddings = torch.randn(24, 16)
    with torch.no_grad():
        model.init_codebook(embeddings)
    return model, embeddings, torch.arange(24)


class WiringTests(unittest.TestCase):

    def test_the_radius_scales_the_tangent_target(self):
        model, embeddings, ids = build()
        radius = torch.linspace(0.1, 1.4, 24)
        with torch.no_grad():
            z = model.encoder(embeddings)
            scaled = model.scale_by_radius(z, radius[ids])
        direction = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        expected = radius[ids].unsqueeze(-1) * direction
        self.assertTrue(torch.allclose(scaled, expected, atol=1e-5))
        self.assertLess(float(scaled.norm(dim=-1).max()), 1.5)

    def test_the_gradient_reaches_the_encoder_through_the_radius(self):
        """The radial path is part of the differentiable graph, not a no_grad scale.

        A stray decorator above the scaling silently zeroed every encoder
        gradient while every value check still passed.
        """
        model, embeddings, ids = build()
        radius = torch.linspace(0.1, 1.4, 24, requires_grad=True)
        z = model.encoder(embeddings)
        scaled = model.scale_by_radius(z, radius[ids])
        self.assertTrue(scaled.requires_grad)
        recon, quant_loss, _, _, reference = model(embeddings, ids, radius=radius)
        loss = torch.nn.functional.mse_loss(recon, reference) + quant_loss
        loss.backward()
        total = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()
                    if p.grad is not None)
        self.assertGreater(total, 0.0)
        # the RADIUS must receive a gradient: the behaviour input is what the
        # mechanism turns into a code, and a detached radius would leave the
        # encoder's non-zero gradient coming from the reconstruction alone.
        self.assertIsNotNone(radius.grad)
        self.assertGreater(float(radius.grad.abs().sum()), 0.0)

    def test_a_different_radius_can_change_the_code(self):
        model, embeddings, ids = build()
        low = torch.full((24,), 0.05)
        high = torch.full((24,), 1.40)
        with torch.no_grad():
            tokens_low = model(embeddings, ids, radius=low)[3]
            tokens_high = model(embeddings, ids, radius=high)[3]
        self.assertFalse(torch.equal(tokens_low, tokens_high))

    def test_behaviour_off_removes_the_radius_difference(self):
        model, embeddings, ids = build(use_behaviour=False)
        radius = torch.linspace(0.05, 1.4, 24)
        from model.behaviour_radius import radii_for_switch
        applied = radii_for_switch(radius, use_behaviour=False)
        with torch.no_grad():
            a = model(embeddings, ids, radius=applied)[3]
            b = model(embeddings, ids,
                      radius=radii_for_switch(radius.flip(0), use_behaviour=False))[3]
        self.assertTrue(torch.equal(a, b))

    def test_geometry_off_uses_the_euclidean_score(self):
        model, embeddings, ids = build(use_curvature=False)
        layer = model.rq.vq_layers[0]
        latent = model.encoder(embeddings)
        with torch.no_grad():
            radial_cost = layer.radial_cost(latent)
            from model.behaviour_radius import squash
            euclid = (squash(latent).unsqueeze(1)
                      - squash(layer.get_code_embs()).unsqueeze(0)).pow(2).sum(-1)
        self.assertTrue(torch.allclose(radial_cost, euclid, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
