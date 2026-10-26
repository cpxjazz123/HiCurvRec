import importlib
import pathlib
import sys
import types
import unittest

import torch


SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1] / 'stage2_RQ-VAE' / 'TIGER_c117_asymmetric_successor_branches'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

hyperbolic = importlib.import_module('model.hyperbolic')
poincare_residual = importlib.import_module('model.poincare_residual')
model_module = importlib.import_module('model.model')


class TestGradientFlow(unittest.TestCase):
    def test_squared_distance_gradcheck_and_finite_difference(self):
        # Stay strictly within the projected ball and away from the arcosh cusp.
        x = torch.tensor([[0.17, -0.11, 0.23]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-0.19, 0.08, 0.04]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: hyperbolic.squared_distance(a, b, c=1.0),
            (x, y), eps=1e-6, atol=2e-5, rtol=2e-4))
        value = hyperbolic.squared_distance(x, y, c=1.0).sum()
        gx, = torch.autograd.grad(value, x)
        step = 1e-6
        direction = torch.tensor([[0.3, -0.2, 0.4]], dtype=torch.double)
        with torch.no_grad():
            plus = hyperbolic.squared_distance(x + step * direction, y, c=1.0).sum()
            minus = hyperbolic.squared_distance(x - step * direction, y, c=1.0).sum()
        finite_difference = (plus - minus) / (2 * step)
        autodiff_directional = (gx * direction).sum()
        self.assertTrue(torch.allclose(autodiff_directional, finite_difference, atol=2e-6, rtol=2e-5))
        self.assertGreater(float(gx.norm()), 1e-5)

    def test_curved_and_flat_distance_have_finite_nonzero_gradients(self):
        for curvature in (0.0, 1.0):
            with self.subTest(curvature=curvature):
                x = torch.tensor([[0.21, -0.13]], dtype=torch.double, requires_grad=True)
                y = torch.tensor([[-0.07, 0.19]], dtype=torch.double, requires_grad=True)
                loss = hyperbolic.squared_distance(x, y, c=curvature).sum()
                gx, gy = torch.autograd.grad(loss, (x, y))
                self.assertTrue(torch.isfinite(gx).all().item())
                self.assertTrue(torch.isfinite(gy).all().item())
                self.assertGreater(float(gx.norm()), 1e-7)
                self.assertGreater(float(gy.norm()), 1e-7)

    def test_ste_reconstruction_reaches_latent_and_selected_codebook(self):
        config = types.SimpleNamespace(codebook_num=2, codebook_size=(3, 3), codebook_dim=2, curvature=1.0, radius=0.9)
        quantizer = poincare_residual.SuccessorBranchQuantiser(config).double()
        with torch.no_grad():
            quantizer.codebooks[0].copy_(torch.tensor([[0.20, 0.00], [-0.30, 0.10], [0.00, -0.35]], dtype=torch.double))
            quantizer.codebooks[1].copy_(torch.tensor([[0.12, 0.03], [-0.20, 0.22], [0.31, -0.08]], dtype=torch.double))
        latent = torch.tensor([[0.37, 0.16], [-0.22, 0.28]], dtype=torch.double, requires_grad=True)
        reconstruction, tokens, residuals, selections = quantizer(latent)
        target = torch.tensor([[0.05, -0.09], [-0.08, 0.13]], dtype=torch.double)
        loss = ((reconstruction - target) ** 2).mean()
        loss.backward()
        self.assertEqual(tuple(tokens.shape), (2, 2))
        self.assertTrue(torch.isfinite(latent.grad).all().item())
        self.assertGreater(float(latent.grad.norm()), 1e-8)
        self.assertTrue(torch.isfinite(residuals).all().item())
        for level, codebook in enumerate(quantizer.codebooks):
            self.assertIsNotNone(codebook.grad)
            self.assertTrue(torch.isfinite(codebook.grad).all().item())
            selected = tokens[:, level].unique()
            selected_grad = codebook.grad.index_select(0, selected)
            self.assertGreater(float(selected_grad.norm()), 1e-9)
            unselected_mask = torch.ones(codebook.shape[0], dtype=torch.bool)
            unselected_mask[selected] = False
            if unselected_mask.any():
                self.assertTrue(torch.equal(codebook.grad[unselected_mask], torch.zeros_like(codebook.grad[unselected_mask])))

    def test_real_model_objective_updates_encoder_and_codes_but_not_frozen_target(self):
        config = types.SimpleNamespace(hidden_sizes=(5,), dropout=0.0, codebook_num=2,
                                       codebook_size=(3, 3), codebook_dim=2, curvature=1.0, radius=0.9,
                                       beta_commit=0.25, beta_code=1.0)
        torch.manual_seed(19)
        model = model_module.RQVAE(config, in_dim=4).double()
        embeddings = torch.tensor([[0.4, -0.2, 0.1, 0.3], [-0.3, 0.5, 0.2, -0.1],
                                   [0.1, 0.2, -0.4, 0.6]], dtype=torch.double)
        model.freeze_targets(embeddings)
        with torch.no_grad():
            for level, codebook in enumerate(model.rq.codebooks):
                codebook.copy_(torch.tensor([[0.25, 0.05], [-0.2, 0.25], [0.0, -0.3]], dtype=torch.double) + level * 0.015)
        loss, rec, commit, code, _tokens = model(embeddings)
        self.assertTrue(all(torch.isfinite(v).item() for v in (loss, rec, commit, code)))
        loss.backward()
        encoder_grads = [p.grad for p in model.encoder.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all().item() for g in encoder_grads))
        self.assertGreater(sum(float(g.norm()) for g in encoder_grads), 1e-9)
        for parameter in model.target_encoder.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        for codebook in model.rq.codebooks:
            self.assertIsNotNone(codebook.grad)
            self.assertTrue(torch.isfinite(codebook.grad).all().item())
            self.assertGreater(float(codebook.grad.norm()), 1e-9)

    def test_disconnected_graph_negative_is_detected(self):
        # A detached branch must not silently masquerade as a valid gradient path.
        source = torch.tensor([0.4, -0.2], requires_grad=True)
        disconnected_loss = source.detach().square().sum()
        self.assertFalse(disconnected_loss.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(disconnected_loss, source)

    def test_nan_gradient_negative_is_detected(self):
        # Verify the test's finite-gradient guard rejects an actual NaN gradient.
        value = torch.tensor(0.0, requires_grad=True)
        bad = torch.sqrt(value)
        gradient, = torch.autograd.grad(bad, value)
        self.assertTrue(torch.isnan(gradient).item())
        self.assertFalse(torch.isfinite(gradient).item())


if __name__ == '__main__':
    unittest.main()
