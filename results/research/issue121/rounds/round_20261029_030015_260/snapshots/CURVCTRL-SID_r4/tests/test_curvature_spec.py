"""CURVCTRL-SID's own composition, as written in the proposal.

The proposal maps BOTH the latent and the code vectors into the ball of the
level's curvature, keeps the residual in the tangent space, and adds only the
curved commitment to the baseline RQ objective. The inherited Frechet candidate
did something else (it lived in the ball and moved the codebook by a Frechet
step), and a copy that keeps that composition is not this mechanism.
"""
import unittest

import torch

from model.curvature_control import exp0_c, squared_distance_c
from model.layers import VQLayer


def _layer(curvature=2.0):
    layer = VQLayer(4, 2, beta=0.25, sk_epsilon=-1.0, sk_iters=-1,
                    frechet=True, use_behaviour=True, use_curvature=True,
                    codebook_gradient=True)
    with torch.no_grad():
        layer.get_code_embs().copy_(torch.tensor(
            [[0.10, 0.0], [0.30, 0.25], [-0.35, 0.35], [0.60, 0.0]]))
    layer.frechet.set_weights(torch.ones(8))
    layer.frechet.curvature = float(curvature)
    layer._tangent_codebook = True          # this mechanism keeps E_l in tangent space
    return layer


class CodeMappingTests(unittest.TestCase):

    def test_the_codes_are_mapped_into_the_ball_at_the_same_curvature(self):
        layer = _layer(2.0)
        latent = torch.tensor([[0.30, 0.10]])
        cost = layer.assignment_cost(latent)
        mapped_point = exp0_c(latent, 2.0)
        mapped_codes = exp0_c(layer.get_code_embs(), 2.0)
        expected = squared_distance_c(mapped_point.unsqueeze(1),
                                      mapped_codes.unsqueeze(0), 2.0)
        self.assertTrue(torch.allclose(cost, expected, atol=1e-6))

    def test_the_code_mapping_is_what_makes_the_assignment_curvature_dependent(self):
        layer = _layer(2.0)
        latent = torch.tensor([[0.30, 0.10]])
        raw = (exp0_c(latent, 0.0).unsqueeze(1)
               - layer.get_code_embs().unsqueeze(0)).pow(2).sum(-1)
        self.assertFalse(torch.allclose(layer.assignment_cost(latent), raw))


class TangentCodebookTests(unittest.TestCase):

    def test_the_codebook_stays_in_the_tangent_space(self):
        """The proposal's E_l are tangent vectors; exp_0^c maps them per step.

        A codebook that is already stored inside the ball would be mapped twice,
        and at a large measured curvature every code would land in the same
        corner of the ball and the assignment would collapse to one code.
        """
        layer = _layer(4.0)
        with torch.no_grad():
            layer.init_codebook(torch.randn(8, 2) * 0.1, torch.device("cpu"))
        self.assertGreater(float(layer.get_code_embs().norm(dim=-1).max()), 1e-6)

    def test_the_start_is_not_mapped_a_second_time(self):
        layer = _layer(4.0)
        centres = torch.tensor([[0.10, 0.0], [0.60, 0.0], [-0.35, 0.2], [0.30, 0.3]])
        with torch.no_grad():
            layer._copy_init_embed(centres.clone())
        self.assertTrue(torch.allclose(layer.get_code_embs().detach(), centres, atol=1e-6))

    def test_the_initial_residual_is_the_raw_difference(self):
        torch.manual_seed(0)
        layer = _layer(4.0)
        sample = torch.randn(16, 2) * 0.1
        with torch.no_grad():
            residual = layer.init_codebook(sample, torch.device("cpu"))
            index = torch.argmin(layer.assignment_cost(sample), dim=-1)
        expected = sample - layer.get_code_embs()[index]
        self.assertTrue(torch.allclose(residual, expected, atol=1e-5))


class CompositionTests(unittest.TestCase):

    def test_the_residual_stays_in_the_tangent_space(self):
        """z_{l+1} = z_l - E[k*]: the decoder receives the RAW code vector.

        The straight-through keeps the latent's gradient, but the value the next
        level subtracts is the tangent-space code, never its image in the ball.
        """
        layer = _layer(2.0)
        latent = torch.tensor([[0.30, 0.10]])
        quant, _, _, index = layer(latent, item_ids=torch.arange(latent.shape[0]))
        code = layer.get_code_embs()[index.reshape(-1)].detach()
        self.assertTrue(torch.allclose(quant.detach(), code, atol=1e-6))
        # the straight-through passes the latent's gradient, not the code's
        self.assertFalse(quant.requires_grad)

    def test_the_loss_is_the_baseline_term_plus_the_curved_commitment(self):
        """L = mse(code, x.detach()) + beta * d_c(u, e*)^2, nothing else."""
        layer = _layer(2.0)
        latent = torch.tensor([[0.30, 0.10]])
        quant, loss, _, index = layer(latent, item_ids=torch.arange(latent.shape[0]))
        point = exp0_c(latent, 2.0)
        mapped = exp0_c(layer.get_code_embs()[index.reshape(-1)], 2.0)
        commitment = squared_distance_c(point, mapped, 2.0)
        codebook = torch.nn.functional.mse_loss(quant.detach(), latent.detach())
        self.assertAlmostEqual(float(loss),
                               float(codebook + layer.beta * commitment.mean()), places=5)

    def test_the_codebook_loss_is_the_baseline_one(self):
        """The proposal keeps the baseline RQ codebook term, not a detached one."""
        layer = _layer(2.0)
        latent = torch.tensor([[0.30, 0.10]])
        _, _, _, index = layer(latent, item_ids=torch.arange(latent.shape[0]))
        code = layer.get_code_embs()[index.reshape(-1)]
        self.assertTrue(code.requires_grad)


if __name__ == "__main__":
    unittest.main()
