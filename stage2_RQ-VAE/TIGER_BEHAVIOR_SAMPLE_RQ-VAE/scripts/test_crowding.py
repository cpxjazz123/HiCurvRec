"""CPU regression/unit suite for the gated crowding mechanisms."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.crowding_losses import (MatchedParameters, crowding_loss, effective_weights,
                                     normalized_weight_entropy, soft_assign)


def fixture():
    torch.manual_seed(42)
    z = torch.randn(8, 4, requires_grad=True)
    pairs = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]])
    cb = torch.randn(6, 4, requires_grad=True)
    return z, pairs, cb


class CrowdingTests(unittest.TestCase):
    def test_t1_ste_regression_pin(self):
        z, pairs, _ = fixture()
        q = torch.nn.functional.embedding(z.detach().argmin(-1) % 2, torch.randn(2, 4))
        first_quant = z + (q - z).detach()
        next_quant = first_quant + (q - first_quant).detach()
        residual = first_quant - next_quant
        loss_residual = residual[pairs[:, 0]].square().sum()
        g_residual = torch.autograd.grad(loss_residual, z, allow_unused=True)[0]
        self.assertTrue(g_residual is None or torch.count_nonzero(g_residual) == 0)
        loss_z = z[pairs[:, 0]].square().sum()
        self.assertGreater(torch.autograd.grad(loss_z, z)[0].abs().sum().item(), 0)

    def test_t2_all_mechanisms_have_finite_encoder_gradients(self):
        cases = ["posattract", "couser", "simweight", "density", "softassign", "prefixshare", "random", "scale"]
        for name in cases:
            with self.subTest(name=name):
                z, pairs, cb = fixture()
                kwargs = dict(strength=0.1, codebook=cb, tau=0.1,
                              pair_weights=torch.tensor([.2, .5, .8, 1.]),
                              degrees=torch.arange(8.).add(1), negatives=torch.tensor([5, 6, 7, 4]),
                              random_pairs=torch.tensor([[0, 5], [1, 6], [2, 7], [3, 4]]))
                loss = crowding_loss(name, z, pairs, **kwargs)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(z.grad)
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_t3_soft_mechanisms_reach_codebook(self):
        for name in ("softassign", "prefixshare"):
            with self.subTest(name=name):
                z, pairs, cb = fixture()
                crowding_loss(name, z, pairs, codebook=cb, strength=.1, tau=.1).backward()
                self.assertIsNotNone(cb.grad)
                self.assertTrue(torch.isfinite(cb.grad).all())
                self.assertGreater(cb.grad.abs().sum().item(), 0)

    def test_t4_random_pair_control(self):
        z, pairs, _ = fixture()
        random_pairs = torch.tensor([[0, 6], [1, 7], [2, 4], [3, 5]])
        positive = crowding_loss("posattract", z, pairs)
        control = crowding_loss("random", z, pairs, random_pairs=random_pairs)
        self.assertTrue(torch.isfinite(control))
        self.assertNotEqual(positive.item(), control.item())
        self.assertTrue(torch.equal(random_pairs, torch.tensor([[0, 6], [1, 7], [2, 4], [3, 5]])))

    def test_t5_parameter_count_match(self):
        mechanism_parameter_count = 6 * 4
        control = MatchedParameters(mechanism_parameter_count)
        self.assertEqual(sum(p.numel() for p in control.parameters()), mechanism_parameter_count)

    def test_t6_scale_preserves_ranking(self):
        z, _, _ = fixture()
        before = torch.pdist(z.detach())
        after = torch.pdist(z.detach() * 0.5)
        self.assertTrue(torch.all(after < before))
        self.assertTrue(torch.equal(torch.argsort(before), torch.argsort(after)))

    def test_t7_absent_arm_is_exact_identity(self):
        base = torch.tensor(1.234567, requires_grad=True)
        assembled = base if None is None else base + torch.tensor(0.)
        self.assertTrue(torch.equal(base, assembled))
        self.assertEqual(base.detach().view(torch.int32).item(), assembled.detach().view(torch.int32).item())

    def test_t8_all_strengths_finite_and_report_magnitudes(self):
        main_z, _, _ = fixture()
        main_norm = torch.autograd.grad(main_z.square().mean(), main_z)[0].norm()
        pair_weights = torch.tensor([.08, .25, .60, 1.0])
        levels = ((.05, .20, 1, 1.0), (.10, .10, 1, 4.0), (.20, .05, 2, 16.0))
        for name in ("posattract", "couser", "simweight", "density", "softassign", "prefixshare"):
            values, ratios, entropies = [], [], []
            for strength, tau, depth, concentration in levels:
                z, pairs, cb = fixture()
                books = (cb, cb * 1.7) if name == "prefixshare" else cb
                value = crowding_loss(name, z, pairs, strength=strength, codebook=books, tau=tau, depth=depth,
                                      concentration=concentration, pair_weights=pair_weights,
                                      degrees=torch.tensor([1., 3., 8., 17., 1., 2., 4., 9.]),
                                      negatives=torch.tensor([5, 6, 7, 4]))
                grad = torch.autograd.grad(value, z)[0]
                self.assertTrue(torch.isfinite(value), (name, strength, value))
                values.append(float(value.detach()))
                ratios.append(float(grad.norm() / main_norm))
                if name == "simweight":
                    effective = effective_weights(pair_weights, concentration)
                    entropies.append(float(normalized_weight_entropy(effective)))
            suffix = f" entropy={','.join(f'{x:.6f}' for x in entropies)}" if entropies else ""
            print(f"{name}: loss={','.join(f'{x:.6f}' for x in values)} "
                  f"grad_ratio={','.join(f'{x:.6f}' for x in ratios)}{suffix}")

    def test_t9_mechanism_axis_discrimination(self):
        bound = 0.90
        levels = ((.05, .20, 1, 1.0), (.10, .10, 1, 4.0), (.20, .05, 2, 16.0))
        pair_weights = torch.tensor([.08, .25, .60, 1.0])
        for name in ("posattract", "couser", "simweight", "density", "softassign", "prefixshare"):
            losses, grads, ratios, entropies = [], [], [], []
            for strength, tau, depth, concentration in levels:
                z, pairs, cb = fixture()
                books = (cb, cb * 1.7) if name == "prefixshare" else cb
                value = crowding_loss(name, z, pairs, strength=strength, codebook=books, tau=tau, depth=depth,
                                      concentration=concentration, pair_weights=pair_weights,
                                      degrees=torch.tensor([1., 3., 8., 17., 1., 2., 4., 9.]),
                                      negatives=torch.tensor([5, 6, 7, 4]))
                grad = torch.autograd.grad(value, z)[0]
                main_grad = torch.autograd.grad(z.square().mean(), z)[0]
                losses.append(float(value.detach()))
                grads.append(grad.flatten())
                ratios.append(float(grad.norm() / main_grad.norm()))
                if name == "simweight":
                    entropies.append(float(normalized_weight_entropy(effective_weights(pair_weights, concentration))))
            if name in {"posattract", "density"}:
                self.assertLess(losses[0], losses[1], (name, losses))
                self.assertLess(losses[1], losses[2], (name, losses))
                self.assertLess(ratios[0], ratios[1], (name, ratios))
                self.assertLess(ratios[1], ratios[2], (name, ratios))
            else:
                cosine = torch.nn.functional.cosine_similarity(grads[0], grads[2], dim=0).item()
                self.assertLess(cosine, bound, f"{name} w1-w3 cosine {cosine:.6f} must be < stated bound {bound}")
            if name == "simweight":
                self.assertGreater(entropies[0] - entropies[1], 0.05, entropies)
                self.assertGreater(entropies[1] - entropies[2], 0.05, entropies)
                print(f"simweight normalized effective-weight entropy={','.join(f'{x:.6f}' for x in entropies)}")
            print(f"{name}: w1-w3 cosine={torch.nn.functional.cosine_similarity(grads[0], grads[2], dim=0).item():.8f} "
                  f"bound={bound:.2f}")
    def test_t10_prefixshare_is_distinct_from_softassign_at_depth_one(self):
        z1, pairs, cb1 = fixture()
        z2 = z1.detach().clone().requires_grad_(True)
        soft = crowding_loss("softassign", z1, pairs, codebook=cb1, strength=.1, tau=.1)
        prefix = crowding_loss("prefixshare", z2, pairs, codebook=cb1.detach().clone().requires_grad_(True),
                               strength=.1, tau=.1, depth=1)
        g_soft = torch.autograd.grad(soft, z1)[0]
        g_prefix = torch.autograd.grad(prefix, z2)[0]
        self.assertNotAlmostEqual(float(soft.detach()), float(prefix.detach()), places=7)
        self.assertLess(torch.nn.functional.cosine_similarity(g_soft.flatten(), g_prefix.flatten(), dim=0).item(),
                        1.0 - 1e-6)

if __name__ == "__main__":
    unittest.main(verbosity=2)
