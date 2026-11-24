"""Independent geometry references and frozen hard/soft assignment contracts."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch


MODEL = Path(__file__).resolve().parent.parent / "model"


def load_module(name):
    spec = importlib.util.spec_from_file_location("issue250_" + name, MODEL / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lorentz_squared(x, y, curvature):
    """Independent hyperboloid exponential and Lorentz-product reference."""
    k = math.sqrt(-curvature)
    rx = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    ry = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    tx, ty = torch.cosh(k * rx), torch.cosh(k * ry)
    sx = torch.sinh(k * rx) * x / rx.clamp_min(1e-30)
    sy = torch.sinh(k * ry) * y / ry.clamp_min(1e-30)
    inner = tx @ ty.T - sx @ sy.T
    return torch.acosh(inner.clamp_min(1)).square() / (-curvature)


def sphere_squared(x, y, curvature):
    """Independent ambient exponential, dot product and acos reference."""
    k = math.sqrt(curvature)
    rx = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    ry = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    ax = torch.cat((torch.cos(k * rx), torch.sin(k * rx) * x / rx.clamp_min(1e-30)), -1)
    ay = torch.cat((torch.cos(k * ry), torch.sin(k * ry) * y / ry.clamp_min(1e-30)), -1)
    return torch.acos((ax @ ay.T).clamp(-1, 1)).square() / curvature


def frozen_center(cost):
    middle = (cost.max() + cost.min()) / 2
    return (cost - middle) / (cost.max() - middle + 1e-5)


def frozen_sinkhorn(cost, epsilon, iterations):
    """Unmodified probability-domain frozen TIGER arithmetic, float64."""
    q = torch.exp(-frozen_center(cost).double() / epsilon)
    b, j = q.shape
    q = q / q.sum()
    for _ in range(iterations):
        q = q / q.sum(0, keepdim=True) / j
        q = q / q.sum(1, keepdim=True) / b
    return q * b


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = load_module("geometry")

    def test_independent_lorentz_sphere_references(self):
        x = torch.tensor([[.3, -.2, .1], [-.4, .1, .2]], dtype=torch.float64)
        y = torch.tensor([[.1, .5, -.2], [.7, -.1, .3], [0., 0., 0.]], dtype=torch.float64)
        actual = self.geo.expert_squared(x, y, scale=1.7)
        references = [lorentz_squared(x / 1.7, y / 1.7, c) if c < 0 else
                      sphere_squared(x / 1.7, y / 1.7, c) if c > 0 else
                      (x[:, None] / 1.7 - y[None] / 1.7).square().sum(-1)
                      for c in (-3, -1, 0, 1, 3)]
        torch.testing.assert_close(actual, torch.stack(references, -1), rtol=1e-10, atol=1e-12)

    def test_hand_answers_common_tangent_and_spherical_wrap(self):
        x = torch.tensor([[0., 0.]], dtype=torch.float64)
        y = torch.tensor([[.5, 0.]], dtype=torch.float64)
        torch.testing.assert_close(self.geo.expert_squared(x, y), torch.full((1, 1, 5), .25, dtype=x.dtype), rtol=1e-12, atol=1e-12)
        # A 2*pi tangent displacement returns to the unit sphere origin.
        wrapped = self.geo.expert_squared(x, torch.tensor([[2 * math.pi, 0.]], dtype=x.dtype))
        self.assertLess(wrapped[0, 0, 3].item(), 1e-26)

    def test_zero_coincident_finite_gradients(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([[0., 0.], [.4, -.2]], dtype=dtype, requires_grad=True)
            d = self.geo.expert_squared(x, x)
            self.assertTrue(torch.equal(d.diagonal(dim1=0, dim2=1), torch.zeros((5, 2), dtype=dtype)))
            d.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_hyperbolic_saturation_keeps_radial_distance(self):
        # tanh(1000/2) rounds to one; clipping ball points loses this distance.
        x = torch.tensor([[1000., 0.], [0., 0.]], requires_grad=True)
        y = torch.tensor([[999., 0.], [-1000., 0.]])
        d = self.geo.expert_squared(x, y)
        self.assertTrue(torch.isfinite(d).all())
        torch.testing.assert_close(d[0, 0, :2], torch.ones(2), rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(d[0, 1, :2], torch.full((2,), 4000000.), rtol=2e-5, atol=1.)
        d.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_fixed_mean_switches_and_broadcasting(self):
        x = torch.tensor([[.7, .2], [-.1, .3]], dtype=torch.float64)
        y = torch.tensor([[1., -.3], [.1, .4], [0., 0.]], dtype=x.dtype)
        experts = self.geo.expert_squared(x, y)
        uniform = torch.full((2, 5), .2, dtype=x.dtype)
        torch.testing.assert_close(self.geo.fixed_squared(x, y), experts.mean(-1))
        torch.testing.assert_close(self.geo.fixed_squared(x, y), self.geo.fixed_squared(y, x).T)
        torch.testing.assert_close(self.geo.fixed_squared(x[:, None], y[None]), experts.mean(-1))
        torch.testing.assert_close(self.geo.mixed_cost(x, y, uniform), experts.mean(-1))
        weights = torch.softmax(torch.arange(10., dtype=x.dtype).reshape(2, 5), -1)
        euclidean = (x[:, None] - y[None]).square().sum(-1)
        torch.testing.assert_close(self.geo.mixed_cost(x, y, weights, curvature_on=False), euclidean)
        for expert in range(5):
            onehot = torch.zeros_like(uniform)
            onehot[:, expert] = 1
            torch.testing.assert_close(self.geo.mixed_cost(x, y, onehot), experts[..., expert])

    def test_smooth_cost_finite_differences(self):
        x = torch.tensor([[.3, -.2], [-.1, .4]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[.6, .1], [-.3, -.1]], dtype=x.dtype, requires_grad=True)
        weights = torch.tensor([[.1, .2, .3, .2, .2], [.3, .1, .1, .2, .3]], dtype=x.dtype, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b, w: self.geo.mixed_cost(a, b, w, scale=1.3), (x, y, weights), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_controlled_weight_intervention_changes_hard_id(self):
        selection = load_module("selection")
        x = torch.tensor([[1., 0.]], dtype=torch.float64)
        codewords = torch.tensor([[1., .6], [1.8, 0.]], dtype=x.dtype)
        euclidean = torch.tensor([[0., 0., 1., 0., 0.]], dtype=x.dtype)
        hyperbolic = torch.tensor([[1., 0., 0., 0., 0.]], dtype=x.dtype)
        ids = []
        for weights in (euclidean, hyperbolic):
            cost = self.geo.mixed_cost(x, codewords, weights)
            ids.append(selection.select(cost, codewords, balanced=False, epsilon=.003, iterations=50)["ids"].item())
        self.assertEqual(ids, [0, 1], "gate must change the actual selected discrete codeword")


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = load_module("selection")

    def test_frozen_hard_rule_exact_and_inference_argmin(self):
        cost = torch.tensor([[0., .001, .002], [.0003, .0002, .002], [.001, .0012, 0.], [.0001, .0009, .001]], dtype=torch.float64)
        codewords = torch.eye(3, dtype=cost.dtype)
        balanced = self.selection.select(cost, codewords, balanced=True, epsilon=.003, iterations=50)
        self.assertTrue(torch.equal(balanced["ids"], frozen_sinkhorn(cost, .003, 50).argmax(-1)))
        torch.testing.assert_close(balanced["probabilities"], frozen_sinkhorn(cost, .003, 50), rtol=1e-11, atol=1e-13)
        for training, use_balance in ((False, True), (True, False), (False, False)):
            result = self.selection.select(cost, codewords, balanced=use_balance, epsilon=.003, iterations=50, training=training)
            self.assertTrue(torch.equal(result["ids"], cost.argmin(-1)))
            torch.testing.assert_close(result["probabilities"], torch.softmax(-frozen_center(cost) / .003, -1))

    def test_balancing_cannot_be_replaced_by_argmin(self):
        cost = torch.tensor([[0., 1e-9], [0., 3e-9]], dtype=torch.float64)
        words = torch.eye(2, dtype=cost.dtype)
        balanced = self.selection.select(cost, words, balanced=True, epsilon=.003, iterations=50)
        self.assertEqual(balanced["ids"].tolist(), [1, 0])
        inference = self.selection.select(cost, words, balanced=True, epsilon=.003, iterations=50, training=False)
        self.assertEqual(inference["ids"].tolist(), [0, 0])

    def test_balanced_marginals_and_hand_answer(self):
        cost = torch.zeros((4, 2), dtype=torch.float64)
        result = self.selection.select(cost, torch.eye(2, dtype=cost.dtype), balanced=True, epsilon=.003, iterations=50)
        self.assertTrue(torch.equal(result["ids"], torch.zeros(4, dtype=torch.long)))
        torch.testing.assert_close(result["probabilities"], torch.full((4, 2), .5, dtype=cost.dtype))
        torch.testing.assert_close(result["probabilities"].sum(1), torch.ones(4, dtype=cost.dtype))
        torch.testing.assert_close(result["probabilities"].sum(0), torch.full((2,), 2., dtype=cost.dtype))
        # Nonconstant but diffuse rectangular matrix checks both normalization axes.
        ramp = torch.arange(12., dtype=cost.dtype).reshape(4, 3) * 1e-10
        p = self.selection.select(ramp, torch.eye(3, dtype=cost.dtype), balanced=True, epsilon=.003, iterations=50)["probabilities"]
        torch.testing.assert_close(p.sum(1), torch.ones(4, dtype=cost.dtype))
        torch.testing.assert_close(p.sum(0), torch.full((3,), 4 / 3, dtype=cost.dtype), rtol=1e-10, atol=1e-10)

    def test_exact_hard_forward_and_only_soft_backward(self):
        cost = torch.tensor([[0., 1e-8, 2e-8], [2e-8, 0., 1e-8]], dtype=torch.float64, requires_grad=True)
        codewords = torch.tensor([[1e16, .1], [-3.7, 1e-16], [7.1, -1e16]], dtype=cost.dtype, requires_grad=True)
        result = self.selection.select(cost, codewords, balanced=False, epsilon=.003, iterations=50)
        self.assertTrue(torch.equal(result["q"], result["hard"]))
        self.assertTrue(torch.equal(result["hard"], codewords[result["ids"]]))
        upstream = torch.tensor([[.1, .2], [-.4, .7]], dtype=cost.dtype)
        actual = torch.autograd.grad((result["q"] * upstream).sum(), (cost, codewords), retain_graph=True)
        expected = torch.autograd.grad(((result["probabilities"] @ codewords) * upstream).sum(), (cost, codewords))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_soft_finite_differences_both_paths(self):
        for balanced in (False, True):
            cost = torch.tensor([[0., 2e-9, 1e-9], [3e-9, -1e-9, 4e-9]], dtype=torch.float64, requires_grad=True)
            words = torch.eye(3, dtype=cost.dtype)
            self.assertTrue(torch.autograd.gradcheck(lambda c: self.selection.select(c, words, balanced=balanced, epsilon=.003, iterations=50)["probabilities"], (cost,), eps=1e-11, atol=2e-3, rtol=2e-4))

    def test_saturated_probabilities_are_finite(self):
        cost = torch.tensor([[0., 1., 2.], [2., 0., 1.], [1., 2., 0.]], requires_grad=True)
        for balanced in (False, True):
            result = self.selection.select(cost, torch.eye(3), balanced=balanced, epsilon=.003, iterations=50)
            self.assertTrue(torch.isfinite(result["probabilities"]).all())
            self.assertTrue(torch.equal(result["q"], result["hard"]))
            gradient, = torch.autograd.grad(result["probabilities"].square().sum(), cost, retain_graph=True)
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
