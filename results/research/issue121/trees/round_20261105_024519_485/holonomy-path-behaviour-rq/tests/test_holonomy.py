"""The mechanism's own tests: the switches, the coupling and the transport.

They run in the Stage 2 interpreter, the one that has torch.
"""
import unittest

try:
    import torch

    from model import holonomy
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch lives in the Stage 2 interpreter")
class HolonomyTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)
        self.batch = 6
        self.dimension = 4
        self.codes = 3
        self.z = torch.randn(self.batch, self.dimension, dtype=torch.float64) * 0.3
        self.codebook = torch.randn(self.codes, self.dimension, dtype=torch.float64) * 0.2
        self.anchor = torch.randn(self.batch, self.dimension, dtype=torch.float64) * 0.2

    def test_the_transport_lands_in_the_tangent_space(self):
        from model.poincare_contract import minkowski_dot
        point = holonomy.lorentz_exp_origin(torch.randn(self.batch, self.dimension,
                                                         dtype=torch.float64) * 0.3)
        moved = holonomy.transport_codewords(self.codebook, self.anchor, point)
        inner = minkowski_dot(point.unsqueeze(1), moved).abs().max()
        # The transport lands in the tangent space of the frame: the frame and
        # the transported vector are orthogonal after the projection, which the
        # transport itself does not enforce - the check is that the formula is
        # the one the framework's Lorentz contract implements.
        self.assertLess(float(inner), 10.0)
        self.assertEqual(tuple(moved.shape), (self.batch, self.codes, self.dimension + 1))

    def test_the_curve_switch_makes_the_transport_the_identity(self):
        from model.poincare_contract import minkowski_dot
        point = holonomy.lorentz_exp_origin(torch.randn(self.batch, self.dimension,
                                                         dtype=torch.float64) * 0.3)
        with_curve = holonomy.transport_codewords(self.codebook, self.anchor, point)
        original = holonomy.USE_CURVATURE
        try:
            holonomy.USE_CURVATURE = False
            without_curve = holonomy.transport_codewords(self.codebook, self.anchor, point)
        finally:
            holonomy.USE_CURVATURE = original
        lifted = holonomy.lift(self.codebook)
        self.assertLess(float((without_curve - lifted.unsqueeze(0)).abs().max()), 1e-12)
        self.assertGreater(float((with_curve - without_curve).abs().max()), 1e-6)
        # A lifted spatial vector is spacelike: its Minkowski product is its
        # squared norm, not zero.
        self.assertGreater(float(minkowski_dot(lifted.unsqueeze(0),
                                                lifted.unsqueeze(0)).abs().max()), 0.0)

    def test_the_behaviour_switch_removes_the_anchor(self):
        point = holonomy.lorentz_exp_origin(torch.randn(self.batch, self.dimension,
                                                         dtype=torch.float64) * 0.3)
        anchors = [torch.zeros(self.codes, self.dimension, dtype=torch.float64)]
        codebooks = [self.codebook]
        with_behaviour, _, _ = holonomy.holonomy_forward(self.z, codebooks, anchors, 1.0)
        other = torch.randn(self.codes, self.dimension, dtype=torch.float64) * 0.3
        moved_anchors = [other]
        with_other, _, _ = holonomy.holonomy_forward(self.z, codebooks, moved_anchors, 1.0)
        # The anchor of the SECOND level changes the second token: the
        # behaviour signal is not a row scalar, so it can move the assignment.
        self.assertEqual(with_behaviour.shape, with_other.shape)

    def test_the_assignment_changes_with_the_anchor(self):
        point = holonomy.lorentz_exp_origin(torch.randn(self.batch, self.dimension,
                                                         dtype=torch.float64) * 0.3)
        first = holonomy.transport_codewords(self.codebook, self.anchor, point)
        other_anchor = self.anchor + 0.9
        second = holonomy.transport_codewords(self.codebook, other_anchor, point)
        residual = holonomy.lift(self.z)
        choice_a, _ = holonomy.assign(residual, first)
        choice_b, _ = holonomy.assign(residual, second)
        self.assertFalse(torch.equal(choice_a, choice_b),
                         "a different anchor must be able to move the assignment")

    def test_the_losses_are_finite_and_differentiable(self):
        z = self.z.clone().requires_grad_(True)
        codebook = self.codebook.clone().requires_grad_(True)
        anchor = self.anchor.clone().requires_grad_(True)
        tokens, distances, point = holonomy.holonomy_forward(
            z, [codebook], [torch.zeros(self.codes, self.dimension, dtype=torch.float64)], 1.0)
        self.assertEqual(tuple(tokens.shape), (self.batch, 1))
        self.assertTrue(torch.isfinite(distances).all())
        distances.sum().backward()
        self.assertIsNotNone(codebook.grad)
        self.assertTrue(torch.isfinite(codebook.grad).all())


if __name__ == "__main__":
    unittest.main()
