"""The stratified mechanism: the curvature schedule, the balanced plan, the switches."""
import unittest

try:
    import torch

    from model import stratified
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch lives in the Stage 2 interpreter")
class StratifiedTests(unittest.TestCase):

    def test_a_bigger_level_scale_means_a_smaller_curvature(self):
        curvature = stratified.curvature_of_scale(torch.tensor([4.0]))
        self.assertLess(float(curvature.abs()), 1.0)
        flat = stratified.curvature_of_scale(torch.tensor([0.5]))
        self.assertGreater(float(flat.abs()), float(curvature.abs()))

    def test_the_exponential_map_lands_inside_the_ball(self):
        torch.manual_seed(3)
        vector = torch.randn(32, 4, dtype=torch.float64)
        curvature = stratified.curvature_of_scale(torch.tensor([1.0]))
        mapped = stratified.exp0_curvature(vector, curvature)
        self.assertLess(float(mapped.norm(dim=-1).max()), 1.0)

    def test_the_euclidean_branch_is_the_identity(self):
        vector = torch.randn(8, 4, dtype=torch.float64) * 0.2
        original = stratified.USE_HYPERBOLIC
        try:
            stratified.USE_HYPERBOLIC = False
            mapped = stratified.exp0_curvature(vector, torch.tensor([1.0]))
            distance = stratified.distance_curvature(
                vector.unsqueeze(1), vector.unsqueeze(0), torch.tensor([1.0]))
        finally:
            stratified.USE_HYPERBOLIC = original
        self.assertLess(float((mapped - vector).abs().max()), 1e-12)
        self.assertGreaterEqual(float(distance.min()), 0.0)
        self.assertAlmostEqual(float(distance.max()),
                               2 * float((vector.unsqueeze(1) - vector.unsqueeze(0)).norm(dim=-1).max()),
                               places=9)

    def test_the_sinkhorn_plan_balances_the_codebook(self):
        torch.manual_seed(6)
        cost = torch.randn(64, 8, dtype=torch.float64)
        plan = stratified.sinkhorn_plan(cost).exp()
        self.assertAlmostEqual(float(plan.sum(dim=0).std()), 0.0, places=3)
        # the marginals are 1/B per row and 1/K per column: the plan is a
        # probability plan and its total mass is one
        self.assertAlmostEqual(float(plan.sum()), 1.0, places=3)

    def test_the_balanced_assignment_spreads_the_items(self):
        torch.manual_seed(8)
        point = torch.randn(64, 4, dtype=torch.float64) * 0.3
        codeword = torch.randn(8, 4, dtype=torch.float64) * 0.3
        curvature = stratified.curvature_of_scale(torch.tensor([1.0]))
        cost = stratified.cost_between(point, codeword, curvature)
        choice, _ = stratified.assign_balanced(cost)
        self.assertGreater(len(set(choice.tolist())), 1)

    def test_the_behaviour_switch_drops_the_centroid_term(self):
        torch.manual_seed(9)
        point = torch.randn(16, 4, dtype=torch.float64) * 0.3
        codeword = torch.randn(5, 4, dtype=torch.float64) * 0.3
        centroid = torch.randn(16, 4, dtype=torch.float64) * 0.3
        curvature = stratified.curvature_of_scale(torch.tensor([1.0]))
        with_centre = stratified.cost_between(point, codeword, curvature, centroid)
        original = stratified.USE_BEHAVIOUR
        try:
            stratified.USE_BEHAVIOUR = False
            without = stratified.cost_between(point, codeword, curvature, centroid)
        finally:
            stratified.USE_BEHAVIOUR = original
        self.assertGreater(float((with_centre - without).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
