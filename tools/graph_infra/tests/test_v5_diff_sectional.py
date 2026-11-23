"""Issue #249: the differentiable statistic, the metric families and the gradient diagnostics.

The expectations are written by hand for the small graphs; the reference implementation in
results/research/issue249/reference.py is loaded by path and never imports the production module.
"""
import importlib.util
import math
import unittest
from pathlib import Path
import numpy as np
import torch

from tools.graph_infra import diff_sectional

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('reference249',
                                              REPO / 'results/research/issue249/reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

torch.set_default_dtype(torch.float64)


def pairwise_metric(metric, vectors, left, right):
    """Distances between two index sets, as a plain float list."""
    out = []
    for i in left:
        for j in right:
            out.append(float(metric(vectors[i], vectors[j])))
    return out


class MetricTests(unittest.TestCase):
    def test_euclidean_hand_computed(self):
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        self.assertAlmostEqual(float(diff_sectional.euclidean_distance(x[0], x[1])), 5.0, places=12)

    def test_zero_distance_is_exactly_zero(self):
        for family, kappa in [('euclidean', 0.0), ('hyperbolic', 1.0), ('spherical', 1.0)]:
            metric = diff_sectional.build_metric(family, kappa)
            x = torch.tensor([0.3, -0.7, 1.1])
            self.assertEqual(float(metric(x, x)), 0.0, msg=family)

    def test_symmetry(self):
        torch.manual_seed(3)
        x = torch.randn(6, 4) * 0.3
        y = torch.randn(6, 4) * 0.3
        for family, kappa in [('euclidean', 0.0), ('hyperbolic', 1.0), ('spherical', 1.0)]:
            metric = diff_sectional.build_metric(family, kappa)
            left = metric(x, y)
            right = metric(y, x)
            self.assertTrue(torch.allclose(left, right, atol=1e-12), msg=family)

    def test_hyperbolic_matches_the_reference(self):
        torch.manual_seed(5)
        x = torch.randn(5, 3) * 0.4
        y = torch.randn(5, 3) * 0.4
        metric = diff_sectional.build_metric('hyperbolic', 1.0)
        produced = metric(x, y).tolist()
        for i in range(5):
            expected = reference.hyperbolic(x[i].tolist(), y[i].tolist(), 1.0)
            self.assertAlmostEqual(produced[i], expected, places=9,
                                   msg=f'hyperbolic {i}')

    def test_spherical_matches_the_reference(self):
        torch.manual_seed(6)
        x = torch.randn(5, 3) * 0.4
        y = torch.randn(5, 3) * 0.4
        metric = diff_sectional.build_metric('spherical', 1.0)
        produced = metric(x, y).tolist()
        for i in range(5):
            expected = reference.spherical(x[i].tolist(), y[i].tolist(), 1.0)
            self.assertAlmostEqual(produced[i], expected, places=9,
                                   msg=f'spherical {i}')

    def test_hyperbolic_and_spherical_reject_a_non_positive_kappa(self):
        x = torch.zeros(2)
        with self.assertRaises(ValueError):
            diff_sectional.hyperbolic_distance(x, x, kappa=0.0)
        with self.assertRaises(ValueError):
            diff_sectional.spherical_distance(x, x, kappa=-1.0)

    def test_mixed_metric_is_a_weighted_blend(self):
        x = torch.tensor([[0.1, 0.2]])
        y = torch.tensor([[0.4, -0.3]])
        weights = torch.tensor([0.5, 0.25, 0.25])
        kappas = torch.tensor([0.0, 1.0, 1.0])
        scales = torch.tensor([1.0, 1.0, 1.0])
        produced = float(diff_sectional.mixed_distance(x, y, weights, kappas, scales))
        expected = (0.5 * float(diff_sectional.euclidean_distance(x[0], y[0]))
                    + 0.25 * float(diff_sectional.hyperbolic_distance(x[0], y[0], 1.0))
                    + 0.25 * float(diff_sectional.spherical_distance(x[0], y[0], 1.0)))
        self.assertAlmostEqual(produced, expected, places=12)


class StatisticTests(unittest.TestCase):
    def test_hand_computed_graph(self):
        # path 0-1-2 with centre 1 and pair (0,2); the reference set is the whole graph.
        d_am = [1.0, 0.0, 1.0]   # d(a, 1)
        d_ab = [0.0, 1.0, 2.0]   # d(a, 0)
        d_ac = [2.0, 1.0, 0.0]   # d(a, 2)
        d_bc = 2.0
        produced = float(diff_sectional.sectional_statistic(
            {'d_am': torch.tensor(d_am), 'd_ab': torch.tensor(d_ab),
             'd_ac': torch.tensor(d_ac), 'd_bc': torch.tensor(d_bc)}))
        self.assertAlmostEqual(produced, 0.0, places=12)
        expected = float(reference.exact_fraction_statistic(d_am, d_ab, d_ac, d_bc))
        self.assertAlmostEqual(produced, expected, places=12)

    def test_star_graph_is_minus_one_half(self):
        # K1,3 with centre 0 and pair (1,2): the closed form gives -0.5.
        d_am = [1.0, 0.0, 1.0, 1.0]
        d_ab = [0.0, 1.0, 2.0, 2.0]
        d_ac = [2.0, 1.0, 0.0, 2.0]
        d_bc = 2.0
        produced = float(diff_sectional.sectional_statistic(
            {'d_am': torch.tensor(d_am), 'd_ab': torch.tensor(d_ab),
             'd_ac': torch.tensor(d_ac), 'd_bc': torch.tensor(d_bc)}))
        self.assertAlmostEqual(produced, -0.5, places=12)

    def test_euclidean_is_not_identically_zero(self):
        # The centre need not be the geometric midpoint, so the Euclidean value is not zero.
        vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
        metric = diff_sectional.build_metric('euclidean')
        value = float(diff_sectional.statistic_from_vectors(
            vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
            torch.tensor([0, 1, 2, 3]), metric))
        self.assertNotAlmostEqual(value, 0.0, places=6,
                                  msg='the Euclidean statistic must not vanish identically')

    def test_closed_form_matches_the_explicit_sum_on_the_real_graph(self):
        # The #248 closed form, recomputed from the persisted distance histogram.
        histogram = np.load(REPO / 'assets/distance_histogram.npz')
        moments = np.load(REPO / 'assets/node_moments.npz')
        offsets = histogram['offsets']
        counts = histogram['counts']
        node = 0
        own = counts[offsets[node]:offsets[node + 1]]
        levels = np.arange(len(own), dtype=np.float64)
        s2 = float((levels ** 2 * own).sum())
        self.assertAlmostEqual(s2, float(moments['s2'][node]), places=9,
                               msg='the histogram must reproduce the stored S2')
        b, c = 1, 2
        for other in (b, c):
            histogram_other = counts[offsets[other]:offsets[other + 1]]
            other_levels = np.arange(len(histogram_other), dtype=np.float64)
            self.assertAlmostEqual(float((other_levels ** 2 * histogram_other).sum()),
                                   float(moments['s2'][other]), places=9)

    def test_weights_are_caller_specified(self):
        d_am = torch.tensor([1.0, 3.0])
        d_ab = torch.tensor([0.0, 2.0])
        d_ac = torch.tensor([2.0, 1.0])
        d_bc = torch.tensor(2.0)
        distances = {'d_am': d_am, 'd_ab': d_ab, 'd_ac': d_ac, 'd_bc': d_bc}
        uniform = float(diff_sectional.sectional_statistic(distances))
        weighted = float(diff_sectional.sectional_statistic(
            distances, weights=torch.tensor([0.0, 1.0])))
        self.assertNotAlmostEqual(uniform, weighted, places=9,
                                  msg='caller weights must change the statistic')
        self.assertAlmostEqual(weighted, float(reference.sectional(
            [1.0, 3.0], [0.0, 2.0], [2.0, 1.0], 2.0, weights=[0.0, 1.0])), places=12)

    def test_mask_removes_a_reference(self):
        distances = {'d_am': torch.tensor([1.0, 5.0]), 'd_ab': torch.tensor([0.0, 3.0]),
                     'd_ac': torch.tensor([2.0, 4.0]), 'd_bc': torch.tensor(2.0)}
        full = float(diff_sectional.sectional_statistic(distances))
        masked = float(diff_sectional.sectional_statistic(
            distances, mask=torch.tensor([True, False])))
        self.assertNotAlmostEqual(full, masked, places=9)
        # the masked mean divides by the surviving references only
        self.assertAlmostEqual(masked, float(reference.sectional(
            [1.0], [0.0], [2.0], 2.0)), places=12)
        self.assertNotAlmostEqual(masked, full / 2, places=9,
                                  msg='the mask must reduce the mass, not only the sum')


class GradientTests(unittest.TestCase):
    def test_autograd_matches_finite_differences(self):
        torch.manual_seed(11)
        vectors = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        metric = diff_sectional.build_metric('euclidean')
        value = diff_sectional.statistic_from_vectors(
            vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
            torch.tensor([0, 1, 2, 3, 4, 5]), metric)
        value.backward()
        analytic = vectors.grad.clone()
        step = 1e-6
        numeric = torch.zeros_like(analytic)
        for index in range(vectors.shape[0]):
            for dim in range(vectors.shape[1]):
                plus = vectors.detach().clone()
                plus[index, dim] += step
                minus = vectors.detach().clone()
                minus[index, dim] -= step
                plus_value = diff_sectional.statistic_from_vectors(
                    plus, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
                    torch.tensor([0, 1, 2, 3, 4, 5]), metric)
                minus_value = diff_sectional.statistic_from_vectors(
                    minus, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
                    torch.tensor([0, 1, 2, 3, 4, 5]), metric)
                numeric[index, dim] = (plus_value - minus_value) / (2 * step)
        self.assertTrue(torch.allclose(analytic, numeric, atol=1e-6, rtol=1e-6),
                        msg=f'analytic {analytic.tolist()} numeric {numeric.tolist()}')

    def test_gradient_is_finite_and_nonzero(self):
        torch.manual_seed(12)
        vectors = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        metric = diff_sectional.build_metric('euclidean')
        value = diff_sectional.statistic_from_vectors(
            vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
            torch.tensor([0, 1, 2, 3, 4, 5]), metric)
        value.backward()
        self.assertTrue(bool(torch.isfinite(vectors.grad).all()))
        self.assertGreater(float(vectors.grad.abs().sum()), 0.0)

    def test_chunked_gradient_equals_the_full_gradient(self):
        torch.manual_seed(13)
        vectors = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        metric = diff_sectional.build_metric('euclidean')
        references = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        full_value = diff_sectional.statistic_from_vectors(
            vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), references, metric)
        full_value.backward()
        full = vectors.grad.clone()
        vectors.grad = None
        chunked_value = diff_sectional.statistic_from_vectors_chunked(
            vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), references, metric,
            chunk=3)
        chunked_value.backward()
        chunked = vectors.grad.clone()
        self.assertAlmostEqual(float(chunked_value), float(full_value), places=12)
        self.assertTrue(torch.allclose(full, chunked, atol=1e-12, rtol=1e-12),
                        msg=f'full {full.tolist()} chunked {chunked.tolist()}')

    def test_identical_vectors_give_a_zero_distance_but_a_defined_statistic(self):
        vectors = torch.zeros(4, 2, dtype=torch.float64, requires_grad=True)
        metric = diff_sectional.build_metric('euclidean')
        value = diff_sectional.statistic_from_vectors(
            vectors, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
            torch.tensor([0, 1, 2, 3]), metric)
        self.assertTrue(bool(torch.isfinite(value)))


if __name__ == '__main__':
    unittest.main()
