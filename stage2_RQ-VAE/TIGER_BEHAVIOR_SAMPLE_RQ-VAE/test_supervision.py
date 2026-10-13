import unittest

import numpy as np
import torch

from analysis.supervision_diagnostic.metrics import (
    binary_auc,
    codebook_occupancy,
    gradient_norms,
    residual_health,
    summarize_representations,
)


class SupervisionMetricsTest(unittest.TestCase):
    def test_binary_auc_handles_ties_and_both_classes(self):
        auc = binary_auc(np.array([0.1, 0.4, 0.4, 0.9]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(auc, 0.875)

    def test_binary_auc_rejects_single_class(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            binary_auc(np.array([0.1, 0.2]), np.array([1, 1]))

    def test_gradient_norms_report_zero_for_unreachable_groups(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        groups = {"reachable": (x,), "unused": (y,)}
        actual = gradient_norms((x.square()).sum(), groups)
        self.assertAlmostEqual(actual["reachable"], float(torch.linalg.vector_norm(2 * x.detach())), places=6)
        self.assertEqual(actual["unused"], 0.0)

    def test_residual_health_detects_constant_level(self):
        residuals = torch.stack((torch.randn(8, 3), torch.ones(8, 3)))
        health = residual_health(residuals)
        self.assertTrue(health[0]["finite"])
        self.assertFalse(health[0]["degenerate"])
        self.assertTrue(health[1]["degenerate"])
    def test_occupancy_is_per_level(self):
        ids = torch.tensor([[0, 3, 5], [0, 3, 6], [1, 3, 5], [2, 4, 6]])
        self.assertEqual(codebook_occupancy(ids), [3, 2, 2])

    def test_representation_summary_reports_single_and_cumulative_auc(self):
        reps = torch.tensor([
            [[0.0], [1.0], [2.0], [3.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ])
        pairs = np.array([[0, 2], [1, 3], [0, 3], [1, 2]])
        labels = np.array([1, 1, 0, 0])
        result = summarize_representations(reps, pairs, labels)
        self.assertEqual(set(result["single_auc"]), {0, 1, 2})
        self.assertEqual(set(result["cumulative_auc"]), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
