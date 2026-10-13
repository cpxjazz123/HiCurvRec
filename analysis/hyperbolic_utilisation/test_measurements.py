import unittest

import numpy as np

import measurements as M


class GeometryMeasurementTests(unittest.TestCase):
    def test_radius_and_angle_statistics_known_construction(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.]])
        stats = M.radius_statistics(points, points)
        self.assertEqual(stats["radius"]["median"], 1.0)
        self.assertEqual(stats["radius"]["below_0.1_fraction"], 0.25)
        angles = M.pairwise_angle_statistics(points[1:])
        self.assertEqual(angles["pair_count"], 3)
        np.testing.assert_allclose(angles["degrees"]["median"], 90.)

    def test_ranking_difference_zero_for_monotone_transform(self):
        euclidean = np.array([[1., 2.], [2., 1.], [0.5, 3.]])
        geodesic = np.square(euclidean)
        self.assertEqual(M.ranking_difference(euclidean, geodesic)["auc_difference"], 0.)

    def test_pipeline_stages_measure_distinct_representations(self):
        vectors = {"encoder": np.array([[0., 0.], [1., 0.], [0., 1.]]),
                   "L1": np.array([[1., 0.], [0., 1.], [-1., 0.]]),
                   "L2": np.array([[0., 1.], [1., 0.], [0., -1.]]),
                   "L3": np.array([[1., 1.], [-1., 1.], [1., -1.]])}
        stages = M.pipeline_spaces(vectors, vectors["L1"] + vectors["L2"] + vectors["L3"])
        self.assertEqual(set(stages), {"continuous_encoder", "per_level_residual", "cumulative_reconstruction"})
        self.assertIsNot(stages["continuous_encoder"], stages["cumulative_reconstruction"])
        self.assertFalse(np.shares_memory(stages["continuous_encoder"], stages["cumulative_reconstruction"]))

    def test_comparable_cross_arm_normalisation_uses_shared_scale(self):
        raw = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 2.]])
        mapped = raw / 4.
        ref = np.array([0, 1, 2, 3])
        scales = M.shared_local_scales(raw, mapped, np.array([0, 1]), ref)
        self.assertTrue(np.array_equal(scales["euclidean"], scales["hyperbolic"] ))


if __name__ == "__main__":
    unittest.main()
