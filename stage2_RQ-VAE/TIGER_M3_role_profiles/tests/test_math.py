"""Issue #142 math: the profile teacher and the relational fit against outside oracles.

The profile coordinates are recomputed here by hand from the definition -- mean,
population standard deviation, three quantiles, ``log1p(degree)`` and the local
closure ratio -- and the standardisation is checked against its own stated
location and scale. The relational fit is checked on data whose line is known.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import role_profiles as module  # noqa: E402
from model.structure_surface import huber_penalty  # noqa: E402


def _graph(rows, cols, weights, nodes):
    return {"item_ids": np.arange(nodes, dtype=np.int64),
            "rows": np.asarray(rows, dtype=np.int64),
            "cols": np.asarray(cols, dtype=np.int64),
            "weights": np.asarray(weights, dtype=np.float64)}


class ProfileTest(unittest.TestCase):
    def test_coordinates_match_a_hand_computation(self):
        # A path 0-1-2-3 with AF3 values 1, 2, 3, and an isolated node 4.
        graph = _graph([0, 1, 2], [1, 2, 3], [1.0, 2.0, 3.0], 5)
        teacher = module.profiles(graph)
        raw = teacher["raw"]
        # Node 1 sees the edges to 0 and to 2, i.e. the values 1 and 2.
        values = np.asarray([1.0, 2.0])
        expected = [values.mean(), values.std(ddof=0),
                    np.quantile(values, 0.1), np.quantile(values, 0.5),
                    np.quantile(values, 0.9), math.log1p(2.0),
                    0.0]
        np.testing.assert_allclose(raw[1], expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(teacher["valid"][1])
        # Nodes 0 and 3 have one incident edge each; node 4 has none.
        self.assertFalse(teacher["valid"][0])
        self.assertFalse(teacher["valid"][3])
        self.assertFalse(teacher["valid"][4])

    def test_standardisation_has_zero_mean_and_unit_scale_over_valid_nodes(self):
        graph = _graph([0, 1, 1, 2], [1, 2, 3, 3], [1.0, 2.0, 4.0, 3.0], 4)
        teacher = module.profiles(graph)
        standardised = teacher["profiles"][teacher["valid"]]
        np.testing.assert_allclose(standardised.mean(0), 0.0, atol=1e-6)
        np.testing.assert_allclose(standardised.std(0), 1.0, atol=1e-6)

    def test_local_closure_is_the_triangle_ratio(self):
        # A triangle 0-1-2 plus a pendant 2-3: node 0 has one triangle and
        # C(2,2) = 1 possible pair, so its ratio is 1.
        graph = _graph([0, 1, 2, 2], [1, 2, 0, 3], [4.0, 4.0, 4.0, 4.0], 4)
        teacher = module.profiles(graph)
        self.assertAlmostEqual(float(teacher["raw"][0][6]), 1.0, places=6)
        self.assertAlmostEqual(float(teacher["raw"][2][6]), 1.0 / 3.0, places=6)


class RelationalFitTest(unittest.TestCase):
    def test_recovers_a_known_line(self):
        nodes = 12
        rng = np.random.default_rng(3)
        profiles = rng.normal(size=(nodes, module.PROFILE_DIM)).astype(np.float32)
        pairs = np.asarray([[index, (index + 1) % nodes] for index in range(nodes)], dtype=np.int64)
        difference = profiles[pairs[:, 0]] - profiles[pairs[:, 1]]
        distance = np.linalg.norm(difference, axis=1)
        target = 0.25 + 2.0 * distance
        state = {"profiles": profiles}
        support = {"pairs": pairs, "is_edge": np.ones(len(pairs), dtype=bool), "distances": target}
        fit = module.relational_fit(state, support)
        self.assertAlmostEqual(fit["a"], 0.25, places=6)
        self.assertAlmostEqual(fit["b"], 2.0, places=6)
        self.assertLess(fit["rmse"], 1e-5)
        self.assertTrue(fit["sign_agrees"])

    def test_a_disagreeing_line_is_refused(self):
        nodes = 12
        rng = np.random.default_rng(5)
        profiles = rng.normal(size=(nodes, module.PROFILE_DIM)).astype(np.float32)
        pairs = np.asarray([[index, (index + 1) % nodes] for index in range(nodes)], dtype=np.int64)
        difference = profiles[pairs[:, 0]] - profiles[pairs[:, 1]]
        distance = np.linalg.norm(difference, axis=1)
        state = {"profiles": profiles}
        support = {"pairs": pairs, "is_edge": np.ones(len(pairs), dtype=bool),
                   "distances": 1.0 - distance}
        fit = module.relational_fit(state, support)
        self.assertFalse(fit["sign_agrees"],
                         "a line that runs against the edges' ordering must not be kept")


class ProfilePenaltyTest(unittest.TestCase):
    def test_equals_the_huber_of_the_coordinate_difference(self):
        predicted = torch.tensor([[0.3], [-2.0]], dtype=torch.float64)
        target = torch.tensor([[-0.1], [0.5]], dtype=torch.float64)
        # The contract surface returns one scalar per row; the raw penalty keeps
        # the coordinate axis.
        torch.testing.assert_close(module.profile_penalty(predicted, target, 1.0),
                                   huber_penalty(predicted - target, 1.0).squeeze(-1),
                                   rtol=0, atol=0)

    def test_gradient_is_the_residual_inside_the_quadratic_region(self):
        predicted = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
        module.profile_penalty(predicted, torch.tensor([[0.0]], dtype=torch.float64), 1.0).sum().backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([[0.4]], dtype=torch.float64),
                                   rtol=0, atol=1e-12)


class SparseBatchTest(unittest.TestCase):
    def test_batch_sizes_are_bounded(self):
        graph = _graph([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                       [1, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1, 2],
                       [1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5], 6)
        pairs = np.asarray([[0, 1], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5],
                            [4, 0], [5, 0], [5, 1], [0, 2]], dtype=np.int64)
        support = {"pairs": pairs, "is_edge": np.ones(len(pairs), dtype=bool), "edges": len(pairs)}
        state = module.prepare_state(graph, support, torch.zeros(6, 4), torch.device("cpu"))
        picked = module.sample_batch(state, np.random.default_rng(1))
        self.assertLessEqual(len(picked["nodes"]), module.NODES_PER_STEP)
        self.assertLessEqual(len(picked["pairs"]), module.EDGES_PER_STEP)
        self.assertTrue(bool(state["valid"][picked["nodes"]].all()))


if __name__ == "__main__":
    unittest.main()
