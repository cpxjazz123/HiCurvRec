"""The behaviour-graph curvature that sets each level's geometry.

The mechanism has to MEASURE something real, put it into the quantizer's
assignment, and survive the three switches. These tests are about the claims
the proposal makes: the curvature is standardised against a size-preserving
null, the deviation is mean-zero and clipped, the level's geometry follows it,
and the assignment changes when the geometry does.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import curvature_allocation as allocation      # noqa: E402


def _two_clusters(size: int = 60, clusters: int = 2, sizes=None):
    """A behaviour graph of dense clusters joined by one weak edge."""
    import scipy.sparse as sp

    sizes = list(sizes) if sizes else [size] * clusters
    starts = []
    cursor = 0
    for value in sizes:
        starts.append(cursor)
        cursor += value
    edges = []
    for cluster in range(clusters):
        start = starts[cluster]
        for first in range(start, start + sizes[cluster]):
            for second in range(first + 1, start + sizes[cluster]):
                edges.append((first, second))
    # the clusters are linked in a ring, and densely, so the quotient is a
    # connected graph and the null relabelling keeps it connected too
    for cluster in range(clusters):
        following = starts[(cluster + 1) % clusters]
        for first in range(starts[cluster], starts[cluster] + sizes[cluster]):
            for second in range(following, following + sizes[(cluster + 1) % clusters]):
                edges.append((first, second))
    rows = [a for a, _ in edges] + [b for _, b in edges]
    cols = [b for _, b in edges] + [a for a, _ in edges]
    data = np.ones(len(rows), dtype=np.float32)
    graph = sp.coo_matrix((data, (rows, cols)),
                          shape=(sum(sizes), sum(sizes))).tocsr()
    return graph


class MeasurementTests(unittest.TestCase):
    def test_the_standardised_value_is_the_deviation_from_the_null(self):
        """The value the mechanism spends is the EXCESS over the null.

        The raw curvature of a quotient graph is dominated by how dense the
        quotient is, so it is not comparable across levels and not comparable
        across arms; the standardised value is the real curvature's deviation
        from a null that keeps the group sizes, in units of that null's spread.
        """
        sizes = [30, 30, 30, 30, 8, 8, 8, 8]
        graph = _two_clusters(clusters=8, sizes=sizes)
        assignment = np.concatenate([np.full(size, index)
                                     for index, size in enumerate(sizes)])
        result = allocation.standardised_curvature(
            graph, assignment, 8, np.random.default_rng(0))
        self.assertIsNotNone(result["null_mean"])
        self.assertIsNotNone(result["null_std"])
        if result["fallback"] is not None:
            self.assertEqual(result["kappa_hat"], 0.0)
            return
        expected = (result["kappa"] - result["null_mean"]) / result["null_std"]
        self.assertAlmostEqual(result["kappa_hat"], expected, places=9)

    def test_the_measurement_is_reproducible_with_the_same_seed(self):
        graph = _two_clusters(size=30, clusters=8)
        assignment = np.repeat(np.arange(8), 30)
        first = allocation.standardised_curvature(graph, assignment, 8,
                                                  np.random.default_rng(7))
        second = allocation.standardised_curvature(graph, assignment, 8,
                                                   np.random.default_rng(7))
        self.assertEqual(first["kappa"], second["kappa"])
        self.assertEqual(first["kappa_hat"], second["kappa_hat"])

    def test_a_level_with_too_few_prefixes_falls_back_to_zero(self):
        graph = _two_clusters()
        assignment = np.repeat(np.arange(2), 60)
        result = allocation.standardised_curvature(
            graph, assignment, 2, np.random.default_rng(0))
        self.assertEqual(result["kappa_hat"], 0.0)
        self.assertIsNotNone(result["fallback"])

    def test_the_null_keeps_the_group_sizes(self):
        graph = _two_clusters(size=30, clusters=8)
        assignment = np.repeat(np.arange(8), 30)
        rng = np.random.default_rng(1)
        result = allocation.standardised_curvature(graph, assignment, 8, rng)
        self.assertEqual(result["occupied_prefixes"], 8)


class ScheduleTests(unittest.TestCase):
    def test_the_deviation_is_mean_zero_and_clipped(self):
        delta = allocation.update_delta(np.zeros(3), np.array([2.0, -1.0, 0.0]))
        self.assertLessEqual(float(np.abs(delta).max()), allocation.DELTA_CLIP)
        self.assertAlmostEqual(float(delta.mean()), 0.0, places=6)

    def test_the_update_is_smoothed(self):
        first = allocation.update_delta(np.zeros(3), np.array([1.0, 0.0, -1.0]))
        second = allocation.update_delta(first, np.array([1.0, 0.0, -1.0]))
        self.assertFalse(np.allclose(first, second), "the update did not move")

    def test_the_level_geometry_follows_the_deviation(self):
        delta = np.array([0.5, 0.0, -0.5])
        curvature, radius = allocation.level_geometry(delta, 2.0)
        self.assertAlmostEqual(float(curvature[0]), float(np.exp(-0.5)), places=6)
        self.assertAlmostEqual(float(radius[0]), float(2.0 * np.exp(0.25)), places=6)
        self.assertAlmostEqual(float(curvature[2]), float(np.exp(0.5)), places=6)

    def test_the_switches_are_declared(self):
        self.assertIsInstance(allocation.BEHAVIOUR_OFF, bool)
        self.assertIsInstance(allocation.GEOMETRY_OFF, bool)


class RefreshTests(unittest.TestCase):
    """The refresh is the mechanism's only clock: it has to run and to bind."""

    def test_the_refresh_measures_and_binds_the_level_geometry(self):
        import train_rqvae as trainer
        from model.model import RQVAE

        graph = _two_clusters(size=40, clusters=8, sizes=[40] * 8)
        nodes = list(range(graph.shape[0]))
        index = {item: item for item in nodes}
        model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        model.train()
        embeddings = torch.randn(graph.shape[0], 768)
        with torch.no_grad():
            model.rq.init_codebook(model.encoder(embeddings), torch.device("cpu"))
        model.set_radii(torch.ones(graph.shape[0]))
        state = {"module": allocation, "nodes": nodes, "index": index, "graph": graph,
                 "rho0": 1.0, "delta": np.zeros(3),
                 "rng": np.random.default_rng(3)}
        trainer._refresh_level_geometry(200, state, model, embeddings,
                                        torch.device("cpu"), 0, 1)
        self.assertEqual(model.level_curvature.shape[0], 3)
        self.assertTrue(bool(torch.isfinite(model.level_curvature).all()))
        self.assertTrue(bool(torch.isfinite(model.level_radius).all()))
        self.assertTrue(bool((model.level_radius > 0).all()))
        for level, layer in enumerate(model.rq.vq_layers):
            self.assertAlmostEqual(float(layer._curvature),
                                   float(model.level_curvature[level]), places=6)


class BroadcastTests(unittest.TestCase):
    """The level geometry is broadcast: the tensors have to be on the device.

    The NCCL backend has no CPU backend, and a CPU tensor in a broadcast ends
    the run in the first epoch on every rank -- which the single-process smoke
    cannot see, because it never broadcasts.
    """

    def test_the_measured_radius_lives_on_the_device(self):
        import train_rqvae as trainer
        from model.model import RQVAE

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = RQVAE(trainer._tokenizer_config(), in_dim=768).to(device)
        embeddings = torch.randn(64, 768)
        radius = trainer._measured_radius(model, embeddings, device)
        self.assertEqual(radius.device.type, device.type)
        self.assertEqual(radius.numel(), 1)

    def test_the_level_geometry_broadcast_tensors_are_on_the_device(self):
        import inspect

        import train_rqvae as trainer

        source = inspect.getsource(trainer._refresh_level_geometry)
        self.assertIn("device=device", source,
                      "the refresh broadcasts tensors that are not on the device")


class QuantizerTests(unittest.TestCase):
    def _model(self):
        import train_rqvae as trainer
        from model.model import RQVAE

        model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        model.train()
        with torch.no_grad():
            embedded = torch.randn(512, 768)
            model.rq.init_codebook(model.encoder(embedded), torch.device("cpu"))
        return model, embedded

    def test_the_level_geometry_changes_the_assignment(self):
        model, embedded = self._model()
        with torch.no_grad():
            encoded = model.encoder(embedded[:64])
            flat = model.rq.vq_layers[0].mapped_cost(
                model.rq.vq_layers[0].point_of(encoded))
            model.rq.vq_layers[0].set_level_geometry(4.0, 1.0)
            curved = model.rq.vq_layers[0].mapped_cost(
                model.rq.vq_layers[0].point_of(encoded))
        self.assertFalse(torch.allclose(flat, curved),
                         "the level's curvature did not change the score")

    def test_the_squashed_tangent_has_the_level_radius(self):
        model, _ = self._model()
        layer = model.rq.vq_layers[0]
        layer.set_level_geometry(1.0, 0.5)
        tangent = torch.tensor([[3.0, 4.0]])
        squashed = layer.squashed_tangent(tangent)
        self.assertLess(float(squashed.norm()), 0.5 + 1e-6)
        self.assertAlmostEqual(float(squashed.norm()), 0.5, places=5)

    def test_the_gradient_reaches_the_codebook(self):
        model, embedded = self._model()
        model.set_level_geometry(torch.tensor([2.0, 1.0, 0.5]),
                                 torch.tensor([1.0, 1.0, 1.0]))
        radius = torch.ones(128)
        out = model(embedded[:128], radius=radius)
        loss, _ = model.compute_loss(out[4], out[0], out[1])
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_the_geometry_is_persistent(self):
        model, _ = self._model()
        model.set_level_geometry(torch.tensor([2.0, 1.0, 0.5]),
                                 torch.tensor([1.0, 1.0, 1.0]))
        state = model.state_dict()
        self.assertIn("level_curvature", state)
        self.assertIn("level_radius", state)
        self.assertAlmostEqual(float(state["level_curvature"][0]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
