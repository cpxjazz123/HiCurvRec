"""Defects the first review found, pinned so they cannot come back.

Each test fails on the revision the review rejected: a device mismatch that would
crash at the first measurement, a subsample that keeps one edge in a million, and
a codebook that is moved by the inherited Frechet step although this mechanism's
specification keeps the baseline RQ update.
"""
import unittest
from pathlib import Path

import torch

from model.curvature_control import freeze_edges


class EdgeSampleTests(unittest.TestCase):

    def test_the_sample_is_capped_and_keeps_a_real_share(self):
        edges = [(index, index + 1) for index in range(0, 4000)]
        kept = freeze_edges(edges, cap=1000)
        self.assertEqual(len(kept), 1000)
        self.assertEqual(kept, freeze_edges(edges, cap=1000))     # deterministic

    def test_a_small_edge_set_is_kept_whole(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        self.assertEqual(len(freeze_edges(edges, cap=1000)), 3)

    def test_the_cap_does_not_depend_on_the_iteration_order(self):
        edges = [(index, index + 1) for index in range(50)]
        self.assertEqual(freeze_edges(edges, cap=10), freeze_edges(list(reversed(edges)), cap=10))


class MeasurementDeviceTests(unittest.TestCase):

    def test_the_endpoint_lookup_works_when_the_table_is_on_the_cpu(self):
        """The measurement gathers rows of a CPU tensor for a CUDA batch."""
        table = torch.randn(20, 4)                     # CPU, as the trainer holds it
        endpoints = [3, 7, 11]
        index = torch.tensor(endpoints)                # CPU index, as it must be
        subset = table[index]
        if torch.cuda.is_available():
            gpu_index = torch.tensor(endpoints, device="cuda:0")
            with self.assertRaises(RuntimeError):
                table[gpu_index]
        self.assertEqual(subset.shape, (3, 4))


class CodebookOwnershipTests(unittest.TestCase):

    def test_the_codebook_keeps_its_gradient_in_this_mechanism(self):
        """CURVCTRL-SID keeps the baseline RQ codebook update on purpose."""
        from model.layers import VQLayer
        layer = VQLayer(4, 2, beta=0.25, sk_epsilon=-1.0, sk_iters=-1,
                        frechet=True, use_behaviour=True, use_curvature=True,
                        codebook_gradient=True)
        self.assertTrue(layer.get_code_embs().requires_grad)

    def test_the_inherited_step_can_be_switched_off(self):
        source = (Path(__file__).resolve().parents[1] / "train_rqvae.py").read_text(
            encoding="utf-8")
        self.assertIn("USE_FRECHET_STEP", source)
        self.assertIn("USE_FRECHET_STEP = False", source)


if __name__ == "__main__":
    unittest.main()
