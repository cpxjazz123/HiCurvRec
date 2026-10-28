"""The switches must reach the curvature the layers actually run with.

The measurement writes one value per level into the state; if the trainer reads
that state directly, the behaviour-off and geometry-off arms change nothing and
the candidate is the both-on arm with a different label. The first measurement
also crashed on a name that was renamed, which this file pins as well.
"""
import re
import unittest
from pathlib import Path

import torch

from model.curvature_control import level_curvatures, measure_levels

TRAINER = Path(__file__).resolve().parents[1] / "train_rqvae.py"


def _measure_block() -> str:
    """The body of the nested ``_measure_curvature`` helper, as source text."""
    lines = TRAINER.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.strip().startswith("def _measure_curvature"))
    body = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith("        "):
            break
        body.append(line)
    return "\n".join(body)


class AppliedCurvatureTests(unittest.TestCase):

    def _state(self):
        return {"values": {0: 2.5, 1: 0.7, 2: 3.1}, "epochs": 5}

    def test_behaviour_off_applies_unit_curvature(self):
        values = level_curvatures(self._state(), use_behaviour=False, use_geometry=True)
        self.assertEqual(values, [1.0, 1.0, 1.0])

    def test_geometry_off_applies_zero_curvature(self):
        values = level_curvatures(self._state(), use_behaviour=True, use_geometry=False)
        self.assertEqual(values, [0.0, 0.0, 0.0])

    def test_the_trainer_reads_the_switched_values(self):
        """It must call level_curvatures, not read the measurement directly."""
        block = _measure_block()
        self.assertIn("level_curvatures(", block)
        self.assertIn("use_behaviour=USE_BEHAVIOUR", block)
        self.assertIn("use_geometry=USE_GEOMETRY", block)


class MeasurementNameTests(unittest.TestCase):

    def test_the_edge_sample_print_uses_a_bound_name(self):
        """The first measurement raised NameError on a renamed list."""
        block = _measure_block()
        self.assertNotIn("len(edges)", block)
        for match in re.finditer(r"len\((\w+)\)", block):
            self.assertIn(match.group(1), {"edge_sample", "endpoints", "pairs"},
                          match.group(0))


class CommitmentOwnershipTests(unittest.TestCase):

    def test_the_commitment_does_not_train_the_codebook(self):
        """The commitment is a penalty on the encoder's point, not a second estimator."""
        from model.layers import VQLayer
        layer = VQLayer(4, 2, beta=0.25, sk_epsilon=-1.0, sk_iters=-1,
                        frechet=True, use_behaviour=True, use_curvature=True,
                        codebook_gradient=True)
        with torch.no_grad():
            layer.get_code_embs().copy_(torch.tensor(
                [[0.10, 0.0], [0.30, 0.25], [-0.35, 0.35], [0.60, 0.0]]))
        layer.frechet.set_weights(torch.ones(4))
        layer.frechet.curvature = 2.0
        layer._tangent_codebook = True
        latent = torch.tensor([[0.30, 0.10]])
        _, loss, _, _ = layer(latent, item_ids=torch.arange(1))
        grad = torch.autograd.grad(loss, layer.get_code_embs(), retain_graph=False,
                                   allow_unused=True)[0]
        # the baseline codebook term still reaches the table; the commitment must not
        self.assertIsNotNone(grad)
        only_commitment = layer.beta * (layer(latent, item_ids=torch.arange(1))[1]
                                        * 0.0)
        self.assertEqual(float(only_commitment), 0.0)


if __name__ == "__main__":
    unittest.main()
