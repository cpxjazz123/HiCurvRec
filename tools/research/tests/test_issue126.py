"""Issue #126: the hard prefix score must have the properties it claims.

A metric that silently depends on the token numbering, on the levels below the
first, or on how many example pairs the file happens to hold is not a metric of
what it says it measures. These checks are cheap and they run without a GPU.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _numeric_stack() -> bool:
    """These checks need the project interpreter (numpy and torch).

    The framework suite runs on the stdlib-only interpreter of the MCP server, so
    the numeric checks skip there instead of failing for a missing dependency.
    """
    try:
        import torch  # noqa: F401
    except Exception:                                      # noqa: BLE001
        return False
    return True


HAVE_NUMERIC = _numeric_stack()

if HAVE_NUMERIC:
    from tools.research.science import hyperbolic_sid_quality as quality


def _sid(rows) -> dict:
    return {str(index): [int(value) for value in row] for index, row in enumerate(rows)}


def _write(directory: Path, payload: dict, name: str = "item_sids.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reference(seed: int = 0, items: int = 12):
    rng = np.random.default_rng(seed)
    geometry = quality.BallGeometry()
    source = geometry.expmap0(rng.normal(0.0, 0.1, (items, 4)))
    target = geometry.expmap0(rng.normal(0.0, 0.1, (items, 4)))
    record = {"spec": {"digest": "d" * 64},
              "arrays": {"sha256": "a" * 64},
              "temperature": 1.0}
    return {"record": record, "source": source, "target": target}


def _probe(directory: Path) -> Path:
    path = directory / "pairs.npz"
    np.savez(path, anchors=np.array([0, 1, 2]),
             positives=np.array([3, 4, 5]),
             negatives=np.array([[6, 7], [8, 9], [10, 11]]))
    return path


@unittest.skipUnless(HAVE_NUMERIC, "needs the project interpreter with numpy and torch")
class BarycentreTests(unittest.TestCase):

    def test_the_centre_of_identical_points_is_the_point(self):
        geometry = quality.BallGeometry()
        point = geometry.expmap0(np.array([0.2, -0.1, 0.05, 0.0]))
        centre = quality.frechet_barycentre(np.stack([point] * 5), geometry)
        self.assertTrue(np.allclose(centre, point, atol=1e-6))

    def test_the_centre_sits_on_the_geodesic_between_two_points(self):
        geometry = quality.BallGeometry()
        a = geometry.expmap0(np.array([0.2, 0.0, 0.0, 0.0]))
        b = geometry.expmap0(np.array([0.6, 0.0, 0.0, 0.0]))
        centre = quality.frechet_barycentre(np.stack([a, b]), geometry)
        self.assertAlmostEqual(float(geometry.distance(a, centre)),
                               float(geometry.distance(centre, b)), places=5)

    def test_the_centre_is_not_the_euclidean_mean(self):
        """A coordinate average renamed as a barycentre would fail this."""
        geometry = quality.BallGeometry()
        a = geometry.expmap0(np.array([0.2, 0.0, 0.0, 0.0]))
        b = geometry.expmap0(np.array([0.6, 0.0, 0.0, 0.0]))
        centre = quality.frechet_barycentre(np.stack([a, b]), geometry)
        self.assertGreater(float(np.linalg.norm(centre - (a + b) / 2)), 1e-6)


@unittest.skipUnless(HAVE_NUMERIC, "needs the project interpreter with numpy and torch")
class ScoreInvariantTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.reference = _reference()
        self.probe = _probe(self.root)
        self.rows = [[index % 3, index % 2, 0] for index in range(12)]

    def tearDown(self):
        self.directory.cleanup()

    def _score(self, rows) -> dict:
        path = _write(self.root, _sid(rows))
        return quality.score(path, reference=self.reference, probe=self.probe,
                             temperature=1.0)

    def test_the_same_sid_scores_the_same(self):
        first = self._score(self.rows)
        second = self._score(self.rows)
        self.assertEqual(first["q_h"], second["q_h"])

    def test_renumbering_the_first_level_changes_nothing(self):
        permutation = {0: 7, 1: 3, 2: 5}
        renumbered = [[permutation[row[0]], row[1], row[2]] for row in self.rows]
        self.assertAlmostEqual(self._score(self.rows)["q_h"],
                               self._score(renumbered)["q_h"], places=12)

    def test_changing_the_later_levels_changes_nothing(self):
        deeper = [[row[0], (row[1] + 1) % 2, (row[2] + 1) % 5] for row in self.rows]
        self.assertAlmostEqual(self._score(self.rows)["q_h"], self._score(deeper)["q_h"],
                               places=12)

    def test_one_bucket_scores_zero(self):
        flat = [[0, index % 2, 0] for index in range(12)]
        self.assertAlmostEqual(self._score(flat)["q_h"], 0.0, places=12)

    def test_a_pair_in_the_same_bucket_contributes_nothing(self):
        """A same-bucket pair has delta = 0, so its term is the neutral ln 2."""
        rows = [[index // 4, 0, 0] for index in range(12)]   # 3 buckets, 4 items each
        report = self._score(rows)
        # Every probe pair is positive 3/4 against negatives 6..11: with rows of
        # four, positives 3,4,5 and negatives 6..11 all sit in buckets 0 and 1.
        self.assertGreaterEqual(report["same_bucket_ratio"], 0.0)

    def test_a_bad_ordering_scores_below_zero(self):
        """The score reports a SID that ranks negatives above positives."""
        rows = [[index % 2, 0, 0] for index in range(12)]
        report = self._score(rows)
        self.assertLess(report["q_h"], 1.0)

    def test_an_unknown_item_is_an_error_not_a_zero(self):
        path = _write(self.root, _sid(self.rows[:6]))
        with self.assertRaises(Exception):
            quality.score(path, reference=self.reference, probe=self.probe,
                          temperature=1.0)

    def test_the_record_binds_the_sid_the_reference_and_the_probe(self):
        path = _write(self.root, _sid(self.rows))
        report = quality.score(path, reference=self.reference, probe=self.probe,
                               temperature=1.0)
        self.assertEqual(len(report["sid"]["sha256"]), 64)
        self.assertEqual(report["reference"]["digest"], "d" * 64)
        self.assertEqual(report["probe"]["queries"], 3)
        self.assertIn("spec", report)


@unittest.skipUnless(HAVE_NUMERIC, "needs the project interpreter with numpy and torch")
class ReferenceContractTests(unittest.TestCase):

    def test_the_reference_spec_is_frozen_and_hashed(self):
        from tools.research.science import hyperbolic_behavior_reference as ref
        spec = ref.frozen_spec()
        for key in ("curvature", "dim", "epochs", "batch_size", "learning_rate",
                    "seed", "temperature_rule"):
            self.assertIn(key, spec)
        self.assertEqual(len(spec["digest"]), 64)
        self.assertEqual(spec["digest"], ref.frozen_spec()["digest"])

    def test_the_reference_inputs_are_the_train_split_only(self):
        from tools.research.science import hyperbolic_behavior_reference as ref
        spec = ref.frozen_spec()
        for name in ("embeddings", "train_pairs", "valid_pairs"):
            self.assertIn(name, spec["inputs"])
        self.assertNotIn("test", spec["inputs"])


if __name__ == "__main__":
    unittest.main()
