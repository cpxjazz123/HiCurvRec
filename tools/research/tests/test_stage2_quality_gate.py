"""Offline Stage2-to-Stage3 quality gate regressions; requires only stdlib."""
import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[3]
SOURCE = HERE / "tools" / "genrec-mcp" / "stage2_quality_gate.py"
spec = importlib.util.spec_from_file_location("quality_gate_under_test", SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def npy(data, shape):
    import array
    header = str({"descr": "<i4", "fortran_order": False, "shape": shape})
    head = (header + " " * ((-(10 + len(header) + 1)) % 16) + "\n").encode()
    values = array.array("i", data)
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(head)) + head + values.tobytes()


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.candidate_path = root / "candidate.json"
        self.baseline_path = root / "baseline.json"
        self.pairs_path = root / "valid.npz"
        self.meta_path = root / "valid.meta.json"
        self.candidate = {str(i): [i, 0, 0] for i in range(13)}
        self.candidate["0"] = [0, 0, 0]
        self.candidate["1"] = [0, 1, 0]
        self.candidate["12"] = [0, 2, 0]
        self.baseline = {str(i): [i % 2, 0, i] for i in range(13)}
        with zipfile.ZipFile(self.pairs_path, "w") as z:
            z.writestr("anchors.npy", npy([0] * 4, (4,)))
            z.writestr("positives.npy", npy([1] * 4, (4,)))
            neg = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12] + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * 3
            z.writestr("negatives.npy", npy(neg, (4, 10)))
        self.meta_path.write_text(json.dumps({
            "seed": 42, "k_negatives": 10,
            "relation": "valid_transition: history[-1] -> target, from valid.parquet",
            "shapes": {"anchors": [4], "positives": [4], "negatives": [4, 10]},
            "checks": {k: True for k in (
                "no_invariant_violations", "all_ids_in_range",
                "positive_is_always_a_valid_successor",
                "no_positive_is_a_train_successor",
                "no_negative_is_a_train_or_valid_successor_of_its_anchor")},
        }))
        self.candidate_path.write_text(json.dumps(self.candidate))
        self.baseline_path.write_text(json.dumps(self.baseline))
        self.patch = patch.multiple(
            gate, EXPECTED_PAIRS=4, FROZEN_BASELINE_L2_PREFIXES=2,
            FROZEN_RESAMPLE_VALID_L1_ER=0.0,
            FROZEN_BASELINE_BLOB_SHA1=gate._git_blob_sha1(self.baseline_path),
            FROZEN_VALID_PAIRS_BLOB_SHA1=gate._git_blob_sha1(self.pairs_path),
            FROZEN_VALID_META_BLOB_SHA1=gate._git_blob_sha1(self.meta_path))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def evaluate(self):
        return gate.evaluate(self.candidate_path, self.baseline_path,
                             self.pairs_path, self.meta_path)

    def test_hitrate_is_recorded_only_and_never_blocks_promotion(self):
        """A missing HitRate threshold must not be required for Stage3."""
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertNotIn("valid_l1_positive_hit_rate_min", result["thresholds"])
        self.assertEqual(result["candidate"]["valid_l1_positive_hit_rate"], 1.0)
        self.assertIn("valid_l1_positive_hit_rate", result["diagnostic_only"])
        self.assertEqual(result["candidate"]["valid_l1_hard_er"], 40)

    def test_pass_only_with_er_and_prefix_gates(self):
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["candidate"]["full_unique_items"], 13)
        self.assertEqual(set(result["sha256"]), {"sid", "baseline", "validation_pairs", "validation_meta"})

    def test_low_prefix_rejected_even_when_er_passes(self):
        self.candidate = {str(i): [0, 0, i] for i in range(13)}
        self.candidate_path.write_text(json.dumps(self.candidate))
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("prefixes", result["refused"])

    def test_er_must_strictly_beat_cutoff(self):
        with patch.object(gate, "MIN_VALID_L1_HARD_ER", 40.0):
            result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("Hard-L1 ER", result["refused"])

    def test_equal_prefix_to_baseline_is_rejected(self):
        self.candidate_path.write_text(json.dumps(self.baseline))
        with patch.object(gate, "MIN_VALID_L1_HARD_ER", -1):
            result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("prefixes", result["refused"])

    def test_missing_pairs_refuses(self):
        self.pairs_path.unlink()
        self.assertFalse(self.evaluate()["passed"])

    def test_tampered_independent_validation_pairs_fail_closed(self):
        with zipfile.ZipFile(self.pairs_path, "a") as z:
            z.writestr("injected.txt", "changed frozen bytes")
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("frozen Git blob", result["refused"])

    def test_missing_baseline_refuses(self):
        self.baseline_path.unlink()
        self.assertFalse(self.evaluate()["passed"])

    def test_recorded_c117_fails_on_real_validation_and_prefixes(self):
        """Reproduce the rejected run from immutable repository evidence."""
        root = HERE / "results"
        candidate = root / "stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id/item_sids.json"
        baseline = root / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
        data = root / "stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
        meta = root / "stage3_T5Train/monotonicity/behavior_pairs_valid_s42.meta.json"
        with patch.multiple(
            gate, EXPECTED_PAIRS=52855, FROZEN_BASELINE_L2_PREFIXES=16145,
            FROZEN_RESAMPLE_VALID_L1_ER=16.38953942232631,
            FROZEN_BASELINE_BLOB_SHA1="d97d24a7a63891682d2b68c09a89164de2031ee9",
            FROZEN_VALID_PAIRS_BLOB_SHA1="11d76dd452f3a27d1f6bf79fe7673bc6505733ef",
            FROZEN_VALID_META_BLOB_SHA1="a0121691dc95168169bffdc3b6018608580468da"):
            verdict = gate.evaluate(candidate, baseline, data, meta)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["candidate"]["level2_distinct_prefixes"], 6233)
        self.assertEqual(verdict["baseline"]["level2_distinct_prefixes"], 16145)
        self.assertAlmostEqual(verdict["baseline"]["valid_l1_hard_er"], 16.38953942232631, places=9)
        self.assertIn("prefixes", verdict["refused"])

    def test_stage3_checks_quality_before_gpu_or_launch(self):
        server = (HERE / "tools/genrec-mcp/server.py").read_text()
        launch = server[server.index("def tool_stage3_start("):]
        self.assertLess(launch.index("stage2_quality_admission(sid)"),
                        launch.index("probe_gpu_layout()"))
        self.assertLess(launch.index("if not quality[\"passed\"]:"),
                        launch.index("subprocess.Popen("))


if __name__ == "__main__":
    unittest.main()
