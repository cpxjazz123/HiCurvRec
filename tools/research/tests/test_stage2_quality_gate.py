"""Stage 2 -> Stage 3 admission regressions under the Issue #120 semantics.

The admission decision is technical validity only. ER, level-two prefix counts,
collisions and usage are measured and reported, and they never refuse a Stage 3
launch; only a malformed, ambiguous or unreadable SID does.
"""
import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[3]
SOURCE = HERE / "tools/genrec-mcp/stage2_quality_gate.py"
spec = importlib.util.spec_from_file_location("quality_gate_under_test", SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def npy(data, shape):
    import array
    header = str({"descr": "<i4", "fortran_order": False, "shape": shape})
    head = (header + " " * ((-(10 + len(header) + 1)) % 16) + "\n").encode()
    values = array.array("i", data)
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(head)) + head + values.tobytes()


class Stage3AdmissionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.candidate_path = root / "candidate.json"
        self.baseline_path = root / "baseline.json"
        self.pairs_path = root / "valid.npz"
        self.meta_path = root / "valid.meta.json"
        # A deliberately poor candidate: every item shares the same first token,
        # so ER collapses and the level-two prefix count is one.
        self.candidate = {str(i): [0, 0, i] for i in range(13)}
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
            gate, EXPECTED_PAIRS=4,
            FROZEN_BASELINE_BLOB_SHA1=gate._git_blob_sha1(self.baseline_path),
            FROZEN_VALID_PAIRS_BLOB_SHA1=gate._git_blob_sha1(self.pairs_path),
            FROZEN_VALID_META_BLOB_SHA1=gate._git_blob_sha1(self.meta_path))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def evaluate(self):
        return gate.evaluate(self.candidate_path, self.baseline_path,
                             self.pairs_path, self.meta_path)

    def test_a_technically_valid_sid_passes_however_poor_its_quality_is(self):
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertIsNone(result["refused"])
        self.assertEqual(result["candidate"]["level2_distinct_prefixes"], 1)
        self.assertLess(result["candidate"]["valid_l1_hard_er"], 10.0)
        self.assertEqual(result["thresholds"], {}, "no promotion floor may be reported")

    def test_quality_numbers_are_still_measured(self):
        result = self.evaluate()
        self.assertIn("valid_l1_hard_er", result["candidate"])
        self.assertIn("valid_l1_positive_hit_rate", result["candidate"])
        self.assertIn("level2_distinct_prefixes", result["candidate"])
        self.assertIn("valid_l1_hard_er", result["diagnostic_only"])
        self.assertEqual(result["admission_policy"], "technical validity only; no quality floor")

    def test_duplicate_full_sid_is_refused(self):
        self.candidate["5"] = self.candidate["4"]
        self.candidate_path.write_text(json.dumps(self.candidate))
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("not unique", result["refused"])

    def test_a_malformed_sid_is_refused(self):
        self.candidate["7"] = [1, -2, 3]
        self.candidate_path.write_text(json.dumps(self.candidate))
        result = self.evaluate()
        self.assertFalse(result["passed"])

    def test_an_unreadable_sid_file_is_refused(self):
        self.candidate_path.write_text("{not json", encoding="utf-8")
        result = self.evaluate()
        self.assertFalse(result["passed"])

    def test_a_missing_reference_does_not_block_a_valid_candidate(self):
        self.baseline_path.unlink()
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["reference_status"])

    def test_tampered_validation_pairs_do_not_block_a_valid_candidate(self):
        with zipfile.ZipFile(self.pairs_path, "w") as z:
            z.writestr("anchors.npy", npy([0] * 4, (4,)))
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["reference_status"])

    def test_stage3_checks_validity_before_gpu_or_launch(self):
        server = (HERE / "tools/genrec-mcp/server.py").read_text()
        launch = server[server.index("def tool_stage3_start("):]
        self.assertIn("stage2_validity_admission(sid)", launch)
        self.assertLess(launch.index("stage2_validity_admission(sid)"),
                        launch.index("probe_gpu_layout()"))
        self.assertLess(launch.index('if not validity["passed"]:'),
                        launch.index("subprocess.Popen("))


if __name__ == "__main__":
    unittest.main()
