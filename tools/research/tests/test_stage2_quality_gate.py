"""Stage 2 -> Stage 3 admission regressions.

Technical validity still refuses malformed, ambiguous or unreadable SIDs. On top
of it, the owner's instruction of 2026-11-03 makes one quality rule a gate: a SID
whose three-level collision exceeds 20% is refused (the previous
technical-validity-only policy is deliberately overridden).
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
        self.metrics_dir = root / "logs"
        self.metrics_dir.mkdir()
        self.write_learned(0.0)
        self.baseline_path.write_text(json.dumps(self.baseline))
        self.patch = patch.multiple(
            gate, EXPECTED_PAIRS=4,
            FROZEN_BASELINE_BLOB_SHA1=gate._git_blob_sha1(self.baseline_path),
            FROZEN_VALID_PAIRS_BLOB_SHA1=gate._git_blob_sha1(self.pairs_path),
            FROZEN_VALID_META_BLOB_SHA1=gate._git_blob_sha1(self.meta_path))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write_learned(self, collision, rows=1):
        lines = [json.dumps({"event": "train", "epoch": 1, "collision": collision})
                 for _ in range(rows)]
        (self.metrics_dir / "training_metrics.jsonl").write_text("\n".join(lines) + "\n")

    def evaluate(self):
        return gate.evaluate(self.candidate_path, self.baseline_path,
                             self.pairs_path, self.meta_path)

    def test_a_technically_valid_sid_under_the_collision_gate_passes(self):
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertIsNone(result["refused"])
        self.assertEqual(result["candidate"]["level2_distinct_prefixes"], 1)
        self.assertLess(result["candidate"]["valid_l1_hard_er"], 10.0)
        self.assertEqual(result["thresholds"], {"l3_collision_max": 0.35})

    def test_quality_numbers_are_still_measured(self):
        result = self.evaluate()
        self.assertIn("valid_l1_hard_er", result["candidate"])
        self.assertIn("valid_l1_positive_hit_rate", result["candidate"])
        self.assertIn("level2_distinct_prefixes", result["candidate"])
        self.assertIn("valid_l1_hard_er", result["diagnostic_only"])
        self.assertIn("LEARNED three-level-collision gate", result["admission_policy"])
        self.assertIn("l3_collision_max", result["quality_gate"])
        self.assertIsNone(result["quality_gate"]["l3_collision_refusal"])
        self.assertIn("l3_collision", result["candidate"])

    def test_a_sid_above_the_collision_gate_is_refused(self):
        # 4 of 13 items share one triple: 4/13 = 0.31 > 0.20.
        self.candidate = {str(i): [0, 0, min(i, 7), i] for i in range(13)}
        self.candidate_path.write_text(json.dumps(self.candidate))
        self.write_learned(gate.three_level_collision(self.candidate))
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("three-level collision", result["refused"])
        self.assertGreater(result["candidate"]["l3_collision"], 0.20)

    def test_a_sid_at_or_below_the_collision_gate_passes(self):
        # 2 of 13 items share a triple: 2/13 = 0.154 <= 0.20.
        self.candidate = {str(i): [0, 0, min(i, 11), i] for i in range(13)}
        self.candidate_path.write_text(json.dumps(self.candidate))
        # the trainer logs 1 - distinct/total, which is what the export check
        # compares; the item-weighted share is the threshold measure.
        self.write_learned(gate.exported_raw_collision(self.candidate))
        result = self.evaluate()
        self.assertTrue(result["passed"], result)
        self.assertLessEqual(result["candidate"]["l3_collision"], 0.20)

    def test_the_collision_gate_boundary(self):
        self.assertIsNone(gate.collision_gate({"l3_collision": 0.35}, learned=0.35,
                                              exported_raw=0.35))
        self.assertIsNone(gate.collision_gate({"l3_collision": 0.0}, learned=0.0,
                                              exported_raw=0.0))
        self.assertIsNotNone(gate.collision_gate({"l3_collision": 0.350001}, learned=0.35))
        self.assertIsNotNone(gate.collision_gate({}, learned=0.0))
        self.assertIsNotNone(gate.collision_gate({"l3_collision": 0.0}, learned=None))
        self.assertIsNotNone(gate.collision_gate({"l3_collision": 0.0}, learned=0.30))
        self.assertIsNotNone(gate.collision_gate({"l3_collision": 0.05}, learned=0.30))

    def test_an_export_deduped_sid_is_refused(self):
        # The exported triples are unique (3 levels, collision 0.0) while the
        # model's own log says it learned 0.15: that is an export post-process.
        self.candidate = {str(i): [0, 0, i] for i in range(13)}
        self.candidate_path.write_text(json.dumps(self.candidate))
        self.write_learned(0.15)
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("post-processing", result["refused"])
        self.assertAlmostEqual(result["quality_gate"]["learned_collision"], 0.15)

    def test_a_learned_collision_above_the_gate_is_refused(self):
        self.write_learned(0.40)
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("must come from Stage2 training", result["refused"])

    def test_a_missing_metrics_file_fails_closed(self):
        (self.metrics_dir / "training_metrics.jsonl").unlink()
        result = self.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("could not be read", result["refused"])

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

    def test_the_gate_imports_under_the_stage2_interpreter(self):
        # stage3_start evaluates the gate with the Stage 2 interpreter (3.9), so a
        # PEP 604 annotation there would make the verifier unavailable.
        import subprocess
        from tools.research.stage2 import contract
        command = [
            contract.STAGE2_PYTHON, "-c",
            "import importlib.util; "
            "spec = importlib.util.spec_from_file_location('g', " + repr(str(SOURCE)) + "); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert module.collision_gate({'l3_collision': 0.20}, learned=0.20, exported_raw=0.20) is None",
        ]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-500:])



if __name__ == "__main__":
    unittest.main()
