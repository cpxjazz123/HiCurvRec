"""Issue #118 P1-D: the failure registry deduplicates by root cause, not prose."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.core import failures


def _entry(**overrides):
    payload = {"stage": "math", "error_class": "MATH", "rule": "FLAT_LIMIT_MISMATCH",
               "expected": "4*||x-y||^2", "actual": "2*||x-y||", "run_id": "run_00000000000000ab",
               "candidate_id": "c1", "status": "open"}
    payload.update(overrides)
    return payload


class RegistryTests(unittest.TestCase):
    def test_the_same_root_cause_twice_is_one_entry_with_two_occurrences(self):
        with tempfile.TemporaryDirectory() as work:
            first = failures.record(Path(work), _entry(run_id="run_00000000000000ab"))
            second = failures.record(Path(work), _entry(run_id="run_00000000000000cd"))
            self.assertEqual(first["failure_id"], second["failure_id"])
            self.assertEqual(second["occurrences"], 2)
            self.assertEqual(second["run_ids"],
                             ["run_00000000000000ab", "run_00000000000000cd"])
            stored = failures.load(Path(work))["entries"]
            self.assertEqual(len(stored), 1)

    def test_a_different_expected_value_is_a_different_root_cause(self):
        with tempfile.TemporaryDirectory() as work:
            failures.record(Path(work), _entry())
            failures.record(Path(work), _entry(expected="0"))
            self.assertEqual(len(failures.load(Path(work))["entries"]), 2)

    def test_unknown_classes_and_statuses_are_refused(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(ValueError):
                failures.record(Path(work), _entry(error_class="MADE_UP"))
            with self.assertRaises(ValueError):
                failures.record(Path(work), _entry(status="done"))

    def test_a_fix_and_its_regression_mark_the_entry_verified(self):
        with tempfile.TemporaryDirectory() as work:
            failures.record(Path(work), _entry())
            updated = failures.record(Path(work), _entry(
                status="verified", fix_commit="abc123",
                regression_test="tools/research/tests/test_contract_preflight.py::MathLayerTests"))
            self.assertEqual(updated["status"], "verified")
            self.assertEqual(updated["fix_commit"], "abc123")
            summary = failures.summary(Path(work))
            self.assertEqual(summary, {"failures": 1, "known_failure_recurrence": 1,
                                       "verified": 1})

    def test_a_preflight_report_is_recorded_by_layer(self):
        with tempfile.TemporaryDirectory() as work:
            report = {"diagnostics": [
                {"layer": "math", "code": "FLAT_LIMIT_MISMATCH", "expected": 1, "actual": 2,
                 "location": "math_definition", "verdict": "MATH_COUNTEREXAMPLE"},
                {"layer": "falsifier", "code": "DIRECTION_CONFLICTS_WITH_EVIDENCE",
                 "expected": "lower", "actual": "higher", "location": "falsifier",
                 "verdict": None},
            ]}
            recorded = failures.record_preflight(Path(work), report, run_id="run_00000000000000ab",
                                                 candidate_id="c1")
            self.assertEqual(len(recorded), 1, "warnings are not failures")
            self.assertEqual(recorded[0]["error_class"], "MATH")
            self.assertEqual(recorded[0]["rule"], "FLAT_LIMIT_MISMATCH")

    def test_the_registry_is_plain_json_on_disk(self):
        with tempfile.TemporaryDirectory() as work:
            failures.record(Path(work), _entry())
            path = Path(work) / failures.REGISTRY
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema"], 1)
            self.assertTrue(payload["entries"])


if __name__ == "__main__":
    unittest.main()


class ClassifierTests(unittest.TestCase):
    def test_messages_map_to_the_documented_classes(self):
        from tools.research.workflows.pipeline import classify_failure
        cases = {
            "ValueError: MATH_COUNTEREXAMPLE (math): math/FLAT_LIMIT_MISMATCH": "MATH",
            "ValueError: falsifier/UNREGISTERED_METRIC at name": "FALSIFIER",
            "ValueError: contract/CAPACITY_MISMATCH": "SCHEMA",
            "RuntimeError: OpenCode failed, cancelled, timed out": "INFRA_TRANSIENT",
            "TypeError: unsupported operand type(s) for |": "PYTHON_RUNTIME",
            "AdmissionError: Agent evaluated different source: math": "HASH_BINDING",
            "ValueError: something else entirely": "SCHEMA",
        }
        for message, expected in cases.items():
            self.assertEqual(classify_failure(message), expected, message)
