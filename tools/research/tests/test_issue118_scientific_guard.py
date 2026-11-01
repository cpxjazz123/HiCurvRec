"""A candidate a reviewer rejected as FATAL/PARKED must not be reviewed again.

The registry showed one candidate sent to review ten times, each time with the
same scientific verdict. The policy is explicit: a scientific rejection is
recorded and the candidate is blocked, and the same reviewer is not asked to
change their vote (CLAUDE.md 12 and 70).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.research.core import failures                      # noqa: E402
from tools.research.workflows import pipeline                 # noqa: E402


class ScientificGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.previous_root = pipeline.settings.STATE_ROOT
        pipeline.settings.STATE_ROOT = self.root

    def tearDown(self):
        pipeline.settings.STATE_ROOT = self.previous_root
        self.tmp.cleanup()

    def _reject(self, candidate_id: str, run_id: str) -> None:
        failures.record(self.root, {
            "stage": "review", "error_class": "SCIENTIFIC_FATAL", "rule": "FATAL",
            "expected": "CPU_ELIGIBLE or MATERIAL", "actual": "FATAL",
            "run_id": run_id, "candidate_id": candidate_id, "status": "open"})

    def test_a_rejected_candidate_is_settled(self):
        self._reject("candidate-1", "run_a")
        self.assertEqual(pipeline.settled_scientific_verdict("candidate-1"), "FATAL")

    def test_another_candidate_is_not_settled(self):
        self._reject("candidate-1", "run_a")
        self.assertIsNone(pipeline.settled_scientific_verdict("candidate-2"))

    def test_the_repeat_is_one_entry_with_two_occurrences(self):
        self._reject("candidate-1", "run_a")
        self._reject("candidate-1", "run_b")
        summary = failures.summary(self.root)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["known_failure_recurrence"], 1)

    def test_a_machine_failure_does_not_settle_a_candidate(self):
        failures.record(self.root, {
            "stage": "review", "error_class": "HASH_BINDING", "rule": "SOURCE_DRIFT",
            "expected": "the frozen source", "actual": "a changed source",
            "run_id": "run_a", "candidate_id": "candidate-1", "status": "open"})
        self.assertIsNone(pipeline.settled_scientific_verdict("candidate-1"))


if __name__ == "__main__":
    unittest.main()
