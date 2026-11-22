"""Issue #124: a completed decision record is never overwritten."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import decision


class DecisionChainTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def _snapshot(self, value: str) -> dict:
        snapshot = {"round_id": "r", "decision_at": value, "selection": {"status": value}}
        snapshot["decision_hash"] = decision.decision_hash(snapshot)
        return snapshot

    def test_the_latest_snapshot_is_the_highest_revision(self):
        first = self._snapshot("a")
        decision.write_snapshot(self.directory, first)
        second = dict(self._snapshot("b"), revision=2)
        second["decision_hash"] = decision.decision_hash(second)
        decision.write_revision(self.directory, second, 2)
        latest = decision.latest_snapshot(self.directory)
        self.assertEqual(latest["decision_hash"], second["decision_hash"])

    def test_a_revision_file_is_never_overwritten(self):
        snapshot = dict(self._snapshot("a"), revision=2)
        decision.write_revision(self.directory, snapshot, 2)
        other = dict(self._snapshot("b"), revision=2)
        with self.assertRaises(RuntimeError):
            decision.write_revision(self.directory, other, 2)

    def test_the_original_decision_stays_untouched(self):
        first = self._snapshot("a")
        decision.write_snapshot(self.directory, first)
        before = (self.directory / decision.DECISION_FILE).read_bytes()
        with self.assertRaises(RuntimeError):
            decision.write_snapshot(self.directory, self._snapshot("b"))
        self.assertEqual((self.directory / decision.DECISION_FILE).read_bytes(), before)

    def test_the_original_decision_is_written_once(self):
        first = self._snapshot("a")
        decision.write_snapshot(self.directory, first)
        payload = json.loads((self.directory / decision.DECISION_FILE).read_text())
        self.assertEqual(payload["decision_hash"], first["decision_hash"])


if __name__ == "__main__":
    unittest.main()
