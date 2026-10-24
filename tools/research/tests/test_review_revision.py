"""A rejected proposal must be revised inside the same run, not kill the run."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.core.dag import TaskStore
from tools.research.workflows import pipeline


class ReviewRevisionTests(unittest.TestCase):
    def _store_and_run(self, work):
        store = TaskStore(Path(work))
        store.add("run__behavior", "DATA", [])
        store.add("run__geometry", "GEOMETRY", ["run__behavior"])
        store.add("run__review", "REVIEW", ["run__geometry"])
        for expected in ("run__behavior", "run__geometry"):
            task_id, _role = store.claim("owner", lease_seconds=60, max_attempts=5)
            assert task_id == expected, (task_id, expected)
            store.complete(task_id, "owner", _artifact(Path(work)))
        return store

    def test_rejected_review_reopens_geometry_with_the_findings(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            store = self._store_and_run(work)
            review_dir = root / "agents" / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "review.json").write_text(json.dumps({
                "candidate_id": "c", "candidate_sha256": "x", "verdict": "MATERIAL",
                "blockers": [{"severity": "MATERIAL", "finding": "flat limit is wrong"}],
            }), encoding="utf-8")
            geometry_dir = root / "agents" / "geometry"
            geometry_dir.mkdir(parents=True, exist_ok=True)
            (geometry_dir / "candidate.json").write_text("{}", encoding="utf-8")
            (geometry_dir / "session.jsonl").write_text("session\n", encoding="utf-8")

            handled = pipeline._send_back_for_revision(
                store, "run__review", root, ValueError("BLOCKED by MATERIAL"))

            self.assertTrue(handled)
            self.assertEqual(store.snapshot()["tasks"]["run__geometry"]["status"], "pending")
            feedback = json.loads((geometry_dir / "revision_feedback.json").read_text())
            self.assertEqual(feedback["review_verdict"], "MATERIAL")
            self.assertTrue((geometry_dir / "revision_feedback.json").is_file())
            self.assertTrue((geometry_dir / "attempt-1" / "session.jsonl").is_file())

    def test_material_can_be_revised_after_many_prior_attempts(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            store = self._store_and_run(work)
            review = root / "agents" / "review"
            review.mkdir(parents=True, exist_ok=True)
            (review / "review.json").write_text(json.dumps({"verdict": "MATERIAL"}))
            geometry = root / "agents" / "geometry"
            geometry.mkdir(parents=True, exist_ok=True)
            for n in range(1, 9):
                (geometry / ("attempt-%d" % n)).mkdir()
            (geometry / "candidate.json").write_text("{}")
            self.assertTrue(pipeline._send_back_for_revision(
                store, "run__review", root, ValueError("MATERIAL")))
            self.assertTrue((geometry / "attempt-9" / "candidate.json").is_file())
            self.assertTrue((review / "attempt-1" / "review.json").is_file())
            self.assertEqual(store.snapshot()["tasks"]["run__geometry"]["status"], "pending")

    def test_fatal_scientific_verdict_is_not_revised(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            store = self._store_and_run(work)
            review = root / "agents" / "review"
            review.mkdir(parents=True, exist_ok=True)
            (review / "review.json").write_text(json.dumps({"verdict": "FATAL"}))
            self.assertFalse(pipeline._send_back_for_revision(
                store, "run__review", root, ValueError("FATAL")))
            self.assertEqual(store.snapshot()["tasks"]["run__geometry"]["status"], "done")

    def test_an_approved_review_is_never_sent_back(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            store = self._store_and_run(work)
            review_dir = root / "agents" / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "review.json").write_text(json.dumps({"verdict": "CPU_ELIGIBLE"}),
                                                    encoding="utf-8")
            handled = pipeline._send_back_for_revision(
                store, "run__review", root, ValueError("x"))
            self.assertFalse(handled)


def _artifact(root: Path) -> Path:
    path = root / "evidence.json"
    path.write_text("{}", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
