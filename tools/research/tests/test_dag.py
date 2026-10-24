import json
import tempfile
import unittest
from pathlib import Path
from tools.research.core.dag import StateError, TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_dependencies_owner_evidence_and_immutable_identity(self):
        with tempfile.TemporaryDirectory() as work:
            store = TaskStore(Path(work))
            store.add("data", "DATA")
            store.add("design", "GEOMETRY", ("data",))
            self.assertEqual(store.claim("w1")[0], "data")
            with self.assertRaises(StateError):
                store.complete("data", "wrong", Path(work) / "absent.json")
            self.assertIsNone(store.claim("w2"))
            artifact = Path(work) / "data.json"
            artifact.write_text('{"measured":true}', encoding="utf-8")
            with self.assertRaises(StateError):
                store.complete("data", "wrong", artifact)
            record = store.complete("data", "w1", artifact)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertEqual(store.claim("w2")[0], "design")
            with self.assertRaises(StateError):
                store.add("design", "LITERATURE", ("data",))
            self.assertEqual(store.snapshot()["tasks"]["data"]["status"], "done")

    def test_unlimited_retries_keep_lease_owner(self):
        with tempfile.TemporaryDirectory() as work:
            store = TaskStore(Path(work))
            store.add("fix", "GEOMETRY")
            for _ in range(8):
                self.assertEqual(store.claim("agent", max_attempts=None)[0], "fix")
                with self.assertRaises(StateError):
                    store.renew("fix", "other", 120)
                store.renew("fix", "agent", 120)
                store.fail("fix", "agent", "retry", retry=True)
            self.assertEqual(store.claim("agent", max_attempts=None)[0], "fix")
            self.assertEqual(store.snapshot()["tasks"]["fix"]["attempts"], 9)

    def test_failure_blocks_task_and_lease_recovers(self):
        with tempfile.TemporaryDirectory() as work:
            store = TaskStore(Path(work))
            store.add("x", "DATA")
            store.claim("a", lease_seconds=1)
            state = store.snapshot()
            self.assertEqual(state["tasks"]["x"]["status"], "running")
            store.fail("x", "a", "bad data")
            self.assertIsNone(store.claim("b"))
            self.assertEqual(store.snapshot()["tasks"]["x"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
