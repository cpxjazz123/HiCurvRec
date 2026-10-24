import tempfile
import unittest
from pathlib import Path

from tools.research.core.dag import TaskStore


class CancelTests(unittest.TestCase):
    def test_cancel_does_not_touch_another_run(self):
        with tempfile.TemporaryDirectory() as temp:
            queue = TaskStore(Path(temp))
            queue.add("run_aaaa__a", "DATA")
            queue.add("run_bbbb__a", "DATA")
            first = queue.claim("owner", prefix="run_aaaa__")
            self.assertEqual(first[0], "run_aaaa__a")
            changed = queue.cancel_prefix("run_aaaa__")
            self.assertEqual(changed, ["run_aaaa__a"])
            self.assertEqual(queue.snapshot()["tasks"]["run_aaaa__a"]["status"], "cancelling")
            self.assertEqual(queue.claim("other", prefix="run_bbbb__")[0], "run_bbbb__a")
            self.assertTrue(queue.finish_cancel(first[0], "owner"))
            self.assertFalse(queue.finish_cancel(first[0], "owner"))

    def test_done_is_preserved_after_cancel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue = TaskStore(root)
            queue.add("r__done", "DATA")
            queue.claim("owner")
            artifact = root / "measured.json"
            artifact.write_text("{}")
            queue.complete("r__done", "owner", artifact)
            self.assertEqual(queue.cancel_prefix("r__"), [])
            self.assertEqual(queue.snapshot()["tasks"]["r__done"]["status"], "done")


if __name__ == "__main__":
    unittest.main()
