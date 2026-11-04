"""The progress channel: the only place an agent records what it did.

The channel is a file the framework owns and the agent writes through one tool.
The MCP server that served it to the CLI is gone with the CLI; the file, its
records and its refusals stay.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.agents import progress_server, runner


class ProgressFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "progress.jsonl"

    def _file(self, deadline=None, margin=60.0):
        return progress_server.ProgressFile(self.path, "task_one", deadline, margin)

    def test_a_record_is_appended_and_confirmed(self):
        progress = self._file()
        saved = progress.append("goal: read the trainer")
        self.assertEqual(saved["seq"], 1)
        records = progress_server.read_records(self.path)
        self.assertEqual(len(records), 1)
        self.assertIn("read the trainer", records[0]["content"])
        self.assertEqual(records[0]["task_id"], "task_one")
        self.assertEqual(records[0]["kind"], "stage")

    def test_an_earlier_record_is_never_rewritten(self):
        progress = self._file()
        progress.append("first stage")
        first = self.path.read_text(encoding="utf-8")
        progress.append("second stage")
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(first))
        records = progress_server.read_records(self.path)
        self.assertEqual([record["content"] for record in records],
                         ["first stage", "second stage"])
        self.assertEqual([record["seq"] for record in records], [1, 2])

    def test_an_empty_record_is_refused(self):
        progress = self._file()
        with self.assertRaises(ValueError):
            progress.append("   ")

    def test_a_record_after_the_deadline_is_refused(self):
        import time
        progress = self._file(deadline=time.time() - 1)
        with self.assertRaises(TimeoutError):
            progress.append("too late")
        self.assertEqual(progress_server.read_records(self.path), [])

    def test_an_incomplete_tail_line_is_ignored(self):
        progress = self._file()
        progress.append("a whole record")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"seq": 2, "content": "half a line')
        records = progress_server.read_records(self.path)
        self.assertEqual(len(records), 1)

    def test_the_tool_takes_the_content_and_nothing_else(self):
        definition = progress_server.tool_definition()
        self.assertEqual(definition["name"], "append_progress")
        self.assertEqual(list(definition["inputSchema"]["properties"]), ["content"])
        self.assertFalse(definition["inputSchema"].get("additionalProperties", True))


if __name__ == "__main__":
    unittest.main()
