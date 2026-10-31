"""The append-only progress channel of a sub-agent.

One agent, one file, append only: the tool takes the record's content and
nothing else, the file it writes to was bound before the agent started, the
records that are already there are never rewritten, and the deadline closes the
file rather than the task's verdict.
"""
import json
import subprocess
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
        definition = progress_server._tool_definition()
        self.assertEqual(definition["name"], "append_progress")
        self.assertEqual(list(definition["inputSchema"]["properties"]), ["content"])
        self.assertFalse(definition["inputSchema"].get("additionalProperties", True))


class ServerTests(unittest.TestCase):
    """The server speaks the protocol and refuses what it must."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "progress.jsonl"
        self.server = Path(progress_server.__file__).resolve()

    def _call(self, *messages, deadline=0.0):
        lines = "".join(json.dumps(message) + "\n" for message in messages)
        completed = subprocess.run(
            ["/usr/bin/python3", str(self.server), "--file", str(self.path),
             "--task-id", "task_one", "--deadline-epoch", str(deadline)],
            input=lines, capture_output=True, text=True, timeout=30)
        return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]

    def test_a_call_appends_exactly_one_record(self):
        answers = self._call(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "append_progress", "arguments": {"content": "stage one"}}})
        self.assertEqual(answers[0]["result"]["serverInfo"]["name"], "progress")
        tools = answers[1]["result"]["tools"]
        self.assertEqual([tool["name"] for tool in tools], ["append_progress"])
        self.assertFalse(answers[2]["result"].get("isError"))
        records = progress_server.read_records(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["content"], "stage one")

    def test_an_unknown_tool_is_refused(self):
        answers = self._call(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "write_file", "arguments": {"path": "/etc/passwd"}}})
        self.assertTrue(answers[0]["result"].get("isError"))
        self.assertEqual(progress_server.read_records(self.path), [])

    def test_a_record_after_the_deadline_is_refused(self):
        import time
        answers = self._call(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "append_progress", "arguments": {"content": "late"}}},
            deadline=time.time() - 1)
        self.assertTrue(answers[0]["result"].get("isError"))
        self.assertIn("deadline", answers[0]["result"]["content"][0]["text"])
        self.assertEqual(progress_server.read_records(self.path), [])

    def test_the_answer_asks_for_a_final_record_before_the_deadline(self):
        import time
        answers = self._call(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "append_progress", "arguments": {"content": "stage"}}},
            deadline=time.time() + 30)
        text = answers[0]["result"]["content"][0]["text"]
        self.assertIn("deadline is near", text)

    def test_two_tasks_write_two_files_and_never_each_other_s(self):
        import time
        other = Path(self.tmp.name) / "other.jsonl"
        first = self._call(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "append_progress", "arguments": {"content": "mine"}}})
        self.assertFalse(first[0]["result"].get("isError"))
        lines = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": "append_progress",
                                        "arguments": {"content": "theirs"}}}) + "\n"
        subprocess.run(["/usr/bin/python3", str(self.server), "--file", str(other),
                        "--task-id", "task_two"],
                       input=lines, capture_output=True, text=True, timeout=30)
        mine = progress_server.read_records(self.path)
        theirs = progress_server.read_records(other)
        self.assertEqual([record["content"] for record in mine], ["mine"])
        self.assertEqual([record["content"] for record in theirs], ["theirs"])
        self.assertNotIn("theirs", self.path.read_text(encoding="utf-8"))




if __name__ == "__main__":
    unittest.main()
