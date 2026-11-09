"""What actually lands on disk, and what the downstream reader gets from it.

These replace the earlier prompt-string assertions: each case runs the real
save path with a fake model turn and then reads the file the consumer reads.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research import settings
from tools.research.agents import runner, runtime


def _run(work, task, role, answer, output="answer.json", progress=None):
    workspace = Path(work) / "agents" / task
    if progress:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / runner.PROGRESS_FILE_NAME).write_text(
            "\n".join(json.dumps({"content": row}) for row in progress) + "\n",
            encoding="utf-8")
    with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"), \
            mock.patch.object(runtime, "build_model", return_value="fake"), \
            mock.patch.object(runtime, "run_turn", side_effect=lambda *a, **k: answer):
        receipt = runtime.run_agent(task, "prompt", output, Path(work),
                                    model="fake:model", role=role, read_paths=[],
                                    allowed_roots=[str(work)], allow_write=True,
                                    timeout=30)
    return Path(receipt), workspace


class AnswerContractTests(unittest.TestCase):
    def test_a_literature_answer_keeps_every_field(self):
        payload = {"sources": [{"title": "A", "status": "ABSTRACT"}],
                   "limitations": ["no full text"], "idea_id": "not-an-idea"}
        with tempfile.TemporaryDirectory() as work:
            receipt, workspace = _run(work, "literature", "literature",
                                      json.dumps(payload),
                                      output="literature.json",
                                      progress=["STEP 1 - workspace inspection"])
            saved = json.loads((workspace / "literature.json").read_text())
            self.assertEqual(saved, payload)
            self.assertTrue(json.loads(Path(receipt).read_text())["verified"])

    def test_a_review_keeps_its_identity_fields(self):
        payload = {"verdict": "PASS", "candidate_id": "c1",
                   "candidate_sha256": "a" * 64, "findings": []}
        with tempfile.TemporaryDirectory() as work:
            receipt, workspace = _run(work, "review", "review",
                                      json.dumps(payload), output="review.json")
            saved = json.loads((workspace / "review.json").read_text())
            self.assertEqual(saved["candidate_id"], "c1")
            self.assertEqual(saved["candidate_sha256"], "a" * 64)

    def test_a_crashed_turn_leaves_a_failed_receipt(self):
        with tempfile.TemporaryDirectory() as work:
            with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"), \
                    mock.patch.object(runtime, "build_model", return_value="fake"), \
                    mock.patch.object(runtime, "run_turn",
                                      side_effect=RuntimeError("provider refused")):
                with self.assertRaises(runner.AgentFailure) as caught:
                    runtime.run_agent("literature", "prompt", "literature.json",
                                      Path(work), model="fake:model", role="literature",
                                      read_paths=[], allowed_roots=[str(work)],
                                      allow_write=True, timeout=30)
            receipt = json.loads(Path(caught.exception.receipt).read_text())
            self.assertFalse(receipt["verified"])
            self.assertIn("provider refused", receipt["failure"])
            self.assertFalse((Path(work) / "agents" / "literature"
                              / "literature.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
