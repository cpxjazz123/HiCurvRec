"""The research loop must retry a rejected proposal inside the same run."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.workflows import pipeline


class RetryTests(unittest.TestCase):
    def test_rejected_attempt_is_archived_and_the_checker_message_is_carried(self):
        calls = {"prompts": [], "runs": 0}

        def fake_run_agent(task, prompt, output_name, root, **kwargs):
            calls["runs"] += 1
            calls["prompts"].append(prompt)
            workspace = Path(root) / "agents" / task
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "candidate.json").write_text("{}", encoding="utf-8")
            (workspace / "session.jsonl").write_text("session\n", encoding="utf-8")
            return workspace / output_name

        def fake_verify(task, root, manifest):
            if calls["runs"] == 1:
                raise ValueError("falsifier is incomplete: mixed_if")
            return None

        original_run, original_verify = pipeline.run_agent, pipeline.verify_science_output
        pipeline.run_agent, pipeline.verify_science_output = fake_run_agent, fake_verify
        try:
            with tempfile.TemporaryDirectory() as work:
                root = Path(work)
                pipeline.run_agent_until_valid("geometry", root, {}, "base prompt")
        finally:
            pipeline.run_agent, pipeline.verify_science_output = original_run, original_verify

        self.assertEqual(calls["runs"], 2)
        self.assertIn("base prompt", calls["prompts"][0])
        self.assertIn("mixed_if", calls["prompts"][1])
        self.assertIn("rejected by the automated checker", calls["prompts"][1])

    def test_every_rejected_attempt_keeps_its_own_receipts(self):
        def fake_run_agent(task, prompt, output_name, root, **kwargs):
            workspace = Path(root) / "agents" / task
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / output_name).write_text("{}", encoding="utf-8")
            (workspace / "session.jsonl").write_text(prompt, encoding="utf-8")
            return workspace / output_name

        def fake_verify(task, root, manifest):
            if len(list((Path(root) / "agents" / task).glob("attempt-*"))) < 1:
                raise ValueError("first defect")
            return None

        original_run, original_verify = pipeline.run_agent, pipeline.verify_science_output
        pipeline.run_agent, pipeline.verify_science_output = fake_run_agent, fake_verify
        try:
            with tempfile.TemporaryDirectory() as work:
                root = Path(work)
                pipeline.run_agent_until_valid("geometry", root, {}, "prompt")
                attempts = sorted((root / "agents" / "geometry").glob("attempt-*"))
                self.assertEqual(len(attempts), 1)
                self.assertTrue((attempts[0] / "session.jsonl").is_file())
        finally:
            pipeline.run_agent, pipeline.verify_science_output = original_run, original_verify

    def test_a_run_gives_up_after_the_last_attempt(self):
        calls = {"runs": 0}

        def fake_run_agent(task, prompt, output_name, root, **kwargs):
            calls["runs"] += 1
            workspace = Path(root) / "agents" / task
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / output_name).write_text("{}", encoding="utf-8")
            return workspace / output_name

        def fake_verify(task, root, manifest):
            raise ValueError("still wrong")

        original_run, original_verify = pipeline.run_agent, pipeline.verify_science_output
        pipeline.run_agent, pipeline.verify_science_output = fake_run_agent, fake_verify
        try:
            with tempfile.TemporaryDirectory() as work:
                with self.assertRaises(ValueError):
                    pipeline.run_agent_until_valid("geometry", Path(work), {}, "prompt")
        finally:
            pipeline.run_agent, pipeline.verify_science_output = original_run, original_verify
        self.assertEqual(calls["runs"], pipeline.MAX_PROPOSAL_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
