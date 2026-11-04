"""The LangChain backend must keep the project's boundary and its contract.

The agents are created with LangChain now; what may not change is what they can
read and write, where their answer goes, and the receipt the framework reads.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.research import settings
from tools.research.agents import runner, runtime


class BoundaryTests(unittest.TestCase):
    """The tools are the boundary in this backend."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)
        self.workspace = self.root / "agents" / "task"
        self.workspace.mkdir(parents=True)
        self.allowed = self.root / "allowed"
        self.allowed.mkdir()
        (self.allowed / "shared.txt").write_text("shared", encoding="utf-8")
        self.secret = self.root / "secret.txt"
        self.secret.write_text("secret", encoding="utf-8")
        self.tools = {tool.__name__: tool for tool in runtime.workspace_tools(
            self.workspace, [self.allowed], allow_write=True)}

    def tearDown(self):
        self.work.cleanup()

    def test_a_read_outside_the_boundary_is_refused(self):
        with self.assertRaises(Exception):
            self.tools["read_file"](str(self.secret))

    def test_a_read_inside_an_allowed_root_is_allowed(self):
        text = self.tools["read_file"](str(self.allowed / "shared.txt"))
        self.assertEqual(text, "shared")

    def test_a_write_outside_the_workspace_is_refused(self):
        with self.assertRaises(Exception):
            self.tools["write_file"](str(self.root / "escape.txt"), "x")

    def test_a_write_inside_the_workspace_is_allowed(self):
        result = self.tools["write_file"]("answer.txt", "ok")
        self.assertTrue((self.workspace / "answer.txt").is_file())
        self.assertIn("answer.txt", result)

    def test_a_long_record_is_split_into_several(self):
        """The channel refuses a record over 4000 characters."""
        from tools.research.agents import progress_server
        progress = progress_server.ProgressFile(
            self.workspace / runner.PROGRESS_FILE_NAME, "task")
        runtime.save_progress(progress, "x" * 9000)
        records = progress_server.read_records(self.workspace / runner.PROGRESS_FILE_NAME)
        self.assertGreaterEqual(len(records), 3)
        self.assertTrue(all(len(str(record.get("content", ""))) <= 4000
                        for record in records))

    def test_a_read_only_task_has_no_write(self):
        tools = {tool.__name__: tool for tool in runtime.workspace_tools(
            self.workspace, [self.allowed], allow_write=False)}
        with self.assertRaises(Exception):
            tools["write_file"]("answer.txt", "x")


class BackendContractTests(unittest.TestCase):
    """The backend refuses to run without a model and keeps the receipt."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def test_the_structured_output_schema_matches_the_framework_fields(self):
        schema = runtime.output_schema("ideate")
        fields = set(schema.model_fields)
        self.assertTrue({"idea_id", "title", "motivation", "behavior_signal",
                         "hyperbolic_operator", "coupling_path"} <= fields)
        ranking = runtime.output_schema("score")
        self.assertIn("ranking", ranking.model_fields)

    def test_a_json_answer_is_rendered_into_the_block_the_parser_reads(self):
        payload = {"idea_id": "alpha-1", "title": "A proposal",
                   "motivation": "m", "novelty": "n", "key_change": "c",
                   "specification": "s", "cost_estimate": "ce",
                   "falsifiable_hypothesis": "h", "risks": ["r"],
                   "behavior_signal": "b", "behavior_computation": "bc",
                   "coupling_path": "cp", "hyperbolic_operator": "ho"}
        block = runtime.render_idea_json(payload)
        self.assertIn("IDEA: alpha-1", block)
        self.assertIn("TITLE: A proposal", block)
        self.assertIn("BEHAVIOR SIGNAL: b", block)
        self.assertTrue(block.rstrip().endswith("END IDEA"))

    def test_a_json_object_is_found_in_a_noisy_answer(self):
        payload = {"idea_id": "alpha-1", "title": "t"}
        self.assertEqual(runtime.extract_json("noise " + json.dumps(payload) + " noise"),
                         payload)

    def test_a_ranking_json_is_rendered_into_the_rank_lines(self):
        payload = {"ranking": [{"idea_id": "a", "position": 1, "reason": "because"}],
                   "objection": "none"}
        lines = runtime.render_rank_json(payload)
        self.assertIn("RANK: a | position=1 | reason=because", lines)
        self.assertIn("OBJECTION: none", lines)

    def test_the_proposal_block_is_extracted_from_the_records(self):
        block = ("IDEA: alpha-1\nTITLE: A proposal\nEND IDEA")
        self.assertEqual(runtime.extract_answer_block("noise\n" + block + "\nmore"),
                         block)

    def test_a_block_that_was_cut_short_is_still_taken(self):
        block = "IDEA: alpha-1\nTITLE: A proposal"
        self.assertEqual(runtime.extract_answer_block(block), block)

    def test_text_without_a_block_yields_nothing(self):
        self.assertIsNone(runtime.extract_answer_block("just a summary"))

    def test_an_answer_that_echoes_the_prompt_is_not_an_answer(self):
        prompt = "Propose exactly 1 DISTINCT, implementable idea for the item ids."
        self.assertTrue(runtime.echoes_the_prompt(prompt, prompt))
        self.assertTrue(runtime.echoes_the_prompt(
            "Propose exactly 1 DISTINCT idea\n...", prompt))
        self.assertFalse(runtime.echoes_the_prompt(
            "IDEA: X\nTITLE: A real proposal", prompt))

    def test_the_backend_refuses_to_run_without_a_chat_model(self):
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", ""):
            with self.assertRaises(runtime.AgentBackendUnavailable) as caught:
                runtime.build_model()
        self.assertIn("AGENT_CHAT_MODEL", str(caught.exception))

    def test_the_agent_writes_its_answer_and_a_receipt(self):
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"), \
                mock.patch.object(runtime, "build_model", return_value="fake"), \
                mock.patch.object(runtime, "run_turn", return_value="the answer"):
            receipt_path = runtime.run_agent(
                "task", "say the answer", "ideas.txt", self.root,
                model="fake:model", role="ideate", read_paths=[],
                allowed_roots=[str(self.root)], allow_write=True)
        receipt = json.loads(Path(receipt_path).read_text())
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["backend"], "openai-agents-sdk")
        self.assertTrue(receipt["boundary_sha256"])
        answer = self.root / "agents" / "task" / "ideas.txt"
        self.assertTrue(answer.is_file())
        self.assertIn("the answer", answer.read_text())

    def test_the_answer_travels_through_the_progress_channel(self):
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"), \
                mock.patch.object(runtime, "build_model", return_value="fake"), \
                mock.patch.object(runtime, "run_turn", return_value="the answer"):
            runtime.run_agent("task", "say the answer", "ideas.txt",
                                        self.root, model="fake:model", role="ideate",
                                        read_paths=[], allowed_roots=[str(self.root)],
                                        allow_write=True)
        records = runner.progress_server.read_records(
            self.root / "agents" / "task" / runner.PROGRESS_FILE_NAME)
        self.assertTrue(records)
        self.assertIn("the answer", str(records[-1].get("content", "")))

    def test_the_model_names_the_boundary(self):
        """There is one backend now; its model is the chat model."""
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"):
            self.assertEqual(runner.effective_model(), "fake:model")

    def test_the_runner_has_no_opencode_path_left(self):
        """The CLI backend is gone: the runner delegates to the agent runtime."""
        source = Path(runner.__file__).read_text()
        for gone in ("opencode run", "bwrap", "sandbox_command", "AGENT_BINARY"):
            self.assertNotIn(gone, source)
        self.assertTrue(hasattr(runner, "run_agent"))
        self.assertTrue(hasattr(runner, "tool_permissions"))


if __name__ == "__main__":
    unittest.main()
