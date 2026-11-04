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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tools.research import settings
from tools.research.agents import langchain_runner, runner


class FakeToolCallingModel(BaseChatModel):
    """A model that answers with one fixed text and accepts tools."""

    reply: str = "the answer"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):        # noqa: D102
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


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
        self.tools = {tool.name: tool for tool in langchain_runner.workspace_tools(
            self.workspace, [self.allowed], allow_write=True)}

    def tearDown(self):
        self.work.cleanup()

    def test_a_read_outside_the_boundary_is_refused(self):
        with self.assertRaises(Exception):
            self.tools["read_file"].invoke({"path": str(self.secret)})

    def test_a_read_inside_an_allowed_root_is_allowed(self):
        text = self.tools["read_file"].invoke({"path": str(self.allowed / "shared.txt")})
        self.assertEqual(text, "shared")

    def test_a_write_outside_the_workspace_is_refused(self):
        with self.assertRaises(Exception):
            self.tools["write_file"].invoke({"path": str(self.root / "escape.txt"),
                                             "content": "x"})

    def test_a_write_inside_the_workspace_is_allowed(self):
        result = self.tools["write_file"].invoke({"path": "answer.txt", "content": "ok"})
        self.assertTrue((self.workspace / "answer.txt").is_file())
        self.assertIn("answer.txt", result)

    def test_a_read_only_task_has_no_write(self):
        tools = {tool.name: tool for tool in langchain_runner.workspace_tools(
            self.workspace, [self.allowed], allow_write=False)}
        with self.assertRaises(Exception):
            tools["write_file"].invoke({"path": "answer.txt", "content": "x"})


class BackendContractTests(unittest.TestCase):
    """The backend refuses to run without a model and keeps the receipt."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)

    def tearDown(self):
        self.work.cleanup()

    def test_the_backend_refuses_to_run_without_a_chat_model(self):
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", ""):
            with self.assertRaises(langchain_runner.AgentBackendUnavailable) as caught:
                langchain_runner.build_model()
        self.assertIn("AGENT_CHAT_MODEL", str(caught.exception))

    def test_the_agent_writes_its_answer_and_a_receipt(self):
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"), \
                mock.patch.object(langchain_runner, "build_model",
                                  return_value=FakeToolCallingModel(reply="the answer")):
            receipt_path = langchain_runner.run_agent(
                "task", "say the answer", "ideas.txt", self.root,
                model="fake:model", role="ideate", read_paths=[],
                allowed_roots=[str(self.root)], allow_write=True)
        receipt = json.loads(Path(receipt_path).read_text())
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["backend"], "langchain")
        self.assertTrue(receipt["boundary_sha256"])
        answer = self.root / "agents" / "task" / "ideas.txt"
        self.assertTrue(answer.is_file())
        self.assertIn("the answer", answer.read_text())

    def test_the_answer_travels_through_the_progress_channel(self):
        with mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"), \
                mock.patch.object(langchain_runner, "build_model",
                                  return_value=FakeToolCallingModel(reply="the answer")):
            langchain_runner.run_agent("task", "say the answer", "ideas.txt",
                                        self.root, model="fake:model", role="ideate",
                                        read_paths=[], allowed_roots=[str(self.root)],
                                        allow_write=True)
        records = runner.progress_server.read_records(
            self.root / "agents" / "task" / runner.PROGRESS_FILE_NAME)
        self.assertTrue(records)
        self.assertIn("the answer", str(records[-1].get("content", "")))

    def test_the_model_of_the_backend_names_the_boundary(self):
        with mock.patch.object(settings, "AGENT_BACKEND", "langchain"), \
                mock.patch.object(settings, "AGENT_CHAT_MODEL", "fake:model"):
            self.assertEqual(runner.effective_model(), "fake:model")
        with mock.patch.object(settings, "AGENT_BACKEND", "opencode"):
            self.assertEqual(runner.effective_model(), settings.AGENT_MODEL)


if __name__ == "__main__":
    unittest.main()
