"""The agent contract: the workspace, the answer file and the prompt.

The agents are LangChain agents now, so the contract is the workspace they get,
the file they answer in, and the boundary their tools enforce.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.research.agents import policy, runner, runtime
from tools.research.agents.runner import run_agent


class AgentTests(unittest.TestCase):
    """The task contract the framework reads."""

    def test_the_agent_gets_its_own_workspace(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            with mock.patch.object(runtime, "build_model", return_value="fake"), \
                    mock.patch.object(runtime, "run_turn", return_value="the answer"):
                try:
                    runtime.run_agent(
                        "task", "answer", "ideas.txt", root, model="fake:model",
                        role="ideate", read_paths=[], allowed_roots=[str(root)])
                except Exception:
                    pass
            self.assertTrue((root / "agents" / "task").is_dir())

    def test_a_read_allowlist_denies_every_other_path(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            allowed = root / "allowed"
            allowed.mkdir()
            outside = root / "elsewhere"
            outside.mkdir()
            with self.assertRaises(policy.PolicyError):
                policy.check_read_paths([outside], roots=[allowed],
                                        task_root=root)

    def test_the_sandbox_hides_every_path_outside_the_allowlist(self):
        """The tool boundary replaces the mount namespace."""
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            workspace = root / "agents" / "task"
            workspace.mkdir(parents=True)
            allowed = root / "allowed"
            allowed.mkdir()
            (allowed / "a.txt").write_text("a", encoding="utf-8")
            (root / "secret.txt").write_text("s", encoding="utf-8")
            tools = {tool.__name__: tool for tool in runtime.workspace_tools(
                workspace, [allowed], allow_write=True)}
            self.assertEqual(tools["read_file"](str(allowed / "a.txt")), "a")
            with self.assertRaises(Exception):
                tools["read_file"](str(root / "secret.txt"))

    def test_the_receipt_names_the_boundary_it_ran_under(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            with mock.patch.object(runtime, "build_model", return_value="fake"), \
                    mock.patch.object(runtime, "run_turn", return_value="the answer"):
                try:
                    receipt = runtime.run_agent(
                        "task", "answer", "ideas.txt", root, model="fake:model",
                        role="ideate", read_paths=[], allowed_roots=[str(root)])
                except Exception:
                    receipt = root / "agents" / "task" / "execution.json"
            evidence = json.loads(Path(receipt).read_text())
            self.assertEqual(evidence["backend"], "openai-agents-sdk")
            self.assertTrue(evidence["boundary_sha256"])


class GeometryPromptThresholdTests(unittest.TestCase):
    """A machine-repairable prompt defect: the threshold keys were removed."""

    def test_geometry_prompt_does_not_raise_with_empty_thresholds(self):
        import json
        from unittest import mock
        from tools.research.agents import prompts
        from tools.research.science import contract

        manifest = {"task_id": "run_test__geometry", "task": "GEOMETRY",
                    "inputs_sha256": "c" * 64,
                    "behavior_evidence_sha256": "a" * 64,
                    "literature_evidence_sha256": "b" * 64,
                    "behavior_evidence": {}, "literature_evidence": {}}
        with mock.patch.object(contract, "THRESHOLDS", {}):
            text = prompts.build_prompt("GEOMETRY", dict(manifest), "candidate.json")
        self.assertIn("NOT admission thresholds", text)
        self.assertIn("technical validity only", text)
        self.assertNotIn("strictly above", text)

    def test_geometry_prompt_uses_configured_thresholds_when_present(self):
        from unittest import mock
        from tools.research.agents import prompts
        from tools.research.science import contract

        manifest = {"task_id": "run_test__geometry", "task": "GEOMETRY",
                    "inputs_sha256": "c" * 64,
                    "behavior_evidence_sha256": "a" * 64,
                    "literature_evidence_sha256": "b" * 64,
                    "behavior_evidence": {}, "literature_evidence": {}}
        configured = {"valid_l1_hard_er_strict_gt": 16.38953942232631,
                      "level2_prefixes_strict_gt": 16145}
        with mock.patch.object(contract, "THRESHOLDS", configured):
            text = prompts.build_prompt("GEOMETRY", dict(manifest), "candidate.json")
        self.assertIn("FROZEN_THRESHOLDS", text)
        self.assertIn("16.38953942232631", text)
        self.assertIn("16145", text)


if __name__ == "__main__":
    unittest.main()
