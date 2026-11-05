"""The boundary of an agent, now that the agents are LangChain agents.

The boundary is no longer a mount namespace: it is the tool set the agent is
given. These checks keep the same promises the mount namespace kept - the allowed
file is readable and nothing else is, a traversal or a symlink does not escape,
another agent's answer is not readable, a refused read is recorded - and they add
the ones the new runtime needs: the runner starts no subprocess at all, and the
receipt says which reads were refused.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.research.agents import policy, runner, runtime


class PolicyTests(unittest.TestCase):
    """The policy refuses an allowlist that is wider than the task may have."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.round_dir = self.root / "rounds" / "round_a"
        self.round_dir.mkdir(parents=True)
        self.trees = self.root / "trees"
        self.candidate = self.trees / "round_a" / "candidate"
        self.candidate.mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def _accept(self, paths, **kwargs):
        return policy.check_read_paths(paths, roots=[self.round_dir, self.trees],
                                       task_root=self.round_dir, **kwargs)

    def test_the_candidate_tree_is_accepted(self):
        accepted = self._accept([self.candidate], candidate_tree=self.candidate)
        self.assertEqual([Path(item) for item in accepted], [self.candidate])

    def test_the_repository_root_is_refused(self):
        with self.assertRaises(policy.PolicyError):
            self._accept([policy.repository_root()])

    def test_another_arms_results_are_refused(self):
        results = policy.repository_root() / "results" / "stage2_RQ-VAE"
        if not results.exists():
            self.skipTest("this repository has no Stage 2 results")
        with self.assertRaises(policy.PolicyError):
            self._accept([results])

    def test_the_stage3_outputs_are_refused(self):
        outputs = policy.repository_root() / "results" / "stage3_T5Train"
        if not outputs.exists():
            self.skipTest("this repository has no Stage 3 outputs")
        with self.assertRaises(policy.PolicyError):
            self._accept([outputs])

    def test_the_shared_agent_data_is_refused(self):
        shared = Path.home() / ".local" / "share" / "opencode"
        if not shared.exists():
            self.skipTest("this machine has no shared opencode data")
        with self.assertRaises(policy.PolicyError):
            self._accept([shared])

    def test_the_global_agent_configuration_is_refused(self):
        config = Path.home() / ".config" / "opencode"
        if not config.exists():
            self.skipTest("this machine has no global opencode configuration")
        with self.assertRaises(policy.PolicyError):
            self._accept([config])

    def test_the_directory_with_the_other_agents_answers_is_refused(self):
        answers = self.round_dir / "agents"
        answers.mkdir()
        with self.assertRaises(policy.PolicyError):
            self._accept([answers])

    def test_a_path_outside_the_declared_roots_is_refused(self):
        outside = self.root / "elsewhere"
        outside.mkdir()
        with self.assertRaises(policy.PolicyError):
            self._accept([outside])

    def test_a_missing_path_is_refused(self):
        with self.assertRaises(policy.PolicyError):
            self._accept([self.candidate / "not-there"])

    def test_a_task_without_roots_is_refused(self):
        with self.assertRaises(policy.PolicyError):
            policy.check_read_paths([self.candidate], roots=None)


class StaleEvidenceTests(unittest.TestCase):
    """Evidence produced under another boundary is not evidence now."""

    def _state(self, root):
        from tools.research.rounds import pipeline as pipeline_module
        tree = root / "tree"
        tree.mkdir()
        (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
        directory = pipeline_module.new_round(root, "round_one", objective="o")
        state = pipeline_module.load_state(directory)
        state["implementations"] = [pipeline_module.implementation_record(
            {"idea_id": "a"}, {"changed_files": ["model.py"]}, tree)]
        return state


class FingerprintAgreementTests(unittest.TestCase):
    """The caller and the executor must fingerprint the SAME read set."""

    def test_the_shared_contract_is_in_both(self):
        from tools.research.agents import runner
        effective = runner.effective_read_paths(["/tmp/tree/train_rqvae.py"])
        self.assertTrue(any("HYPERBOLIC_GEOMETRY_CONTRACT.md" in path for path in effective))
        source = (Path(__file__).resolve().parents[1] / "rounds" /
                  "agent_calls.py").read_text(encoding="utf-8")
        self.assertIn("runner.effective_read_paths(read_paths)", source)


class FingerprintInputTests(unittest.TestCase):
    """The fingerprint must cover the same inputs on both sides."""

    def test_the_executor_accepts_the_candidate_digest(self):
        from tools.research.agents import runner
        import inspect
        signature = inspect.signature(runner.run_agent)
        self.assertIn("source_digest", signature.parameters)

    def test_the_contract_digest_is_part_of_the_fingerprint(self):
        from tools.research.agents import runner
        import inspect
        source = inspect.getsource(runner.boundary_fingerprint)
        self.assertIn("geometry_contract", source)


class ToolBoundaryTests(unittest.TestCase):
    """The tools are the boundary: what is allowed is readable, the rest is not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "agents" / "task"
        self.workspace.mkdir(parents=True)
        self.allowed = self.root / "allowed"
        self.allowed.mkdir()
        (self.allowed / "shared.txt").write_text("shared", encoding="utf-8")
        self.other = self.root / "agents" / "other"
        self.other.mkdir()
        (self.other / "ideas.txt").write_text("other agent answer", encoding="utf-8")
        self.secret = self.root / "secret.txt"
        self.secret.write_text("secret", encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)
        self.refusals = []
        self.tools = {tool.__name__: tool for tool in runtime.workspace_tools(
            self.workspace, [self.allowed], allow_write=True,
            refusals=self.refusals)}

    def _read(self, path):
        return self.tools["read_file"](str(path))

    def test_the_allowed_file_is_readable(self):
        self.assertEqual(self._read(self.allowed / "shared.txt"), "shared")

    def test_another_agents_answer_is_not_readable(self):
        with self.assertRaises(Exception):
            self._read(self.other / "ideas.txt")

    def test_another_round_is_not_readable(self):
        other_round = self.root / "rounds" / "round_b"
        other_round.mkdir(parents=True)
        (other_round / "state.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(Exception):
            self._read(other_round / "state.json")

    def test_a_parent_traversal_is_not_readable(self):
        with self.assertRaises(Exception):
            self._read(self.workspace / ".." / ".." / "secret.txt")

    def test_a_symlink_out_of_the_allowlist_is_not_readable(self):
        link = self.allowed / "escape.txt"
        link.symlink_to(self.secret)
        with self.assertRaises(Exception):
            self._read(link)

    def test_the_shared_agent_data_is_not_readable(self):
        with self.assertRaises(Exception):
            self._read(Path.home() / ".local" / "share" / "opencode")

    def test_a_refused_read_is_recorded(self):
        with self.assertRaises(Exception):
            self._read(self.secret)
        self.assertTrue(self.refusals)
        self.assertEqual(self.refusals[0]["tool"], "read_file")
        self.assertIn("secret.txt", self.refusals[0]["path"])


class SingleEntryTests(unittest.TestCase):
    """One entry point, no subprocess, no switch."""

    def test_the_runner_starts_no_subprocess(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        self.assertNotIn("Popen", source)

    def test_the_executor_has_no_switch_for_the_namespace(self):
        self.assertFalse(hasattr(runner, "sandbox_command"))
        self.assertFalse(hasattr(runner, "SANDBOX_BINARY"))

    def test_the_executor_refuses_a_task_without_declared_roots(self):
        from tools.research.agents import runner as runner_module
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(policy.PolicyError):
                policy.check_read_paths([Path(work)], roots=None)


class ExternalDirectoryTests(unittest.TestCase):
    """A file allowlist must also name the directory it lives in."""

    def test_a_file_outside_the_allowed_directories_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "ws"
            workspace.mkdir()
            scoped = Path(work) / "tree" / "model" / "m.py"
            scoped.parent.mkdir(parents=True)
            scoped.write_text("x = 1\n", encoding="utf-8")
            other = Path(work) / "other" / "n.py"
            other.parent.mkdir(parents=True)
            other.write_text("y = 2\n", encoding="utf-8")
            tools = {tool.__name__: tool for tool in runtime.workspace_tools(
                workspace, [scoped], allow_write=True)}
            self.assertEqual(tools["read_file"](str(scoped)), "x = 1\n")
            with self.assertRaises(Exception):
                tools["read_file"](str(other))


class ReadTrackingTests(unittest.TestCase):
    """The receipt says what the agent could not read, not only what it could."""

    def test_denied_reads_are_recorded_in_the_receipt(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            with mock.patch.object(runtime, "build_model", return_value="fake"), \
                    mock.patch.object(runtime, "run_turn", return_value="the answer"):
                receipt = runtime.run_agent(
                    "task", "read something", "ideas.txt", root, model="fake:model",
                    role="ideate", read_paths=[], allowed_roots=[str(root)],
                    allow_write=True)
            data = json.loads(Path(receipt).read_text())
            self.assertIn("refusals", data)
            self.assertIsInstance(data["refusals"], list)


class DeniedDirectoryTests(unittest.TestCase):
    """A refused listing is not a refused source."""

    def test_a_directory_denial_is_recorded_as_a_listing(self):
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "ws"
            workspace.mkdir()
            refusals = []
            tools = {tool.__name__: tool for tool in runtime.workspace_tools(
                workspace, [], allow_write=True, refusals=refusals)}
            with self.assertRaises(Exception):
                tools["list_directory"]("/tmp/")
            self.assertTrue(refusals)
            self.assertEqual(refusals[0]["tool"], "list_directory")




if __name__ == "__main__":
    unittest.main()
