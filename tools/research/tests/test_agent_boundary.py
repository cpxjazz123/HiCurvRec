"""The acceptance suite for the agent filesystem boundary.

The standard the framework holds itself to: a task names the research files it
may read, and the agent can reach exactly those through any tool it has - no
other round, no other agent's answer, no parent traversal, no symlink out of
the allowlist, and no channel that reads the file for it (the global opencode
configuration, which can declare MCP servers, is not mounted at all).

Two layers are checked here. The policy refuses a wider allowlist before a
single process starts, and the mount namespace then makes the refusal true: the
paths that were not allowed do not exist inside it.
"""
import inspect
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.research.agents import policy, runner

BWRAP = shutil.which("bwrap")


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

    def test_the_shared_opencode_data_is_refused(self):
        shared = Path.home() / ".local" / "share" / "opencode"
        if not shared.exists():
            self.skipTest("this machine has no shared opencode data")
        with self.assertRaises(policy.PolicyError):
            self._accept([shared])

    def test_the_global_opencode_configuration_is_refused(self):
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


class MountTests(unittest.TestCase):
    """The mount namespace makes the policy's refusal true."""

    def setUp(self):
        if not BWRAP:
            self.skipTest("bubblewrap is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.round_dir = root / "rounds" / "round_a"
        self.workspace = self.round_dir / "agents" / "task"
        self.workspace.mkdir(parents=True)
        # Another agent's answer, in the directory above this workspace.
        (self.workspace.parent / "other_agent.txt").write_text("EARLIER ANSWER",
                                                                 encoding="utf-8")
        # Another round.
        other_round = root / "rounds" / "round_b"
        other_round.mkdir(parents=True)
        (other_round / "other_round.txt").write_text("OTHER ROUND", encoding="utf-8")
        # The candidate tree: the one directory the task may read.
        self.candidate = root / "trees" / "round_a" / "candidate"
        self.candidate.mkdir(parents=True)
        (self.candidate / "allowed.txt").write_text("ALLOWED", encoding="utf-8")
        # A link inside the allowlist that points out of it.
        (self.candidate / "escape").symlink_to(other_round / "other_round.txt")
        self.private = root / "private"
        (self.private / ".local" / "share" / "opencode").mkdir(parents=True)
        (self.private / ".cache" / "opencode").mkdir(parents=True)

    def _inside(self, script):
        command = runner.sandbox_command(["/bin/sh", "-c", script], self.workspace,
                                         [self.candidate], Path("/bin/sh"),
                                         self.private)
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr[-600:])
        return completed.stdout

    def test_the_allowed_file_is_readable(self):
        output = self._inside("cat " + str(self.candidate / "allowed.txt"))
        self.assertIn("ALLOWED", output)

    def test_another_agents_answer_is_not_readable(self):
        output = self._inside("cat " + str(self.workspace.parent / "other_agent.txt")
                              + " 2>/dev/null || echo DENIED")
        self.assertIn("DENIED", output)
        self.assertNotIn("EARLIER ANSWER", output)

    def test_another_round_is_not_readable(self):
        output = self._inside("cat " + str(self.round_dir.parent / "round_b"
                                           / "other_round.txt")
                              + " 2>/dev/null || echo DENIED")
        self.assertIn("DENIED", output)
        self.assertNotIn("OTHER ROUND", output)

    def test_a_parent_traversal_is_not_readable(self):
        output = self._inside("cat ../other_agent.txt 2>/dev/null || echo DENIED")
        self.assertIn("DENIED", output)
        self.assertNotIn("EARLIER ANSWER", output)

    def test_a_symlink_out_of_the_allowlist_is_not_readable(self):
        output = self._inside("cat " + str(self.candidate / "escape")
                              + " 2>/dev/null || echo DENIED")
        self.assertIn("DENIED", output)
        self.assertNotIn("OTHER ROUND", output)

    def test_the_global_opencode_configuration_is_not_mounted(self):
        output = self._inside("cat " + str(Path.home() / ".config" / "opencode"
                                           / "opencode.jsonc")
                              + " 2>/dev/null || echo DENIED")
        self.assertIn("DENIED", output)

    def test_the_shared_opencode_data_is_not_mounted(self):
        output = self._inside("ls " + str(Path.home() / ".local" / "share" / "opencode")
                              + " 2>/dev/null | head -3 || echo DENIED")
        self.assertNotIn("opencode.db", output)


class BindDestinationTests(unittest.TestCase):
    """No bind destination may run through a symlinked component.

    The namespace creates the parents of a bind destination inside its own
    root. When the destination runs through a symlink - the repository is
    reachable through a symlinked home - that creates the link as a directory,
    and the operation that wanted to create it as a link fails. The failure
    looked like an unrelated sandbox error, so the rule is pinned here.
    """

    def test_no_destination_has_a_symlinked_ancestor(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            workspace = root / "workspace"
            workspace.mkdir()
            private = root / "private"
            (private / ".local" / "share" / "opencode").mkdir(parents=True)
            (private / ".cache" / "opencode").mkdir(parents=True)
            command = runner.sandbox_command(["/bin/sh", "-c", "true"], workspace,
                                             [], Path("/bin/sh"), private)
            offenders = []
            for index, item in enumerate(command):
                if item not in ("--bind", "--ro-bind", "--ro-bind-try"):
                    continue
                destination = Path(command[index + 2])
                for ancestor in destination.parents:
                    if ancestor == Path("/"):
                        break
                    if ancestor.is_symlink():
                        offenders.append(str(destination) + " through " + str(ancestor))
                        break
            self.assertEqual(offenders, [], "bind destinations through a symlink: " +
                             repr(offenders))

    def test_the_progress_server_is_mounted_by_a_resolved_path(self):
        self.assertEqual(runner.PROGRESS_SERVER, runner.PROGRESS_SERVER.resolve())


class SingleEntryTests(unittest.TestCase):
    """There is one launcher, and the boundary cannot be switched off."""

    def test_only_the_runner_starts_an_opencode_process(self):
        needle = '"opencode", ' + '"run"'
        offenders = []
        for path in Path("tools").rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if needle in text:
                offenders.append(str(path))
        self.assertEqual(offenders, ["tools/research/agents/runner.py"])

    def test_the_executor_has_no_switch_for_the_namespace(self):
        parameters = inspect.signature(runner.run_agent).parameters
        self.assertNotIn("sandbox", parameters)
        self.assertNotIn("boundary", parameters)

    def test_the_executor_refuses_a_task_without_declared_roots(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(policy.PolicyError):
                runner.run_agent("t", "prompt", "answer.txt", Path(work),
                                 read_paths=[], allowed_roots=None)


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

    def test_a_review_under_another_boundary_cannot_be_accepted(self):
        from tools.research.rounds import pipeline as pipeline_module
        with tempfile.TemporaryDirectory() as work:
            state = self._state(Path(work))
            reports = [{"verifier": "v" + str(index), "verdict": "PASS", "checks": [],
                        "boundary_version": "agent-boundary/older",
                        "boundary_sha256": "old"} for index in range(1, 6)]
            state = pipeline_module.apply_verification(state, "a", reports)
            with self.assertRaises(pipeline_module.RoundError) as caught:
                pipeline_module.accept_verification(state, "a", "looks fine")
            self.assertIn("boundary", str(caught.exception))

    def test_the_verifiers_must_have_run_under_one_boundary(self):
        from tools.research.rounds import pipeline as pipeline_module
        with tempfile.TemporaryDirectory() as work:
            state = self._state(Path(work))
            reports = [{"verifier": "v" + str(index), "verdict": "PASS", "checks": [],
                        "boundary_version": policy.BOUNDARY_VERSION,
                        "boundary_sha256": "b"} for index in range(1, 5)]
            reports.append({"verifier": "v5", "verdict": "PASS", "checks": [],
                            "boundary_version": "agent-boundary/older",
                            "boundary_sha256": "b"})
            with self.assertRaises(pipeline_module.RoundError) as caught:
                pipeline_module.apply_verification(state, "a", reports)
            self.assertIn("boundary", str(caught.exception))

    def test_a_cached_answer_from_another_boundary_is_not_reused(self):
        from tools.research.rounds import plaintext, worker
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            workspace = root / "agents" / "task"
            workspace.mkdir(parents=True)
            (workspace / "verification.txt").write_text("VERDICT: PASS\n", encoding="utf-8")
            (workspace / "execution.json").write_text(
                json.dumps({"verified": True, "boundary_sha256": "old"}), encoding="utf-8")
            calls = []

            def fake_run(*args, **kwargs):
                calls.append(kwargs)
                raise AssertionError("the cached answer was reused")

            with mock.patch.object(worker, "run_agent", side_effect=fake_run):
                with self.assertRaises(AssertionError):
                    worker._run_agent(root, "task", "prompt", "verification.txt",
                                      plaintext.parse_verification, role="verify")
            # The stale answer was moved aside, and the agent runs again.
            self.assertTrue((workspace / "unverified" / "verification.txt.unverified").is_file())


if __name__ == "__main__":
    unittest.main()


class ExternalDirectoryTests(unittest.TestCase):
    """A file allowlist must also declare the directory it lives in."""

    def test_the_containing_directory_is_declared(self):
        from tools.research.agents import runner
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "ws"
            workspace.mkdir()
            scoped = Path(work) / "tree" / "model" / "m.py"
            scoped.parent.mkdir(parents=True)
            scoped.write_text("x = 1\n", encoding="utf-8")
            permissions = runner.isolation_permissions(workspace, "verify", [str(scoped)])
            external = permissions["external_directory"]
            directory_pattern = str(scoped.parent.resolve()) + "/**"
            self.assertEqual(external.get(directory_pattern), "allow")
            self.assertEqual(external.get(str(scoped.resolve())), "allow")
            self.assertEqual(permissions["read"].get(str(scoped.resolve())), "allow")


class ReadTrackingTests(unittest.TestCase):
    """The receipt says what the agent could read, not only what it was allowed to."""

    def test_denied_reads_are_recorded(self):
        import json
        from tools.research.agents import runner
        with tempfile.TemporaryDirectory() as work:
            session = Path(work) / "session.jsonl"
            lines = [
                {"type": "tool_use", "part": {"tool": "read", "state": {
                    "status": "error", "input": {"path": "/tmp/x.py"},
                    "error": "Permission denied: read"}}},
                {"type": "tool_use", "part": {"tool": "read", "state": {
                    "status": "completed", "input": {"path": "/tmp/y.py"}}}},
                {"type": "text", "part": {"text": "answer"}},
            ]
            session.write_text("\n".join(json.dumps(line) for line in lines),
                               encoding="utf-8")
            outcome = runner.session_reads(session)
        self.assertEqual(len(outcome["reads_denied"]), 1)
        self.assertEqual(outcome["reads_denied"][0]["target"], "/tmp/x.py")
        self.assertEqual(outcome["files_read"], ["/tmp/y.py"])
