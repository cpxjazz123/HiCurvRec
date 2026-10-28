import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tools.research import settings
from tools.research.agents import runner
from tools.research.agents.runner import (isolation_permissions, run_agent,
                                          sandbox_command)
from tools.research.core.evidence import read_json


@contextlib.contextmanager
def plain_process():
    """The executor without the namespace.

    The namespace is the subject of its own test; these tests are about the
    answer contract, so the fake binary runs directly. There is no production
    seam: ``run_agent`` always builds the namespace.
    """
    with mock.patch.object(runner, "sandbox_command",
                           side_effect=lambda command, *args, **kwargs: list(command)), \
         mock.patch.object(runner, "agent_sandbox_home",
                           side_effect=lambda workspace: Path(tempfile.mkdtemp())):
        yield


class AgentTests(unittest.TestCase):
    def test_the_boundary_is_written_in_the_schema_opencode_reads(self):
        """opencode v2 reads a ``permission`` object, not a rules array.

        The earlier array shape was silently ignored, so every agent kept a full
        shell; the test now pins the shape the binary actually honours.
        """
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "agents" / "task"
            verifier = isolation_permissions(workspace, "verify")
            self.assertEqual(verifier["bash"], "deny")
            self.assertEqual(verifier["webfetch"], "deny")
            self.assertEqual(verifier["external_directory"], "deny")
            # The verifier and the repair planner may not write anything: they
            # review and plan, and the operator applies.
            for role in ("verify", "repair"):
                reader = isolation_permissions(workspace, role)
                self.assertEqual(reader["edit"], "deny")
                self.assertEqual(reader["write"], "deny")
                self.assertEqual(reader["bash"], "deny")
            worker = isolation_permissions(workspace, "implementation")
            self.assertEqual(worker["bash"], "allow")
            self.assertEqual(worker["edit"], "allow")
            self.assertEqual(worker["webfetch"], "deny")
            self.assertEqual(worker["external_directory"], "deny")

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is not installed")
    def test_pwd_is_the_workspace_so_the_project_config_applies(self):
        """A child inherits the server's PWD; opencode resolves the project from it.

        The old runner passed ``cwd=workspace`` but left ``PWD`` at the repository
        root, so opencode loaded the repository project, ignored the workspace
        ``opencode.json`` and ran read-only roles with a shell. That is how a
        verifier read the previous round's answer.
        """
        with tempfile.TemporaryDirectory() as work:
            binary = Path(work) / "fake-opencode"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "Path('pwd_probe.json').write_text(json.dumps("
                "{'pwd': os.environ.get('PWD'), 'cwd': os.getcwd()}))\n"
                "Path('answer.json').write_text(json.dumps({'ok': True}))\n"
                "print(json.dumps({'type': 'mock'}))\n",
                encoding="utf-8")
            binary.chmod(0o700)
            with plain_process():
                receipt = run_agent("pwd_probe", "write answer", "answer.json",
                                    Path(work) / "out", binary=str(binary), timeout=10,
                                    read_paths=[], allowed_roots=[Path(work)])
            probe = json.loads((receipt.parent / "pwd_probe.json").read_text())
            self.assertEqual(probe["pwd"], str(receipt.parent))
            self.assertEqual(Path(probe["cwd"]).resolve(), receipt.parent.resolve())

    def test_a_read_allowlist_denies_every_other_path(self):
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "agents" / "task"
            tree = Path(work) / "trees" / "candidate"
            tree.mkdir(parents=True)
            policy = isolation_permissions(workspace, "verify", [tree])
            self.assertEqual(policy["read"]["*"], "deny")
            self.assertEqual(policy["read"][str(tree.resolve()) + "/**"], "allow")
            self.assertEqual(policy["read"][str(workspace.resolve()) + "/**"], "allow")
            self.assertEqual(policy["external_directory"]["*"], "deny")
            self.assertEqual(policy["external_directory"][str(tree.resolve()) + "/**"],
                             "allow")
            # A reviewer may not delegate to a subagent with different rules.
            self.assertEqual(policy["task"], "deny")

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is not installed")
    def test_the_sandbox_hides_every_path_outside_the_allowlist(self):
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "ws"
            workspace.mkdir()
            allowed = Path(work) / "allowed"
            allowed.mkdir()
            (allowed / "ok.txt").write_text("ALLOWED", encoding="utf-8")
            forbidden = Path(work) / "forbidden"
            forbidden.mkdir()
            (forbidden / "secret.txt").write_text("SECRET", encoding="utf-8")
            private = Path(work) / "home"
            (private / ".local" / "share" / "opencode").mkdir(parents=True)
            (private / ".cache" / "opencode").mkdir(parents=True)
            script = ("cat " + str(allowed / "ok.txt") + "; "
                      "cat " + str(forbidden / "secret.txt") + " 2>/dev/null "
                      "|| echo DENIED")
            command = sandbox_command(["/bin/sh", "-c", script], workspace,
                                      [allowed], Path("/bin/sh"), private)
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
            self.assertIn("ALLOWED", completed.stdout)
            self.assertIn("DENIED", completed.stdout)
            self.assertNotIn("SECRET", completed.stdout)

    def test_a_read_only_role_answers_through_the_session_stream(self):
        """A role that may not write still delivers its answer."""
        with tempfile.TemporaryDirectory() as work:
            binary = Path(work) / "fake-opencode"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:4] == ['run','--standalone','--format']\n"
                "print(json.dumps({'type': 'text', 'part': "
                "{'text': 'VERDICT: PASS'}}))\n",
                encoding="utf-8",
            )
            binary.chmod(0o700)
            with plain_process():
                receipt = run_agent("test_readonly", "answer", "verification.txt",
                                    Path(work) / "out", binary=str(binary), timeout=10,
                                    role="verify", read_paths=[],
                                    allowed_roots=[Path(work)])
            evidence = read_json(receipt)
            self.assertTrue(evidence["verified"])
            answer = receipt.parent / "verification.txt"
            self.assertTrue(answer.is_file())
            self.assertIn("VERDICT: PASS", answer.read_text(encoding="utf-8"))

    def test_fake_opencode_is_real_subprocess_not_a_marker(self):
        with tempfile.TemporaryDirectory() as work:
            binary = Path(work) / "fake-opencode"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:4] == ['run','--standalone','--format']\n"
                "Path('answer.json').write_text(json.dumps({'ok': True}))\n"
                "print(json.dumps({'type': 'mock', 'tool': 'none'}))\n",
                encoding="utf-8",
            )
            binary.chmod(0o700)
            with plain_process():
                receipt = run_agent("test_agent", "Write answer", "answer.json",
                                    Path(work) / "out", binary=str(binary), timeout=10,
                                    read_paths=[], allowed_roots=[Path(work)])
            evidence = read_json(receipt)
            self.assertTrue(evidence["verified"])
            self.assertGreater(evidence["pid"], 0)
            self.assertEqual(evidence["exit_code"], 0)
            self.assertTrue((receipt.parent / "session.jsonl").exists())
            # Every agent run carries its own permission boundary, and the receipt
            # names the boundary it ran under.
            config = receipt.parent / evidence["isolation_config"]
            self.assertTrue(config.is_file())
            written = json.loads(config.read_text(encoding="utf-8"))
            self.assertIn("permission", written)
            self.assertIn("bash", written["permission"])
            # The workspace must be its own opencode project, otherwise opencode
            # resolves the repository as the project and ignores the boundary.
            self.assertEqual(evidence["project_marker"], ".git")
            self.assertTrue((receipt.parent / ".git").is_dir())


    def test_archived_candidate_cannot_be_reused_as_new_agent_output(self):
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "out" / "agents" / "geometry"
            archived = workspace / "attempt-1"
            archived.mkdir(parents=True)
            (archived / "candidate.json").write_text('{"old":true}', encoding="utf-8")
            binary = Path(work) / "fake-opencode-no-output"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "print(json.dumps({'type':'mock','status':'no_output'}))\n",
                encoding="utf-8")
            binary.chmod(0o700)
            # The contract is that the archived answer is NOT adopted as this
            # run's output; the wording of the failure is not the contract.
            from tools.research.agents.runner import AgentFailure
            with self.assertRaises(AgentFailure) as caught:
                with plain_process():
                    run_agent("geometry", "Write candidate", "candidate.json",
                              Path(work) / "out", binary=str(binary), timeout=10,
                              read_paths=[], allowed_roots=[Path(work)])
            self.assertIn(caught.exception.failure_class, {"empty", "unusable", "crash"})
            self.assertFalse((workspace / "candidate.json").is_file(),
                             "an archived candidate was adopted as the new answer")
            receipt = read_json(workspace / "execution.json")
            self.assertFalse(receipt["verified"])
            self.assertIsNone(receipt["output_sha256"])

    def test_prompt_names_the_absolute_output_path(self):
        """A relative output path is resolved against the git root, not the workspace."""
        from tools.research.agents.prompts import build_prompt
        manifest = {"task_id": "literature", "inputs_sha256": "abc"}
        root = Path("/tmp/run_root")
        prompt = build_prompt("LITERATURE", manifest, "literature.json", root=root)
        expected = str(root / "agents" / "literature" / "literature.json")
        self.assertIn(expected, prompt)
        self.assertNotIn("in your working directory", prompt)


if __name__ == "__main__":
    unittest.main()
