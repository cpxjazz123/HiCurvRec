import json
import os
import tempfile
import unittest
from pathlib import Path
from tools.research import settings
from tools.research.agents.runner import isolation_permissions, run_agent
from tools.research.core.evidence import read_json


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
            receipt = run_agent("test_readonly", "answer", "verification.txt",
                                Path(work) / "out", binary=str(binary), timeout=10,
                                role="verify")
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
            receipt = run_agent("test_agent", "Write answer", "answer.json",
                                Path(work) / "out", binary=str(binary), timeout=10)
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
                run_agent("geometry", "Write candidate", "candidate.json",
                          Path(work) / "out", binary=str(binary), timeout=10)
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
