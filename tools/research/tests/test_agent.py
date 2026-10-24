import json
import os
import tempfile
import unittest
from pathlib import Path
from tools.research import settings
from tools.research.agents.runner import isolation_permissions, run_agent
from tools.research.core.evidence import read_json


class AgentTests(unittest.TestCase):
    def test_agent_is_confined_to_its_own_workspace(self):
        """The permission boundary must deny shell, Code Mode and every escape path."""
        with tempfile.TemporaryDirectory() as work:
            workspace = Path(work) / "agents" / "task"
            rules = isolation_permissions(workspace)
            effects = {(rule["action"], rule["resource"]): rule["effect"]
                       for rule in rules}
            self.assertEqual(effects[("shell", "*")], "deny")
            self.assertEqual(effects[("execute", "*")], "deny")
            self.assertEqual(effects[("external_directory", "*")], "deny")
            self.assertEqual(effects[("edit", "*")], "allow")
            self.assertEqual(effects[("edit", str(settings.ROOT) + "/*")], "deny")
            self.assertEqual(effects[("edit", str(workspace.resolve()) + "/*")], "allow")
            # The traversal denies must come last: opencode v2 lets the LAST match win.
            for action in ("../*", "*/../*", "*..*"):
                self.assertEqual(effects[("edit", action)], "deny")
                self.assertEqual([rule["resource"] for rule in rules].index(action),
                                 len(rules) - 3 + ["../*", "*/../*", "*..*"].index(action))

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
            self.assertIn("permissions", written)
            self.assertTrue(written["permissions"])
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
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                run_agent("geometry", "Write candidate", "candidate.json",
                          Path(work) / "out", binary=str(binary), timeout=10)
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
