"""The acceptance scenarios the review asked for.

Three things have to hold, and they are checked here rather than argued:

* a candidate the framework created can reach the verification phase: the
  permission roots and the place the trees are created are the same path;
* an agent that ran out of its budget hands its saved records to the main
  agent, marked as a partial delivery, and a partial review is never a verdict;
* one round has one worker: a second call does not start a second one.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.research.agents import policy, runner
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import plaintext, worker


class CandidatePathTests(unittest.TestCase):
    """The tree the framework creates is a tree the policy lets a reviewer read."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.issue = self.repo / "results" / "research" / "issue121"
        self.round_dir = self.issue / "rounds" / "round_one"
        self.round_dir.mkdir(parents=True)
        self.trees = self.issue / "trees" / "round_one"
        self.trees.mkdir(parents=True)
        self.candidate = self.trees / "idea_a"
        self.candidate.mkdir()

    def test_the_roots_contain_the_candidate_the_framework_creates(self):
        roots = worker._round_roots(self.round_dir)
        accepted = policy.check_read_paths([self.candidate], roots=roots,
                                          task_root=self.round_dir,
                                          candidate_tree=self.candidate)
        self.assertEqual([Path(item) for item in accepted], [self.candidate])

    def test_the_roots_are_the_round_and_the_round_trees(self):
        roots = worker._round_roots(self.round_dir)
        self.assertEqual(Path(roots[0]).resolve(), self.round_dir.resolve())
        self.assertEqual(Path(roots[1]).resolve(), self.trees.resolve())

    def test_a_round_directory_holds_the_other_answers_and_is_refused(self):
        roots = worker._round_roots(self.round_dir)
        with self.assertRaises(policy.PolicyError):
            policy.check_read_paths([self.round_dir], roots=roots,
                                     task_root=self.round_dir)

    def test_another_round_is_refused(self):
        other = self.issue / "rounds" / "round_two"
        other.mkdir()
        roots = worker._round_roots(self.round_dir)
        with self.assertRaises(policy.PolicyError):
            policy.check_read_paths([other], roots=roots, task_root=self.round_dir)


class PartialDeliveryTests(unittest.TestCase):
    """A task that ran out of time still delivers what it saved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _agent_that_times_out(self, records: list):
        """A fake executor that fails like a task that hit its deadline."""
        def fake_run(task_id, prompt, output_name, root, **kwargs):
            workspace = Path(root) / "agents" / task_id
            workspace.mkdir(parents=True, exist_ok=True)
            progress = workspace / runner.PROGRESS_FILE_NAME
            progress.write_text("".join(json.dumps(record) + "\n" for record in records),
                                encoding="utf-8")
            receipt = workspace / "execution.json"
            receipt.write_text(json.dumps({"task_id": task_id, "verified": False,
                                           "failure_class": "timeout"}), encoding="utf-8")
            raise runner.AgentFailure("timeout", str(receipt),
                                      "the agent did not finish")
        return fake_run

    def test_the_saved_records_reach_the_main_agent(self):
        block = ("IDEA: idea_a\nTITLE: t\nMOTIVATION: m\nNOVELTY: n\nCHANGE: c\n"
                 "SPECIFICATION: s\nCOST: c\nHYPOTHESIS: h\nRISKS: r\n"
                 "BEHAVIOR SIGNAL: b\nBEHAVIOR COMPUTATION: bc\n"
                 "LEAKAGE BOUNDARY: l\nHYPERBOLIC OPERATOR: o\n")
        records = [{"seq": 1, "task_id": "ideation_1", "kind": "stage",
                    "content": "stage one"},
                   {"seq": 2, "task_id": "ideation_1", "kind": "stage", "content": block}]
        with mock.patch.object(worker, "run_agent",
                               side_effect=self._agent_that_times_out(records)):
            report = worker._run_agent(self.root, "ideation_1", "prompt", "ideas.txt",
                                       plaintext.parse_ideas, role="ideate",
                                       read_paths=[],
                                       allowed_roots=[self.root])
        self.assertTrue(report["partial"])
        self.assertEqual(report["failure_class"], "timeout")
        self.assertEqual(report["progress_count"], 2)
        self.assertEqual(report["ideas"][0]["idea_id"], "idea_a")

    def test_a_partial_review_is_never_a_verdict(self):
        reports = [{"verifier": "v" + str(index), "verdict": "PASS", "checks": [],
                    "partial": True, "timed_out": True} for index in range(1, 6)]
        state = {"ideas": [], "implementations": [], "step": "verification"}
        with self.assertRaises(pipeline_module.RoundError):
            pipeline_module.apply_verification(state, "a", reports)

    def test_a_complete_review_is_still_a_verdict(self):
        reports = [{"verifier": "v" + str(index), "verdict": "PASS", "checks": [],
                    "boundary_version": policy.BOUNDARY_VERSION,
                    "boundary_sha256": "b"} for index in range(1, 6)]
        with tempfile.TemporaryDirectory() as work:
            tree = Path(work) / "tree"
            tree.mkdir()
            (tree / "model.py").write_text("x = 1\n", encoding="utf-8")
            state = {"ideas": [], "implementations": [], "step": "verification",
                     "events": []}
            state["implementations"] = [pipeline_module.implementation_record(
                {"idea_id": "a"}, {"changed_files": ["model.py"]}, tree)]
            state = pipeline_module.apply_verification(state, "a", reports)
        self.assertEqual(state["implementations"][0]["status"], "verified")


class OneWorkerTests(unittest.TestCase):
    """A round has one worker."""

    def test_the_liveness_check_finds_the_worker_by_its_round_id(self):
        import importlib.util
        server_path = Path(__file__).resolve().parents[2] / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_server", str(server_path))
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        with tempfile.TemporaryDirectory() as work:
            script = Path(work) / "round_worker.py"
            script.write_text("print('worker')\n", encoding="utf-8")
            process = subprocess.Popen(
                ["/usr/bin/python3", str(script), json.dumps({"round_id": "round_one"})],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                # The round id is in the command line, so the check finds the
                # worker; the log path is not, so a check against it would
                # never find one and would start a second.
                self.assertTrue(server.state_process_live(process.pid, "round_one"))
                self.assertFalse(server.state_process_live(process.pid,
                                                         str(work) + "/round.log"))
            finally:
                process.terminate()
                process.wait(timeout=10)


class DeadlineTests(unittest.TestCase):
    """The budget covers the whole call, not only the part after the start."""

    def test_a_run_that_overran_the_budget_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            binary = root / "fake-opencode"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import time, sys\n"
                "time.sleep(2)\n"
                "print('{}')\n", encoding="utf-8")
            binary.chmod(0o700)
            with mock.patch.object(runner, "sandbox_command",
                                   side_effect=lambda command, *args, **kwargs: list(command)), \
                 mock.patch.object(runner, "agent_sandbox_home",
                                   side_effect=lambda workspace: Path(tempfile.mkdtemp())):
                with self.assertRaises(runner.AgentFailure) as caught:
                    runner.run_agent("deadline_probe", "prompt", "answer.txt", root,
                                     binary=str(binary), timeout=1, role="verify",
                                     read_paths=[], allowed_roots=[root],
                                     deadline_seconds=1)
            self.assertEqual(caught.exception.failure_class, "timeout")
            receipt = json.loads((root / "agents" / "deadline_probe" / "execution.json")
                                  .read_text())
            self.assertFalse(receipt["verified"])
            self.assertTrue(receipt["deadline_passed"])


class ProgressTailTests(unittest.TestCase):
    """A truncated tail does not swallow the next record."""

    def test_a_fragment_is_isolated_and_the_next_record_stands_alone(self):
        from tools.research.agents import progress_server
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "progress.jsonl"
            path.write_text('{"seq": 1, "content": "whole"}\n{"seq": 2, "content": "half',
                            encoding="utf-8")
            progress = progress_server.ProgressFile(path, "task_one")
            progress.append("a new record")
            records = progress_server.read_records(path)
            self.assertEqual([record["content"] for record in records],
                             ["whole", "a new record"])
            self.assertEqual(records[-1]["seq"], 2)


if __name__ == "__main__":
    unittest.main()
