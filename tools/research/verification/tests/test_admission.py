import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.research.verification import admission as A


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = self.root / "issue116"
        self.cases = self.root / "cases"
        self.tree = self.root / "proposed_hyperbolic"
        self.tree.mkdir()
        self.run_id = "run_0123456789abcdef"
        record = self.evidence / "runs" / self.run_id / "run.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"run_id": self.run_id}))
        (self.tree / "train_rqvae.py").write_text("print('real trainer placeholder')\n")

    def test_missing_manifest_is_fail_closed(self):
        with patch.object(A, "EVIDENCE_ROOT", self.evidence):
            with patch.object(A, "CASES_ROOT", self.cases):
                result = A.run_preflight(self.tree, "novel_behavior_a", "/usr/bin/python3")
        self.assertFalse(result["passed"])
        self.assertIn("evidence", result["refused"])

    def test_stale_source_rejected_and_baselines_protected(self):
        with patch.object(A, "EVIDENCE_ROOT", self.evidence):
            with patch.object(A, "CASES_ROOT", self.cases):
                base = self.root / "TIGER_RQ-VAE"
                base.mkdir()
                (base / "train_rqvae.py").write_text("pass\n")
                with self.assertRaises(A.AdmissionError):
                    A.validate_evidence(base, "new_hyp")
                self.assertNotEqual(A.source_digest(self.tree), "")
                old = A.source_digest(self.tree)
                (self.tree / "model.py").write_text("changed=True\n")
                self.assertNotEqual(old, A.source_digest(self.tree))

    def test_missing_independent_agent_evidence_rejected(self):
        cid = "new_math_model"
        case_dir = self.cases / cid
        case_dir.mkdir(parents=True)
        tests = {}
        for kind in A.SUITES:
            path = case_dir / ("test_" + kind + ".py")
            path.write_text("import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n")
            tests[kind] = A.sha256(path)
        proposal = self.evidence / "runs" / self.run_id / "agents" / "geometry" / "candidate.json"
        proposal.parent.mkdir(parents=True)
        proposal.write_text(json.dumps({"candidate_id": cid}))
        manifest_dir = self.evidence / "verification" / cid
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.json").write_text(json.dumps({
            "schema": 1, "candidate_id": cid, "run_id": self.run_id,
            "source_tree": self.tree.name,
            "source_sha256": A.source_digest(self.tree),
            "proposal_sha256": A.sha256(proposal), "tests": tests,
        }))
        with patch.object(A, "EVIDENCE_ROOT", self.evidence):
            with patch.object(A, "CASES_ROOT", self.cases):
                with self.assertRaisesRegex(A.AdmissionError, "evidence"):
                    A.validate_evidence(self.tree, cid)

    def _complete_fixture(self, override=None):
        """Synthetic test-runner fixtures only; not proof of any real mechanism."""
        cid = "new_behavior_case"
        folder = self.cases / cid
        folder.mkdir(parents=True)
        frozen_tests = {}
        for name in A.SUITES:
            path = folder / ("test_" + name + ".py")
            code = (
                "import unittest\n"
                "class IndependentFixture(unittest.TestCase):\n"
                "    def test_arithmetic(self): self.assertEqual(2, 1+1)\n"
            )
            if override and name in override:
                code = override[name]
            path.write_text(code, encoding="utf-8")
            frozen_tests[name] = A.sha256(path)
        proposal = self.evidence / "agents" / "geometry" / "candidate.json"
        proposal.parent.mkdir(parents=True)
        proposal.write_text(json.dumps({"candidate_id": cid}))
        code_sha = A.source_digest(self.tree)
        proposal_sha = A.sha256(proposal)
        path = self.evidence / "verification" / cid / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "schema": 1, "candidate_id": cid, "source_tree": self.tree.name,
            "source_sha256": code_sha, "proposal_sha256": proposal_sha,
            "tests": frozen_tests,
        }), encoding="utf-8")
        for pos, role in enumerate(A.AGENT_ROLES):
            session = self.evidence / "runs" / self.run_id / "agents" / (cid + "_" + role)
            session.mkdir(parents=True)
            output = session / (role + ".json")
            content = {"candidate_id": cid, "source_sha256": code_sha}
            if role == "review":
                content.update({
                    "verdict": "APPROVE_TESTS",
                    "proposal_sha256": proposal_sha,
                    "test_sha256": frozen_tests,
                })
            output.write_text(json.dumps(content), encoding="utf-8")
            stream = session / "session.jsonl"
            stream.write_text('{"event":"fixture_only"}\n', encoding="utf-8")
            (session / "execution.json").write_text(json.dumps({
                "verified": True, "exit_code": 0, "timed_out": False,
                "pid": 2000 + pos, "output_name": output.name,
                "output_sha256": A.sha256(output), "session_sha256": A.sha256(stream),
            }), encoding="utf-8")
        return cid

    def _run_fixture(self, cid):
        with patch.object(A, "EVIDENCE_ROOT", self.evidence):
            with patch.object(A, "CASES_ROOT", self.cases):
                return A.run_preflight(self.tree, cid, sys.executable)

    def test_approved_fixture_executes_all_four_real_subprocess_suites(self):
        cid = self._complete_fixture()
        report = self._run_fixture(cid)
        self.assertTrue(report["passed"], report)
        self.assertEqual(set(report["suites"]), set(A.SUITES))
        self.assertTrue(all(item["tests_run"] >= 1 for item in report["suites"].values()))

    def test_unittest_skips_are_not_admission(self):
        cid = self._complete_fixture({"ddp": (
            "import unittest\n"
            "class Dummy(unittest.TestCase):\n"
            " @unittest.skip('no real DDP')\n"
            " def test_must_run(self): self.assertTrue(True)\n"
        )})
        report = self._run_fixture(cid)
        self.assertFalse(report["passed"])
        self.assertFalse(report["suites"]["ddp"]["passed"])
        self.assertIn("skipped", report["refused"])

    def test_unittest_zero_tests_are_not_admission(self):
        cid = self._complete_fixture({"math": "import unittest\n"})
        report = self._run_fixture(cid)
        self.assertFalse(report["passed"])
        self.assertEqual(report["suites"]["math"]["tests_run"], 0)

    def test_source_drift_after_approval_is_rejected(self):
        cid = self._complete_fixture()
        (self.tree / "model.py").write_text("changed=42\n", encoding="utf-8")
        report = self._run_fixture(cid)
        self.assertFalse(report["passed"])
        self.assertIn("trainer source", report["refused"])

    def test_invalid_ids_fail(self):
        for cid in ("../escape", "A/B", "", "a", "bad name"):
            with self.assertRaises(A.AdmissionError):
                A.validate_evidence(self.tree, cid)


if __name__ == "__main__":
    unittest.main()
