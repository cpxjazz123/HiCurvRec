import json
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
        proposal = self.evidence / "agents" / "geometry" / "candidate.json"
        proposal.parent.mkdir(parents=True)
        proposal.write_text(json.dumps({"candidate_id": cid}))
        manifest_dir = self.evidence / "verification" / cid
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.json").write_text(json.dumps({
            "schema": 1, "candidate_id": cid, "source_tree": self.tree.name,
            "source_sha256": A.source_digest(self.tree),
            "proposal_sha256": A.sha256(proposal), "tests": tests,
        }))
        with patch.object(A, "EVIDENCE_ROOT", self.evidence):
            with patch.object(A, "CASES_ROOT", self.cases):
                with self.assertRaisesRegex(A.AdmissionError, "evidence"):
                    A.validate_evidence(self.tree, cid)

    def test_invalid_ids_fail(self):
        for cid in ("../escape", "A/B", "", "a", "bad name"):
            with self.assertRaises(A.AdmissionError):
                A.validate_evidence(self.tree, cid)


if __name__ == "__main__":
    unittest.main()
