import tempfile
import unittest
from pathlib import Path
from tools.research.agents.prompts import build_prompt
from tools.research.core.evidence import digest_file
from tools.research.science.contracts import check_review_identity, check_candidate


class ContractTests(unittest.TestCase):
    def test_stale_review_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as work:
            candidate = Path(work) / "candidate.json"
            candidate.write_text('{"candidate_id":"behavior-116-1"}', encoding="utf-8")
            sha = digest_file(candidate)
            manifest = {"candidate_id": "behavior-116-1", "candidate_sha256": sha,
                        "forbidden_stale_formulas": ["UNSIGNED WEDGE"]}
            check_review_identity(candidate, manifest, "behavior-116-1 " + sha)
            with self.assertRaises(ValueError):
                check_review_identity(candidate, manifest,
                                      "behavior-116-1 " + sha + " UNSIGNED WEDGE")
            with self.assertRaises(ValueError):
                check_review_identity(candidate, {"candidate_id": "other",
                                                  "candidate_sha256": sha}, "other " + sha)

    def test_geometry_requires_behavior(self):
        with self.assertRaises(ValueError):
            build_prompt("GEOMETRY", {"task_id": "geometry", "inputs_sha256": "x"}, "candidate.json")

    def test_incomplete_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            check_candidate({"candidate_id": "toy"})


if __name__ == "__main__":
    unittest.main()
