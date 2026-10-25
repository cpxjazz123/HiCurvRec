"""Issue #124: the candidate's own path check must discriminate."""
from __future__ import annotations

import unittest

from tools.research.rounds import candidate_checks


class CandidateCheckIdentityTests(unittest.TestCase):
    def test_the_verdicts_are_distinct_and_named(self):
        verdicts = {candidate_checks.VERDICT_VERIFIED,
                    candidate_checks.VERDICT_DEGENERATE,
                    candidate_checks.VERDICT_INCONCLUSIVE}
        self.assertEqual(len(verdicts), 3)
        self.assertIn("DEGENERATE", candidate_checks.VERDICT_DEGENERATE)

    def test_a_missing_candidate_is_inconclusive_not_verified(self):
        report = candidate_checks.run_candidate_check("/tmp/does-not-exist-issue124")
        self.assertEqual(report["verdict"], candidate_checks.VERDICT_INCONCLUSIVE)
        self.assertIn("reason", report)

    def test_the_behaviour_control_replaces_the_signal_with_one_value(self):
        import torch
        curvature = torch.tensor([0.2, 0.5, 0.9])
        controlled = candidate_checks._interventions(curvature, True)
        self.assertEqual(len(set(controlled.tolist())), 1)
        self.assertAlmostEqual(float(controlled[0]), float(curvature.mean()), places=6)

    def test_the_untouched_control_returns_the_signal(self):
        import torch
        curvature = torch.tensor([0.2, 0.5, 0.9])
        self.assertTrue(torch.equal(candidate_checks._interventions(curvature, False),
                                    curvature))


if __name__ == "__main__":
    unittest.main()
