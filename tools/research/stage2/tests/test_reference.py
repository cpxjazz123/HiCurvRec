"""The formula reference: the real distance agrees, both defect classes fail.

These tests need torch, sympy and sympytorch, which live in the Stage 2
interpreter; under any other interpreter they are skipped rather than failed.
"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    import sympytorch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

from tools.research.stage2 import reference as reference_module

DISTANCE_STATEMENTS = [
    "x2 = x0**2 + x1**2",
    "y2 = y0**2 + y1**2",
    "xy = x0*y0 + x1*y1",
    "s = 1 + 2*c*(-xy) + c*y2",
    "t = 1 - c*x2",
    "den = 1 + 2*c*(-xy) + c*c*x2*y2",
    "mx = (s*(-x0) + t*y0)/den",
    "my = (s*(-x1) + t*y1)/den",
    "norm = sqrt(mx**2 + my**2)",
    "d = 2/sqrt(c)*atanh(sqrt(c)*norm)",
]


def _contract(callable_path):
    return {
        "calls": [{
            "name": "poincare_distance",
            "callable": callable_path,
            "symbols": ["x0", "x1", "y0", "y1", "c"],
            "args": [{"name": "x", "width": 2, "radius": 0.4, "samples": 12},
                     {"name": "y", "width": 2, "radius": 0.4, "samples": 12},
                     {"name": "c", "scalar": 1.0}],
            "statements": DISTANCE_STATEMENTS,
            "output": "d",
        }],
        "excluded": [{"surface": "quantizer argmin",
                      "reason": "an argmin is not differentiable"}],
    }


def _tree(contract):
    directory = Path(tempfile.mkdtemp(prefix="reference_"))
    (directory / reference_module.CONTRACT_NAME).write_text(json.dumps(contract))
    return directory


@unittest.skipUnless(HAVE_TORCH, "torch and sympytorch live in the Stage 2 interpreter")
class ReferenceAgreementTests(unittest.TestCase):

    def test_the_real_distance_agrees_with_its_formula(self):
        verdict = reference_module.check_tree(
            _tree(_contract("tools.research.geometry.poincare_contract:distance")))
        self.assertEqual(verdict["status"], "OK", verdict.get("reason"))
        call = verdict["calls"][0]
        self.assertLess(call["value_max_abs_diff"], 1e-12)
        self.assertLess(call["gradient_max_abs_diff"], 1e-10)
        self.assertTrue(call["gradcheck_candidate"])
        self.assertTrue(call["gradcheck_reference"])

    def test_a_formula_with_the_wrong_sign_fails_on_the_values(self):
        verdict = reference_module.check_tree(
            _tree(_contract("tools.research.stage2.tests.fake_mechanisms:wrong_distance")))
        self.assertEqual(verdict["status"], "FAIL")
        self.assertGreater(verdict["calls"][0]["value_max_abs_diff"], 0.1)

    def test_a_wrong_gradient_fails_even_though_the_value_is_right(self):
        verdict = reference_module.check_tree(
            _tree(_contract("tools.research.stage2.tests.fake_mechanisms:wrong_gradient")))
        self.assertEqual(verdict["status"], "FAIL")
        call = verdict["calls"][0]
        self.assertLess(call.get("value_max_abs_diff", 1.0), 1e-12)
        self.assertGreater(call.get("gradient_max_abs_diff", 0.0), 1.0)

    def test_a_tree_without_a_contract_is_recorded_not_passed(self):
        verdict = reference_module.check_tree(Path(tempfile.mkdtemp(prefix="empty_")))
        self.assertEqual(verdict["status"], "NOT_APPLICABLE")
        self.assertIn("reference_contract.json", verdict["reason"])


if __name__ == "__main__":
    unittest.main()
