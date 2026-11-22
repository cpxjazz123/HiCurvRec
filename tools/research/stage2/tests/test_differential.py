"""The differential verifier: agreement is evidence, disagreement stops the round.

The interesting cases are the ones a single reference cannot see: two
derivations with the SAME gradient and a different value (`x**2` and
`x**2 + 10`), and a derivation whose code disagrees with its own statements.
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

from tools.research.rounds import differential as module

SYMBOLS = ["x0", "x1", "p0", "p1", "n0", "n1", "m"]
ARGS = [{"name": "x", "width": 2, "radius": 0.4},
        {"name": "p", "width": 2, "radius": 0.4},
        {"name": "n", "width": 2, "radius": 0.4},
        {"name": "m", "scalar": 0.2}]

STATEMENTS_A = [
    "dx = x0 - p0", "dy = x1 - p1", "ex = x0 - n0", "ey = x1 - n1",
    "dp = sqrt(dx**2 + dy**2)", "dn = sqrt(ex**2 + ey**2)",
    "L = log(1 + exp(-(dp - dn + m)))",
]
STATEMENTS_B = [
    "dp = sqrt((x0 - p0)**2 + (x1 - p1)**2)",
    "dn = sqrt((x0 - n0)**2 + (x1 - n1)**2)",
    "L = log(1 + exp(-(dp - dn + m)))",
]
STATEMENTS_C = [
    "dp = sqrt((x0 - p0)**2 + (x1 - p1)**2)",
    "dn = sqrt((x0 - n0)**2 + (x1 - n1)**2)",
    "z = dn - dp - m",
    "L = log(1 + exp(z))",
]
STATEMENTS_SIGN = [
    "dp = sqrt((x0 - p0)**2 + (x1 - p1)**2)",
    "dn = sqrt((x0 - n0)**2 + (x1 - n1)**2)",
    "L = log(1 + exp(dp - dn + m))",
]

CODE = '''
import torch


def compute(x, p, n, m):
    dp = ((x - p) ** 2).sum(dim=-1).sqrt()
    dn = ((x - n) ** 2).sum(dim=-1).sqrt()
    return torch.log(1 + torch.exp(-(dp - dn + m)))
'''

CODE_WRONG = '''
import torch


def compute(x, p, n, m):
    dp = ((x - p) ** 2).sum(dim=-1).sqrt()
    dn = ((x - n) ** 2).sum(dim=-1).sqrt()
    return 2 * torch.log(1 + torch.exp(-(dp - dn + m)))
'''


def _derivation(name, statements, code=CODE):
    return {"name": name, "symbols": SYMBOLS, "args": ARGS,
            "statements": statements, "output": "L", "code": code}


class SymbolicDifferenceTests(unittest.TestCase):

    def test_a_constant_offset_is_not_equivalent_even_though_the_gradients_match(self):
        left = {"symbols": ["x"], "args": [{"name": "x", "width": 1}],
                "statements": ["f = x**2"], "output": "f"}
        right = {"symbols": ["x"], "args": [{"name": "x", "width": 1}],
                 "statements": ["f = x**2 + 10"], "output": "f"}
        verdict = module.symbolic_difference(left, right)
        self.assertFalse(verdict["equivalent"])
        self.assertIn("10", verdict["residual"])

    def test_two_orders_of_the_same_algebra_are_equivalent(self):
        left = {"symbols": ["x", "y"], "args": [], "statements": ["f = x**2 + 2*x*y + y**2"],
                "output": "f"}
        right = {"symbols": ["x", "y"], "args": [], "statements": ["f = (x + y)**2"],
                 "output": "f"}
        self.assertTrue(module.symbolic_difference(left, right)["equivalent"])


@unittest.skipUnless(HAVE_TORCH, "torch lives in the Stage 2 interpreter")
class DifferentialVerdictTests(unittest.TestCase):

    def test_three_independent_derivations_agree(self):
        verdict = module.compare({"implementations": [
            _derivation("a", STATEMENTS_A), _derivation("b", STATEMENTS_B),
            _derivation("c", STATEMENTS_C)]})
        self.assertEqual(verdict["status"], "AGREE", verdict.get("reason"))
        self.assertEqual(len(verdict["pairs"]), 3)
        self.assertTrue(all(entry["status"] == "OK" for entry in verdict["self_checks"]))

    def test_a_sign_error_is_named(self):
        verdict = module.compare({"implementations": [
            _derivation("a", STATEMENTS_A), _derivation("sign", STATEMENTS_SIGN)]})
        self.assertEqual(verdict["status"], "FAIL")
        self.assertFalse(verdict["pairs"][0]["symbolic"]["equivalent"])

    def test_code_that_disagrees_with_its_own_statements_is_refused(self):
        verdict = module.compare({"implementations": [
            _derivation("a", STATEMENTS_A), _derivation("b", STATEMENTS_B),
            _derivation("wrong", STATEMENTS_C, code=CODE_WRONG)]})
        self.assertEqual(verdict["status"], "FAIL")
        failing = [entry for entry in verdict["self_checks"] if entry["status"] == "FAIL"]
        self.assertEqual([entry["name"] for entry in failing], ["wrong"])
        self.assertGreater(failing[0]["value_max_abs_diff"], 0.01)

    def test_a_single_derivation_is_not_a_verdict(self):
        verdict = module.compare({"implementations": [_derivation("a", STATEMENTS_A)]})
        self.assertEqual(verdict["status"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
