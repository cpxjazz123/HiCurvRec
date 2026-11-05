"""The formula compiler's acceptance: it must REJECT, not merely accept.

The framework's value is not that correct code passes; it is that wrong code
does not. These tests inject the four classes of error the review asked for -
a sign error, a wrong gradient direction, a wrong scale, a wrong assignment -
and require the deterministic verifier to refuse each of them, while three
correct implementations agree.
"""
import unittest

try:
    import torch

    import sympytorch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

from tools.research.rounds import specification as spec_module

STATEMENTS = [
    "x2 = x0**2 + x1**2",
    "d = sqrt(x2)",
]
ARGS = [{"name": "x", "width": 2, "radius": 0.6}]
SYMBOLS = ["x0", "x1"]

CORRECT = '''
import torch


def compute(x):
    return torch.sqrt((x ** 2).sum(dim=-1))
'''

SIGN_ERROR = '''
import torch


def compute(x):
    return torch.sqrt(-(x ** 2).sum(dim=-1).clamp_min(1e-8))
'''

SCALE_ERROR = '''
import torch


def compute(x):
    return 2.0 * torch.sqrt((x ** 2).sum(dim=-1))
'''

GRADIENT_ERROR = '''
import torch


def compute(x):
    value = torch.sqrt((x ** 2).sum(dim=-1))
    return value.detach() - (value - value.detach())
'''


def _implementation(name, code):
    return {"name": name, "symbols": SYMBOLS, "args": ARGS,
            "statements": STATEMENTS, "output": "d", "code": code}


class SpecificationTests(unittest.TestCase):

    def test_a_complete_specification_has_no_problems(self):
        spec = {"mechanism": "m",
                "computations": [{"name": "distance", "definition": "d = |x|",
                                  "inputs": ["x"], "output": "d"}],
                "inputs": ["x"], "outputs": ["d"], "parameters": ["none"],
                "dimensions": {"d": 2},
                "gradient_rules": [{"operator": "sqrt", "rule": "the framework's"}],
                "training_integration": {"loss": "the distance", "reconstruction":
                                         "the baseline", "optimizer": "adamw"},
                "switches": ["none"], "excluded": ["the argmin"],
                "continuous_surface": {"callable": "m:compute", "symbols": SYMBOLS,
                                       "args": ARGS, "statements": STATEMENTS,
                                       "output": "d"}}
        self.assertEqual(spec_module.validate(spec), [])

    def test_an_incomplete_specification_names_what_is_missing(self):
        problems = spec_module.validate({"mechanism": "m"})
        self.assertTrue(any("computations" in problem for problem in problems))
        self.assertTrue(any("gradient_rules" in problem for problem in problems))
        self.assertTrue(any("dimensions" in problem for problem in problems))

    def test_the_frozen_hash_matches_the_file(self):
        import tempfile
        from pathlib import Path
        directory = Path(tempfile.mkdtemp(prefix="spec_"))
        frozen = spec_module.freeze(directory, {"mechanism": "m", "a": 1})
        loaded = spec_module.load(directory)
        self.assertEqual(frozen["sha256"], loaded["sha256"])
        self.assertTrue(spec_module.matches_frozen(directory, frozen["sha256"])["matches"])
        path = directory / spec_module.SPEC_NAME
        path.write_text(path.read_text() + "\n")
        self.assertFalse(spec_module.matches_frozen(directory, frozen["sha256"])["matches"])


@unittest.skipUnless(HAVE_TORCH, "torch and sympytorch live in the Stage 2 interpreter")
class AcceptanceTests(unittest.TestCase):

    def _verdict(self, *implementations):
        from tools.research.rounds import differential
        return differential.compare({"implementations": list(implementations)})

    def test_three_correct_implementations_agree(self):
        verdict = self._verdict(_implementation("a", CORRECT),
                                _implementation("b", CORRECT),
                                _implementation("c", CORRECT))
        self.assertEqual(verdict["status"], "AGREE", verdict.get("reason"))

    def test_a_sign_error_is_refused(self):
        verdict = self._verdict(_implementation("ok", CORRECT),
                                _implementation("sign", SIGN_ERROR))
        self.assertEqual(verdict["status"], "FAIL")
        kinds = [pair.get("kind") for pair in verdict.get("pairs") or []]
        self.assertIn("contradiction", kinds)

    def test_a_scale_error_is_refused(self):
        verdict = self._verdict(_implementation("ok", CORRECT),
                                _implementation("scale", SCALE_ERROR))
        self.assertEqual(verdict["status"], "FAIL")
        self.assertGreater(float(verdict["pairs"][0]["value_max_abs_diff"]), 1e-3)

    def test_a_wrong_gradient_is_refused_even_though_the_value_is_right(self):
        verdict = self._verdict(_implementation("ok", CORRECT),
                                _implementation("grad", GRADIENT_ERROR))
        self.assertEqual(verdict["status"], "FAIL")
        pair = verdict["pairs"][0]
        self.assertLess(float(pair["value_max_abs_diff"]), 1e-12)
        self.assertGreater(float(pair["gradient_max_abs_diff"]), 1e-6)

    def test_a_missing_implementation_is_not_a_pass(self):
        verdict = self._verdict(_implementation("ok", CORRECT))
        self.assertNotEqual(verdict["status"], "AGREE")


if __name__ == "__main__":
    unittest.main()
