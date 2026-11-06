"""The derivation phase: the instance, the runtime contract and the comparison.

These tests cover the faults the derivation phase showed in the round of
2026-11-06: the instance was a loss-shaped fixed one that a mechanism which
changes an assignment cannot use; the acceptance check compared names and not
widths; a missing agent shifted the other agents' answers; the previous answer
sat inside the agent's own readable workspace; and a layout mismatch was
recorded as a contradiction, which it is not.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.research.rounds import differential, step5_derivation

HAVE_TORCH = True
try:
    import torch
except Exception:                                          # noqa: BLE001
    HAVE_TORCH = False


SURFACE = {
    "args": [{"name": "x", "width": 2, "radius": 0.4},
             {"name": "y", "width": 2, "radius": 0.4},
             {"name": "curvature", "scalar": 1.0}],
    "symbols": ["x0", "x1", "y0", "y1", "c"],
    "statements": [
        "xy = -(x0*y0 + x1*y1)",
        "xn = x0**2 + x1**2",
        "yn = y0**2 + y1**2",
        "num0 = (1 + 2*c*xy + c*yn)*(-x0) + (1 - c*xn)*y0",
        "num1 = (1 + 2*c*xy + c*yn)*(-x1) + (1 - c*xn)*y1",
        "den = 1 + 2*c*xy + c**2*xn*yn + 1e-12",
        "mx0 = num0/den",
        "mx1 = num1/den",
        "mn = mx0**2 + mx1**2",
        "d = 2/sqrt(c)*atanh(Min(sqrt(c)*sqrt(mn), 1 - 1e-6))",
    ],
    "output": "d",
}

CODE = """import math
import torch

def compute(x, y, curvature):
    c = float(curvature)
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    x0, x1 = x[..., 0], x[..., 1]
    y0, y1 = y[..., 0], y[..., 1]
    xy = -(x0 * y0 + x1 * y1)
    xn = x0 ** 2 + x1 ** 2
    yn = y0 ** 2 + y1 ** 2
    num0 = (1 + 2 * c * xy + c * yn) * (-x0) + (1 - c * xn) * y0
    num1 = (1 + 2 * c * xy + c * yn) * (-x1) + (1 - c * xn) * y1
    den = 1 + 2 * c * xy + c * c * xn * yn + 1e-12
    mx0 = num0 / den
    mx1 = num1 / den
    mn = mx0 ** 2 + mx1 ** 2
    inner = math.sqrt(c) * torch.sqrt(torch.clamp(mn, min=0.0))
    return (2.0 / math.sqrt(c)) * torch.atanh(torch.clamp(inner, max=1.0 - 1e-6))
"""


def payload(**overrides) -> dict:
    record = {"symbols": list(SURFACE["symbols"]),
              "args": [dict(entry) for entry in SURFACE["args"]],
              "statements": list(SURFACE["statements"]),
              "output": SURFACE["output"],
              "code": CODE}
    record.update(overrides)
    return record


class TheInstanceFollowsTheFrozenSurface(unittest.TestCase):
    def test_the_signature_is_the_specification_one(self):
        args, symbols = step5_derivation.derivation_signature(
            {"continuous_surface": SURFACE})
        self.assertEqual([entry["name"] for entry in args],
                         ["x", "y", "curvature"])
        self.assertEqual(symbols, ["x0", "x1", "y0", "y1", "c"])

    def test_a_mechanism_without_a_loss_still_has_an_instance(self):
        # A mechanism that changes an assignment has no loss-shaped surface;
        # the instance must not fall back to the fixed loss instance.
        args, _ = step5_derivation.derivation_signature(
            {"continuous_surface": SURFACE})
        self.assertNotIn("theta_1_1", [entry["name"] for entry in args])

    def test_the_prompt_carries_the_frozen_surface(self):
        prompt = step5_derivation._derivation_prompt(
            {"title": "t", "specification": "s", "spec_equations": "e"},
            {"continuous_surface": SURFACE})
        self.assertIn("THE FROZEN SURFACE", prompt)
        self.assertIn("xy = -(x0*y0 + x1*y1)", prompt)
        self.assertNotIn("theta_1_1", prompt)

    def test_the_prompt_states_the_runtime_contract(self):
        prompt = step5_derivation._derivation_prompt(
            {"title": "t", "specification": "s", "spec_equations": "e"},
            {"continuous_surface": SURFACE})
        # A scalar arrives as a Python float, a vector as a float64 tensor.
        self.assertIn("PYTHON FLOAT", prompt)
        self.assertIn("float64", prompt)


class TheInstanceIsChecked(unittest.TestCase):
    def test_the_right_answer_passes(self):
        self.assertEqual(
            step5_derivation._instance_problems(payload(), {"continuous_surface": SURFACE}), [])

    def test_a_vector_declared_as_a_scalar_is_refused(self):
        wrong = payload()
        wrong["args"] = [dict(SURFACE["args"][0], scalar=1.0),
                         dict(SURFACE["args"][1]),
                         dict(SURFACE["args"][2])]
        problems = step5_derivation._instance_problems(wrong, {"continuous_surface": SURFACE})
        self.assertTrue(problems)
        self.assertIn("widths", " ".join(problems))

    def test_the_scalar_declared_as_a_vector_is_refused(self):
        wrong = payload()
        wrong["args"] = [dict(SURFACE["args"][0]),
                         dict(SURFACE["args"][1]),
                         {"name": "curvature", "width": 1}]
        problems = step5_derivation._instance_problems(wrong, {"continuous_surface": SURFACE})
        self.assertTrue(problems)

    def test_the_code_must_accept_the_declared_arguments(self):
        self.assertTrue(step5_derivation._code_accepts(
            payload(), step5_derivation.derivation_signature(
                {"continuous_surface": SURFACE})[0]))


class TheAnswersKeepTheirTask(unittest.TestCase):
    def test_a_missing_agent_does_not_shift_the_others(self):
        from tools.research.rounds import agent_calls

        class Boom(Exception):
            failure_class = "unknown"

        def worker(name):
            if name == "derivation_1":
                raise Boom("the first agent is gone")
            return {"statements": ["a = 1"]}

        with self.assertRaises(agent_calls.AgentPhaseIncomplete) as caught:
            agent_calls._run_agents_concurrently(
                [("derivation_1",), ("derivation_2",), ("derivation_3",)], worker)
        delivered = caught.exception.delivered
        self.assertEqual([entry["task"] for entry in delivered],
                         ["derivation_2", "derivation_3"])
        self.assertEqual([entry["result"]["statements"] for entry in delivered],
                         [["a = 1"], ["a = 1"]])


class ThePreviousAnswerLeavesTheRound(unittest.TestCase):
    def test_the_workspace_is_archived_outside_the_round(self):
        directory = Path(tempfile.mkdtemp(prefix="round_"))
        try:
            workspace = directory / "agents" / "derivation_1"
            workspace.mkdir(parents=True)
            (workspace / "derivation.json").write_text("{}", encoding="utf-8")
            (workspace / "progress.jsonl").write_text("", encoding="utf-8")
            archived = step5_derivation.archive_previous_workspace(
                directory, 1)
            self.assertFalse(workspace.exists())
            self.assertTrue((archived / "derivation.json").is_file())
            # The archive is outside the round: no agent may read it.
            self.assertFalse(str(archived).startswith(str(directory)))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


@unittest.skipUnless(HAVE_TORCH, "the differential needs the project interpreter")
class ALayoutMismatchIsNotAContradiction(unittest.TestCase):
    def test_the_layout_mismatch_is_incomparable(self):
        left = {"name": "left", **payload()}
        right = {"name": "right", **payload()}
        right["args"] = [dict(SURFACE["args"][0], scalar=1.0),
                         dict(SURFACE["args"][1]),
                         dict(SURFACE["args"][2])]
        report = differential.compare({"implementations": [left, right]})
        self.assertEqual(report["status"], "FAIL")
        kinds = {pair.get("kind") for pair in report.get("pairs") or []}
        self.assertEqual(kinds, {"incomparable"})
        for pair in report.get("pairs") or []:
            self.assertNotEqual(pair.get("kind"), "contradiction")

    def test_two_derivations_of_the_same_surface_agree(self):
        left = {"name": "left", **payload()}
        right = {"name": "right", **payload()}
        report = differential.compare({"implementations": [left, right]})
        self.assertEqual(report["status"], "AGREE", report.get("reason"))


class TheImplementationPhaseNamesWhatItCalls(unittest.TestCase):
    def test_the_phase_calls_resolve(self):
        from tools.research.rounds import step6_implementation

        for name in ("phase_formulation", "phase_coding", "phase_derivation"):
            self.assertTrue(callable(getattr(step6_implementation, name)), name)

    def test_the_module_imports_without_a_round(self):
        import importlib

        importlib.reload(importlib.import_module("tools.research.rounds.step6_implementation"))


if __name__ == "__main__":
    unittest.main()
