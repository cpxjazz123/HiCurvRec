"""The eval gate: a defect a round caught can never be repeated silently.

Every eval in ``tools/research/rounds/evals.py`` is proven here twice:

* ``test_seeded_defect_is_caught`` - it fails on the defect it was seeded from,
  so the check is not decorative;
* ``test_frozen_specification_passes`` - it leaves the frozen specification of
  the round that produced it alone, so the gate can never retroactively
  invalidate the authority a round is already training against.

The gate itself is proven at the freeze: a specification that trips a check is
not frozen, whatever the reviewers say.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.research.rounds import evals

REPO = Path(__file__).resolve().parents[3]
FROZEN = (REPO / "results" / "research" / "issue121" / "rounds"
          / "round_20261105_144909_130" / "math_spec.json")


def _frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_the_frozen_specification_passes_every_eval():
    """The round's own authority passes: no eval may fire retroactively."""
    result = evals.run(_frozen())
    assert result["passed"], result["failures"]
    assert result["checked"] == len(evals.EVALS) + 1


def _without(eval_id: str, spec: dict) -> dict:
    """Copy the specification with one property removed."""
    return copy.deepcopy(spec)


# ---- each eval is proven on the defect it came from -------------------------

def test_distance_clamp_is_caught():
    spec = _without("distance-clamp", _frozen())
    spec["continuous_surface"]["statements"] = [
        line.replace(", max=1.0 - 1e-6", "").replace(
            "torch.clamp(", "").replace(")", "") if "atanh" in line else line
        for line in spec["continuous_surface"]["statements"]]
    # a dropped clamp on the atanh argument
    spec["continuous_surface"]["statements"] = [
        "def dist(x, y, c): return 2.0 * _norm(x - y) if c == 0.0 else "
        "(2.0 / math.sqrt(c)) * torch.atanh(math.sqrt(c) * _norm(mobius(-x, y, c)))"]
    problems = evals._distance_clamp(spec)
    assert problems and "clamp" in problems[0]


def test_lift_zero_dead_branch_is_caught():
    spec = _frozen()
    spec["continuous_surface"]["statements"] = [
        "def _norm(v): return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0) + eps)",
        "def lift(v, c): return v if c == 0.0 else v * torch.where("
        "_norm(v) <= eps, torch.zeros_like(_norm(v)), "
        "torch.tanh(math.sqrt(c) * _norm(v)) / (math.sqrt(c) * _norm(v) + eps)).unsqueeze(-1)"]
    problems = evals._lift_zero(spec)
    assert problems and "dead" in problems[0]


def test_curvature_zero_branch_is_caught():
    spec = _frozen()
    spec["switches"] = ["the geometry switch: with it off the lift is the identity"]
    problems = evals._curvature_zero_branch(spec)
    assert len(problems) == 2


def test_behaviour_off_false_arithmetic_is_caught():
    spec = _frozen()
    spec["switches"] = ["BEHAVIOUR_OFF: the demand is zero, so b_k = 0 and every "
                        "offset is zero, and the assignment is the plain one"]
    problems = evals._behaviour_off_arithmetic(spec)
    assert any("b_k = 0" in problem for problem in problems)
    assert any("k-independent" in problem for problem in problems)


def test_per_batch_demand_is_caught():
    spec = _frozen()
    for entry in spec["computations"]:
        if entry.get("name") == "behaviour demand":
            entry["definition"] = ("p_ik = softmax_k(-D_ik / tau); "
                                   "b_raw_k = sum_i p_ik * m_i / (sum_i p_ik + eps)")
    spec["continuous_surface"]["statements"] = [
        "b_raw_k = (p_ik * m_i[:, None]).sum(0) / (p_ik.sum(0) + eps)"]
    problems = evals._epoch_aggregation(spec)
    assert problems


def test_undefined_epoch_one_weight_is_caught():
    spec = _frozen()
    for entry in spec["computations"]:
        if entry.get("name") == "offset weight":
            entry["definition"] = ("mean_scale = the running mean of min_k D_k; "
                                   "w = rho_off * mean_scale / max_k log(1 + b_k)")
    problems = evals._epoch_one_weight(spec)
    assert problems and "epoch 1" in problems[0]


def test_unstated_demand_surface_is_caught():
    spec = _frozen()
    for entry in spec["computations"]:
        if entry.get("name") == "behaviour demand":
            entry["definition"] = "p_ik = softmax_k(-D_ik / tau)"
    problems = evals._demand_surface(spec)
    assert problems


def test_ambiguous_scale_is_caught():
    spec = _frozen()
    for entry in spec["computations"]:
        if entry.get("name") == "scale":
            entry["definition"] = "s is the RMS of the latents"
    problems = evals._scale_definition(spec)
    assert len(problems) == 2


def test_lifted_reconstruction_is_caught():
    spec = _frozen()
    spec["training_integration"]["reconstruction"] = (
        "the chosen codeword, lifted into the ball")
    problems = evals._reconstruction_unlifted(spec)
    assert problems


# ---- the numeric eval executes the specification's own geometry -------------

def test_numeric_eval_catches_a_wrong_limit():
    """A distance that forgets the factor 2 is not the mechanism's metric."""
    spec = _frozen()
    spec["continuous_surface"]["statements"] = [
        line.replace("return 2.0 * _norm(x - y)", "return _norm(x - y)")
        for line in spec["continuous_surface"]["statements"]]
    failure = evals.run_numeric(spec)
    assert failure and any("Euclidean" in problem for problem in failure["problems"])


def test_numeric_eval_catches_a_lift_that_leaves_the_ball():
    spec = _frozen()
    spec["continuous_surface"]["statements"] = [
        line.replace("def lift(v, c): return v if c == 0.0 else v * (",
                     "def lift(v, c): return v if c == 0.0 else 9.0 * (")
        for line in spec["continuous_surface"]["statements"]]
    failure = evals.run_numeric(spec)
    assert failure and any("ball" in problem for problem in failure["problems"])


def test_numeric_eval_reports_a_missing_interpreter(monkeypatch):
    spec = _frozen()
    failure = evals.run_numeric(spec, python="/nonexistent/python")
    assert failure and "could not run" in failure["problems"][0]
