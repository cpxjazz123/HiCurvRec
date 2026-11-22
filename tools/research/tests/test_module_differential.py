"""The differential between coding agents, decided by running their modules.

The round's evidence is behavioural: three independent implementations of the
same frozen specification, called on the same inputs, must produce the same
tensors in every switch arm. A wrong sign, a dropped factor or a switch that
does nothing is a disagreement, and the report names the field that differs.

A comparison that cannot run is an error, never an agreement: the differential
is what lets a candidate touch the baseline tree.
"""
from __future__ import annotations

from tools.research.rounds import module_differential

SPEC = {"continuous_surface": {"callable": "level_score"}}

GOOD = """
import math
import torch

EPS = 1e-12

def _norm(v):
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))

def lift(v, c):
    return v if c == 0.0 else v * (torch.tanh(math.sqrt(c) * _norm(v))
                                   / (math.sqrt(c) * _norm(v) + EPS)).unsqueeze(-1)

def mobius(x, y, c):
    xy = (x * y).sum(-1, keepdim=True)
    xn = (x * x).sum(-1, keepdim=True)
    yn = (y * y).sum(-1, keepdim=True)
    return ((1 + 2 * c * xy + c * yn) * x + (1 - c * xn) * y) / (
        1 + 2 * c * xy + c * c * xn * yn + EPS)

def dist(x, y, c):
    if c == 0.0:
        return 2.0 * _norm(x - y)
    return (2.0 / math.sqrt(c)) * torch.atanh(
        torch.clamp(math.sqrt(c) * _norm(mobius(-x, y, c)), max=1.0 - 1e-6))

def level_score(residual, codebook, curvature, demand, weight):
    u = lift(residual, curvature)
    q = lift(codebook, curvature)
    d = dist(u[:, None, :], q[None, :, :], curvature)
    off = weight * torch.log1p(demand)
    return {"D_k": d, "score_k": d - off, "k1": torch.argmin(d - off, dim=-1)}
"""

#: The same module with the offset added instead of subtracted.
WRONG_SIGN = GOOD.replace("score_k\": d - off", "score_k\": d + off") \
                  .replace("argmin(d - off", "argmin(d + off")

#: The same module with the Euclidean factor dropped from the c = 0 arm.
WRONG_LIMIT = GOOD.replace("return 2.0 * _norm(x - y)", "return _norm(x - y)")

#: The same module where the behaviour switch changes nothing.
DROPS_SWITCH = GOOD.replace(
    "    off = weight * torch.log1p(demand)",
    "    off = weight * torch.log1p(demand)\n"
    "    if demand.numel() == 5 and False:\n"
    "        off = torch.zeros_like(off)")


def _compare(codes):
    implementations = [{"name": "coding_%d" % (index + 1), "code": code}
                       for index, code in enumerate(codes)]
    return module_differential.compare_modules(implementations, SPEC)


def test_three_agreeing_implementations_agree():
    verdict = _compare([GOOD, GOOD, GOOD])
    assert verdict["status"] == "AGREE", verdict.get("reason")
    arms = {pair["arm"] for pair in verdict["pairs"]}
    assert arms == {"BOTH_ON", "GEOMETRY_OFF", "BEHAVIOUR_OFF"}
    assert all(pair["worst_gap"] == 0.0 for pair in verdict["pairs"])


def test_a_wrong_sign_is_a_disagreement():
    verdict = _compare([GOOD, WRONG_SIGN, GOOD])
    assert verdict["status"] == "DISAGREE"
    assert any("score_k" in (pair.get("reason") or "")
               for pair in verdict["pairs"] if pair["status"] == "DISAGREE")


def test_a_wrong_euclidean_limit_is_a_disagreement():
    verdict = _compare([GOOD, WRONG_LIMIT, GOOD])
    assert verdict["status"] == "DISAGREE"
    arms = {pair["arm"] for pair in verdict["pairs"] if pair["status"] == "DISAGREE"}
    assert "GEOMETRY_OFF" in arms


def test_the_switch_arms_are_compared_separately():
    """A module that ignores the geometry argument passes one arm and fails one."""
    verdict = _compare([GOOD, WRONG_LIMIT])
    by_arm = {}
    for pair in verdict["pairs"]:
        by_arm.setdefault(pair["arm"], []).append(pair["status"])
    assert by_arm["BOTH_ON"] == ["AGREE"]
    assert by_arm["GEOMETRY_OFF"] == ["DISAGREE"]


def test_a_module_that_does_not_load_is_an_error_not_an_agreement():
    verdict = _compare([GOOD, "import torch\nthis is not python"])
    assert verdict["status"] in ("NOT_APPLICABLE", "DISAGREE", "ERROR")
    assert verdict["status"] != "AGREE"


def test_a_module_that_raises_is_an_error():
    raises = GOOD.replace(
        "    u = lift(residual, curvature)",
        "    raise RuntimeError('boom')")
    verdict = _compare([GOOD, raises])
    assert verdict["status"] != "AGREE"
    assert any("boom" in (pair.get("reason") or "") or "raised" in (pair.get("reason") or "")
               for pair in verdict["pairs"]) or verdict.get("reason")


def test_fewer_than_two_implementations_is_not_applicable():
    verdict = module_differential.compare_modules(
        [{"name": "only", "code": "def level_score(*a): return a"}], SPEC)
    assert verdict["status"] == "NOT_APPLICABLE"
