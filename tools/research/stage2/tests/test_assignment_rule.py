"""The baseline's Sinkhorn-balanced assignment is not the mechanism's to change.

Three places enforce it, and each is proved here on the arm that broke it:

* the **scope gate** refuses a proposal that takes the level index by a plain
  argmin over its own score;
* the **specification eval** refuses a frozen surface that does it;
* the **admission check** refuses a candidate tree whose code does it.

The evidence is the arm of round_20261105_144909_130: it replaced the balanced
assignment with ``k1 = argmin_k(score_k)``, the level-1 codebook collapsed (42 of
256 codes used against the reference arm's 241, the largest code holding 31% of
the items), and the owner's learned collision gate refused the SID.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ARM = (REPO / "results" / "research" / "issue121" / "trees"
       / "round_20261105_144909_130" / "pmi-geomargin-l1-rq")


def _idea(**overrides) -> dict:
    idea = {
        "title": "a level-1 offset",
        "specification": "the level-1 score is D_k - off_k",
        "spec_equations": "k1 = argmin_k score_k ; r_next = r - a[k1]",
        "coupling_equation": "k1(i) = argmin_k ( d_c(u_i, q_k) - w log(1 + demand_k) )",
        "coupling_path": "the curved distance and the demand meet inside one argmin",
        "behavior_signal": "the training-only transition demand per codeword",
        "behavior_computation": "the demand is counted over the training split",
        "leakage_boundary": "only the training split is read",
        "hyperbolic_operator": "the Poincare distance at curvature -c",
        "hyperbolic_location": "the level-1 quantisation step",
        "ablation_behavior_off": "the demand is zero, so the offset is zero",
        "ablation_geometry_off": "c = 0, so the distance is the Euclidean one",
        "key_change": "the level-1 index becomes argmin_k (d_c(u, q_k) - w log(1 + demand_k))",
        "novelty": "a popularity-adjusted surplus subtracted from the curved distance",
    }
    idea.update(overrides)
    return idea


def test_the_scope_gate_refuses_a_plain_argmin_assignment():
    from tools.research.rounds import scope

    verdict = scope.check_scope(_idea())
    assert not verdict["in_scope"]
    assert any("balanced assignment" in reason for reason in verdict["reasons"])


def test_the_scope_gate_accepts_a_mechanism_that_keeps_the_balancing():
    from tools.research.rounds import scope

    idea = _idea(
        spec_equations=("the Sinkhorn cost matrix is C_k = D_k - off_k ; "
                        "k1 = the baseline's balanced assignment of C"),
        coupling_equation=("k1(i) = the baseline's Sinkhorn-balanced assignment "
                           "of ( d_c(u_i, q_k) - w log(1 + demand_k) )"),
        key_change=("the mechanism's score becomes the cost matrix of the "
                    "baseline's Sinkhorn-balanced assignment"))
    verdict = scope.check_scope(idea)
    assert not any("balanced assignment" in reason for reason in verdict["reasons"])


def test_the_specification_eval_refuses_the_arm_that_collapsed():
    from tools.research.rounds import evals

    spec = json.loads((REPO / "results" / "research" / "issue121" / "rounds"
                        / "round_20261105_144909_130" / "math_spec.json")
                       .read_text(encoding="utf-8"))
    problems = evals._assignment_keeps_the_balancing(spec)
    assert problems and "Sinkhorn" in problems[0]


def test_the_admission_check_refuses_the_arm_and_accepts_the_baseline():
    from tools.research.stage2 import checks

    arm = checks.check_assignment_rule(ARM)
    assert arm["status"] == "FAIL"
    assert "argmin" in arm["detail"]
    baseline = checks.check_assignment_rule(REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE")
    assert baseline["status"] == "PASS"


def test_a_tree_that_keeps_the_balancing_passes_the_admission():
    import tempfile

    from tools.research.stage2 import checks

    directory = Path(tempfile.mkdtemp(prefix="assignment_"))
    (directory / "model").mkdir()
    (directory / "model" / "layers.py").write_text(
        "import torch\n"
        "def sinkhorn(cost, epsilon, iterations):\n"
        "    return cost\n"
        "class VQLayer:\n"
        "    def forward(self, x):\n"
        "        cost = self.mechanism.score(x, self.codebook)\n"
        "        return self._indices(cost, infer_use_sk=False)\n",
        encoding="utf-8")
    verdict = checks.check_assignment_rule(directory)
    assert verdict["status"] == "PASS", verdict["detail"]
