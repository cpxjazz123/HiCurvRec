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
#: The frozen specification of the round that produced the balancing eval is a
#: FIXTURE: the research state it came from is not part of the repository, and a
#: test that depends on a scratch directory stops running when it is cleaned.
FROZEN = (REPO / "tools" / "research" / "tests" / "fixtures"
          / "round_20261105_144909_130" / "math_spec.json")


def _frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_the_frozen_specification_passes_every_eval():
    """The round's own authority passes: no eval may fire retroactively.

    The exception is the balancing eval: it was written after this round, and it
    FIRES on this round's specification on purpose - that specification is the
    measured evidence that replacing the Sinkhorn-balanced assignment collapses
    the codebook. It is checked on its own below.
    """
    result = evals.run(_frozen())
    # Two checks were written AFTER this round: the balancing rule and the
    # behaviour-chain shape. They judge the live specifications, not the history.
    historical = {"assignment-keeps-the-balancing", "behaviour-off-arithmetic",
                  "behaviour-chain"}
    failures = [failure for failure in result["failures"]
                if failure["id"] not in historical]
    assert not failures, failures
    assert result["checked"] == len(evals.EVALS) + 1


def test_the_balancing_eval_fires_on_the_specification_that_collapsed():
    """The round that replaced the balanced assignment is the eval's evidence."""
    problems = evals._assignment_keeps_the_balancing(_frozen())
    assert problems and "Sinkhorn" in problems[0]


def test_the_balancing_eval_accepts_a_specification_that_keeps_it():
    spec = _frozen()
    spec["continuous_surface"]["executable_statements"] = [
        line for line in spec["continuous_surface"]["executable_statements"]
        if not line.startswith("k1 = torch.argmin(score_k")]
    spec["continuous_surface"]["executable_statements"].append(
        "k1 = sinkhorn_assignment(score_k, sk_epsilon, sk_iters)")
    assert evals._assignment_keeps_the_balancing(spec) == []


def _without(eval_id: str, spec: dict) -> dict:
    """Copy the specification with one property removed."""
    return copy.deepcopy(spec)


# ---- each eval is proven on the defect it came from -------------------------

def test_distance_clamp_is_caught():
    """The clamp on the artanh argument is what keeps a boundary latent finite."""
    spec = _frozen()
    spec["continuous_surface"]["executable_statements"] = [
        "def dist(x, y, c): return 2.0 * _norm(x - y) if c == 0.0 else "
        "(2.0 / math.sqrt(c)) * torch.atanh(math.sqrt(c) * _norm(mobius(-x, y, c)))"]
    problems = evals._distance_clamp(spec)
    assert problems and "clamp" in problems[0]


def test_lift_zero_dead_branch_is_caught():
    spec = _frozen()
    spec["continuous_surface"]["executable_statements"] = [
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


def test_a_surface_without_the_behaviour_chain_is_caught():
    """The check asks for the chain's SHAPE, not for one mechanism's names."""
    spec = _frozen()
    spec["continuous_surface"]["executable_statements"] = [
        "C_ik = D_k",
        "k1 = torch.argmax(P_ik, dim=-1)",
    ]
    problems = evals._behaviour_chain(spec)
    assert problems
    assert any("per-item behaviour" in problem for problem in problems)


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
    spec["continuous_surface"]["executable_statements"] = [
        line.replace("return 2.0 * _norm(x - y)", "return _norm(x - y)")
        for line in spec["continuous_surface"]["executable_statements"]]
    failure = evals.run_numeric(spec)
    assert failure and any("Euclidean" in problem for problem in failure["problems"])


def test_numeric_eval_catches_a_lift_that_leaves_the_ball():
    spec = _frozen()
    spec["continuous_surface"]["executable_statements"] = [
        line.replace("def lift(v, c): return v if c == 0.0 else v * (",
                     "def lift(v, c): return v if c == 0.0 else 9.0 * (")
        for line in spec["continuous_surface"]["executable_statements"]]
    failure = evals.run_numeric(spec)
    assert failure and any("ball" in problem for problem in failure["problems"])


def test_numeric_eval_reports_a_missing_interpreter(monkeypatch):
    spec = _frozen()
    failure = evals.run_numeric(spec, python="/nonexistent/python")
    assert failure and "could not run" in failure["problems"][0]


# ---- the ledger: a caught defect is decided, never merely remembered --------

def test_every_ledger_entry_names_a_real_eval():
    """A ledger entry that points at a missing eval is a promise, not a gate."""
    known = {item.id for item in evals.EVALS}
    for entry in evals.ledger():
        if entry.get("decision") == "eval":
            assert entry.get("eval_id") in known, entry
        else:
            assert entry.get("decision") == "not-decidable", entry
            assert entry.get("note"), entry


def test_the_round_s_own_findings_are_all_decided():
    """The defects this round caught map to evals, so the next round inherits them."""
    findings = [
        "computations['lift']: the stated 'zero-norm branch' does not exist in the code",
        "the definition drops the idea's clamp artanh(min(..., 1 - 1e-6))",
        "continuous_surface has sqrt(c) in the denominators and no branch for c = 0",
        "the BEHAVIOUR_OFF justification is mathematically false: b_k = log 2, not 0",
        "b_raw_k is computed on a SINGLE minibatch, i.e. a per-batch mean",
        "w for epoch 1 needs mean_scale, which does not exist yet",
        "the specification must say which surface the soft assignment uses",
        "s is only 'the RMS of the TRAINING-SPLIT latents', which is ambiguous",
        "reconstruction must use a_{k1} UNLIFTED",
        "the claim that levels 2 and 3 see exactly the baseline's inputs is wrong",
        "parameters list 'the frozen demand b_raw (K1,)' while it is updated per epoch",
        "rho is called the radius-to-scale ratio, but R_c = s/rho",
    ]
    assert evals.undecided(findings) == []


def test_an_unknown_finding_blocks_the_freeze():
    """A defect with no eval and no recorded reason is not silently forgotten."""
    findings = ["the proposal silently changes the research mechanism"]
    assert evals.undecided(findings) == findings


def test_the_gate_refuses_when_a_finding_is_undecided(monkeypatch):
    """The freeze is refused while a caught defect has no decision."""
    from tools.research.rounds import worker

    undone = ["a defect nobody decided"]

    def fake_undecided(findings, entries=None):
        return undone

    monkeypatch.setattr(evals, "undecided", fake_undecided)
    assert worker.evals_module.undecided(undone) == undone


# ---- the surface evals: the listing must be a program ------------------------

def _draft() -> dict:
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    return json.loads((repo / "tools" / "research" / "tests" / "fixtures"
                       / "round_20261105_144909_130" / "math_spec_draft.json")
                      .read_text(encoding="utf-8"))


def test_the_surface_evals_accept_the_frozen_surface():
    for item in evals.EVALS:
        if item.id.startswith("surface") or item.id.startswith("symbolic"):
            assert not item.check(_draft()), (item.id, item.check(_draft()))


def test_a_name_the_listing_never_assigns_is_refused():
    spec = _draft()
    spec["continuous_surface"]["executable_statements"] = [
        line for line in spec["continuous_surface"]["executable_statements"]
        if not line.startswith("off_k = ")]
    spec["continuous_surface"]["executable_statements"].append(
        "score_k = D_k - off[None, :]")
    problems = evals._surface_defines_what_it_uses(spec)
    assert problems and "off" in problems[0]


def test_a_surface_without_the_balanced_assignment_is_refused():
    """The cost may change; the assignment rule may not (CLAUDE.md 19h)."""
    spec = _draft()
    spec["continuous_surface"]["executable_statements"] = [
        line for line in spec["continuous_surface"]["executable_statements"]
        if "sinkhorn" not in line.lower()]
    problems = evals._behaviour_chain(spec)
    assert problems and any("balanced assignment" in problem for problem in problems)


def test_a_surface_that_never_lifts_is_refused():
    spec = _draft()
    spec["continuous_surface"]["executable_statements"] = [
        line for line in spec["continuous_surface"]["executable_statements"]
        if "lift(" not in line]
    problems = evals._surface_applies_the_lift(spec)
    assert problems


def test_a_surface_without_a_cost_is_refused():
    spec = _draft()
    spec["continuous_surface"]["executable_statements"] = [
        line for line in spec["continuous_surface"]["executable_statements"]
        if not line.strip().startswith("C_ik = ")]
    problems = evals._behaviour_chain(spec)
    assert problems and any("cost matrix" in problem for problem in problems)


def test_a_symbolic_surface_without_the_guards_is_refused():
    spec = _draft()
    spec["continuous_surface"]["statements"] = [
        line.replace(" + 1e-12", "") if line.strip().startswith("den") else
        line.replace("atanh(Min(sqrt(c)*sqrt(mn), 1 - 1e-6))", "atanh(sqrt(c)*sqrt(mn))")
        for line in spec["continuous_surface"]["statements"]]
    problems = evals._symbolic_distance_keeps_the_guards(spec)
    assert len(problems) == 2


def test_a_surface_that_names_another_callable_is_refused():
    spec = _draft()
    spec["continuous_surface"]["callable"] = "level_score(residual, codebook, curvature, demand, weight)"
    problems = evals._surface_interface_is_consistent(spec)
    assert problems


def test_a_surface_that_ignores_a_declared_argument_is_refused():
    spec = _draft()
    spec["continuous_surface"]["args"] = [{"name": "x", "width": 2},
                                          {"name": "y", "width": 2},
                                          {"name": "curvature", "scalar": 0.25},
                                          {"name": "demand", "width": 1}]
    problems = evals._surface_uses_its_arguments(spec)
    assert problems and "demand" in problems[0]
