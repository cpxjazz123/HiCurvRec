"""The joint behaviour x hyperbolic research scope (Issue #121).

The issue is explicit about what may count as a mechanism: training-only user
behaviour and a curved operation must be *computationally coupled*, so that
removing either changes the core computation. A behaviour suffix on an
unchanged Euclidean quantiser, a standalone hyperbolic loss beside an unrelated
behaviour loss, or inherited hyperbolic code that nothing calls are all out of
scope, however novel they sound.

This module checks that contract against the *content* an agent wrote, not
against a boolean it set: every field has to name the behaviour signal, the
curved operator, the place in training where they meet, and the equation that
connects them. The reasons it returns are what the agents are told when an idea
is sent back, so they have to name the missing part.
"""
from __future__ import annotations

import re

#: Every idea carries these fields, on top of the generic ones.
REQUIRED_SCOPE_FIELDS = (
    # behaviour
    "behavior_signal", "behavior_computation", "leakage_boundary",
    # geometry
    "hyperbolic_operator", "hyperbolic_location",
    # the coupling itself
    "coupling_path", "coupling_equation",
    # what should change, and how it would be seen
    "observables", "ablation_behavior_off", "ablation_geometry_off",
    "ablation_both_on",
    # the minimal implementable specification
    "spec_equations", "spec_inputs_outputs", "safeguards",
)

#: Words that name a curved operation. The list is about the concept, not about
#: one library: an idea may name any of them as long as it also says where the
#: operation acts.
CURVED_TERMS = (
    "poincaré", "poincare", "lorentz", "hyperbol", "negative curvature",
    "curvature", "geodesic", "exp map", "exp_map", "log map", "log_map",
    "möbius", "mobius", "curved", "constant-curvature", "curved space",
)

#: Words that name a training-only behaviour signal.
BEHAVIOUR_TERMS = (
    "train-only", "train only", "training", "user", "behavio", "co-consumption",
    "co-occurrence", "transition", "sequence", "interaction", "session",
    "popularity", "neighbour", "neighbor", "graph", "branch", "click",
)

#: Words that show the two sides are used *together* rather than side by side.
COUPLING_TERMS = (
    "×", "*", "weight", "weighted", "modulat", "gate", "gated", "condition",
    "joint", "coupl", "combin", "interact", "product of", "reweight",
    "geometry-aware", "behaviour-aware", "behavior-aware", "transport",
)

#: Words that place the operation inside the training or quantization path.
LOCATION_TERMS = (
    "quantiz", "residual", "assign", "codebook", "encoder", "decoder", "loss",
    "gradient", "forward", "step", "batch", "update", "objective",
)

#: A field shorter than this is a name, not an explanation.
MIN_EXPLANATION = 24


class ScopeError(ValueError):
    """A proposal does not meet the joint scope contract."""


def _text(idea: dict, field: str) -> str:
    value = idea.get(field, "")
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _has_any(text: str, terms) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in terms)


def _both_sides(text: str) -> bool:
    """Whether one piece of text names a behaviour signal and a curved operator."""
    return _has_any(text, BEHAVIOUR_TERMS) and _has_any(text, CURVED_TERMS)


def _clause_holds_both(text: str) -> bool:
    """Whether one clause of an equation uses the behaviour signal and the curve.

    Two unrelated losses, ``L_rec + l_b L_behaviour + l_h L_hyperbolic``, name
    both families but no single term carries both: that is a sum of two ideas,
    not a coupling. A coupled term, ``w_behaviour(x) * d_poincare(v, c)``, does.
    """
    clauses = re.split(r"[+,;]|\band\b", str(text))
    return any(_both_sides(clause) for clause in clauses)


def check_scope(idea: dict) -> dict:
    """Check one proposal against the joint scope contract.

    Returns ``{"in_scope": bool, "reasons": [...], "checks": {...}}``. The
    reasons name what is missing, in the words the agent used, so a rejected
    idea can be sent back with something actionable.
    """
    reasons: list[str] = []
    checks: dict[str, bool] = {}

    for field in REQUIRED_SCOPE_FIELDS:
        text = _text(idea, field).strip()
        if not text:
            reasons.append(field + " is missing")
            checks[field] = False
        else:
            checks[field] = True

    # --- behaviour -------------------------------------------------------
    signal = _text(idea, "behavior_signal")
    computation = _text(idea, "behavior_computation")
    boundary = _text(idea, "leakage_boundary").lower()
    if signal and len(signal) < MIN_EXPLANATION:
        reasons.append("behavior_signal does not name a signal, it names a label")
    if computation and not _has_any(computation, ("train", "interaction", "sequence",
                                                  "transition", "session", "graph")):
        reasons.append("behavior_computation does not say how the signal is computed")
    if boundary and not _has_any(boundary, ("train-only", "train only", "train split",
                                            "training split", "no validation",
                                            "no test", "without validation",
                                            "without test", "only the train")):
        reasons.append("leakage_boundary does not keep the signal on the training split")

    # --- geometry --------------------------------------------------------
    operator = _text(idea, "hyperbolic_operator")
    location = _text(idea, "hyperbolic_location")
    if operator and not _has_any(operator, CURVED_TERMS):
        reasons.append("hyperbolic_operator names no curved operation")
    if location and not _has_any(location, LOCATION_TERMS):
        reasons.append("hyperbolic_location does not say where the operation acts")

    # --- the coupling ----------------------------------------------------
    path = _text(idea, "coupling_path")
    equation = _text(idea, "coupling_equation")
    if path:
        if not _has_any(path, BEHAVIOUR_TERMS):
            reasons.append("coupling_path never mentions the behaviour signal")
        if not _has_any(path, CURVED_TERMS):
            reasons.append("coupling_path never mentions the curved operation")
        if not _has_any(path, COUPLING_TERMS):
            reasons.append("coupling_path does not say how the two are combined")
    if equation:
        if not _both_sides(equation):
            reasons.append("coupling_equation does not contain both a behaviour term "
                           "and a curved term")
        elif not _clause_holds_both(equation):
            reasons.append("coupling_equation puts the two side by side instead of "
                           "connecting them")
    if path and equation and not _both_sides(path):
        reasons.append("coupling_path does not connect the behaviour signal with the "
                       "curved operation")

    # --- effect and ablation ---------------------------------------------
    off_behaviour = _text(idea, "ablation_behavior_off").strip()
    off_geometry = _text(idea, "ablation_geometry_off").strip()
    both_on = _text(idea, "ablation_both_on").strip()
    if off_behaviour and off_geometry and off_behaviour.lower() == off_geometry.lower():
        reasons.append("the two ablations are the same experiment")
    if off_behaviour and _has_any(off_behaviour, CURVED_TERMS) and \
            not _has_any(off_behaviour, BEHAVIOUR_TERMS):
        reasons.append("ablation_behavior_off does not switch the behaviour off")
    if off_geometry and _has_any(off_geometry, BEHAVIOUR_TERMS) and \
            not _has_any(off_geometry, CURVED_TERMS):
        reasons.append("ablation_geometry_off does not switch the geometry off")

    return {"in_scope": not reasons, "reasons": reasons, "checks": checks}


def scope_summary(idea: dict) -> str:
    """A short, quotable scope statement for a report or an issue comment."""
    verdict = check_scope(idea)
    if verdict["in_scope"]:
        return ("IN_SCOPE: " + _text(idea, "coupling_path")[:200])
    return "OUT_OF_SCOPE: " + "; ".join(verdict["reasons"][:6])
