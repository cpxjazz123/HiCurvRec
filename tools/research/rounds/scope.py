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
    "add", "added", "adds", "into", "turn", "turns", "enter", "enters",
    "same", "mix", "blend", "apply", "applied", "drive", "drives", "inform",
    "shape", "shapes", "affect", "affects", "supply", "supplies", "scale",
)

#: Words that place the operation inside the training or quantization path.
LOCATION_TERMS = (
    "quantiz", "residual", "assign", "codebook", "encoder", "decoder", "loss",
    "gradient", "forward", "step", "batch", "update", "objective",
)

#: Words that show a signal is kept on the training split.
EXCLUSION_TERMS = ("only", "no ", "never", "without", "exclusiv", "freeze",
                   "frozen", "before", "not ")

#: A field shorter than this is a name, not an explanation.
MIN_EXPLANATION = 24

#: A symbol like ``b_i``, ``d_c`` or ``exp_0^c`` names a quantity in the
#: equations without naming the concept in words.
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,11}$")

#: Words that are never a quantity on their own.
_NOT_SYMBOLS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "then", "than", "when",
    "where", "which", "while", "into", "each", "same", "train", "training", "only",
    "using", "used", "use", "via", "one", "two", "all", "any", "not", "but",
})


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


def _clause_holds_both(text: str, behaviour_terms, curved_terms,
                       behaviour_symbols, curved_symbols) -> bool:
    """Whether one clause of an equation uses the behaviour signal and the curve.

    Two unrelated losses, ``L_rec + l_b L_behaviour + l_h L_hyperbolic``, name
    both families but no single term carries both: that is a sum of two ideas,
    not a coupling. A coupled term, ``b_i * d_c(v, c)``, does.
    """
    clauses = re.split(r"[+,;]", str(text))
    return any(_mentions(clause, behaviour_terms, behaviour_symbols)
               and _mentions(clause, curved_terms, curved_symbols)
               for clause in clauses)


def _symbols(text: str) -> set:
    """The quantity names a text introduces, so an equation can be read with them."""
    found = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{0,11}", str(text)):
        lowered = token.lower()
        if lowered in _NOT_SYMBOLS or len(lowered) < 2:
            continue
        if _SYMBOL.match(token) and (("_" in token) or any(ch.isdigit() for ch in token)
                                     or (len(token) == 1 and token.isalpha())):
            found.add(token)
    return found


def _mentions(text: str, terms, symbols) -> bool:
    """Whether a text names a family of concepts, in words or in symbols."""
    if _has_any(text, terms):
        return True
    lowered = str(text).lower()
    return any(symbol.lower() in lowered for symbol in symbols)


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

    behaviour_terms = set(BEHAVIOUR_TERMS)
    curved_terms = set(CURVED_TERMS)
    # The quantity names the idea itself introduces, so an equation may use
    # "b_i * d_c(...)" instead of spelling the concepts out.
    behaviour_symbols = (_symbols(_text(idea, "behavior_signal"))
                         | _symbols(_text(idea, "behavior_computation"))
                         | _symbols(_text(idea, "coupling_path"))
                         | _symbols(_text(idea, "specification"))
                         | _symbols(_text(idea, "spec_equations")))
    curved_symbols = (_symbols(_text(idea, "hyperbolic_operator"))
                      | _symbols(_text(idea, "coupling_path"))
                      | _symbols(_text(idea, "specification"))
                      | _symbols(_text(idea, "spec_equations")))

    # --- behaviour -------------------------------------------------------
    signal = _text(idea, "behavior_signal")
    computation = _text(idea, "behavior_computation")
    boundary = _text(idea, "leakage_boundary").lower()
    if signal and len(signal) < MIN_EXPLANATION:
        reasons.append("behavior_signal does not name a signal, it names a label")
    if computation and not _has_any(computation, ("train", "interaction", "sequence",
                                                  "transition", "session", "graph")):
        reasons.append("behavior_computation does not say how the signal is computed")
    if boundary and not (_has_any(boundary, ("train", "training"))
                         and _has_any(boundary, EXCLUSION_TERMS)):
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
        if not _mentions(path, behaviour_terms, behaviour_symbols):
            reasons.append("coupling_path never mentions the behaviour signal")
        if not _mentions(path, curved_terms, curved_symbols):
            reasons.append("coupling_path never mentions the curved operation")
        if not _has_any(path, COUPLING_TERMS):
            reasons.append("coupling_path does not say how the two are combined")
    if equation:
        has_behaviour = _mentions(equation, behaviour_terms, behaviour_symbols)
        has_curved = _mentions(equation, curved_terms, curved_symbols)
        if not (has_behaviour and has_curved):
            reasons.append("coupling_equation does not contain both a behaviour term "
                           "and a curved term")
        elif not _clause_holds_both(equation, behaviour_terms, curved_terms,
                                    behaviour_symbols, curved_symbols):
            reasons.append("coupling_equation puts the two side by side instead of "
                           "connecting them")

    # --- effect and ablation ---------------------------------------------
    off_behaviour = _text(idea, "ablation_behavior_off").strip()
    off_geometry = _text(idea, "ablation_geometry_off").strip()
    if off_behaviour and off_geometry and off_behaviour.lower() == off_geometry.lower():
        reasons.append("the two ablations are the same experiment")
    if off_geometry and not _has_any(off_geometry, ("euclidean", "flat", "straight",
                                                     "curvature 0", "zero curvature",
                                                     "euclid")):
        reasons.append("ablation_geometry_off does not say what replaces the curve")
    if off_behaviour and not _has_any(off_behaviour, ("0", "off", "one", "none", "without",
                                                      "disable", "remove", "zero", "not used",
                                                      "no ", "ignore", "drop", "uniform")):
        reasons.append("ablation_behavior_off does not switch the behaviour off")

    return {"in_scope": not reasons, "reasons": reasons, "checks": checks}


def scope_summary(idea: dict) -> str:
    """A short, quotable scope statement for a report or an issue comment."""
    verdict = check_scope(idea)
    if verdict["in_scope"]:
        return ("IN_SCOPE: " + _text(idea, "coupling_path")[:200])
    return "OUT_OF_SCOPE: " + "; ".join(verdict["reasons"][:6])
