"""The frozen mathematical specification of a mechanism.

The Formula Agent writes the specification, the Math Reviewers check it
against the idea it came from, and the freeze writes the file and its hash.
After the freeze the specification is the ONLY authority the implementation may
follow: the reference contract takes its statements from it, and the
implementation may not reinterpret it.

Three responsibilities are kept apart on purpose:

* the IDEA agent decides WHAT is researched;
* the FORMULA agent decides HOW the idea is written as mathematics - and it
  may not change the idea, only complete it; if the idea lacks a definition the
  mechanism needs, it answers ``SPECIFICATION_INCOMPLETE`` and says which
  definitions are missing;
* the CODING agent implements the frozen specification faithfully.

The specification must cover every computation the mechanism touches, not only
a loss: the metric, the behaviour's entry point, the assignment, the gradient
path, the reconstruction, the switches and the surfaces that are deliberately
out of scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPEC_NAME = "math_spec.json"

#: The sections a specification must have. A missing one is a specification
#: that would leave an implementation free to interpret.
REQUIRED_SECTIONS = ("mechanism", "computations", "inputs", "outputs",
                     "parameters", "dimensions", "gradient_rules",
                     "training_integration", "conventions", "switches", "excluded")

#: The conventions of the framework's own geometry. A specification that does
#: not state them states a formula in an unknown parameterisation, and the
#: parameterisation is where the factors live: the ball radius 1/sqrt(c) against
#: a ball of radius R, the origin factor 2 of the exponential map, the
#: normalisation of the distance. The framework owns the geometry, so it states
#: the conventions itself and the reviewers check the specification against
#: them.
CANONICAL_CONVENTIONS = {
    "curvature": "the ball is of curvature kappa = -c with c > 0; the ball "
                 "radius is R_c = 1/sqrt(c) - a radius R with kappa = -1/R^2 "
                 "is the same family under c = 1/R^2, and every distance, "
                 "exponential map and radius must use ONE of them",
    "distance": "d_c(x, y) = (2/sqrt(c)) artanh(sqrt(c) ||(-x) (+) y||), with "
                "(+) the canonical Mobius sum; at c = 0 the limit is "
                "2||x - y||, and the factor 2 is the metric's normalisation",
    "exponential_map": "exp_0^c(v) = tanh(sqrt(c)||v||) v / (sqrt(c)||v||), and "
                        "exp_x^c(v) = x (+) [tanh(sqrt(c) lambda_x^c ||v|| / 2) "
                        "v / (sqrt(c)||v||)] with lambda_x^c = 2/(1 - c||x||^2)",
    "logarithm": "log_0^c(y) = artanh(sqrt(c)||y||) y / (sqrt(c)||y||)",
    "tangent_norm": "a tangent vector's Euclidean coordinate norm is not its "
                    "Riemannian norm: ||v||_{0,c} = lambda_0^c ||v|| = 2||v||, so "
                    "d_c(0, exp_0^c(v)) = 2||v|| and log_0^c(exp_0^c(v)) = v",
    "fréchet_mean": "the Frechet mean of a weighted set is the point minimising "
                     "the weighted sum of squared distances; a coordinate "
                     "average may not be called a Frechet mean",
    "straight_through": "the assignment is an argmin and has no gradient; the "
                        "reconstruction gradient reaches the encoder through the "
                        "straight-through estimator x_q = x + (x_q - x).detach()",
    "source": "tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md, which "
              "the implementation must cite and the reviewers must check",
}

#: The answer of a Formula Agent that finds the idea incomplete.
INCOMPLETE = "SPECIFICATION_INCOMPLETE"


def validate(spec: dict) -> list:
    """The problems of a specification; an empty list means it is complete."""
    problems = []
    if not isinstance(spec, dict):
        return ["the specification is not an object"]
    for section in REQUIRED_SECTIONS:
        if section not in spec:
            problems.append("the section %r is missing" % section)
        elif not spec[section]:
            problems.append("the section %r is empty" % section)
    computations = spec.get("computations")
    if isinstance(computations, list):
        if not computations:
            problems.append("no computation is stated")
        for entry in computations:
            if not isinstance(entry, dict):
                problems.append("a computation is not an object")
                continue
            for key in ("name", "definition", "inputs", "output"):
                if not entry.get(key):
                    problems.append("the computation %r has no %r"
                                    % (entry.get("name", "?"), key))
    dimensions = spec.get("dimensions")
    if isinstance(dimensions, dict):
        for name, size in dimensions.items():
            if not str(name):
                problems.append("a dimension has no name")
            if size in (None, "", 0):
                problems.append("the dimension %r has no size" % name)
    else:
        problems.append("the dimensions are not stated as a mapping")
    rules = spec.get("gradient_rules")
    if isinstance(rules, list):
        if not rules:
            problems.append("no gradient rule is stated")
        for rule in rules:
            if not isinstance(rule, dict) or not rule.get("operator")                     or not rule.get("rule"):
                problems.append("a gradient rule does not name its operator and its rule")
    else:
        problems.append("the gradient rules are not stated as a list")
    conventions = spec.get("conventions")
    if not conventions:
        spec["conventions"] = dict(CANONICAL_CONVENTIONS)
        spec["conventions_derived_by"] = "the framework: the canonical geometry contract"
    integration = spec.get("training_integration")
    if isinstance(integration, dict):
        for key in ("loss", "reconstruction", "optimizer"):
            if not integration.get(key):
                problems.append("the training integration does not state %r" % key)
    else:
        problems.append("the training integration is not stated")
    surface = spec.get("continuous_surface")
    if isinstance(surface, dict):
        statements = surface.get("statements")
        if not statements:
            problems.append("the continuous surface states no statements")
        for statement in statements or []:
            if "=" not in str(statement):
                problems.append("the statement %r is not an assignment" % statement)
        if not surface.get("symbols"):
            problems.append("the continuous surface states no symbols")
        if not surface.get("args"):
            problems.append("the continuous surface states no arguments")
        if not surface.get("callable"):
            problems.append("the continuous surface names no callable")
    else:
        problems.append("the continuous surface is not stated")
    return problems


def _canonical(spec: dict) -> str:
    """The one serialisation the hash is taken over: written and read alike."""
    return json.dumps(spec, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def complete(spec: dict) -> dict:
    """Fill the three mechanical sections the model tends to leave out.

    A model that writes a rich formula and omits "dimensions",
    "gradient_rules" and "training_integration" is a common failure of the
    formulation role. The framework can state the mechanical parts - the widths
    come from the surface's own arguments, the gradient rules and the training
    integration from the mechanism family's conventions - but it must SAY that
    it did: every derived section carries ``derived_by``, and the two Math
    Reviewers judge the result as a whole, so a derived section that misstates
    the idea is sent back like any other.
    """
    spec = dict(spec or {})
    surface = spec.get("continuous_surface") or {}

    dimensions = spec.get("dimensions")
    if not dimensions:
        derived = {}
        for argument in surface.get("args") or []:
            name = str(argument.get("name") or "")
            if not name:
                continue
            width = argument.get("width")
            derived[name] = int(width) if width else "scalar"
        derived["symbols"] = len(surface.get("symbols") or [])
        spec["dimensions"] = derived
        spec["dimensions_derived_by"] = "the framework: the widths come from the surface's arguments"

    rules = spec.get("gradient_rules")
    if not rules:
        spec["gradient_rules"] = [
            {"operator": "the continuous surface",
             "rule": "the gradient follows autograd through the surface's statements; "
                      "the framework's reference check verifies it against a rebuild "
                      "and a gradcheck on both sides"},
            {"operator": "the argmin of the assignment",
             "rule": "no gradient: it decides the token, and the straight-through "
                      "estimator carries the reconstruction gradient to the encoder"},
        ]
        spec["gradient_rules_derived_by"] = "the framework: the conventions of the family"

    conventions = spec.get("conventions")
    if not conventions:
        spec["conventions"] = dict(CANONICAL_CONVENTIONS)
        spec["conventions_derived_by"] = "the framework: the canonical geometry contract"
    integration = spec.get("training_integration")
    if not integration:
        spec["training_integration"] = {
            "loss": "the baseline's objective plus the mechanism's contribution",
            "reconstruction": "the codeword the assignment chose",
            "optimizer": "the baseline's optimizer and schedule",
        }
        spec["training_integration_derived_by"] = "the framework: the conventions of the family"
    return spec


def freeze(directory: Path, spec: dict) -> dict:
    """Write the specification and its hash; the hash is what the code cites."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = _canonical(spec)
    path = directory / SPEC_NAME
    path.write_text(payload, encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def load(directory: Path) -> dict:
    """The frozen specification of a round, with the hash of the bytes on disk."""
    path = Path(directory) / SPEC_NAME
    if not path.is_file():
        return {}
    payload = path.read_text(encoding="utf-8")
    return {"spec": json.loads(payload),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "path": str(path)}


def matches_frozen(directory: Path, recorded_sha256: str) -> dict:
    """Whether the frozen file on disk is the one the record cites."""
    frozen = load(directory)
    if not frozen:
        return {"matches": False, "reason": "no frozen specification on disk"}
    if frozen["sha256"] != recorded_sha256:
        return {"matches": False,
                "reason": "the frozen file changed after the freeze (%s -> %s)"
                          % (str(recorded_sha256)[:12], frozen["sha256"][:12])}
    return {"matches": True, "sha256": frozen["sha256"], "spec": frozen["spec"]}


def statements(spec: dict) -> list:
    """The SymPy statements of the continuous surface, in the frozen order."""
    surface = (spec or {}).get("continuous_surface") or {}
    return list(surface.get("statements") or [])


def output_name(spec: dict) -> str:
    """The name of the surface's output expression."""
    surface = (spec or {}).get("continuous_surface") or {}
    return str(surface.get("output") or "")
