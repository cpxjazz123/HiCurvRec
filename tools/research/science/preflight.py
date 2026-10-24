"""Deterministic preflight in front of the scientific review (Issue #118 P0-B).

The layers run in dependency order and stop at the first layer that cannot be
trusted, so a failure names the smallest thing that has to be repaired:

1. ``output``    the file exists, parses and carries the required top-level keys.
2. ``contract``  capacity, data boundary, global SID, control shape, falsifier.
3. ``math``      the declared operator is evaluated by the audited reference:
                 its flat limit, its behaviour at zero, its domain margin.
4. ``falsifier`` the decision rule references registered metrics and a NULL rule.

Verdict classes are distinct on purpose. ``MACHINE_INVALID`` and
``MATH_COUNTEREXAMPLE`` are repairable engineering defects; only
``SCIENTIFIC_REVIEW_REQUIRED`` may consume an independent reviewer.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import contract as contract_module
from .contract import (
    BEHAVIOURAL_METRICS,
    DIAGNOSTIC_METRICS,
    FROZEN,
    METRICS,
    THRESHOLDS,
    build_operator,
    canonical_metric,
    compile_contract,
)

PASS = "PASS"
MACHINE_INVALID = "MACHINE_INVALID"
MATH_COUNTEREXAMPLE = "MATH_COUNTEREXAMPLE"
SCIENTIFIC_REVIEW_REQUIRED = "SCIENTIFIC_REVIEW_REQUIRED"

REQUIRED_KEYS = (
    "candidate_id", "behavior_problem_id", "behavior_evidence_sha256",
    "literature_evidence_sha256", "next_item_hypothesis", "math_definition",
    "hyperbolic_necessity", "euclidean_control", "effect_site",
    "real_trigger_measurement", "training_only_inputs", "stable_global_sid",
    "protocol", "falsifier",
)


def diagnostic(layer: str, code: str, location: str, expected, actual,
               repair: str, *, verdict: str = MACHINE_INVALID) -> dict:
    return {"layer": layer, "code": code, "location": location,
            "expected": expected, "actual": actual, "repair": repair,
            "verdict": verdict}


# --------------------------------------------------------------------- layers
def check_output(candidate) -> list:
    problems = []
    if not isinstance(candidate, dict) or not candidate:
        return [diagnostic("output", "EMPTY_OR_NOT_OBJECT", "<root>", "a JSON object",
                           type(candidate).__name__, "write the candidate object")]
    missing = [key for key in REQUIRED_KEYS if not candidate.get(key)]
    if missing:
        problems.append(diagnostic("output", "MISSING_KEYS", ",".join(missing),
                                   list(REQUIRED_KEYS), missing,
                                   "add the missing top-level fields"))
    return problems


def check_contract(candidate) -> list:
    problems = []
    if candidate.get("stable_global_sid") is not True:
        problems.append(diagnostic("contract", "SID_NOT_GLOBAL", "stable_global_sid",
                                   True, candidate.get("stable_global_sid"),
                                   "one global three-token SID per item"))
    protocol = candidate.get("protocol")
    if not isinstance(protocol, dict):
        problems.append(diagnostic("contract", "PROTOCOL_MISSING", "protocol",
                                   "object", type(protocol).__name__, "declare the frozen protocol"))
    else:
        capacity = FROZEN["capacity"]
        observed = (protocol.get("latent_dim"), protocol.get("levels"),
                    protocol.get("codes_per_level"))
        expected = (capacity["latent_dim"], capacity["levels"], capacity["codes_per_level"])
        if observed != expected:
            problems.append(diagnostic("contract", "CAPACITY_MISMATCH", "protocol",
                                       expected, observed, "use the frozen SID capacity"))
        if protocol.get("hyperparameter_sweep"):
            problems.append(diagnostic("contract", "SWEEP_FORBIDDEN", "protocol.hyperparameter_sweep",
                                       False, True, "remove the sweep"))
        if protocol.get("uses_future_events"):
            problems.append(diagnostic("contract", "FUTURE_LEAKAGE", "protocol.uses_future_events",
                                       False, True, "train-only inputs only"))
    math_definition = candidate.get("math_definition")
    if isinstance(math_definition, dict):
        # The assignment and residual path must be declared as named fields, not
        # as prose: prose proposals kept contradicting themselves, and a reviewer
        # then could not recompute Q, the labels or the losses.
        assignment = math_definition.get("assignment_algorithm")
        if not isinstance(assignment, dict) or not assignment.get("cost_and_transport")                 or not assignment.get("discrete_label_extraction"):
            problems.append(diagnostic(
                "contract", "ASSIGNMENT_NOT_DECLARED", "math_definition.assignment_algorithm",
                "cost_and_transport and discrete_label_extraction",
                sorted(assignment) if isinstance(assignment, dict) else None,
                "declare how the cost is built, how the transport is solved and how the hard "
                "labels are extracted"))
        if not math_definition.get("projection_order"):
            problems.append(diagnostic(
                "contract", "PROJECTION_ORDER_MISSING", "math_definition.projection_order",
                "the order of map, project, assign, score and subtract",
                None, "state the order of the per-level steps"))
        if not math_definition.get("tensor_gradient_rules"):
            problems.append(diagnostic(
                "contract", "GRADIENT_RULES_MISSING", "math_definition.tensor_gradient_rules",
                "which tensors are frozen and which are differentiable", None,
                "state the gradient rule for every tensor"))
    control = str(candidate.get("euclidean_control") or "")
    if not control:
        problems.append(diagnostic("contract", "CONTROL_MISSING", "euclidean_control",
                                   "a runnable control arm", None,
                                   "define the control arm"))
    # Whether the control really differs only in the distance is a scientific
    # judgement about the described agents, not a keyword question: the preflight
    # checks that a control exists and leaves comparability to the reviewer.
    return problems


def _poincare_squared_distance(x, y, curvature, delta):
    diff2 = sum((a - b) ** 2 for a, b in zip(x, y))
    if curvature <= 0:
        return 4.0 * diff2
    x2 = sum(a * a for a in x)
    y2 = sum(b * b for b in y)
    denominator = max((1.0 - curvature * x2) * (1.0 - curvature * y2), 1e-12)
    u = 1.0 + 2.0 * curvature * diff2 / denominator
    u = max(u, 1.0 + delta)
    excess = u - 1.0
    arcosh = math.log1p(excess + math.sqrt(excess * (u + 1.0)))
    return (arcosh * arcosh) / curvature


def check_math(candidate) -> list:
    """Evaluate the declared operator with the audited reference, not the prose."""
    operator = build_operator(candidate.get("math_definition"))
    if operator.get("error"):
        return [diagnostic("math", "OPERATOR_UNRESOLVED", "math_definition.operator",
                           sorted(contract_module.OPERATOR_FAMILIES), operator["error"],
                           "name a supported operator family and freeze c and delta")]
    curvature = operator["curvature_c"]
    delta = operator["delta"]
    problems = []
    # (a) Flat limit at separated points. The limit of the CLAMPED expression does
    # not exist at fixed delta - the clamp floor 2*delta/c diverges as c -> 0 - so
    # the limit claim is checked on the unclamped formula, which is what the
    # proposal states, and the clamp is checked separately for whether it
    # dominates the declared curvature.
    x, y = (0.11, -0.07, 0.23), (0.19, 0.02, 0.31)
    separation_squared = sum((a - b) ** 2 for a, b in zip(x, y))
    reference = _poincare_squared_distance(x, y, 1e-7, 0.0)
    expected = 4.0 * separation_squared
    if not math.isclose(reference, expected, rel_tol=2e-4, abs_tol=2e-6):
        problems.append(diagnostic(
            "math", "FLAT_LIMIT_MISMATCH", "math_definition.operator.flat_limit",
            expected, reference,
            "state the assignment distance as squared d_c^2, whose limit is 4||x-y||^2",
            verdict=MATH_COUNTEREXAMPLE))
    # (a2) A stated flat limit must be the audited one. This is the defect that
    # reached real reviews twice: calling 4||x-y||^2 the limit of the unsquared
    # distance, whose limit is 2||x-y||.
    declared = (candidate.get("math_definition") or {}).get("operator")
    declared_limit = declared.get("flat_limit") if isinstance(declared, dict) else None
    if isinstance(declared_limit, str) and declared_limit.strip():
        text = declared_limit.replace(" ", "").lower()
        canonical = ("4*||x-y||^2", "4*|x-y|^2", "4||x-y||^2", "4|x-y|^2")
        if text not in canonical:
            problems.append(diagnostic(
                "math", "FLAT_LIMIT_DECLARATION_MISMATCH",
                "math_definition.operator.flat_limit", canonical[0], declared_limit,
                "the squared distance's limit is 4*||x-y||^2; the distance itself is 2*||x-y||",
                verdict=MATH_COUNTEREXAMPLE))
    # (a2b) A fixed arcosh floor lifts the small-separation regime: the squared
    # distance stops approaching zero and is replaced by the constant 2*delta/c.
    # The stabilisation must therefore be max(u, 1.0), which leaves the measurable
    # regime untouched; a floor of 1+delta is reported with the numbers.
    probe = 1e-6
    floor_value = _poincare_squared_distance((0.0, 0.0, 0.0), (probe, 0.0, 0.0), curvature, delta)
    true_value = 4.0 * probe * probe
    if floor_value > 1.01 * true_value:
        problems.append(diagnostic(
            "math", "FLOOR_LIFTS_DISTANCE", "math_definition.operator.delta",
            f"about {true_value:.3e} at separation {probe:g}", f"{floor_value:.3e}",
            "clamp the arcosh argument to at least 1.0 (max(u, 1.0)); a 1+delta floor makes "
            "the distance constant 2*delta/c near coincidence",
            verdict=MATH_COUNTEREXAMPLE))
    # (a3) The objective must say what stops the representation from collapsing.
    # A candidate whose frozen objective can be lowered by mapping every item to
    # one code has already produced a single distinct SID in a real 3000-epoch
    # run, so the declaration is required before any GPU time is spent.
    objective = (candidate.get("math_definition") or {}).get("objective")
    guard = str((objective or {}).get("anti_collapse") or "") if isinstance(objective, dict) else ""
    if not guard.strip():
        problems.append(diagnostic(
            "math", "COLLAPSE_UNGUARDED", "math_definition.objective.anti_collapse",
            "a term or mechanism that keeps the latent informative",
            "not declared",
            "name the anti-collapse term (decoder reconstruction, usage balance, "
            "entropy or margin) and where it enters the loss"))
    # The clamp may only act where the arcosh argument is numerically degenerate;
    # at the declared curvature a separated pair must clear it.
    if curvature > 0 and curvature * separation_squared < 10.0 * delta:
        problems.append(diagnostic(
            "math", "CLAMP_DOMINATES_DOMAIN", "math_definition.operator.delta",
            f"c*|x-y|^2 >> delta at c={curvature}",
            f"c*|x-y|^2={curvature * separation_squared:.3e} vs delta={delta:.1e}",
            "reduce delta or state the curvature regime where the clamp is inactive",
            verdict=MATH_COUNTEREXAMPLE))
    # (b) Coincident points: the clamp sets the value, which must stay finite.
    coincident = _poincare_squared_distance(x, x, curvature, delta)
    if not math.isfinite(coincident) or coincident < 0.0:
        problems.append(diagnostic(
            "math", "ZERO_VECTOR_UNSTABLE", "math_definition.operator.delta",
            "finite non-negative squared distance at x=y", coincident,
            "clamp the arcosh argument to 1+delta with delta > 0",
            verdict=MATH_COUNTEREXAMPLE))
    # (c) Domain: both arms must keep vectors strictly inside the ball.
    radius = operator.get("radius", 1.0 - 1e-5)
    if curvature > 0 and radius >= 1.0 / math.sqrt(curvature):
        problems.append(diagnostic(
            "math", "DOMAIN_ESCAPE", "math_definition.operator.radius",
            f"radius < {1.0 / math.sqrt(curvature):.6f}", radius,
            "project into the open ball before scoring", verdict=MATH_COUNTEREXAMPLE))
    return problems


def check_falsifier(candidate) -> list:
    falsifier = candidate.get("falsifier")
    if not isinstance(falsifier, dict):
        return [diagnostic("falsifier", "FALSIFIER_MISSING", "falsifier", "object",
                           type(falsifier).__name__, "declare the falsifier object")]
    problems = []
    diagnostics = falsifier.get("stage2_diagnostics") or []
    if not isinstance(diagnostics, list) or not diagnostics:
        problems.append(diagnostic("falsifier", "NO_DIAGNOSTICS", "falsifier.stage2_diagnostics",
                                   "at least one metric", diagnostics,
                                   "name the Stage2 metrics the mechanism should move"))
    registered = set()
    warnings = []
    for entry in diagnostics:
        label = (entry or {}).get("name") if isinstance(entry, dict) else None
        name = canonical_metric(label) if label is not None else None
        if name is None:
            problems.append(diagnostic("falsifier", "UNREGISTERED_METRIC",
                                       "falsifier.stage2_diagnostics[].name",
                                       sorted(METRICS), label,
                                       "use a registry metric id or a documented label alias"))
            continue
        if METRICS[name]["role"] == "behavioural_test":
            problems.append(diagnostic("falsifier", "BEHAVIOURAL_AS_STAGE2",
                                       f"falsifier.stage2_diagnostics[{name}]",
                                       "a Stage2 metric", name,
                                       "behavioural metrics belong to the Stage3 registration"))
        direction = str(entry.get("direction") or "")
        expectation = _direction_of(direction)
        if expectation and expectation != METRICS[name]["direction"]:
            # Not a machine defect: the measured direction is evidence the
            # reviewer must weigh, so this is carried as a warning.
            warnings.append(diagnostic(
                "falsifier", "DIRECTION_CONFLICTS_WITH_EVIDENCE",
                f"falsifier.stage2_diagnostics[{name}].direction",
                METRICS[name]["direction"], direction,
                "justify the direction against the measured evidence", verdict=None))
        registered.add(name)
        for key in ("direction", "minimum_effect", "seed_rule"):
            if not entry.get(key):
                problems.append(diagnostic("falsifier", "INCOMPLETE_DIAGNOSTIC",
                                           f"falsifier.stage2_diagnostics[{name}].{key}",
                                           "non-empty", entry.get(key),
                                           "give direction, minimum_effect and seed_rule"))
    problems.extend(entry for entry in warnings if entry)
    behavioural = falsifier.get("stage3_behavioral_test") or {}
    metric = behavioural.get("metric")
    if metric not in BEHAVIOURAL_METRICS:
        problems.append(diagnostic("falsifier", "BEHAVIOURAL_METRIC_INVALID",
                                   "falsifier.stage3_behavioral_test.metric",
                                   list(BEHAVIOURAL_METRICS), metric,
                                   "register test_R@10 or test_NDCG@10"))
    classification = falsifier.get("classification") or {}
    text = " ".join(str(classification.get(key)) for key in
                    ("falsified_if", "passed_if", "mixed_if"))
    for key in ("falsified_if", "passed_if", "mixed_if"):
        if not classification.get(key):
            problems.append(diagnostic("falsifier", "CLASSIFICATION_INCOMPLETE",
                                       f"falsifier.classification.{key}", "non-empty", None,
                                       "define all three outcome classes"))
    if "null" not in text.lower():
        problems.append(diagnostic("falsifier", "NULL_OUTCOME_MISSING",
                                   "falsifier.classification", "a NULL outcome rule", text[:80],
                                   "classify undefined or non-finite outcomes as NULL"))
    # Thresholds are declared once: the falsifier is their only home.
    scattered = [key for key in ("real_trigger_measurement",)
                 if isinstance(candidate.get(key), str)
                 and any(number in candidate[key] for number in
                         (f"{THRESHOLDS['valid_l1_hard_er_strict_gt']}",))]
    if scattered:
        problems.append(diagnostic("falsifier", "THRESHOLD_DUPLICATED", ",".join(scattered),
                                   "thresholds live only in the falsifier",
                                   "same threshold restated in prose",
                                   "reference the falsifier instead of restating the number"))
    del registered
    return problems


LAYERS = (("output", check_output), ("contract", check_contract),
          ("math", check_math), ("falsifier", check_falsifier))


def run(candidate) -> dict:
    """Run the layers in order and return the first decisive verdict.

    Entries carrying ``verdict=None`` are warnings: they record a conflict the
    reviewer must weigh (for example an expected direction that contradicts the
    measured evidence) but they never mask or manufacture a machine failure.
    """
    warnings = []
    for name, check in LAYERS:
        found = check(candidate)
        warnings.extend(entry for entry in found if entry.get("verdict") is None)
        failures = [entry for entry in found if entry.get("verdict") is not None]
        if not failures:
            continue
        verdict = MATH_COUNTEREXAMPLE if any(
            entry["verdict"] == MATH_COUNTEREXAMPLE for entry in failures) else MACHINE_INVALID
        return {"verdict": verdict, "stopped_at": name,
                "diagnostics": warnings + failures, "warnings": warnings,
                "scientific_review_allowed": False}
    contract = compile_contract(candidate)
    return {"verdict": SCIENTIFIC_REVIEW_REQUIRED, "stopped_at": None,
            "diagnostics": warnings, "warnings": warnings,
            "scientific_review_allowed": True,
            "contract": contract, "contract_sha256": contract["contract_sha256"]}


def run_file(path: Path) -> dict:
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:                     # noqa: BLE001 - reported, not raised
        return {"verdict": MACHINE_INVALID, "stopped_at": "output",
                "diagnostics": [diagnostic("output", "UNREADABLE", str(path), "valid JSON",
                                           type(exc).__name__ + ": " + str(exc),
                                           "write a parseable candidate file")],
                "scientific_review_allowed": False}
    return run(payload)

def _direction_of(text: str):
    """The expected direction a proposal states, when it states one.

    "curved minus flat > 0" expects the candidate metric to be higher;
    "flat minus curved > 0" expects it to be lower. Anything else is prose the
    preflight does not try to interpret.
    """
    lowered = str(text or "").lower()
    if "flat minus curved" in lowered or "flat - curved" in lowered:
        return "lower"
    if "curved minus flat" in lowered or "curved - flat" in lowered:
        return "higher"
    return None
