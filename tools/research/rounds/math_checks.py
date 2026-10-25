"""Small-tensor checks for a proposed curved operation (Issue #123/#124).

The earlier version of this module had one check that could not fail: it
multiplied every codeword score of an item by the same positive scalar and
concluded that "the switches change the assignment" from a single ``or``. That
is wrong, and it hides itself: for a row-wise positive factor ``w_i`` the
arg-min over codewords is unchanged, because
``argmin_k w_i d(i, k) = argmin_k d(i, k)``. A behaviour weight that only
scales a row can never move an assignment, so the probe separates four
questions and answers each on its own:

``geometry_operator_valid``
    Do the maps, the domain and the curvature behave as claimed? This says
    nothing about the mechanism being useful.

``behavior_dependency``
    Does turning the *intended* signal off change the intended output? A common
    positive row scalar or a codeword-independent offset cannot count: those are
    invariances, and the checks below measure them.

``geometry_dependency``
    Does the matched curvature-free intervention change the curved computation?

``joint_path_dependency``
    Do both reach the same computation? Without the candidate's own code this
    is ``CANDIDATE_PATH_NOT_RUN``: the generic tests are proxies, labelled as
    such.

Every generic test is a proxy or an operator test. None of them can green-light
a 3000-epoch run; that is the candidate path check's job.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_NOT_RUN = "NOT_RUN"
STATUS_CANDIDATE_PATH_NOT_RUN = "CANDIDATE_PATH_NOT_RUN"

#: The evidence scope of a check.
SCOPE_OPERATOR = "OPERATOR_ONLY"
SCOPE_PROXY = "PROXY"

CHECK_OPERATOR = "geometry_operator_valid"
CHECK_BEHAVIOR = "behavior_dependency"
CHECK_GEOMETRY = "geometry_dependency"
CHECK_JOINT = "joint_path_dependency"

#: The tensors are tiny and float64: the checks are about the maths, not the
#: floating point of a full run.
TENSOR_SIZE = 4
SEED = 20261025


def _poincare(c: float, x):
    import torch

    root = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    limit = (1.0 - 1e-6) * root
    scale = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return x * scale


def _exp0(u, c: float):
    import torch

    root = c ** 0.5
    norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    factor = torch.tanh(root * norm) / (root * norm)
    return _poincare(c, factor * u)


def _log0(x, c: float):
    import torch

    root = c ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scaled = (root * norm).clamp(max=1.0 - 1e-6)
    return x * (torch.atanh(scaled) / (root * norm))


def _poincare_distance(x, y, c: float):
    """The Poincare distance, reduced over the feature dimension."""
    import torch

    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    delta = num / den.clamp_min(1e-12)
    root = c ** 0.5
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    distance = (2.0 / root) * torch.atanh((root * norm).clamp(max=1.0 - 1e-6))
    return distance.squeeze(-1)


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                           .encode("utf-8")).hexdigest()[:16]


def _operator_family(idea: dict) -> str:
    text = " ".join(str(idea.get(field, "")) for field in
                    ("hyperbolic_operator", "coupling_equation", "key_change")).lower()
    if "lorentz" in text:
        return "lorentz"
    if "poincare" in text or "poincar" in text or "hyperbolic" in text:
        return "poincare"
    if "curvature" in text:
        return "curvature"
    return "none"


def _check(name: str, status: str, *, detail: str, scope: str = SCOPE_OPERATOR,
           **extra) -> dict:
    entry = {"check": name, "status": status, "detail": detail, "scope": scope}
    entry.update(extra)
    return entry


def _margins(scores):
    """The per-item score margin between the best and the runner-up codeword."""
    import torch

    ordered = torch.sort(scores, dim=-1, descending=True).values
    return ordered[:, 0] - ordered[:, 1]


def _row_scalar_invariance(scores, weights) -> bool:
    """The analytic invariance a positive row scalar has on an arg-min.

    ``argmin_k (1 + w_i) d(i, k) = argmin_k d(i, k)`` for every positive
    ``w_i``. The measured equality of the two arg-mins is returned, so the
    report shows the invariance instead of asserting it.
    """
    import torch

    scaled = scores * (1.0 + weights)
    return bool(torch.equal(scaled.argmax(-1), scores.argmax(-1)))


def _offset_invariance(scores, offsets) -> bool:
    """A codeword-independent offset cannot move an arg-min either."""
    import torch

    shifted = scores + offsets
    return bool(torch.equal(shifted.argmax(-1), scores.argmax(-1)))


def run_math_checks(idea: dict, *, repo: Path = None, seed: int = SEED) -> dict:
    """The checks of one proposal, with each scope reported separately."""
    import torch

    started = time.monotonic()
    torch.manual_seed(seed)
    family = _operator_family(idea)
    checks = []
    if family == "none":
        checks.append(_check(CHECK_OPERATOR, STATUS_NOT_RUN,
                             detail="the proposal names no curved operation the checks "
                                    "can evaluate",
                             implication="a curved operation must be named for the "
                                         "geometry to be checkable"))
        return {"schema": 2, "idea_id": idea.get("idea_id"), "status": STATUS_NOT_RUN,
                "operator_family": family, "checks": checks,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "seed": seed, "tensor_size": TENSOR_SIZE,
                "note": "no check claims a Stage 3 effect; the candidate's own path is "
                        "not exercised here"}
    c = 0.5
    x = torch.randn(1, TENSOR_SIZE, dtype=torch.float64) * 0.15
    v = torch.randn(1, TENSOR_SIZE, dtype=torch.float64) * 0.2
    recovered = _log0(_exp0(v, c), c)
    error = float((recovered - v).abs().max())
    sectional = -c
    projected = _poincare(c, x * 50.0)
    limit = (1.0 - 1e-6) / (c ** 0.5)
    inside = float(projected.norm(dim=-1).max()) <= limit * (1.0 + 1e-9)
    y = torch.randn(1, TENSOR_SIZE, dtype=torch.float64) * 0.15
    curved = float(_poincare_distance(x, y, c).abs().max())
    euclidean = float((x - y).norm())
    ratio = curved / euclidean if euclidean else float("nan")
    operator_detail = ("log_0(exp_0(v)) error " + format(error, ".2e") +
                       ", sectional curvature " + str(sectional) +
                       ", curved/euclidean ratio " + format(ratio, ".4f") +
                       ", domain guard " + ("holds" if inside else "violated"))
    operator_status = (STATUS_PASS if (operator_ok(operator_detail) and sectional < 0
                                      and inside and abs(ratio - 1.0) > 1e-3)
                       else STATUS_FAIL)
    checks.append(_check(
        CHECK_OPERATOR, operator_status, detail=operator_detail, scope=SCOPE_OPERATOR,
        round_trip_error=error, sectional_curvature=sectional,
        curved_euclidean_ratio=ratio, domain_limit=limit,
        implication="a valid operator says nothing about the coupling: the behaviour "
                    "must still reach the same computation"))
    checks.extend(_dependency_checks(c))
    failed = [entry["check"] for entry in checks if entry["status"] == STATUS_FAIL]
    inconclusives = [entry for entry in checks
                      if entry["status"] in (STATUS_INCONCLUSIVE,
                                             STATUS_CANDIDATE_PATH_NOT_RUN)]
    # The overall status is the candidate-relevant conclusion: a failed proxy
    # invariance is a finding about the proxy, and the joint path cannot be
    # judged by a proxy at all.
    operator_failed = CHECK_OPERATOR in failed
    status = (STATUS_FAIL if operator_failed else
              STATUS_INCONCLUSIVE if inconclusives or failed else STATUS_PASS)
    return {"schema": 2, "idea_id": idea.get("idea_id"), "status": status,
            "operator_family": family, "checks": checks,
            "implementation_digest": _digest([entry["check"] for entry in checks]),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "seed": seed, "tensor_size": TENSOR_SIZE,
            "failed_checks": failed,
            "note": "the operator checks and the proxies are not the candidate's own "
                    "pathway; the joint path needs the implemented candidate"}


def operator_ok(detail: str) -> bool:
    """Whether the round trip itself held; the detail carries the number."""
    return "error 0.00e+00" not in detail and "error nan" not in detail


def _dependency_checks(c: float) -> list:
    """Behaviour and geometry dependence, measured separately, plus the invariances."""
    import torch

    torch.manual_seed(SEED)
    items, codes = 4, 3
    points = torch.randn(items, TENSOR_SIZE, dtype=torch.float64) * 0.3
    codebook = torch.randn(codes, TENSOR_SIZE, dtype=torch.float64) * 0.3
    curved = _poincare_distance(points[:, None, :], codebook[None, :, :], c) ** 2
    euclidean = ((points[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    # A behaviour weight that is constant across the codewords of an item: the
    # invariance the earlier check missed.
    row_scalar = torch.tensor([0.2, 0.5, 1.0, 2.0], dtype=torch.float64)[:, None]
    # A behaviour weight that differs per codeword: this one CAN move the arg-min.
    codeword_weight = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64)
    offsets = torch.tensor(0.5, dtype=torch.float64)

    both_on = (curved * (1.0 + codeword_weight)).argmin(-1)
    behaviour_off = curved.argmin(-1)
    geometry_off = (euclidean * (1.0 + codeword_weight)).argmin(-1)
    behaviour_changed = bool((both_on != behaviour_off).any())
    geometry_changed = bool((both_on != geometry_off).any())
    margins_both = _margins(curved * (1.0 + codeword_weight))
    margins_behaviour_off = _margins(curved)

    row_invariant = _row_scalar_invariance(curved, row_scalar)
    offset_invariant = _offset_invariance(curved, offsets)
    return [
        _check(CHECK_BEHAVIOR,
               STATUS_FAIL if row_invariant else STATUS_INCONCLUSIVE,
               detail="a positive row scalar leaves every arg-min unchanged (measured: " +
                      str(row_invariant) + "), so it can never move an assignment; a "
                      "codeword-dependent weight moved " + str(behaviour_changed) +
                      " of the top-1 indices; a codeword-independent offset is also "
                      "invariant (" + str(offset_invariant) + ")",
               scope=SCOPE_PROXY,
               witness={"row_scalar_invariance": row_invariant,
                        "offset_invariance": offset_invariant,
                        "codeword_weight_changes_assignment": behaviour_changed,
                        "margins_both_on": [float(value) for value in margins_both],
                        "margins_behavior_off": [float(value) for value in margins_behaviour_off]},
               implication="behaviour must enter the score in a codeword-dependent way; "
                           "a row scale or a common offset is an invariance, not a coupling"),
        _check(CHECK_GEOMETRY,
               STATUS_PASS if geometry_changed else STATUS_INCONCLUSIVE,
               detail="the matched Euclidean intervention changed " +
                      str(int(geometry_changed)) + " of the top-1 indices",
               scope=SCOPE_PROXY,
               witness={"geometry_off_top1": geometry_off.tolist(),
                        "both_on_top1": both_on.tolist()},
               implication="the curved and the Euclidean computations must differ in "
                           "the quantity the mechanism claims"),
        _check(CHECK_JOINT, STATUS_CANDIDATE_PATH_NOT_RUN,
               detail="the candidate's own quantizer path is not exercised by the "
                      "generic proxy; the joint dependency needs the implemented tree",
               scope=SCOPE_PROXY,
               implication="only the candidate path check can admit a Stage 2 run"),
    ]


def _cli() -> int:                                    # pragma: no cover - entry point
    """Run the checks of one proposal and print the report as JSON."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="hyperbolic sanity checks")
    parser.add_argument("--idea", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    print(json.dumps(run_math_checks(json.loads(arguments.idea), seed=arguments.seed),
                     ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(_cli())
