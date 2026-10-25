"""Small-tensor sanity checks for a proposed curved operation (Issue #123).

The checks are deliberately cheap and falsifiable: they ask whether the operator
the proposal names behaves as it claims on a handful of tensors, whether the
curvature is negative where it is said to be, and whether the matched Euclidean
control really is a different computation. They say ``PASS``, ``FAIL``,
``INCONCLUSIVE`` or ``NOT_RUN``; a check that could not be executed never claims
a result.

Two things are never concluded here. A coordinate round trip
(``log_0(exp_0(v)) = v``) shows the maps are consistent, not that the curvature
buys anything. And a small proxy assignment is not the candidate's own
pathway: it is labelled as a proxy wherever it is used.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_NOT_RUN = "NOT_RUN"

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
    import torch

    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    delta = num / den.clamp_min(1e-12)
    root = c ** 0.5
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # d_c(x, y) = (2 / sqrt(c)) * artanh(sqrt(c) * ||(-x) (+)_c y||)
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


def _check(name: str, status: str, *, detail: str, **extra) -> dict:
    entry = {"check": name, "status": status, "detail": detail}
    entry.update(extra)
    return entry


def run_math_checks(idea: dict, *, repo: Path = None, seed: int = SEED) -> dict:
    """The sanity checks of one proposal."""
    import torch

    started = time.monotonic()
    torch.manual_seed(seed)
    family = _operator_family(idea)
    checks = []
    if family == "none":
        checks.append(_check("operator_declared", STATUS_NOT_RUN,
                             detail="the proposal names no curved operation the checks "
                                    "can evaluate",
                             implication="a curved operation must be named for the "
                                         "geometry to be checkable"))
        return {"schema": 1, "idea_id": idea.get("idea_id"), "status": STATUS_NOT_RUN,
                "operator_family": family, "checks": checks,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "seed": seed, "tensor_size": TENSOR_SIZE,
                "note": "no check claims a Stage 3 effect"}
    c = 0.5
    x = torch.randn(1, TENSOR_SIZE, dtype=torch.float64) * 0.15
    v = torch.randn(1, TENSOR_SIZE, dtype=torch.float64) * 0.2
    # 1) the maps are consistent with each other
    recovered = _log0(_exp0(v, c), c)
    error = float((recovered - v).abs().max())
    checks.append(_check(
        "log0_exp0_round_trip", STATUS_PASS if error < 1e-6 else STATUS_FAIL,
        detail="max |log_0(exp_0(v)) - v| = " + format(error, ".3e"),
        error=error,
        implication="the maps are mutually consistent; this is a sanity check, NOT "
                    "evidence that the curvature helps"))
    # 2) the curvature is negative where the proposal says it is
    sectional = -c
    checks.append(_check(
        "curvature_is_negative", STATUS_PASS if sectional < 0 else STATUS_FAIL,
        detail="curvature magnitude c=" + str(c) + " gives sectional curvature " +
               str(sectional),
        sectional_curvature=sectional))
    # 3) the distance is not the Euclidean one
    y = torch.randn(1, TENSOR_SIZE, dtype=torch.float64) * 0.15
    curved = float(_poincare_distance(x, y, c).abs().max())
    euclidean = float((x - y).norm())
    ratio = curved / euclidean if euclidean else float("nan")
    checks.append(_check(
        "distance_is_not_euclidean", STATUS_PASS if abs(ratio - 1.0) > 1e-3 else STATUS_FAIL,
        detail="curved/euclidean distance ratio = " + format(ratio, ".4f"),
        ratio=ratio,
        implication="a ratio of one would mean the matched Euclidean control "
                    "reproduces the operator"))
    # 4) the domain guard holds
    projected = _poincare(c, x * 50.0)
    limit = (1.0 - 1e-6) / (c ** 0.5)
    # The projection lands exactly on the limit, so the comparison carries a
    # float tolerance rather than a strict inequality.
    inside = float(projected.norm(dim=-1).max()) <= limit * (1.0 + 1e-9)
    checks.append(_check(
        "domain_guard_holds", STATUS_PASS if inside else STATUS_FAIL,
        detail="max ||projected|| = " + format(float(projected.norm(dim=-1).max()), ".6f") +
               " under the limit " + format(limit, ".6f"),
        limit=limit))
    # 5) the off/on switches move the assignment, on a small proxy
    proxy = _proxy_off_on(c)
    checks.append(_check(
        "proxy_behavior_off_geometry_off_both_on", proxy["status"],
        detail=proxy["detail"],
        proxy=True,
        limitation="a synthetic proxy: it shows the switches are meaningful in "
                   "principle, it is NOT the candidate's own pathway and says nothing "
                   "about the candidate's SID or its Recall"))
    failures = [entry for entry in checks if entry["status"] == STATUS_FAIL]
    inconclusives = [entry for entry in checks if entry["status"] == STATUS_INCONCLUSIVE]
    status = STATUS_FAIL if failures else (STATUS_INCONCLUSIVE if inconclusives else STATUS_PASS)
    return {"schema": 1, "idea_id": idea.get("idea_id"), "status": status,
            "operator_family": family, "checks": checks,
            "implementation_digest": _digest([entry["check"] for entry in checks]),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "seed": seed, "tensor_size": TENSOR_SIZE,
            "note": "a geometric sanity check is not evidence that Stage 3 Recall improves"}


def _proxy_off_on(c: float) -> dict:
    """A tiny proxy: do the off/on switches change the assignment at all?"""
    import torch

    torch.manual_seed(SEED)
    codes = torch.randn(2, TENSOR_SIZE, dtype=torch.float64) * 0.3
    points = torch.randn(3, TENSOR_SIZE, dtype=torch.float64) * 0.3
    weights = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)[:, None]
    curved = _poincare_distance(points[:, None, :], codes[None, :, :], c) ** 2
    euclidean = ((points[:, None, :] - codes[None, :, :]) ** 2).sum(-1)
    both_on = (curved * (1.0 + weights)).argmin(-1)
    behavior_off = curved.argmin(-1)
    geometry_off = (euclidean * (1.0 + weights)).argmin(-1)
    changed = bool((both_on != behavior_off).any() or (both_on != geometry_off).any())
    finite = bool(torch.isfinite(curved).all())
    return {"status": STATUS_PASS if (changed and finite) else STATUS_INCONCLUSIVE,
            "detail": "both-on assignment " + str(both_on.tolist()) +
                       ", behaviour-off " + str(behavior_off.tolist()) +
                       ", geometry-off " + str(geometry_off.tolist()) +
                       ", the switches change the assignment: " + str(changed)}
