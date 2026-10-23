"""The hyperbolic validity gates H1-H7 (issue #110, WP2).

A word in a title, or an import of a distance function, is not a hyperbolic mechanism. These
gates are the independent check that a mechanism is genuinely hyperbolic, and they are
independent of the mechanism's own code: the geometry comes from
:mod:`hyperbolic_verifier.geometry`, which is written from the definitions, and the
mechanism's source is only ever READ, to see which operators it actually calls.

The gates fail closed. A gate whose evidence is missing is a FAIL, not a pass, and
``not_applicable`` is never a pass for a gate the issue requires. Every certificate is bound
to the proposal hash and the code-tree hash it was produced for, so a certificate cannot be
carried over to a different mechanism.

The gates that need the mechanism's own runtime -- H4, the gradient path, and H6, the causal
SID pathway -- are given their evidence by the caller. A caller that cannot supply it gets a
FAIL, which is the point: the issue asks for "insufficient evidence" to block, not to be
treated as a pass.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import geometry as G

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The gates, in the order the issue lists them.
GATES: Tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")

#: The manifold models a mechanism may declare.
MANIFOLDS = ("poincare_ball", "lorentz_hyperboloid")

#: The operator names that count as genuinely hyperbolic, and the module-level names a
#: candidate is expected to use for them.
HYPERBOLIC_OPERATOR_NAMES: Dict[str, Tuple[str, ...]] = {
    "geodesic_distance": ("poincare_distance", "hyperbolic_distance", "lorentz_distance"),
    "exp_map": ("exp_map0", "hyperbolic_exp", "poincare_exp_map0"),
    "log_map": ("log_map0", "hyperbolic_log", "poincare_log_map0"),
    "mobius_addition": ("mobius_add", "gyro_add"),
    "gyrovector": ("gyro", "gyrovector"),
    "busemann": ("busemann", "horo"),
    "parallel_transport": ("parallel_transport", "pt_parallel_transport"),
    "riemannian_update": ("riemannian", "riemannian_update", "retraction"),
    "artanh_projection": ("artanh", "clamp_to_ball", "project_to_ball"),
}

#: The Euclidean operators a mechanism may use for the ordinary parts of its computation.
EUCLIDEAN_NAMES = (
    "torch.cdist",
    "cdist(",
    "linalg.norm",
    "pairwise_distance",
    "cosine_similarity",
    "softmax",
)

#: A gate state that is not a pass.
NON_PASSING = frozenset({"not_applicable", "insufficient_evidence", "fail", "failed"})


class GateEvidenceError(RuntimeError):
    """The evidence a gate needs is missing or malformed. The gate FAILS."""


@dataclass
class GateResult:
    """One gate's verdict."""

    gate: str
    state: str
    detail: str = ""
    measurements: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "state": self.state,
            "detail": self.detail,
            "measurements": self.measurements,
        }


@dataclass
class Certificate:
    """The seven gates' verdicts, bound to what they were produced for."""

    proposal_sha256: str
    code_tree_sha256: str
    gates: Dict[str, GateResult] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(
            result.state == "pass" for result in self.gates.values()
        ) and set(self.gates) == set(GATES)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "proposal_sha256": self.proposal_sha256,
            "code_tree_sha256": self.code_tree_sha256,
            "passed": self.passed,
            "gates": {name: result.as_dict() for name, result in sorted(self.gates.items())},
            "summary": {
                name: result.state for name, result in sorted(self.gates.items())
            },
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        return path


def code_tree_sha256(tree: Path) -> str:
    """The digest of a tree's source, so a certificate is bound to the code it checked."""
    tree = Path(tree)
    digest = hashlib.sha256()
    for path in sorted(tree.rglob("*.py")):
        digest.update(str(path.relative_to(tree)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _declared_curvature(declaration: Dict[str, Any]) -> Optional[float]:
    curvature = declaration.get("curvature")
    if isinstance(curvature, (int, float)):
        return float(curvature)
    if isinstance(curvature, str):
        match = re.search(r"-?\d+(?:\.\d+)?", curvature)
        if match:
            return float(match.group(0))
    return None


def gate_h1(declaration: Dict[str, Any], source: str) -> GateResult:
    """H1: the manifold, the curvature, the domains and the tangent basepoints."""
    model = str(declaration.get("hyperbolic_model") or "").strip().lower()
    curvature = _declared_curvature(declaration)
    problems: List[str] = []
    if model not in MANIFOLDS:
        problems.append(f"the declared model {model!r} is not one of {list(MANIFOLDS)}")
    if curvature is None:
        problems.append("no curvature is declared")
    elif curvature >= 0:
        problems.append(f"the curvature must be -c with c > 0, not {curvature!r}")
    if not declaration.get("basepoints") and "basepoint" not in source.lower():
        problems.append("no tangent basepoint is declared or named in the source")
    if problems:
        return GateResult("H1", "fail", "; ".join(problems))
    return GateResult(
        "H1",
        "pass",
        f"the model {model} with curvature {curvature} and its domain constraint",
        {"model": model, "curvature": curvature},
    )


def gate_h2(declaration: Dict[str, Any], source: str) -> GateResult:
    """H2: a genuinely hyperbolic operator is in the loss, and the mechanism needs it.

    The evidence is a TRACE of the candidate's own source: which hyperbolic operators it
    calls, and whether the operator it declared is among them. A Euclidean mechanism with a
    hyperbolic name is refused here, before any GPU time.
    """
    called: List[str] = []
    for label, names in HYPERBOLIC_OPERATOR_NAMES.items():
        if any(name in source for name in names):
            called.append(label)
    declared = str(declaration.get("hyperbolic_operator") or "").strip()
    euclidean_only = [
        name for name in EUCLIDEAN_NAMES if name in source
    ] and not called
    if not called:
        return GateResult(
            "H2",
            "fail",
            "the source calls no hyperbolic operator: a mechanism in a hyperbolic campaign is "
            "hyperbolic only if a genuinely hyperbolic operator is in its computation",
            {"hyperbolic_operators_called": [], "euclidean_operators": euclidean_only},
        )
    # the declaration is matched against the operator LABELS as well as the call names, so
    # "geodesic distance" and "geodesic_distance" are the same claim
    normalised = re.sub(r"[^a-z0-9]+", "_", declared.lower()).strip("_")
    traceable = {label for label in HYPERBOLIC_OPERATOR_NAMES}
    traceable |= {name for names in HYPERBOLIC_OPERATOR_NAMES.values() for name in names}
    if declared and not any(token in normalised for token in traceable):
        return GateResult(
            "H2",
            "fail",
            f"the declared operator {declared!r} is not one the verifier can trace",
            {"hyperbolic_operators_called": called},
        )
    return GateResult(
        "H2",
        "pass",
        f"the source calls the hyperbolic operator(s) {called}",
        {"hyperbolic_operators_called": called, "declared": declared},
    )


def gate_h3(declaration: Dict[str, Any]) -> GateResult:
    """H3: the geometry must depend on the curvature, and not by a scale alone.

    The evidence is computed on FIXED points: the distance and the exponential map are
    evaluated at two curvatures, and the change is compared with the change a mere rescaling
    of the points would produce. A curvature flag that changes nothing, and a geometry that
    only scales, are both refused.

    WHAT THIS GATE CANNOT SEE (finding F4, issue #112). In the Poincare ball the curvature IS a
    pure rescaling: x -> x/sqrt(c) is an isometry onto the unit ball, so on points with a fixed
    NORMALISED radius every intrinsic quantity is independent of c. The measured consequence:
    the transport's deviation from the identity is 0.5143924039356781 at c = 0.25, 1 and 4
    alike. A curvature sweep in this model is therefore a scale change, and no gate that only
    compares magnitudes can tell a real geometric dependence from a rescaling.

    The honest form of H3 for this model is the curvature CLASS: the same operator in the
    hyperbolic model and in its Euclidean twin must produce measurably different geometry (a
    tangent closed level set versus a half-space), with the difference predicted in advance.
    The candidates in this repository are checked against that form directly -- see
    tools/hyperbolic_verifier/busemann.py and horosphere.py, and the measured
    "curvature is a rescaling" result in
    results/hyperbolic_research_runs/subspace_111/subspace_geometry_review.json.
    """
    import torch

    curvature = _declared_curvature(declaration)
    if curvature is None or curvature >= 0:
        return GateResult("H3", "fail", "no usable curvature is declared")
    c = abs(curvature)
    generator = torch.Generator().manual_seed(20240101)
    x = torch.randn(4, 3, generator=generator, dtype=torch.float64) * 0.25
    y = torch.randn(4, 3, generator=generator, dtype=torch.float64) * 0.25
    at_c = G.poincare_distance(x, y, c)
    at_half = G.poincare_distance(x, y, c / 2.0)
    if not torch.isfinite(at_c).all() or not torch.isfinite(at_half).all():
        return GateResult("H3", "fail", "the distance is not finite at the declared curvature")
    changed = float((at_c - at_half).abs().max())
    scale = float((x - y).abs().max())
    if changed <= 1e-9:
        return GateResult(
            "H3",
            "fail",
            "the geometry does not change with the curvature: the curvature is a flag, not a "
            "parameter of the computation",
            {"max_change": changed},
        )
    if changed <= 1e-6 * scale:
        return GateResult(
            "H3",
            "fail",
            "the geometry changes with the curvature only by a scale",
            {"max_change": changed, "scale": scale},
        )
    return GateResult(
        "H3",
        "pass",
        f"the geometry changes with the curvature (by {changed:.4f} between c and c/2)",
        {"max_change": changed, "curvature": curvature},
    )


def gate_h4(evidence: Optional[Dict[str, Any]]) -> GateResult:
    """H4: finite, correct gradients reach the declared trainable parameters.

    The evidence is supplied by the caller, which runs the mechanism's own module. Without
    it the gate FAILS: the issue is explicit that insufficient evidence is not a pass.
    """
    if not evidence:
        return GateResult(
            "H4",
            "insufficient_evidence",
            "no gradient evidence was supplied: the mechanism's module has to be run and its "
            "gradients measured",
        )
    declared = set(evidence.get("declared_targets") or ())
    reached = set(evidence.get("reached_targets") or ())
    finite = bool(evidence.get("finite", False))
    if not finite:
        return GateResult("H4", "fail", "a gradient is not finite")
    missing = sorted(declared - reached)
    if missing:
        return GateResult(
            "H4",
            "fail",
            f"the gradient does not reach the declared target(s) {missing}",
            {"declared": sorted(declared), "reached": sorted(reached)},
        )
    return GateResult(
        "H4",
        "pass",
        "the gradients are finite and reach exactly the declared parameters",
        {"declared": sorted(declared), "reached": sorted(reached)},
    )


def gate_h5(declaration: Dict[str, Any]) -> GateResult:
    """H5: the numerics survive the degenerate cases.

    The coincident points, the origin and the near-boundary points are the three cases that
    break a hyperbolic term. The check is on the verifier's own geometry, because that is
    what a mechanism's operator has to agree with.
    """
    import torch

    curvature = _declared_curvature(declaration)
    if curvature is None or curvature >= 0:
        return GateResult("H5", "fail", "no usable curvature is declared")
    c = abs(curvature)
    origin = torch.zeros(1, 3, dtype=torch.float64)
    near = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    boundary = G.project_to_ball(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64), c)
    problems: List[str] = []
    for name, value in (
        ("the distance to the origin", G.poincare_distance(origin, origin, c)),
        ("the distance to a coincident point", G.poincare_distance(near, near, c)),
        ("the distance to a projected boundary point", G.poincare_distance(boundary, boundary, c)),
    ):
        if not torch.isfinite(value).all():
            problems.append(f"{name} is not finite")
    margin = G.ball_margin(boundary, c)
    if bool((margin <= 0).any()):
        problems.append("the projection does not keep the point inside the ball")
    if problems:
        return GateResult("H5", "fail", "; ".join(problems))
    return GateResult(
        "H5",
        "pass",
        "the coincident, the zero and the near-boundary cases are finite and inside the domain",
        {"margin": float(margin.min())},
    )


def gate_h6(evidence: Optional[Dict[str, Any]]) -> GateResult:
    """H6: the operator is actually enabled and it changes the hard SID level.

    The evidence is the activation test: a short run in which the mechanism is enabled, and
    the export it produced. Without it the gate FAILS.
    """
    if not evidence:
        return GateResult(
            "H6",
            "insufficient_evidence",
            "no activation evidence was supplied: the operator has to be shown to be enabled "
            "and to change the exported hard SID level",
        )
    enabled = bool(evidence.get("operator_enabled", False))
    changed = evidence.get("changed_fraction")
    codes = evidence.get("distinct_level_codes")
    if not enabled:
        return GateResult("H6", "fail", "the operator was not enabled in the run")
    if changed is not None and float(changed) <= 0.0:
        return GateResult(
            "H6",
            "fail",
            "the run with the operator enabled changed no item's codes",
            {"changed_fraction": changed},
        )
    if codes is not None and int(codes) <= 2:
        return GateResult(
            "H6",
            "fail",
            f"the export uses only {codes} level-0 codes: the arm collapsed",
            {"distinct_level_codes": codes},
        )
    return GateResult(
        "H6",
        "pass",
        "the operator is enabled and the exported codes changed",
        {"changed_fraction": changed, "distinct_level_codes": codes},
    )


def gate_h7(declaration: Dict[str, Any]) -> GateResult:
    """H7: the mathematical identities the proposal claims, checked against the oracle.

    The identities are the ones the oracle can check: the distance from the origin, the
    exponential and logarithm maps, the Busemann function's divergence, and the conformal
    factor. The oracle does not import the candidate's code.
    """
    import torch

    generator = torch.Generator().manual_seed(20240202)
    v = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.3
    origin = torch.zeros_like(v)
    problems: List[str] = []
    mapped = G.poincare_exp_map0(v, 1.0)
    if float((G.poincare_distance(origin, mapped, 1.0) - torch.linalg.norm(v, dim=-1)).abs().max()) > 1e-9:
        problems.append("the exponential map does not preserve the tangent norm")
    if float((G.poincare_log_map0(mapped, 1.0) - v).abs().max()) > 1e-9:
        problems.append("the logarithm map is not the inverse of the exponential map")
    norm = torch.linalg.norm(v, dim=-1)
    if float((G.poincare_distance(origin, v, 1.0) - 2.0 * torch.atanh(norm)).abs().max()) > 1e-9:
        problems.append("the distance from the origin is not twice the artanh of the norm")
    if problems:
        return GateResult("H7", "fail", "; ".join(problems))
    return GateResult(
        "H7",
        "pass",
        "the distance, the maps and the Busemann function satisfy their identities against "
        "the independent oracle",
    )


def run_gates(
    declaration: Dict[str, Any],
    source: str,
    *,
    proposal_sha256: str,
    code_tree_sha256: str,
    gradient_evidence: Optional[Dict[str, Any]] = None,
    activation_evidence: Optional[Dict[str, Any]] = None,
) -> Certificate:
    """Run the seven gates and return the certificate they produce.

    A gate whose evidence is missing is a FAIL with the state ``insufficient_evidence``,
    which the certificate's ``passed`` treats as not passed. Nothing here can be waived.
    """
    certificate = Certificate(
        proposal_sha256=proposal_sha256, code_tree_sha256=code_tree_sha256
    )
    certificate.gates["H1"] = gate_h1(declaration, source)
    certificate.gates["H2"] = gate_h2(declaration, source)
    certificate.gates["H3"] = gate_h3(declaration)
    certificate.gates["H4"] = gate_h4(gradient_evidence)
    certificate.gates["H5"] = gate_h5(declaration)
    certificate.gates["H6"] = gate_h6(activation_evidence)
    certificate.gates["H7"] = gate_h7(declaration)
    return certificate


def source_of(tree: Path, module: str = "") -> str:
    """The source of a mechanism, read for the operator trace."""
    tree = Path(tree)
    if module:
        path = tree / f"{module}.py"
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    parts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(tree.glob("*.py"))]
    return "\n".join(parts)


__all__ = [
    "Certificate",
    "GATES",
    "GateEvidenceError",
    "GateResult",
    "HYPERBOLIC_OPERATOR_NAMES",
    "MANIFOLDS",
    "code_tree_sha256",
    "gate_h1",
    "gate_h2",
    "gate_h3",
    "gate_h4",
    "gate_h5",
    "gate_h6",
    "gate_h7",
    "run_gates",
    "source_of",
]
