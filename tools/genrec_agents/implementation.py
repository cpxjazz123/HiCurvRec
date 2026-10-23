"""Agent I -- the implementation contract (issue #110, WP1).

The implementation agent writes code for ONE frozen, reviewed proposal. It does not design:
it implements the mathematics the proposal froze, exactly, and it cannot quietly substitute
something easier. When the proposal is ambiguous or a tensor it needs does not exist, the
agent returns ``RESEARCH_REVISION_REQUIRED`` -- the proposal goes back to the research agent
as a NEW version with a fresh novelty review -- and it never falls back to a Euclidean
proxy.

Three rules are enforced here, because they are the ones that would otherwise be enforced
by good intentions:

* the frozen files are not writable: the CGV tooling, the datasets, the baselines, the
  archived evidence and the control files;
* the implementation declares the mathematics it implemented, and it is compared with the
  proposal's -- a change of the mathematics is a new mechanism identity, not a fix;
* a repair is only allowed for an independent, verified code bug under the SAME frozen
  mathematics, it is counted, and it never runs an official cold training.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The paths the implementation may not write: the verifier, the data, the baselines, the
#: archived evidence and the control files.
FROZEN_PATHS: Tuple[str, ...] = (
    "tools/cgv",
    "tools/mvg",
    "results/stage3_T5Train/monotonicity",
    "results/stage0_build_parquet",
    "results/harness3_runs",
    "scripts/history_probe.py",
    "scripts/history_probe_runner.py",
    "stage2_RQ-VAE/TIGER_r8_control",
)

#: The status an implementation returns.
IMPLEMENTED = "implemented"
REVISION_REQUIRED = "RESEARCH_REVISION_REQUIRED"
REFUSED = "refused"

#: How many repairs one implementation may have, and what a repair is for.
MAX_REPAIRS = 3

#: The terms a mechanism implementation may declare.
DECLARED_TERMS = ("equation", "symbols", "parameters", "gradient_targets", "target_paths")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ImplementationRequest:
    """What the implementation agent is given: one frozen proposal and nothing else."""

    mechanism_id: str
    proposal: Dict[str, Any]
    proposal_sha256: str
    tree: Path
    frozen_protocol_hash: str = ""
    repairs_used: int = 0
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "proposal_sha256": self.proposal_sha256,
            "tree": str(self.tree),
            "frozen_protocol_hash": self.frozen_protocol_hash,
            "repairs_used": self.repairs_used,
            "notes": self.notes,
        }


@dataclass
class ImplementationResult:
    """What the implementation agent returns."""

    status: str
    mechanism_id: str
    proposal_sha256: str
    implementation_sha256: str = ""
    detail: str = ""
    revision_request: Optional[Dict[str, Any]] = None
    repaired: bool = False
    repairs_used: int = 0
    written: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "mechanism_id": self.mechanism_id,
            "proposal_sha256": self.proposal_sha256,
            "implementation_sha256": self.implementation_sha256,
            "detail": self.detail,
            "revision_request": self.revision_request,
            "repaired": self.repaired,
            "repairs_used": self.repairs_used,
            "written": list(self.written),
            "evidence": dict(self.evidence),
        }


def write_is_allowed(tree: Path, path: Path) -> bool:
    """Whether the implementation may write this path inside its tree."""
    try:
        relative = Path(path).resolve().relative_to(Path(tree).resolve())
    except ValueError:
        return False
    text = str(relative)
    return not any(text.startswith(frozen) for frozen in FROZEN_PATHS)


def check_against_frozen_files(tree: Path, changed: Sequence[Path]) -> List[str]:
    """The frozen files the implementation touched, if any."""
    touched: List[str] = []
    for path in changed:
        try:
            relative = Path(path).resolve().relative_to(Path(tree).resolve())
        except ValueError:
            continue
        text = str(relative)
        if any(text.startswith(frozen) for frozen in FROZEN_PATHS):
            touched.append(text)
    return touched


def compare_mathematics(
    proposal: Dict[str, Any], implementation: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare the mathematics the proposal froze with the mathematics that was implemented.

    A change of the equation, of the symbols, of the parameters, of the gradient targets or
    of the target paths is a change of the mathematics, and the issue is explicit that such a
    change needs a NEW mechanism identity and a fresh novelty review.
    """
    differences: List[str] = []
    for term in DECLARED_TERMS:
        left = proposal.get(term)
        right = implementation.get(term)
        if term == "parameters":
            left = {str(k): float(v) for k, v in (left or {}).items()}
            right = {str(k): float(v) for k, v in (right or {}).items()}
        if left != right:
            differences.append(term)
    return {
        "identical": not differences,
        "differences": differences,
        "verdict": (
            "the implementation is the frozen mathematics"
            if not differences
            else "the implementation changed the mathematics: it needs a new mechanism identity"
        ),
    }


def request_revision(
    request: ImplementationRequest, reason: str, *, missing: Optional[Dict[str, Any]] = None
) -> ImplementationResult:
    """The revision request: the proposal goes back to the research agent, unchanged."""
    return ImplementationResult(
        status=REVISION_REQUIRED,
        mechanism_id=request.mechanism_id,
        proposal_sha256=request.proposal_sha256,
        detail=reason,
        revision_request={
            "reason": reason,
            "missing": missing or {},
            "proposal_sha256": request.proposal_sha256,
            "requested_utc": now_utc(),
            "instruction": (
                "produce a NEW proposal version with a fresh novelty review; a silent "
                "substitution of a Euclidean proxy is not allowed"
            ),
        },
        repairs_used=request.repairs_used,
    )


def implement(
    request: ImplementationRequest,
    implementation: Optional[Dict[str, Any]] = None,
    *,
    written: Optional[Sequence[Path]] = None,
    frozen_files_touched: Optional[Sequence[str]] = None,
) -> ImplementationResult:
    """Check an implementation against the frozen proposal and its rules.

    The function does not write the code: the agent does that, and this decides whether what
    it wrote may be sealed. It refuses a changed mathematics, a touched frozen file, a
    proposal the implementation could not follow, and a repair that has run out.
    """
    implementation = implementation or {}
    written = list(written or ())
    touched = list(frozen_files_touched or ())

    if touched:
        return ImplementationResult(
            status=REFUSED,
            mechanism_id=request.mechanism_id,
            proposal_sha256=request.proposal_sha256,
            detail=f"the implementation touched the frozen file(s) {touched}",
            repairs_used=request.repairs_used,
        )

    comparison = compare_mathematics(request.proposal, implementation)
    if not comparison["identical"]:
        return ImplementationResult(
            status=REVISION_REQUIRED,
            mechanism_id=request.mechanism_id,
            proposal_sha256=request.proposal_sha256,
            detail=(
                f"the implementation changed {comparison['differences']}, which is a change of "
                "the mathematics and needs a new mechanism identity and a fresh novelty review"
            ),
            revision_request={
                "reason": "the mathematics changed",
                "missing": {term: implementation.get(term) for term in comparison["differences"]},
                "proposal_sha256": request.proposal_sha256,
                "requested_utc": now_utc(),
                "instruction": (
                    "produce a NEW proposal version with a fresh novelty review; a silent "
                    "substitution of a Euclidean proxy is not allowed"
                ),
            },
            repairs_used=request.repairs_used,
        )

    if not implementation.get("equation"):
        return request_revision(
            request, "the implementation declares no equation, so the proposal was not followed"
        )

    digests = {
        str(path): sha256_file(path) for path in written if Path(path).is_file()
    }
    implementation_sha256 = sha256_bytes(
        json.dumps(digests, sort_keys=True).encode("utf-8")
    )
    return ImplementationResult(
        status=IMPLEMENTED,
        mechanism_id=request.mechanism_id,
        proposal_sha256=request.proposal_sha256,
        implementation_sha256=implementation_sha256,
        detail="the implementation is the frozen mathematics",
        repairs_used=request.repairs_used,
        written=[str(path) for path in written],
        evidence={
            "file_digests": digests,
            "mathematics_comparison": comparison,
            "frozen_files_touched": touched,
            "sealed_utc": now_utc(),
            "official_cold_run_performed": False,
        },
    )


def repair_allowed(
    result: ImplementationResult, *, is_code_bug: bool, same_mathematics: bool
) -> Tuple[bool, str]:
    """Whether a repair may be made, and why.

    A repair is for an independent, verified code bug under the SAME frozen mathematics. It
    is counted, and it never runs an official cold training.
    """
    if not is_code_bug:
        return False, "a repair is for a verified code bug, not for a change of the mathematics"
    if not same_mathematics:
        return False, "a repair must keep the frozen mathematics"
    if result.repairs_used >= MAX_REPAIRS:
        return False, f"the repair budget of {MAX_REPAIRS} is spent"
    return True, "a code-bug repair under the same mathematics"


__all__ = [
    "DECLARED_TERMS",
    "FROZEN_PATHS",
    "IMPLEMENTED",
    "ImplementationRequest",
    "ImplementationResult",
    "MAX_REPAIRS",
    "REFUSED",
    "REVISION_REQUIRED",
    "check_against_frozen_files",
    "compare_mathematics",
    "implement",
    "repair_allowed",
    "request_revision",
    "write_is_allowed",
]
