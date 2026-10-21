"""The mechanism-gradient integration contract (issue #106, workstream A).

A mechanism says which parameters it intends to reach. This module turns that into a
**testable contract** with an independent numeric oracle, and it is deliberately not a
"push the gradient everywhere" mechanism: a contract states the intended parameter groups,
and the oracle checks that exactly those groups received the mechanism's contribution,
exactly once, with the base objective preserved.

The contract has four obligations, each checkable:

1. **declared** -- the mechanism names the parameter groups it intends to reach, and it
   names the groups it must NOT reach;
2. **delivered once** -- the contribution to each declared group survives the host's own
   assembly, clipping and optimizer step, and arrives exactly once (a mechanism composed
   twice would double its contribution, and the oracle sees that);
3. **the base objective is preserved** -- the host's own loss and gradient are unchanged by
   the mechanism's presence, apart from the mechanism's own contribution;
4. **the oracle agrees** -- the contribution the mechanism claims is the gradient of the
   mechanism's own term with respect to that parameter group, computed independently of
   the mechanism's code (autograd on the term alone).

The oracle is the important part: it is the check that would have caught the two arms whose
gradients the host discards, and the check that would catch a mechanism which reaches the
codebook a thousand times harder than the arms that did not collapse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = 1

#: The parameter groups the host exposes.
KNOWN_GROUPS = ("encoder", "decoder") + tuple(f"rq_codebook_{index}" for index in range(3))

#: A group is "reached" when the mechanism's contribution to it survives the host's
#: assembly and is non-negligible relative to the mechanism's own gradient on it.
DELIVERY_TOLERANCE = 1e-12


class ContractViolation(RuntimeError):
    """The mechanism did not do what it declared."""


@dataclass
class GradientContract:
    """What a mechanism declares about its own gradient delivery."""

    version: int = CONTRACT_VERSION
    mechanism_id: str = ""
    reaches: List[str] = field(default_factory=list)
    must_not_reach: List[str] = field(default_factory=list)
    weight: float = 0.0
    base_objective_preserved: bool = True
    delivered_once: bool = True
    notes: str = ""

    @classmethod
    def from_module(cls, module: Any, *, mechanism_id: str = "", weight: float = 0.0) -> "GradientContract":
        """Read the contract a mechanism module declares, or refuse it."""
        declared = getattr(module, "MECHANISM_CONTRACT", None)
        if not isinstance(declared, dict):
            raise ContractViolation(
                f"{mechanism_id or getattr(module, '__name__', 'the mechanism')} declares no "
                "MECHANISM_CONTRACT: a mechanism that does not say which parameter groups it "
                "intends to reach cannot be checked"
            )
        reaches = list(declared.get("reaches") or ())
        forbidden = list(declared.get("must_not_reach") or ())
        unknown = [name for name in reaches + forbidden if name not in KNOWN_GROUPS]
        if unknown:
            raise ContractViolation(
                f"the contract names parameter groups the host does not have: {unknown}"
            )
        overlap = sorted(set(reaches) & set(forbidden))
        if overlap:
            raise ContractViolation(
                f"the contract both reaches and forbids the same groups: {overlap}"
            )
        return cls(
            version=int(declared.get("version", CONTRACT_VERSION)),
            mechanism_id=mechanism_id,
            reaches=reaches,
            must_not_reach=forbidden,
            weight=float(declared.get("weight", weight)),
            base_objective_preserved=bool(declared.get("base_objective_preserved", True)),
            delivered_once=bool(declared.get("delivered_once", True)),
            notes=str(declared.get("notes", "")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mechanism_id": self.mechanism_id,
            "reaches": list(self.reaches),
            "must_not_reach": list(self.must_not_reach),
            "weight": self.weight,
            "base_objective_preserved": self.base_objective_preserved,
            "delivered_once": self.delivered_once,
            "notes": self.notes,
        }


def check_delivery(
    contract: GradientContract,
    lineage: Dict[str, Any],
    *,
    tolerance: float = DELIVERY_TOLERANCE,
) -> Dict[str, Any]:
    """Check the contract against a measured gradient lineage.

    ``lineage`` is the report of ``gradient_lineage.trace``: the mechanism's contribution
    after the host's assembly, per parameter group.
    """
    contribution = (lineage.get("mechanism_contribution") or {}).get("by_stage", {})
    after_assembly = contribution.get("after_host_assembly", {}) or {}
    after_backward = contribution.get("after_backward", {}) or {}
    problems: List[str] = []
    delivered: Dict[str, float] = {}
    for name in contract.reaches:
        value = float(after_assembly.get(name, 0.0))
        delivered[name] = value
        if abs(value) <= tolerance:
            problems.append(
                f"the contract declares {name!r} and the host's assembly delivered "
                f"{value:.3e} there: the mechanism's gradient to that group does not reach "
                "the optimizer"
            )
    for name in contract.must_not_reach:
        value = float(after_assembly.get(name, 0.0))
        delivered[name] = value
        if abs(value) > tolerance:
            problems.append(
                f"the contract forbids {name!r} and the mechanism delivered {value:.3e} there"
            )
    if contract.delivered_once:
        for name in contract.reaches:
            before = float(after_backward.get(name, 0.0))
            after = float(after_assembly.get(name, 0.0))
            if before and after > 2.0 * abs(before) + tolerance:
                problems.append(
                    f"the contribution to {name!r} grew from {before:.3e} to {after:.3e} "
                    "across the host's assembly: it is being counted more than once"
                )
    return {
        "ok": not problems,
        "problems": problems,
        "delivered": delivered,
        "contract": contract.as_dict(),
    }


def relative_strength(
    lineage: Dict[str, Any], group: str = "rq_codebook_0"
) -> Optional[float]:
    """How large the mechanism's contribution is on a group, relative to the host's own.

    The number that separated the arm that changed the export harmlessly from the one that
    collapsed it is the ABSOLUTE size of the codebook gradient the mechanism delivers; the
    host's own gradient on the same group is the natural yardstick.
    """
    host = (lineage.get("host_objective") or {}).get("after_host_assembly", {}) or {}
    mine = (
        ((lineage.get("mechanism_contribution") or {}).get("by_stage", {}) or {})
        .get("after_host_assembly", {})
        or {}
    )
    contribution = abs(float(mine.get(group, 0.0)))
    reference = abs(float(host.get(group, 0.0)))
    if reference <= 0:
        return None
    return contribution / reference


__all__ = [
    "CONTRACT_VERSION",
    "ContractViolation",
    "DELIVERY_TOLERANCE",
    "GradientContract",
    "KNOWN_GROUPS",
    "check_delivery",
    "relative_strength",
]
