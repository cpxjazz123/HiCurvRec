"""The MVG utilities, exposed as judges Harness2 can select.

The gates were written as a mandatory sequence: M0 to M5, every one of them, for every
mechanism. Issue #102 retires that shape. What survives is the *content* of the checks --
an independent numerical oracle, derivative evidence, declared properties, an on/off
integration proof -- and the rule that the code under test may not edit any of it.

So the checks are registered here as named judges, and a mechanism's own frozen
declaration says which of them apply to it. Harness2 selects from that list, runs the
selected judges, and treats a missing or failed critical judge as a block.

Nothing here decides whether a mechanism is good. The judges measure; the controller
decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from . import gates as G
from . import spec as SPEC


@dataclass(frozen=True)
class Judge:
    """One independent check, with the mechanism features it applies to."""

    id: str
    question: str
    applies_to: Tuple[str, ...]
    run: Callable[[G.MechanismUnderTest], G.GateResult]
    critical: bool = True


JUDGES: Tuple[Judge, ...] = (
    Judge(
        id="contract_audit",
        question=(
            "does the implementation agree with the frozen declaration about what it is, and "
            "does it contain a substitution the declaration forbids?"
        ),
        applies_to=("forbidden_substitutions", "integration"),
        run=G.gate_m0,
    ),
    Judge(
        id="forward_oracle",
        question=(
            "does the implementation compute the declared objective, forward value and "
            "intermediates, where an independent implementation computes the same numbers?"
        ),
        applies_to=("objective",),
        run=G.gate_m1,
    ),
    Judge(
        id="derivative_evidence",
        question=(
            "is the gradient the derivative of the declared loss, does it reach the declared "
            "targets, and does it survive the degenerate cases?"
        ),
        applies_to=("gradient_targets",),
        run=G.gate_m2,
    ),
    Judge(
        id="property_checks",
        question="do the declared metamorphic and semantic properties hold?",
        applies_to=("invariants",),
        run=G.gate_m3,
    ),
    Judge(
        id="integration_on_off",
        question=(
            "is the mechanism-off tree the frozen host, and does the mechanism-on tree move "
            "the parameters the export reads?"
        ),
        applies_to=("integration",),
        run=G.gate_m4,
    ),
)

BY_ID: Dict[str, Judge] = {judge.id: judge for judge in JUDGES}

#: The registry and the validator must agree on the names; a declaration that asks for a
#: judge the validator accepts but the registry does not have would fail late.
_missing = sorted(set(SPEC.KNOWN_JUDGES) - set(BY_ID))
if _missing:  # pragma: no cover - a development error, not a runtime condition
    raise RuntimeError(f"the judge registry is missing {_missing}")


class JudgeSelectionError(ValueError):
    """A declaration that asks for a judge that does not exist."""


def select_judges(
    spec_doc: dict,
    *,
    required: Sequence[str] = (),
) -> List[Judge]:
    """The judges a mechanism's own declaration calls for.

    The declaration may name judges explicitly. When it does not, the selection follows the
    declaration's structure: a mechanism with an objective needs the forward oracle, one
    with gradient targets needs the derivative evidence, and every mechanism needs the
    contract audit. ``required`` adds judges the controller cannot do without.
    """
    named = list(spec_doc.get("judges") or [])
    unknown = [name for name in named if name not in BY_ID]
    if unknown:
        raise JudgeSelectionError(
            f"the declaration asks for judge(s) that do not exist: {unknown}; "
            f"known judges are {sorted(BY_ID)}"
        )
    chosen = set(named) | set(required)
    if not chosen:
        if spec_doc.get("objective"):
            chosen.add("forward_oracle")
        if spec_doc.get("gradient_targets"):
            chosen.add("derivative_evidence")
        if spec_doc.get("invariants"):
            chosen.add("property_checks")
        if spec_doc.get("integration"):
            chosen.add("integration_on_off")
    chosen.add("contract_audit")
    return [judge for judge in JUDGES if judge.id in chosen]


def run_selected(
    mut: G.MechanismUnderTest,
    judges: Sequence[Judge],
) -> Dict[str, G.GateResult]:
    """Run the selected judges. A judge that cannot run is ``blocked``, never a pass."""
    results: Dict[str, G.GateResult] = {}
    for judge in judges:
        try:
            results[judge.id] = judge.run(mut)
        except Exception as exc:  # pragma: no cover - defensive
            results[judge.id] = G.GateResult(
                "?", G.BLOCKED, f"the judge could not run: {type(exc).__name__}: {exc}", {}, [str(exc)]
            )
    return results


def verdict(results: Dict[str, G.GateResult]) -> str:
    """``passed`` only when every selected judge passed."""
    if not results:
        return "blocked"
    states = {result.state for result in results.values()}
    if G.FAILED in states:
        return "failed"
    if G.BLOCKED in states:
        return "blocked"
    return "passed"


__all__ = [
    "BY_ID",
    "JUDGES",
    "Judge",
    "JudgeSelectionError",
    "run_selected",
    "select_judges",
    "verdict",
]
