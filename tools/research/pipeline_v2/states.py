"""The states of one candidate, and the transitions between them.

Seven states, mutually exclusive, with one rule that the old loop broke: an
infrastructure failure (a timeout, a refused read, a crashed service) says nothing
about the mechanism and is never filed as a defect of the candidate.
"""
from __future__ import annotations

IMPLEMENTED = "IMPLEMENTED"            # the source snapshot is frozen
TEST_FAILED = "TEST_FAILED"            # a deterministic check failed
REVIEW_REQUIRED = "REVIEW_REQUIRED"    # tests passed, the science review is next
NEEDS_FIX = "NEEDS_FIX"                # a reviewer named a defect with evidence
VERIFIED = "VERIFIED"                  # tests and the review agree
INFRA_ERROR = "INFRA_ERROR"            # a timeout, a refused read, a crash
REJECTED = "REJECTED"                  # the mechanism is wrong, or the budget ran out

ALL = (IMPLEMENTED, TEST_FAILED, REVIEW_REQUIRED, NEEDS_FIX, VERIFIED,
       INFRA_ERROR, REJECTED)

#: Which state may follow which. A transition that is not here is a bug, not a
#: judgement call.
TRANSITIONS = {
    # A new source snapshot while a candidate is already IMPLEMENTED is a new
    # version of the same candidate, not a violation.
    IMPLEMENTED: {IMPLEMENTED, TEST_FAILED, REVIEW_REQUIRED, INFRA_ERROR},
    TEST_FAILED: {IMPLEMENTED, REJECTED, INFRA_ERROR},
    REVIEW_REQUIRED: {VERIFIED, NEEDS_FIX, INFRA_ERROR},
    NEEDS_FIX: {IMPLEMENTED, REJECTED, INFRA_ERROR},
    VERIFIED: {REJECTED},
    INFRA_ERROR: {IMPLEMENTED, REVIEW_REQUIRED, TEST_FAILED, NEEDS_FIX},
    REJECTED: set(),
}

#: The error classes are exclusive: a source defect, a scientific objection, an
#: infrastructure fault and a budget stop are different facts.
ERROR_CLASSES = {
    "source_defect": {TEST_FAILED},
    "scientific_objection": {NEEDS_FIX, REJECTED},
    "infrastructure": {INFRA_ERROR},
    "budget": {REJECTED},
}

#: How many times the automatic loop may repair one candidate before a human
#: decision. The old loop had no bound that a slow reviewer could not extend.
REPAIR_BUDGET = 2


class TransitionError(RuntimeError):
    """A transition the state machine does not allow."""


def check(current: str, target: str) -> str:
    """Refuse a transition that is not in the table."""
    if current not in ALL:
        raise TransitionError("unknown state: " + str(current))
    if target not in TRANSITIONS.get(current, set()):
        raise TransitionError("cannot move from " + current + " to " + target)
    return target


def error_class(state: str) -> str:
    """The class a state belongs to, or ``""`` for a working state."""
    for name, members in ERROR_CLASSES.items():
        if state in members:
            return name
    return ""
