"""The states of one candidate, and the transitions between them.

Five states, mutually exclusive. The scientific case for a mechanism is argued
during ideation and scoring, by independent research agents, before it is
implemented; verification here answers only whether the CODE is correct, and a
program answers that. There is no review agent after the implementation.

The rule the old loop broke stays: an infrastructure failure (a timeout, a missing
GPU, a crashed service) says nothing about the mechanism and is never filed as a
defect of the candidate.
"""
from __future__ import annotations

IMPLEMENTED = "IMPLEMENTED"            # the source snapshot is frozen
TEST_FAILED = "TEST_FAILED"            # a deterministic check failed
VERIFIED = "VERIFIED"                  # every deterministic check passed
INFRA_ERROR = "INFRA_ERROR"            # a timeout, a missing device, a crash
REJECTED = "REJECTED"                  # the mechanism is wrong, or the budget ran out

ALL = (IMPLEMENTED, TEST_FAILED, VERIFIED, INFRA_ERROR, REJECTED)

#: Which state may follow which. A transition that is not here is a bug, not a
#: judgement call. IMPLEMENTED -> IMPLEMENTED is a new source version of the same
#: candidate, which is not a violation.
TRANSITIONS = {
    IMPLEMENTED: {IMPLEMENTED, TEST_FAILED, VERIFIED, INFRA_ERROR},
    TEST_FAILED: {IMPLEMENTED, REJECTED, INFRA_ERROR},
    VERIFIED: {REJECTED},
    INFRA_ERROR: {IMPLEMENTED, TEST_FAILED, VERIFIED},
    REJECTED: set(),
}

#: The error classes are exclusive: a source defect, an infrastructure fault and a
#: rejection are different facts and never convert into one another.
ERROR_CLASSES = {
    "source_defect": {TEST_FAILED},
    "infrastructure": {INFRA_ERROR},
    "rejection": {REJECTED},
}

#: How many times the automatic loop may repair one candidate before a human
#: decision.
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
