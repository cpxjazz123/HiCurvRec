"""The state the research graph carries between its nodes."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    """What the research graph knows while a round is in its first three phases.

    The list fields use a reducer, so a node that answers with one agent's
    result adds to the state instead of replacing it: that is what makes the
    parallel branches of the graph compose.
    """

    round_id: str
    objective: str
    root: str
    extra: bool
    per_agent: Annotated[list, operator.add]
    reports: Annotated[list, operator.add]
    errors: Annotated[list[str], operator.add]
    ideas: list
    bundles: dict
    selection: dict
    probe_phase: dict
    round_state: dict
    phase_results: dict
