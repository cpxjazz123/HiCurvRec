"""The phases of a round, in order, with the round's own state machine."""
from __future__ import annotations

from pathlib import Path

from tools.research.rounds import worker

PHASES = ("ideation", "probe", "scoring", "derivation")


def run_flow(root: Path, round_id: str, *, objective: str = "",
             extra: bool = False, phases=PHASES, log=print) -> dict:
    """Run the requested phases of a round and report what each of them did.

    The phases are the framework's own: they run the agents (through the OpenAI
    Agents SDK), merge their answers with the round's state machine, and persist
    the result. Nothing is orchestrated by a graph engine: the order and the
    boundaries are the round's.
    """
    wanted = [phase for phase in PHASES if phase in phases]
    if not wanted:
        raise ValueError("no phase selected")
    results = {}
    for phase in wanted:
        if phase == "ideation":
            results["ideation"] = worker.phase_ideation(Path(root), round_id, log=log)
        elif phase == "probe":
            results["probe"] = worker.phase_probe(Path(root), round_id, log=log)
        elif phase == "derivation":
            results["derivation"] = worker.phase_derivation(Path(root), round_id, log=log)
        else:
            results["scoring"] = worker.phase_scoring(Path(root), round_id, log=log)
    return {"round_id": round_id, "phases": results}
