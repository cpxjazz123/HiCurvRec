"""The phases of a round, in order, with the round's own state machine."""
from __future__ import annotations

from pathlib import Path

from tools.research.rounds import worker

PHASES = ("ideation", "probe", "scoring", "formulation", "coding",
          "derivation", "implementation")


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
    # One runner per round. The MCP takes this lock for the phases it drives;
    # the runner has to take it too, or two runs of the same round overlap and
    # both write the state. A run that finds the round busy is refused, not
    # silently queued: the caller knows what it asked for.
    from tools.research.rounds import lock as round_lock
    from tools.research.rounds import pipeline as round_pipeline

    directory = round_pipeline.find_round_directory(Path(root), round_id)
    guard = round_lock.RoundLock(directory)
    try:
        guard.acquire(owner="runner:" + ",".join(wanted))
    except round_lock.RoundBusy as exc:
        raise SystemExit("the round is already running: " + str(exc))
    try:
        return _run_phases(Path(root), round_id, wanted, objective=objective, log=log)
    finally:
        guard.release()


def _run_phases(root: Path, round_id: str, wanted: list, *, objective: str = "",
                log=print) -> dict:
    results = {}
    for phase in wanted:
        if phase == "ideation":
            results["ideation"] = worker.phase_ideation(Path(root), round_id, log=log)
        elif phase == "probe":
            results["probe"] = worker.phase_probe(Path(root), round_id, log=log)
        elif phase == "formulation":
            results["formulation"] = worker.phase_formulation(Path(root), round_id, log=log)
        elif phase == "coding":
            results["coding"] = worker.phase_coding(Path(root), round_id, log=log)
        elif phase == "derivation":
            results["derivation"] = worker.phase_derivation(Path(root), round_id, log=log)
        elif phase == "implementation":
            results["implementation"] = worker.phase_implementation(Path(root), round_id, log=log)
        else:
            results["scoring"] = worker.phase_scoring(Path(root), round_id, log=log)
    return {"round_id": round_id, "phases": results}
