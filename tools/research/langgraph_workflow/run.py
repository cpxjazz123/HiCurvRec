"""Run the first three phases of a round through the LangGraph graph.

Usage:
    python3 -m tools.research.langgraph_workflow.run --round <round_id> \
        [--objective "<text>"] [--repo <path>]

The round keeps its own directory, its own state file and its own artefacts; the
graph only decides who runs in parallel and in which order the phases run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.research.langgraph_workflow.graph import default_checkpointer, run_phase1
from tools.research.rounds import pipeline as pipeline_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, help="the round id")
    parser.add_argument("--objective", default="", help="the round objective")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--extra", action="store_true",
                        help="ask the ideation agents for one more idea each")
    parser.add_argument("--phases", default="ideation,probe,scoring",
                        help="the phases to run, comma separated")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    phases = tuple(part.strip() for part in args.phases.split(",") if part.strip())
    checkpointer = default_checkpointer()
    try:
        state = run_phase1(repo, args.round, objective=args.objective,
                            extra=args.extra, checkpointer=checkpointer,
                            phases=phases)
    finally:
        closer = getattr(checkpointer, "close", None)
        if closer is not None:
            closer()
    print(json.dumps({
        "round_id": state.get("round_id"),
        "ideas": len(state.get("ideas") or []),
        "reports": len(state.get("reports") or []),
        "errors": list(state.get("errors") or []),
        "selection": (state.get("selection") or {}).get("status"),
        "step": (state.get("round_state") or {}).get("step"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
