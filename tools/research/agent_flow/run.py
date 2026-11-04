"""Run the phases of a round.

Usage:
    python3 -m tools.research.agent_flow.run --round <round_id> [--phases ideation,probe,scoring]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.research.agent_flow.flow import PHASES, run_flow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, help="the round id")
    parser.add_argument("--objective", default="", help="the round objective")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--phases", default=",".join(PHASES),
                        help="the phases to run, comma separated")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    phases = tuple(part.strip() for part in args.phases.split(",") if part.strip())
    outcome = run_flow(repo, args.round, objective=args.objective, phases=phases)
    print(json.dumps(outcome, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
