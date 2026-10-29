"""The command line of Pipeline V2: one candidate, one verification, one record.

The checks take minutes (a two-rank run, three negative controls, the mechanism
contract), so the MCP tool starts this in the background and the caller polls the
record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .controller import Controller
from .run import ROOT, verify_candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Research Pipeline V2 verification")
    parser.add_argument("tree", help="the candidate tree to verify")
    parser.add_argument("--record-dir", default=None,
                        help="where the record lives (default results/research/pipeline_v2/<arm>)")
    parser.add_argument("--checks", default="",
                        help="comma separated subset of the checks to run")
    arguments = parser.parse_args(argv)
    tree = Path(arguments.tree).resolve()
    controller = Controller(Path(arguments.record_dir) if arguments.record_dir
                            else (ROOT / tree.name))
    checks = [item for item in (arguments.checks or "").split(",") if item] or None
    record = verify_candidate(tree, checks=checks, controller=controller)
    print(json.dumps({"state": record.get("state"),
                      "version": record.get("version"),
                      "source_digest": record.get("source_digest"),
                      "checks": {name: entry.get("status") for name, entry in
                                 (record.get("tests", {}).get("checks") or {}).items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
