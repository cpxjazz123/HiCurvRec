"""The mechanism-specific independent checks, as a command.

The checks need torch and the campaign needs the framework's interpreter, and the
framework needs Python 3.12. The two are separated by a subprocess and a JSON report,
which is the same boundary the rest of this repository uses: nothing crosses it except
data.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/invariant_check.py --tree <arm> --proposal <proposal.json> --out <report.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mvg import invariants  # noqa: E402


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.mvg.invariant_check", description=__doc__)
    parser.add_argument("--tree", required=True, help="the arm tree to check")
    parser.add_argument("--proposal", required=True, help="the frozen proposal (JSON)")
    parser.add_argument("--module", default="", help="the mechanism module's name")
    parser.add_argument("--entrypoint", default="", help="the mechanism's entry point")
    parser.add_argument("--out", default="", help="where to write the report")
    args = parser.parse_args(argv)

    tree = Path(args.tree)
    if not tree.is_absolute():
        tree = REPO_ROOT / tree
    proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
    report = invariants.verify_mechanism(
        tree, proposal, module=args.module, entrypoint=args.entrypoint
    )
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
