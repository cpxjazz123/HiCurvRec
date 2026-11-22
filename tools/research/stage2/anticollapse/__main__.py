"""`python -m anticollapse <tree> [--family mixed|euclid] [--floor N] [--no-force] [--check-only]`

Patch a Stage2 candidate tree so its assignment can never leave a codeword empty, then prove
it by running the patched quantiser on a collapsed cost matrix in a fresh interpreter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .patch_tree import FAMILIES, UnknownTree, apply_floor, verify_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path, help="Stage2 candidate tree to protect")
    parser.add_argument("--family", choices=sorted(FAMILIES), default=None,
                        help="quantiser family (inferred from the tree when omitted)")
    parser.add_argument("--floor", type=int, default=1,
                        help="items each codeword must receive (default 1)")
    parser.add_argument("--no-force", action="store_true",
                        help="keep the Sinkhorn path instead of replacing it everywhere")
    parser.add_argument("--check-only", action="store_true",
                        help="verify the tree without patching it")
    args = parser.parse_args(argv)

    try:
        if not args.check_only:
            report = apply_floor(args.tree, args.family,
                                 force_constrained=not args.no_force, floor=args.floor)
            print(json.dumps(report, indent=2, sort_keys=True))
        verdict = verify_tree(args.tree, args.family)
    except UnknownTree as error:
        print(f"anticollapse: {error}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
