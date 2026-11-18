"""One command that answers: is this Stage2 tree protected against codebook collapse?

    python tools/research/stage2/anticollapse/selfcheck.py <tree> [--sid <item_sids.json>]

Exit code 0 means every check that can be made from outside the mechanism passed:

    install     the floor is present (installed now if it was missing)
    probe       the patched quantiser covers every codeword in all three regimes:
                plain argmin, the baseline's balanced argmax, and the full forward
    artifact    the exported identifier uses at least two codewords at every level
                (skipped, and reported as such, when no identifier is given)

Exit code 1 prints the reason for the first failure and never a summary that hides it.
The command writes nothing outside the tree it is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # run as a file: put `tools/research` on the path so `stage2.anticollapse` resolves
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.anticollapse import (ReadOnlyBaseline, UnknownTree, apply_floor,  # noqa: E402
                                 audit_sid, verify_tree)


def check(tree: Path, sid: Path | None = None) -> dict:
    report: dict = {"tree": str(tree)}
    try:
        report["install"] = apply_floor(tree)
    except ReadOnlyBaseline as error:
        report["install"] = {"ok": False, "detail": str(error)}
        report["ok"] = False
        return report
    except UnknownTree as error:
        report["install"] = {"ok": False, "detail": str(error)}
        report["ok"] = False
        return report

    try:
        verdict = verify_tree(tree)
    except UnknownTree as error:
        report["probe"] = {"ok": False, "detail": str(error)}
        report["ok"] = False
        return report
    report["probe"] = verdict
    if not verdict["ok"]:
        report["ok"] = False
        return report

    if sid is not None:
        artifact = audit_sid(sid)
        report["artifact"] = artifact
        if artifact["status"] != "PASS":
            report["ok"] = False
            return report
    else:
        report["artifact"] = {"status": "SKIPPED", "detail": "no identifier given"}
    report["ok"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", type=Path)
    parser.add_argument("--sid", type=Path, default=None,
                        help="the exported item_sids.json to audit")
    args = parser.parse_args(argv)
    result = check(args.tree.resolve(), args.sid)
    print(json.dumps(result, indent=1, sort_keys=True))
    if result.get("ok"):
        print("PROTECTED: " + str(args.tree))
        return 0
    print("NOT PROTECTED: " + str(args.tree))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
