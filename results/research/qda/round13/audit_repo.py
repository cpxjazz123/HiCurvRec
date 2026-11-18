"""Read-only audit: which trees in this repository are protected against codebook collapse.

`selfcheck.py` installs the guard, which is right for a tree that is about to train. This
audit must not mutate anything, so it runs `verify_tree` alone: an unpatched tree reports
NOT PROTECTED because it genuinely has no floor, and that is the point of the table.

    python results/research/qda/round13/audit_repo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "research"))

from stage2.anticollapse import ReadOnlyBaseline, UnknownTree, verify_tree  # noqa: E402

CANDIDATES = [
    *sorted((REPO / "stage2_RQ-VAE").glob("TIGER_*")),
    *sorted((REPO / "stage2_RQ-VAE" / "baseline").glob("TIGER_*")),
    *sorted((REPO / "results" / "research" / "qda" / "trees").glob("*")),
]


def main() -> int:
    rows = []
    for tree in CANDIDATES:
        if not tree.is_dir():
            continue
        try:
            verdict = verify_tree(tree)
        except ReadOnlyBaseline:
            rows.append((tree.name, "READ-ONLY", "the frozen copy source, never trained"))
            continue
        except UnknownTree as error:
            text = str(error).splitlines()[0]
            # A directory that is not a tree at all (a leftover log dir) is not a candidate.
            status = "NOT A TREE" if "no known quantiser" in text else "NOT PROTECTED"
            rows.append((tree.name, status, text[:110]))
            continue
        if verdict["ok"]:
            rows.append((tree.name, "PROTECTED",
                         " ".join(f"{name} {entry['used']}/{entry['used_forward']}/"
                                  f"{entry['used_balanced']}"
                                  for name, entry in verdict["layers"].items())))
        else:
            rows.append((tree.name, "NOT PROTECTED", verdict["reason"][:110]))
    width = max(len(row[0]) for row in rows) if rows else 10
    for name, status, detail in rows:
        print(f"{name:<{width}}  {status:<14}  {detail}")
    protected = sum(1 for _, status, _ in rows if status == "PROTECTED")
    print(f"\n{protected}/{len(rows)} trees protected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
