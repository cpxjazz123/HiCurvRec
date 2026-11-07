"""Two admissions of the same tree must not share a scratch directory.

The admission copies the candidate tree before it runs the entry check and
deletes the copy afterwards. The scratch name used to be derived from the tree
alone, so a dispatch that ran while a manual check was under way reused the
directory: the first to finish deleted the copy the second was still reading,
and the second reported a missing source file that looked like a defect of the
candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import checks  # noqa: E402


def test_two_scratch_directories_for_one_tree_are_distinct() -> str:
    tree = REPO / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0"
    if not tree.is_dir():
        return "the candidate tree is not present, nothing to check"
    first = checks._admission_tree(tree)
    second = checks._admission_tree(tree)
    try:
        if first == second:
            raise AssertionError("two admission runs shared one scratch directory: " + str(first))
        if not first.is_dir() or not second.is_dir():
            raise AssertionError("a scratch directory was not created")
    finally:
        checks._discard_admission_tree(first)
        checks._discard_admission_tree(second)
    return "each admission run gets its own scratch directory"


def main() -> int:
    failures = 0
    try:
        print("PASS test_two_scratch_directories_for_one_tree_are_distinct: "
              + test_two_scratch_directories_for_one_tree_are_distinct())
    except AssertionError as exc:
        failures += 1
        print("FAIL test_two_scratch_directories_for_one_tree_are_distinct: " + str(exc))
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print("ERROR: " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
