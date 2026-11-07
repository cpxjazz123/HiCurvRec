"""The admission copy must survive a tree that carries a dangling symlink.

Every candidate tree inherits the frozen dataset directory, whose
``item_emb.parquet`` is a symlink to a file that is not in the repository. The
admission copies the tree to a scratch directory before it runs the entry
check, and a plain copy follows the link and fails on the missing target, so
the admission could not run on any tree at all -- the frozen baseline included.

This test builds a tree with exactly that shape and asserts the copy succeeds
and keeps the link a link.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import checks  # noqa: E402


def test_the_admission_copy_survives_a_dangling_symlink() -> str:
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        tree = work / "TIGER_copy_probe"
        (tree / "dataset" / "Amazon_2023_Instruments").mkdir(parents=True)
        (tree / "train_rqvae.py").write_text("# probe\n", encoding="utf-8")
        link = tree / "dataset" / "Amazon_2023_Instruments" / "item_emb.parquet"
        link.symlink_to(work / "does_not_exist.parquet")
        if not link.is_symlink():
            raise AssertionError("the probe did not build a symlink")
        copied = checks._admission_tree(tree)
        try:
            copied_link = copied / "dataset" / "Amazon_2023_Instruments" / "item_emb.parquet"
            if not copied_link.is_symlink():
                raise AssertionError("the copy followed the link instead of keeping it")
            if copied_link.resolve() != link.resolve():
                raise AssertionError("the copy points somewhere else")
        finally:
            checks._discard_admission_tree(copied)
    return "the admission copy keeps a dangling symlink a symlink"


def main() -> int:
    failures = 0
    for test in (test_the_admission_copy_survives_a_dangling_symlink,):
        try:
            print("PASS " + test.__name__ + ": " + test())
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
