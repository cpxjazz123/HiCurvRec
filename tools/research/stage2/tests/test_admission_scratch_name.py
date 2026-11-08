"""Regression: the entry check and the identifier check must share one scratch.

The admission copy's name grew a pid and a uuid when the copy became per-run, and
the pattern that turns a copy name back into the tree name kept stripping only the
digest. The entry check then wrote its identifier into
``_conformance/<tree>_<digest>_<pid>_<uuid>/`` while the identifier check read
``_conformance/<tree>/`` and reported NOT_RUN on a tree that had exported one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import checks  # noqa: E402


class ScratchNameTest(unittest.TestCase):
    def test_the_per_run_suffix_resolves_to_the_tree_name(self):
        name = "_admission_TIGER_demo_d1816c1ce6_2875846_a08b59b8"
        self.assertEqual(checks._DIGEST_SUFFIX.sub("", name), "_admission_TIGER_demo")

    def test_the_plain_digest_still_resolves(self):
        self.assertEqual(checks._DIGEST_SUFFIX.sub("", "_admission_TIGER_demo_d1816c1ce6"),
                         "_admission_TIGER_demo")

    def test_a_tree_name_with_hex_like_tail_is_untouched(self):
        self.assertEqual(checks._DIGEST_SUFFIX.sub("", "TIGER_mixed_rq_F0"), "TIGER_mixed_rq_F0")

    def test_both_checks_agree_on_the_scratch_directory(self):
        tree = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
        copy = REPO / "stage2_RQ-VAE" / "_admission_TIGER_RQ-VAE_d1816c1ce6_1234_deadbeef"
        self.assertEqual(checks._scratch_dir(tree).name, checks._scratch_dir(copy).name)


if __name__ == "__main__":
    unittest.main()
