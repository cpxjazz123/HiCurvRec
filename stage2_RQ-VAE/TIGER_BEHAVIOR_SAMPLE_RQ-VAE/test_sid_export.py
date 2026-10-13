"""E3 audit of the SID export path and the prefix structure it hands to Stage 3.

``_extend_collisions`` appends a fourth token that is a pure function of an item's
position inside its (L1, L2, L3) collision group -- ``768 + occurrence`` where 768 is
the sum of the three codebook sizes. It carries nothing the quantiser learned, and the
export asserts that the result is unique, so a collapsed codebook cannot show up as
duplicate SIDs. It shows up as a large fraction of items whose fourth token is an
arbitrary disambiguator instead of the constant 768 that singletons receive.

These tests pin that behaviour, because every Stage 3 number in this family is read
through this function.
"""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

import train_rqvae


class CollisionExtensionTests(unittest.TestCase):
    def test_t1_all_rows_are_unique_after_extension(self):
        tokens = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6]])
        extended = train_rqvae._extend_collisions(tokens, [256, 256, 256])
        self.assertEqual(extended.shape, (4, 4))
        self.assertEqual(len(np.unique(extended, axis=0)), 4)

    def test_t2_the_extra_level_is_the_group_offset_not_a_learned_code(self):
        tokens = np.array([[7, 7, 7], [7, 7, 7], [9, 9, 9]])
        extended = train_rqvae._extend_collisions(tokens, [256, 256, 256])
        # Singletons sit at the base offset; later members of a group step by one in
        # the order they appear, which is the only thing the extra level encodes.
        self.assertEqual(list(extended[0]), [7, 7, 7, 768])
        self.assertEqual(list(extended[1]), [7, 7, 7, 769])
        self.assertEqual(list(extended[2]), [9, 9, 9, 768])

    def test_t3_the_first_three_levels_are_passed_through_unchanged(self):
        rng = np.random.default_rng(42)
        tokens = rng.integers(0, 256, size=(500, 3))
        extended = train_rqvae._extend_collisions(tokens, [256, 256, 256])
        np.testing.assert_array_equal(extended[:, :3], tokens)

    def test_t4_singletons_all_carry_the_same_fourth_token(self):
        tokens = np.arange(300, dtype=np.int64).reshape(100, 3) % 256
        extended = train_rqvae._extend_collisions(tokens, [256, 256, 256])
        self.assertTrue(np.all(extended[:, 3] == 768))

    def test_t5_extension_is_deterministic(self):
        tokens = np.array([[3, 3, 3], [3, 3, 3], [5, 1, 2], [3, 3, 3]])
        first = train_rqvae._extend_collisions(tokens, [256, 256, 256])
        second = train_rqvae._extend_collisions(tokens, [256, 256, 256])
        np.testing.assert_array_equal(first, second)

    def test_t6_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            train_rqvae._extend_collisions(np.zeros((4, 2), dtype=np.int64), [256, 256, 256])

    def test_t7_a_collapsed_codebook_shows_up_as_a_disambiguation_burden(self):
        # 100 items squeezed into two prefixes produce two groups of fifty, and every
        # one of them needs a fourth token to be told apart -- the structural signal
        # that separates a collapsed arm from a healthy one, since the export itself
        # guarantees uniqueness either way.
        collapsed = np.tile(np.array([[1, 1, 1], [2, 2, 2]]), (50, 1))
        extended = train_rqvae._extend_collisions(collapsed, [256, 256, 256])
        group_sizes = sorted(Counter(map(tuple, extended[:, :3].tolist())).values())
        self.assertEqual(group_sizes, [50, 50])
        # 768 does NOT mean "singleton": the first member of any group lands there too,
        # which is exactly why the singleton count has to come from the group sizes and
        # not from the token value. Two of a hundred rows carry it here.
        self.assertEqual(float(np.mean(extended[:, 3] == 768)), 0.02)
        self.assertEqual(sorted(set(extended[:, 3].tolist())), list(range(768, 818)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
