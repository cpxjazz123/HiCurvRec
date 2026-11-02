"""Failing-first tests for the corrected inference of Issue #137 (B1, B2)."""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import stats_core as core  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402


def exhaustive_p(x: np.ndarray, y: np.ndarray, statistic) -> float:
    observed = abs(statistic(x, y))
    exceed = 0
    total = 0
    for permutation in itertools.permutations(range(len(y))):
        total += 1
        if abs(statistic(x, y[list(permutation)])) >= observed - 1e-12:
            exceed += 1
    return exceed / total


class PermutationTests(unittest.TestCase):
    def test_every_draw_is_a_bijection(self):
        draws = core.permutation_bijections(7, 50, seed=3)
        self.assertEqual(draws.shape, (50, 7))
        for row in draws:
            self.assertEqual(sorted(row.tolist()), list(range(7)))

    def test_the_null_preserves_the_outcome_multiset(self):
        rng = np.random.default_rng(0)
        y = np.asarray([1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 9.0])
        for row in core.permutation_bijections(y.size, 20, seed=1):
            self.assertEqual(sorted(y[row].tolist()), sorted(y.tolist()))

    def test_perfect_positive_and_negative_ranks_hit_the_floor(self):
        x = np.arange(12, dtype=float)
        positive = core.permutation_p(x, x, draws=2000, seed=5)
        negative = core.permutation_p(x, -x, draws=2000, seed=5)
        for result in (positive, negative):
            self.assertEqual(result["spearman"], 1 / 2001)
            self.assertEqual(result["pearson"], 1 / 2001)
            self.assertEqual(result["draws"], 2000)
            self.assertFalse(result["exhaustive"])

    def test_exact_permutation_agrees_with_manual_enumeration_n5(self):
        rng = np.random.default_rng(2)
        x = np.arange(5, dtype=float)
        y = np.asarray([0.4, 2.1, 1.0, 3.7, 2.9])
        exact = core.exact_permutation_p(x, y)
        manual_rho = exhaustive_p(x, y, core.spearman)
        manual_r = exhaustive_p(x, y, core.pearson)
        self.assertAlmostEqual(exact["spearman"], manual_rho, places=12)
        self.assertAlmostEqual(exact["pearson"], manual_r, places=12)
        self.assertEqual(exact["permutations"], 120)
        self.assertTrue(exact["exhaustive"])

    def test_exact_permutation_handles_ties_n6(self):
        x = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
        y = np.asarray([0.5, 1.5, 2.5, 2.5, 4.0, 5.0])
        exact = core.exact_permutation_p(x, y)
        manual = exhaustive_p(x, y, core.spearman)
        self.assertAlmostEqual(exact["spearman"], manual, places=12)
        self.assertEqual(exact["permutations"], 720)

    def test_agreement_with_scipy_permutation_test(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=6)
        y = rng.normal(size=6)
        exact = core.exact_permutation_p(x, y)

        def statistic(a, b):
            return core.spearman(np.asarray(a), np.asarray(b))

        scipy_result = scipy_stats.permutation_test(
            (x, y), statistic, permutation_type="pairings", alternative="two-sided",
            n_resamples=np.inf)
        self.assertAlmostEqual(exact["spearman"], float(scipy_result.pvalue), places=10)

    def test_the_old_bootstrap_null_is_not_a_permutation_null(self):
        x = np.arange(10, dtype=float)
        y = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        correct = core.permutation_p(x, y, draws=5000, seed=4)["spearman"]
        rng = np.random.default_rng(4)
        observed = abs(core.spearman(x, y))
        exceed = 0
        for _ in range(50):
            picks = rng.integers(0, y.size, (100, y.size))       # the B1 defect
            sample = y[picks]
            centre = sample - sample.mean(axis=1, keepdims=True)
            denominator = np.sqrt((centre ** 2).sum(axis=1))
            values = (x - x.mean()) @ centre.T / (np.sqrt(((x - x.mean()) ** 2).sum())
                                                  * denominator)
            exceed += int((np.abs(values) >= observed - 1e-12).sum())
        bootstrap = (exceed + 1) / 5001
        self.assertNotAlmostEqual(correct, bootstrap, places=4)


class BootstrapTests(unittest.TestCase):
    def test_cis_are_paired_and_with_replacement(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=40)
        y = x + rng.normal(size=40) * 0.1
        result = core.bootstrap_ci(x, y, draws=2000, seed=2)
        self.assertEqual(result["resampling"], "with_replacement")
        self.assertEqual(result["draws"], 2000)
        self.assertGreater(result["pearson"][0], 0.9)
        self.assertGreater(result["spearman"][0], 0.9)
        repeat = core.bootstrap_ci(x, y, draws=2000, seed=2)
        self.assertEqual(result, repeat)

    def test_identical_inputs_give_a_degenerate_ci(self):
        x = np.arange(15, dtype=float)
        result = core.bootstrap_ci(x, x, draws=1000, seed=1)
        self.assertAlmostEqual(result["pearson"][0], 1.0, places=9)
        self.assertAlmostEqual(result["spearman"][1], 1.0, places=9)


class HolmTests(unittest.TestCase):
    def test_textbook_oracle(self):
        adjusted = core.holm([0.01, 0.02, 0.04])
        self.assertEqual([round(value, 10) for value in adjusted], [0.03, 0.04, 0.04])

    def test_ties_are_monotone_and_bounded(self):
        adjusted = core.holm([0.02, 0.02, 0.5])
        self.assertTrue(all(a <= 1.0 for a in adjusted))
        self.assertLessEqual(adjusted[0], adjusted[1])
        self.assertLessEqual(adjusted[1], adjusted[2])

    def test_columns_are_adjusted_independently(self):
        pearson = [0.001, 0.2, 0.3]
        spearman = [0.4, 0.003, 0.0005]
        holm_r, holm_rho = core.holm_columns(pearson, spearman)
        self.assertEqual([round(v, 10) for v in holm_r], [0.003, 0.4, 0.4])
        self.assertEqual([round(v, 10) for v in holm_rho], [0.4, 0.006, 0.0015])
        self.assertNotAlmostEqual(holm_r[0], holm_rho[0])

    def test_the_mixed_minimum_column_is_not_reused(self):
        # a test whose Pearson p is small but Spearman p is large must not carry
        # the Pearson adjustment into the Spearman column
        pearson = [0.0001, 0.5]
        spearman = [0.5, 0.0001]
        holm_r, holm_rho = core.holm_columns(pearson, spearman)
        self.assertAlmostEqual(holm_r[0], 0.0002)
        self.assertAlmostEqual(holm_r[1], 0.5)
        self.assertAlmostEqual(holm_rho[0], 0.5)
        self.assertAlmostEqual(holm_rho[1], 0.0002)

    def test_family_counts(self):
        family = core.family_counts(potential_metric_levels=66, outcomes=4,
                                    defined_metric_levels=65)
        self.assertEqual(family["potential"], 528)
        self.assertEqual(family["defined"], 520)
        self.assertEqual(family["undefined"], 8)

    def test_520_test_gate(self):
        # 260 defined metric/outcome entries, two coefficient columns each
        rng = np.random.default_rng(0)
        p_values = np.clip(rng.random(260) ** 4, 1e-5, 1.0)
        holm_r, holm_rho = core.holm_columns(p_values.tolist(), p_values.tolist())
        self.assertEqual(len(holm_r), 260)
        self.assertEqual(len(holm_rho), 260)
        # the smallest raw p cannot be adjusted below raw
        self.assertGreaterEqual(min(holm_r), min(p_values))
        self.assertAlmostEqual(holm_r[0], holm_rho[0])


class RankTests(unittest.TestCase):
    def test_midranks(self):
        self.assertEqual(core.midranks(np.array([3.0, 1.0, 2.0, 2.0])).tolist(),
                         [3.0, 0.0, 1.5, 1.5])

    def test_spearman_is_pearson_on_midranks_with_ties(self):
        x = np.asarray([1.0, 2.0, 2.0, 3.0, 5.0])
        y = np.asarray([2.0, 2.0, 4.0, 4.0, 9.0])
        self.assertAlmostEqual(core.spearman(x, y),
                               core.pearson(core.midranks(x), core.midranks(y)),
                               places=12)


if __name__ == "__main__":
    unittest.main()
