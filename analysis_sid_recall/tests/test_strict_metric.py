"""The strict structural score must reject what the loose one pays partial credit for.

Constructed distances only: every case states which pairs are near and which are far, so the expected
loose and strict scores follow from the definition rather than from a run.
"""

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis_under_test", HERE.parent / "analysis.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def distance_from(table):
    return lambda left, right: table[(min(left, right), max(left, right))]


def test_two_group_strict_requires_every_near_pair_closer():
    instance = {"nodes": (0, 1, 2), "core": [(0, 1), (0, 2)], "tail": [(1, 2)]}
    groups = study.instance_distances("K4_tail", instance, instance["nodes"],
                                      distance_from({(0, 1): 1.0, (0, 2): 3.0, (1, 2): 2.0}))
    assert study.hypothesis("K4_tail", groups) == 0.5          # means are both 2.0
    assert study.strict_indicator("K4_tail", groups) == 0.0    # one core pair sits beyond a tail pair

    tight = study.instance_distances("K4_tail", instance, instance["nodes"],
                                     distance_from({(0, 1): 1.0, (0, 2): 1.5, (1, 2): 3.0}))
    assert study.hypothesis("K4_tail", tight) == 1.0
    assert study.strict_indicator("K4_tail", tight) == 1.0


def test_p5_strict_requires_the_whole_chain():
    instance = {"nodes": (0, 1, 2, 3, 4),
                "D1": [(0, 1)], "D2": [(0, 2)], "D3": [(0, 3)], "D4": [(0, 4)]}
    table = {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 4.0}
    groups = study.instance_distances("P5_hops", instance, instance["nodes"], distance_from(table))
    assert study.hypothesis("P5_hops", groups) == 1.0
    assert study.strict_indicator("P5_hops", groups) == 1.0

    # D2 and D3 each contain a pair that inverts the chain: the means still increase, the strict
    # ordering does not hold, which is exactly the 2/3-credit case the review described.
    broken = {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 4.0,
              (2, 3): 9.0}
    instance_broken = {"nodes": (0, 1, 2, 3, 4),
                       "D1": [(0, 1)], "D2": [(0, 2), (2, 3)], "D3": [(0, 3)], "D4": [(0, 4)]}
    groups_broken = study.instance_distances("P5_hops", instance_broken, instance_broken["nodes"],
                                             distance_from(broken))
    # means still rise (1 < 5.5, 5.5 > 3, 3 < 4): two of three comparisons hold, so the loose score
    # pays 2/3 while the strict one fails on the extra pair inside D2
    assert abs(study.hypothesis("P5_hops", groups_broken) - 2.0 / 3.0) < 1e-12
    assert study.strict_indicator("P5_hops", groups_broken) == 0.0


def test_k5_strict_counts_wide_pairs():
    nodes = (0, 1, 2, 3, 4)
    pairs = list(combinations(range(5), 2))
    tight = {(min(a, b), max(a, b)): 0.1 * (index + 1) for index, (a, b) in enumerate(pairs)}
    for offset in (10, 20, 30, 40):
        for a, b in pairs:
            tight[(offset + a, offset + b)] = 5.0
    loose_draws = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24],
                   [30, 31, 32, 33, 34], [40, 41, 42, 43, 44]]
    scored = study.instance_scores("K5", {"nodes": nodes}, loose_draws, distance_from(tight))
    assert scored[2] == 1.0        # every pair of the clique is below the random pair distance
    assert scored[0] == 1.0        # and the loose mean comparison agrees


if __name__ == "__main__":
    test_two_group_strict_requires_every_near_pair_closer()
    test_p5_strict_requires_the_whole_chain()
    test_k5_strict_counts_wide_pairs()
    print("all strict-metric tests pass")
