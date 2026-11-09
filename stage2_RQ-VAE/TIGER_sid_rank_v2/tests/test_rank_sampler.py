"""Triplet sampling checks: strata balance, no neighbour negatives, degree matching."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank_sampler as S

TRAIN = Path(__file__).resolve().parents[3] / "results" / "stage0_build_parquet" / "train.parquet"
GRAPH = S.load_graph(TRAIN)


def test_strata_are_balanced():
    triplets = S.sample_triplets(GRAPH, 200, random.Random(0))
    tags = triplets["stratum"]
    assert abs(int((tags == 0).sum()) - int((tags == 1).sum())) <= 4


def test_negatives_are_never_true_neighbours():
    triplets = S.sample_triplets(GRAPH, 300, random.Random(1))
    for left, negative in zip(triplets["left"], triplets["negative"]):
        assert int(negative) not in GRAPH.neighbours.get(int(left), set())


def test_positives_are_real_edges():
    triplets = S.sample_triplets(GRAPH, 200, random.Random(2))
    pairs = {tuple(sorted(pair)) for pair in map(tuple, GRAPH.edges.tolist())}
    for left, right in zip(triplets["left"], triplets["right"]):
        assert tuple(sorted((int(left), int(right)))) in pairs


def test_negatives_sit_in_the_same_degree_bucket():
    triplets = S.sample_triplets(GRAPH, 300, random.Random(3))
    matched = 0
    for left, negative in zip(triplets["left"], triplets["negative"]):
        li = int(np.searchsorted(GRAPH.nodes, int(left)))
        ni = int(np.searchsorted(GRAPH.nodes, int(negative)))
        if int(GRAPH.degree[li]) // S.DEGREE_BUCKET == int(GRAPH.degree[ni]) // S.DEGREE_BUCKET:
            matched += 1
    assert matched >= 0.9 * len(triplets["left"])


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("ok   ", name)
            except AssertionError:
                failures += 1
                print("FAIL ", name)
    raise SystemExit(1 if failures else 0)
