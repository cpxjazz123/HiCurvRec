"""Behavioural neighbourhood rank preservation: how close is a real neighbour inside the ranking?

The owner's metric, and the primary one from here on: for an anchor A and one of its real behaviour
neighbours B, where does B sit in the distance ordering of A over the whole catalogue?  A percentile
of 0 means B is the nearest item; 0.5 means B ranks like a random item; 1 means the farthest.

    percentile(A, B) = (midrank - 1) / (n - 1)

The midrank -- the average of the positions a tie occupies -- is what keeps the SID tree honest: its
distance has only four levels, so without it a tie would look like a win.

The anchor's neighbours are averaged per anchor before anything is pooled, so an anchor with three
neighbours does not count three times.  Every structure uses the same rule, which is the point: the
sum-of-distances alignment rate of the previous report compares sums whose length and spread differ
by structure, so it cannot rank structures; a percentile can.

Two controls, because a hub's neighbour is easy to rank and a rare item's neighbour is not:

    matched   each true neighbour is compared with random non-neighbours of the anchor drawn from the
              same degree bucket, so popularity is held fixed
    stratified   the same percentile table per degree quintile

    python3 results/research/issue215/behavioural_neighbourhood_rank.py [ARM,...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "behavioural_neighbourhood_rank.json"
ARMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("TIGER",)
SEED = 42
MAX_INSTANCES = 2000
MAX_ANCHORS = 2000            # per structure, per representation
MATCHED_DRAWS = 5             # degree-matched non-neighbours per true neighbour
DEGREE_BUCKETS = 5
STRUCTURES = ("edge", "path3", "triangle", "path4", "star4", "cycle4", "triangle_tail",
              "diamond", "K4")
REPRESENTATIONS = ("q", "sid", "fused")


def load_behaviour() -> tuple[int, list[set[int]]]:
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return n, adjacency


def build_instances(adjacency: list[set[int]], rng: np.random.Generator) -> dict:
    """Every structure as {nodes, edges} -- the same builder the alignment report uses."""
    neighbours = [sorted(adjacency[i]) for i in range(len(adjacency))]
    degrees = np.array([len(w) for w in neighbours])
    instances: dict[str, list[dict]] = {name: [] for name in STRUCTURES}

    def pick(values) -> int:
        values = list(values)
        return int(values[rng.integers(len(values))])

    def anchor_at(degree: int) -> int | None:
        candidates = np.flatnonzero(degrees >= degree)
        return None if candidates.size == 0 else int(pick(candidates))

    for _ in range(MAX_INSTANCES):
        a = anchor_at(1)
        if a is None:
            continue
        b = pick(neighbours[a])
        instances["edge"].append({"nodes": [a, b], "edges": [(a, b)]})

    for _ in range(MAX_INSTANCES):
        b = anchor_at(2)
        if b is None:
            continue
        a = pick(neighbours[b])
        rest = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest:
            continue
        c = pick(rest)
        instances["path3"].append({"nodes": [a, b, c], "edges": [(b, a), (b, c)]})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(2)
        if a is None:
            continue
        b = pick(neighbours[a])
        common = [w for w in neighbours[a] if w != b and w in adjacency[b]]
        if not common:
            continue
        c = pick(common)
        instances["triangle"].append({"nodes": [a, b, c],
                                      "edges": [(a, b), (a, c), (b, c)]})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(1)
        if a is None:
            continue
        b = pick(neighbours[a])
        rest_b = [w for w in neighbours[b] if w != a]
        if not rest_b:
            continue
        c = pick(rest_b)
        rest_c = [w for w in neighbours[c] if w not in (b, a) and w not in adjacency[a]]
        if not rest_c:
            continue
        d = pick(rest_c)
        instances["path4"].append({"nodes": [a, b, c, d],
                                   "edges": [(a, b), (b, c), (c, d)]})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        leaves = [neighbours[a][int(index)] for index in picks]
        if any(leaves[i] in adjacency[leaves[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["star4"].append({"nodes": [a] + leaves,
                                   "edges": [(a, leaf) for leaf in leaves]})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(2)
        if a is None:
            continue
        b = pick(neighbours[a])
        rest_b = [w for w in neighbours[b] if w != a]
        if not rest_b:
            continue
        c = pick(rest_b)
        rest_c = [w for w in neighbours[c] if w not in (b, a) and w in adjacency[a]]
        if not rest_c:
            continue
        d = pick(rest_c)
        instances["cycle4"].append({"nodes": [a, b, c, d],
                                    "edges": [(a, b), (b, c), (c, d), (d, a)]})

    for _ in range(MAX_INSTANCES):
        c = anchor_at(2)
        if c is None:
            continue
        a = pick(neighbours[c])
        common = [w for w in neighbours[c] if w != a and w in adjacency[a]]
        if not common:
            continue
        b = pick(common)
        rest = [w for w in neighbours[c]
                if w not in (a, b) and w not in adjacency[a] and w not in adjacency[b]]
        if not rest:
            continue
        d = pick(rest)
        instances["triangle_tail"].append({"nodes": [a, b, c, d],
                                           "edges": [(a, b), (a, c), (b, c), (c, d)]})

    for _ in range(MAX_INSTANCES):
        b = anchor_at(2)
        if b is None:
            continue
        a = pick(neighbours[b])
        rest = [w for w in neighbours[b] if w != a and w in adjacency[a]]
        if not rest:
            continue
        c = pick(rest)
        candidates = [w for w in neighbours[c]
                      if w not in (a, b) and w in adjacency[b] and w not in adjacency[a]]
        if not candidates:
            continue
        d = pick(candidates)
        instances["diamond"].append({"nodes": [a, b, c, d],
                                     "edges": [(a, b), (a, c), (b, c), (b, d), (c, d)]})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        others = [neighbours[a][int(index)] for index in picks]
        if not all(others[i] in adjacency[others[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        nodes = [a] + others
        instances["K4"].append({"nodes": nodes, "edges": list(combinations(nodes, 2))})

    return instances


def midrank_percentiles(values: np.ndarray) -> np.ndarray:
    """Percentile of every item by its value, ties sharing the average position (vectorised)."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]]) - 1
    average = (starts + ends) / 2.0
    ranks_sorted = np.repeat(average, ends - starts + 1)
    ranks = np.empty(n)
    ranks[order] = ranks_sorted
    return ranks / max(n - 1, 1)


def tree_distance_from(tokens: np.ndarray, anchor: int) -> np.ndarray:
    match1 = tokens[:, 0] == tokens[anchor, 0]
    match2 = match1 & (tokens[:, 1] == tokens[anchor, 1])
    match3 = match2 & (tokens[:, 2] == tokens[anchor, 2])
    return 1.0 - (match1.astype(np.int16) + match2.astype(np.int16)
                  + match3.astype(np.int16)) / 3.0


def summarise(percentiles: list[float]) -> dict:
    values = np.array(percentiles)
    return {"count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75))}


def main() -> int:
    n, adjacency = load_behaviour()
    rng = np.random.default_rng(SEED)
    instances = build_instances(adjacency, rng)
    degrees = np.array([len(adjacency[i]) for i in range(n)])
    order = np.argsort(degrees, kind="mergesort")
    bucket_of = np.empty(n, dtype=np.int64)
    bucket_of[order] = np.minimum(
        (np.arange(n) * DEGREE_BUCKETS) // max(n, 1), DEGREE_BUCKETS - 1)
    bucket_pool = [np.flatnonzero(bucket_of == b) for b in range(DEGREE_BUCKETS)]

    journal: dict = {"seed": SEED, "max_instances": MAX_INSTANCES, "max_anchors": MAX_ANCHORS,
                     "matched_draws": MATCHED_DRAWS, "degree_buckets": DEGREE_BUCKETS,
                     "note": "percentile 0 = nearest item, 0.5 = random; midranks; anchors pooled "
                             "after averaging over each anchor's neighbours; matched control draws "
                             "non-neighbours from the anchor's own degree bucket",
                     "arms": {}}

    for arm in ARMS:
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)
        if len(q) != n:
            raise SystemExit(f"{arm}: {len(q)} vectors for {n} items")

        anchors_by_structure: dict[str, list[tuple[int, set[int]]]] = {}
        for name in STRUCTURES:
            seen: dict[int, set[int]] = defaultdict(set)
            for instance in instances[name]:
                for left, right in instance["edges"]:
                    seen[left].add(right)
                    seen[right].add(left)
            anchors_by_structure[name] = list(seen.items())[:MAX_ANCHORS]

        sampled_pairs = [(anchor, other) for name in STRUCTURES
                         for anchor, truth in anchors_by_structure[name] for other in truth]
        q_values = np.array([np.linalg.norm(q[a] - q[b]) for a, b in sampled_pairs])
        sid_values = np.array([tree_distance_from(tokens, a)[b] for a, b in sampled_pairs])
        q_mean, q_std = float(q_values.mean()), float(q_values.std() or 1.0)
        sid_mean, sid_std = float(sid_values.mean()), float(sid_values.std() or 1.0)

        block: dict = {}
        for name in STRUCTURES:
            entry: dict = {"anchors": len(anchors_by_structure[name]), "representations": {}}
            per_rep: dict[str, list[float]] = {rep: [] for rep in REPRESENTATIONS}
            matched: dict[str, list[float]] = {rep: [] for rep in REPRESENTATIONS}
            by_bucket: dict[str, dict[int, list[float]]] = {
                rep: defaultdict(list) for rep in REPRESENTATIONS}
            neighbour_counts: dict[int, list[float]] = defaultdict(list)
            for anchor, truth in anchors_by_structure[name]:
                euclid = np.linalg.norm(q - q[anchor], axis=1)
                tree = tree_distance_from(tokens, anchor)
                distances = {"q": euclid,
                             "sid": tree,
                             "fused": (euclid - q_mean) / q_std + (tree - sid_mean) / sid_std}
                blocked = set(truth) | {anchor}
                for representation in REPRESENTATIONS:
                    percentiles = midrank_percentiles(distances[representation])
                    anchor_values = [percentiles[item] for item in truth]
                    per_rep[representation].append(float(np.mean(anchor_values)))
                    for item in truth:
                        by_bucket[representation][int(bucket_of[item])].append(
                            float(percentiles[item]))
                    if representation == "q":
                        neighbour_counts[len(truth)].append(float(np.mean(anchor_values)))
                    for item in truth:
                        pool = bucket_pool[int(bucket_of[item])]
                        draws = 0
                        for _ in range(64):
                            if draws >= MATCHED_DRAWS:
                                break
                            candidate = int(pool[rng.integers(pool.size)])
                            if candidate in blocked or candidate in adjacency[anchor]:
                                continue
                            matched[representation].append(float(percentiles[candidate]))
                            draws += 1
            for representation in REPRESENTATIONS:
                entry["representations"][representation] = {
                    "anchor_level": summarise(per_rep[representation]),
                    "matched_non_neighbours": summarise(matched[representation])
                        if matched[representation] else None,
                    "by_degree_quintile": {
                        str(bucket): summarise(values)
                        for bucket, values in sorted(by_bucket[representation].items())},
                }
            entry["by_neighbour_count"] = {
                str(count): summarise(values)
                for count, values in sorted(neighbour_counts.items())}
            block[name] = entry
        journal["arms"][arm] = block

    JOURNAL.write_text(json.dumps(journal, indent=1, default=float))
    for arm, block in journal["arms"].items():
        print(f"\n=== {arm}: percentile of a real neighbour in the anchor's distance ordering "
              f"(0 = nearest, 0.5 = random); midranks; anchors pooled ===")
        print(f"{'structure':<14}{'anchors':>9}" + "".join(f"{rep:>22}" for rep in REPRESENTATIONS))
        for name in STRUCTURES:
            line = f"{name:<14}{block[name]['anchors']:>9}"
            for representation in REPRESENTATIONS:
                summary = block[name]["representations"][representation]["anchor_level"]
                line += f"{summary['mean']:.3f}/{summary['median']:.3f}".rjust(22)
            print(line)
        print("  (cell = mean/median; matched non-neighbour baseline below)")
        print(f"{'structure':<14}{'':>9}" + "".join(f"{rep:>22}" for rep in REPRESENTATIONS))
        for name in STRUCTURES:
            line = f"{name:<14}{'':>9}"
            for representation in REPRESENTATIONS:
                summary = block[name]["representations"][representation]["matched_non_neighbours"]
                line += (f"{summary['mean']:.3f}/{summary['median']:.3f}".rjust(22)
                         if summary else "n/a".rjust(22))
            print(line)
        print("mean percentile by degree quintile (q only):")
        for name in STRUCTURES:
            cells = " ".join(
                f"{bucket}:{summary['mean']:.3f}"
                for bucket, summary in
                block[name]["representations"]["q"]["by_degree_quintile"].items())
            print(f"  {name:<14} {cells}")
        print("mean percentile by neighbour count (q only):")
        for name in STRUCTURES:
            cells = " ".join(f"k={count}:{summary['mean']:.3f}"
                             for count, summary in block[name]["by_neighbour_count"].items())
            print(f"  {name:<14} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
