"""Structure alignment gain: each structure's own matched null, not a shared 50% line.

The support rate of a hypothesis is not comparable across structures by itself: a star compares three
edges against three leaf pairs, a diamond compares five edges against one pair, a triangle compares a
sum of three distances against a matched set, and the variance of those statistics differs.  So each
structure gets its own null: a random node set of the same size whose members are drawn from the same
degree buckets as the real roles, evaluated with the *same* role partition and the *same* direction.

    real support    x%      the share of real instances that satisfy the hypothesis
    null support    y%      the same statistic on degree-matched random role assignments
    gain            x - y   what the behaviour structure adds on top of a random group of that shape

A gain near zero means the geometry would have satisfied the hypothesis for a random group of the same
degrees and role shape just as often -- the structure itself contributes nothing.  A large gain is the
evidence that the quantized vector expresses *that* structure.

The null is drawn once per instance per draw index and shared by the three representations, so the
comparison is paired; the gain's interval comes from resampling instances, not comparisons.

    python3 results/research/issue215/structure_alignment_gain.py [ARM,...]
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
JOURNAL = OUT / "structure_alignment_gain.json"
sys.path.insert(0, str(OUT))

import structure_hypothesis_tests as sht  # noqa: E402  (the builders, the validation, the metric)

ARMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("TIGER",)
SEED = 42
NULL_DRAWS = 5
REPRESENTATIONS = ("q", "sid", "fused")


def role_partition(instances: list[dict]) -> dict:
    """The structure's pair pattern expressed in role positions, checked to be identical everywhere."""
    pattern: dict | None = None
    for instance in instances:
        index = {node: position for position, node in enumerate(instance["nodes"])}
        current = {"closer": [(index[a], index[b]) for a, b in instance.get("closer", [])],
                   "farther": [(index[a], index[b]) for a, b in instance.get("farther", [])],
                   "hop1": [(index[a], index[b]) for a, b in instance.get("hop1", [])],
                   "hop2": [(index[a], index[b]) for a, b in instance.get("hop2", [])],
                   "hop3": [(index[a], index[b]) for a, b in instance.get("hop3", [])]}
        current = {key: sorted(value) for key, value in current.items()}
        if pattern is None:
            pattern = current
        elif current != pattern:
            raise AssertionError("the structure's role partition is not constant across instances")
    return pattern


def main() -> int:
    n, adjacency = sht.load_behaviour()
    rng = np.random.default_rng(SEED)
    instances = sht.sample_instances(adjacency, rng)
    cliques = sht.enumerate_cliques(n, adjacency)
    cliques.sort()
    chosen = rng.choice(len(cliques), size=min(len(cliques), sht.MAX_INSTANCES), replace=False)
    instances["K4"] = [{"nodes": list(cliques[int(index)])} for index in sorted(chosen)]
    violations = sht.validate(instances, adjacency)

    degrees = np.array([len(adjacency[i]) for i in range(n)])
    order = np.argsort(degrees, kind="mergesort")
    bucket_of = np.empty(n, dtype=np.int64)
    bucket_of[order] = np.minimum((np.arange(n) * 5) // max(n, 1), 4)
    bucket_pool = [np.flatnonzero(bucket_of == b) for b in range(5)]

    patterns = {name: role_partition(instances[name])
                for name in sht.STRUCTURES if instances[name]}

    # one null draw set per instance per draw index, shared by every representation
    null_sets: dict[str, list[list[tuple[int, ...]]]] = {}
    for name, pattern in patterns.items():
        blocks: list[list[tuple[int, ...]]] = []
        for instance in instances[name]:
            real = set(instance["nodes"])
            sets: list[tuple[int, ...]] = []
            adjacency_free = name in ("triangle", "K4")
            for _ in range(NULL_DRAWS):
                drawn: list[int] = []
                for node in instance["nodes"]:
                    pool = bucket_pool[int(bucket_of[node])]
                    for _ in range(64):
                        candidate = int(pool[rng.integers(pool.size)])
                        if candidate in real or candidate in drawn:
                            continue
                        if adjacency_free and candidate in adjacency[node]:
                            continue
                        drawn.append(candidate)
                        break
                if len(drawn) == len(instance["nodes"]):
                    sets.append(tuple(drawn))
            blocks.append(sets)
        null_sets[name] = blocks

    journal: dict = {"seed": SEED, "null_draws": NULL_DRAWS,
                     "structural_violations": violations,
                     "note": "real support uses the real instances; null support uses the same role "
                             "partition on degree-matched random node sets of the same size; gain = "
                             "real - null; intervals come from resampling instances",
                     "arms": {}}

    for arm in ARMS:
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)
        if len(q) != n:
            raise SystemExit(f"{arm}: {len(q)} vectors for {n} items")

        pairs = [(a, b) for name in sht.STRUCTURES for instance in instances[name]
                 for group in ("closer", "farther", "hop1", "hop2", "hop3")
                 for a, b in instance.get(group, [])]
        q_values = np.array([np.linalg.norm(q[a] - q[b]) for a, b in pairs])
        sid_values = np.array([sht.tree_distance(tokens, a, b) for a, b in pairs])
        q_mean, q_std = float(q_values.mean()), float(q_values.std() or 1.0)
        sid_mean, sid_std = float(sid_values.mean()), float(sid_values.std() or 1.0)

        def distance(representation: str, left: int, right: int) -> float:
            euclid = float(np.linalg.norm(q[left] - q[right]))
            tree = sht.tree_distance(tokens, left, right)
            if representation == "q":
                return euclid
            if representation == "sid":
                return tree
            if representation == "fused":
                return (euclid - q_mean) / q_std + (tree - sid_mean) / sid_std
            raise KeyError(representation)

        block: dict = {}
        for name, pattern in patterns.items():
            entry: dict = {"instances": len(instances[name]), "representations": {}}
            for representation in REPRESENTATIONS:
                real_support: list[float] = []
                null_support: list[float] = []
                effects: list[float] = []
                for index, instance in enumerate(instances[name]):
                    roles = instance["nodes"]
                    sets = null_sets[name][index]
                    if not sets:
                        continue

                    def statistic(nodes: tuple[int, ...]) -> float | None:
                        if name == "path4":
                            hop1 = float(np.mean([distance(representation, nodes[a], nodes[b])
                                                  for a, b in pattern["hop1"]]))
                            hop2 = float(np.mean([distance(representation, nodes[a], nodes[b])
                                                  for a, b in pattern["hop2"]]))
                            hop3 = float(np.mean([distance(representation, nodes[a], nodes[b])
                                                  for a, b in pattern["hop3"]]))
                            return 1.0 if hop1 < hop2 < hop3 else 0.0
                        if name in ("triangle", "K4"):
                            return float(np.mean([distance(representation, nodes[a], nodes[b])
                                                  for a, b in combinations(range(len(nodes)), 2)]))
                        closer = float(np.mean([distance(representation, nodes[a], nodes[b])
                                                for a, b in pattern["closer"]]))
                        farther = float(np.mean([distance(representation, nodes[a], nodes[b])
                                                 for a, b in pattern["farther"]]))
                        return closer - farther

                    if name in ("triangle", "K4"):
                        inside = statistic(tuple(roles))
                        null_inside = [statistic(drawn) for drawn in sets]
                        support = 1.0 if inside < float(np.mean(null_inside)) else 0.0
                        # the null replaces the real group by another matched group and keeps the
                        # same reference: by symmetry its support rate is the honest baseline
                        null_scores = []
                        for position, first in enumerate(sets):
                            others = [statistic(drawn) for index_other, drawn in enumerate(sets)
                                      if index_other != position]
                            if not others:
                                continue
                            null_scores.append(1.0 if statistic(first) < float(np.mean(others))
                                               else 0.0)
                        null_rate = float(np.mean(null_scores)) if null_scores else 0.5
                        effects.append(float(inside) - float(np.mean(null_inside)))
                    elif name == "path4":
                        support = statistic(tuple(roles))
                        null_rate = float(np.mean([statistic(drawn) for drawn in sets]))
                        effects.append(0.0)
                    else:
                        value = statistic(tuple(roles))
                        support = 1.0 if value < 0 else 0.0
                        null_rate = float(np.mean([1.0 if statistic(drawn) < 0 else 0.0
                                                   for drawn in sets]))
                        effects.append(-value)
                    real_support.append(support)
                    null_support.append(null_rate)

                real = float(np.mean(real_support)) if real_support else None
                null = float(np.mean(null_support)) if null_support else None
                gain = (real - null) if (real is not None and null is not None) else None
                if real_support:
                    real_array = np.array(real_support)
                    null_array = np.array(null_support)
                    bootstrap = np.random.default_rng(SEED + 11).integers(
                        0, len(real_array), size=(2000, len(real_array)))
                    gains = (real_array[bootstrap].mean(axis=1)
                             - null_array[bootstrap].mean(axis=1))
                    low = float(np.percentile(gains, 2.5))
                    high = float(np.percentile(gains, 97.5))
                else:
                    low = high = float("nan")
                entry["representations"][representation] = {
                    "instances_used": len(real_support),
                    "real_support": real,
                    "null_support": null,
                    "gain": gain,
                    "gain_ci95": [low, high],
                    "verdict": ("GAIN" if low > 0 else
                                "NO_GAIN" if high < 0 else "INCONCLUSIVE"),
                }
            block[name] = entry
        journal["arms"][arm] = block

    JOURNAL.write_text(json.dumps(journal, indent=1, default=float))
    print("structural violations:", violations)
    for arm, block in journal["arms"].items():
        print(f"\n=== {arm}: structure alignment gain = real support - matched null support ===")
        for representation in REPRESENTATIONS:
            print(f"\n-- {representation} --")
            print(f"  {'structure':<14}{'n':>6}{'real':>8}{'null':>8}{'gain':>8}"
                  f"{'gain 95% CI':>20}  verdict")
            for name in patterns:
                row = block[name]["representations"][representation]
                low, high = row["gain_ci95"]
                print(f"  {name:<14}{row['instances_used']:>6}{row['real_support']:>8.3f}"
                      f"{row['null_support']:>8.3f}{row['gain']:>8.3f}"
                      f"{f'[{low:+.3f},{high:+.3f}]':>20}  {row['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
