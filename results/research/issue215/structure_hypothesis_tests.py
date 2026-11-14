"""One distance hypothesis per structure, tested against the structure's own connection pattern.

The owner's second question, separate from "are real neighbours close": does the *whole* distance
relation inside a structure match the pattern its connections imply?  Each structure gets one stated
hypothesis, and the hypothesis is tested, not assumed -- a complementary pair can be behaviour-linked
and still far apart in semantics, and the report says so wherever the data refuses the pattern.

    structure   hypothesis                                   statistic
    star        centre to leaf  <  leaf to leaf              mean(spoke) - mean(leaf,leaf)   < 0
    path3       adjacent < end to end                        mean(spoke) - d(end,end)        < 0
    cycle4      adjacent < opposite                          mean(adjacent) - mean(opposite) < 0
    diamond     the one missing pair > the five edges        d(missing) - mean(edges)        > 0
    triangle    the three nodes form a tight group           mean inside < degree-matched random set
    K4          the four nodes form a tight group            mean inside < degree-matched random set
    path4       one hop < two hops < three hops              hop means strictly increasing
    triangle_tail  inside the triangle < tail to the core    mean(internal) - mean(tail,core) < 0
    edge        not applicable: one distance, no relation

Two things learned from the previous sections, both respected here:

    * instances are sampled WITHOUT replacement and the distinct count is reported -- the four-clique
      row of an earlier table rested on 2,000 draws from 19 distinct anchors
    * the statistic is expressed in the representation's own distance spread, because a fused score is
      not a metric and its raw units mean nothing across representations

A structure's verdict is only issued when the 95% Wilson interval of its support share excludes 0.5 and
the distinct count is at least MIN_INSTANCES; otherwise it is UNDERPOWERED, never "no effect".

    python3 results/research/issue215/structure_hypothesis_tests.py [ARM,...]
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "structure_hypothesis_tests.json"
ARMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("TIGER",)
SEED = 42
MAX_INSTANCES = 1000
MAX_ATTEMPTS = 60_000
MIN_INSTANCES = 100
MATCHED_DRAWS = 5
REPRESENTATIONS = ("q", "sid", "fused")
STRUCTURES = ("star4", "path3", "cycle4", "diamond", "triangle", "K4", "path4", "triangle_tail")
HYPOTHESIS = {
    "star4": "centre to leaf closer than leaf to leaf",
    "path3": "adjacent closer than end to end",
    "cycle4": "adjacent closer than opposite",
    "diamond": "the missing pair farther than the five edges",
    "triangle": "the three nodes form a tight group",
    "K4": "the four nodes form a tight group",
    "path4": "distance grows with the hop count",
    "triangle_tail": "inside the triangle closer than the tail to the core",
}


def load_behaviour() -> tuple[int, list[set[int]]]:
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return n, adjacency


def sample_instances(adjacency: list[set[int]], rng: np.random.Generator) -> dict:
    """Distinct instances per structure, sampled without replacement."""
    n = len(adjacency)
    neighbours = [sorted(adjacency[i]) for i in range(n)]
    degrees = np.array([len(w) for w in neighbours])
    pools = {name: {} for name in STRUCTURES}

    def pick(values) -> int:
        values = list(values)
        return int(values[rng.integers(len(values))])

    def anchor_at(degree: int) -> int | None:
        candidates = np.flatnonzero(degrees >= degree)
        return None if candidates.size == 0 else int(pick(candidates))

    def keep(name: str, nodes: list[int], block: dict) -> None:
        key = frozenset(nodes)
        if key in pools[name] or len(pools[name]) >= MAX_INSTANCES:
            return
        pools[name][key] = {"nodes": nodes, **block}

    for _ in range(MAX_ATTEMPTS):
        if len(pools["star4"]) >= MAX_INSTANCES:
            break
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        leaves = [neighbours[a][int(index)] for index in picks]
        if any(leaves[i] in adjacency[leaves[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        keep("star4", [a] + leaves, {"groups": {
            "near": [(a, leaf) for leaf in leaves], "far": list(combinations(leaves, 2))}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["path3"]) >= MAX_INSTANCES:
            break
        b = anchor_at(2)
        if b is None:
            continue
        a = pick(neighbours[b])
        rest = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest:
            continue
        c = pick(rest)
        keep("path3", [a, b, c], {"groups": {"near": [(b, a), (b, c)], "far": [(a, c)]}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["cycle4"]) >= MAX_INSTANCES:
            break
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
        keep("cycle4", [a, b, c, d], {"groups": {
            "near": [(a, b), (b, c), (c, d), (d, a)], "far": [(a, c), (b, d)]}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["diamond"]) >= MAX_INSTANCES:
            break
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
        keep("diamond", [a, b, c, d], {"groups": {
            "near": [(a, b), (a, c), (b, c), (b, d), (c, d)], "far": [(a, d)]}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["triangle"]) >= MAX_INSTANCES:
            break
        a = anchor_at(2)
        if a is None:
            continue
        b = pick(neighbours[a])
        common = [w for w in neighbours[a] if w != b and w in adjacency[b]]
        if not common:
            continue
        c = pick(common)
        keep("triangle", [a, b, c], {"groups": {}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["K4"]) >= MAX_INSTANCES:
            break
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        others = [neighbours[a][int(index)] for index in picks]
        if not all(others[i] in adjacency[others[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        keep("K4", [a] + others, {"groups": {}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["path4"]) >= MAX_INSTANCES:
            break
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
        keep("path4", [a, b, c, d], {"groups": {
            "hop1": [(a, b), (b, c), (c, d)], "hop2": [(a, c), (b, d)], "hop3": [(a, d)]}})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["triangle_tail"]) >= MAX_INSTANCES:
            break
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
        keep("triangle_tail", [a, b, c, d], {"groups": {
            "near": [(a, b), (a, c), (b, c)], "far": [(d, a), (d, b), (d, c)]}})

    print("distinct instances: " + ", ".join(f"{name}={len(blocks)}"
                                             for name, blocks in pools.items()), flush=True)
    return {name: list(blocks.values()) for name, blocks in pools.items()}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (centre - half, centre + half)


def tree_distance(tokens: np.ndarray, left: int, right: int) -> float:
    count = 0
    for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
        if x != y:
            break
        count += 1
    return 1.0 - count / 3.0


def main() -> int:
    n, adjacency = load_behaviour()
    rng = np.random.default_rng(SEED)
    instances = sample_instances(adjacency, rng)
    degrees = np.array([len(adjacency[i]) for i in range(n)])
    order = np.argsort(degrees, kind="mergesort")
    bucket_of = np.empty(n, dtype=np.int64)
    bucket_of[order] = np.minimum((np.arange(n) * 5) // max(n, 1), 4)
    bucket_pool = [np.flatnonzero(bucket_of == b) for b in range(5)]

    journal: dict = {"seed": SEED, "max_instances": MAX_INSTANCES, "min_instances": MIN_INSTANCES,
                     "matched_draws": MATCHED_DRAWS, "hypotheses": HYPOTHESIS,
                     "note": "instances sampled without replacement; effect sizes in the "
                             "representation's own distance spread; verdicts need a Wilson interval "
                             "excluding 0.5 and at least 100 distinct instances",
                     "arms": {}}

    for arm in ARMS:
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)
        if len(q) != n:
            raise SystemExit(f"{arm}: {len(q)} vectors for {n} items")

        pairs = [(a, b) for name in STRUCTURES for instance in instances[name]
                 for group in instance["groups"].values() for a, b in group]
        pairs += [(instance["nodes"][0], other) for name in STRUCTURES
                  for instance in instances[name] for other in instance["nodes"][1:]]
        q_values = np.array([np.linalg.norm(q[a] - q[b]) for a, b in pairs])
        sid_values = np.array([tree_distance(tokens, a, b) for a, b in pairs])
        q_mean, q_std = float(q_values.mean()), float(q_values.std() or 1.0)
        sid_mean, sid_std = float(sid_values.mean()), float(sid_values.std() or 1.0)

        def distance(representation: str, left: int, right: int) -> float:
            euclid = float(np.linalg.norm(q[left] - q[right]))
            tree = tree_distance(tokens, left, right)
            if representation == "q":
                return euclid
            if representation == "sid":
                return tree
            if representation == "fused":
                return (euclid - q_mean) / q_std + (tree - sid_mean) / sid_std
            raise KeyError(representation)

        block: dict = {}
        for name in STRUCTURES:
            entry: dict = {"instances": len(instances[name]), "representations": {}}
            for representation in REPRESENTATIONS:
                scale = {"q": q_std, "sid": sid_std, "fused": math.hypot(1.0, 1.0)}[representation]
                supports = 0
                ties = 0
                values: list[float] = []
                scores: list[float] = []
                hop_first = 0
                hop_second = 0
                for instance in instances[name]:
                    nodes = instance["nodes"]
                    groups = instance["groups"]
                    if name in ("triangle", "K4"):
                        inside = [distance(representation, a, b) for a, b in combinations(nodes, 2)]
                        matched: list[float] = []
                        for _ in range(MATCHED_DRAWS):
                            drawn: list[int] = []
                            for node in nodes:
                                pool = bucket_pool[int(bucket_of[node])]
                                for _ in range(64):
                                    candidate = int(pool[rng.integers(pool.size)])
                                    if candidate in nodes or candidate in adjacency[node]:
                                        continue
                                    drawn.append(candidate)
                                    break
                            if len(drawn) == len(nodes):
                                matched.append(float(np.mean(
                                    [distance(representation, a, b)
                                     for a, b in combinations(drawn, 2)])))
                        if not matched:
                            continue
                        value = float(np.mean(inside)) - float(np.mean(matched))
                        supported = value < 0
                    elif name == "path4":
                        hop1 = float(np.mean([distance(representation, a, b)
                                              for a, b in groups["hop1"]]))
                        hop2 = float(np.mean([distance(representation, a, b)
                                              for a, b in groups["hop2"]]))
                        hop3 = float(np.mean([distance(representation, a, b)
                                              for a, b in groups["hop3"]]))
                        value = hop3 - hop1
                        hop_first += int(hop1 < hop2)
                        hop_second += int(hop2 < hop3)
                        supported = hop1 < hop2 < hop3
                    else:
                        near = float(np.mean([distance(representation, a, b)
                                              for a, b in groups["near"]]))
                        far = float(np.mean([distance(representation, a, b)
                                             for a, b in groups["far"]]))
                        value = near - far
                        supported = (value > 0) if name == "diamond" else (value < 0)
                    values.append(value / scale)
                    if abs(value) < 1e-12:
                        ties += 1
                        scores.append(0.5)
                    elif supported:
                        supports += 1
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                total = len(values)
                low, high = wilson(supports, total)
                # a discrete representation ties constantly; a tie is neither support nor rejection,
                # so the tie-aware share scores it 0.5 and is bootstrapped over instances
                if total:
                    score_array = np.array(scores)
                    picks = np.random.default_rng(SEED + 7).integers(0, total, size=(2000, total))
                    boot = score_array[picks].mean(axis=1)
                    boot_low = float(np.percentile(boot, 2.5))
                    boot_high = float(np.percentile(boot, 97.5))
                else:
                    boot_low = boot_high = float("nan")
                if total < MIN_INSTANCES:
                    verdict = "UNDERPOWERED"
                elif boot_low > 0.5:
                    verdict = "SUPPORTED"
                elif boot_high < 0.5:
                    verdict = "REJECTED"
                else:
                    verdict = "INCONCLUSIVE"
                entry["representations"][representation] = {
                    "instances_used": total,
                    "support_share": supports / total if total else None,
                    "strict_ci95": [low, high],
                    "tie_share": ties / total if total else None,
                    "tie_aware_share": (supports + 0.5 * ties) / total if total else None,
                    "tie_aware_ci95": [boot_low, boot_high],
                    "hop_one_lt_two": hop_first / total if total and name == "path4" else None,
                    "hop_two_lt_three": hop_second / total if total and name == "path4" else None,
                    "mean_effect_sd": float(np.mean(values)) if values else None,
                    "median_effect_sd": float(np.median(values)) if values else None,
                    "verdict": verdict,
                }
            block[name] = entry
        journal["arms"][arm] = block

    JOURNAL.write_text(json.dumps(journal, indent=1, default=float))
    for arm, block in journal["arms"].items():
        print(f"\n=== {arm}: one distance hypothesis per structure (instances sampled without "
              f"replacement) ===")
        for name in STRUCTURES:
            print(f"\n{name} (n={block[name]['instances']} distinct) -- {HYPOTHESIS[name]}")
            print(f"  {'rep':<6}{'support':>8}{'strict CI':>17}{'ties':>7}{'tie-aware':>10}"
                  f"{'aware CI':>17}{'mean sd':>9}{'med sd':>8}  verdict")
            for representation in REPRESENTATIONS:
                row = block[name]["representations"][representation]
                if not row["instances_used"]:
                    continue
                low, high = row["strict_ci95"]
                blow, bhigh = row["tie_aware_ci95"]
                print(f"  {representation:<6}{row['support_share']:>8.3f}"
                      f"{f'[{low:.3f},{high:.3f}]':>17}{row['tie_share']:>7.2f}"
                      f"{row['tie_aware_share']:>10.3f}{f'[{blow:.3f},{bhigh:.3f}]':>17}"
                      f"{row['mean_effect_sd']:>9.3f}{row['median_effect_sd']:>8.3f}"
                      f"  {row['verdict']}")
                if name == "path4":
                    print(f"         hop one<two {row['hop_one_lt_two']:.3f}, "
                          f"two<three {row['hop_two_lt_three']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
