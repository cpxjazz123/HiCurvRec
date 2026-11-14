"""One distance hypothesis per structure, tested after the code review that found three defects.

The owner reviewed the previous version and found, correctly:

    * the diamond's direction was inverted -- `near` holds the five real edges and `far` the single
      missing pair, so the hypothesis "the missing pair is farther" is `value < 0`, not `value > 0`.
      Every structure now uses one rule: value = mean(closer) - mean(farther), supported iff < 0.
    * the four-cycle sampler never checked the diagonals, so some instances were diamonds or cliques.
      A cycle now requires a-c and b-d to be absent.
    * the four-node path sampler never excluded a-c and b-d, so some instances were triangles with a
      tail, or diamonds. A path now requires a-c, a-d and b-d to be absent.
    * the clique count of 109 was what random sampling found, not how many exist. Cliques are now
      enumerated exhaustively and the true count is reported.
    * the tightness null could draw the same item twice, and drew a fresh set per representation.
      The matched set is now drawn once per instance, with distinct items, and shared by every
      representation.

A validation pass re-checks every sampled instance against its definition and records the number of
violations; a non-zero count is a bug, not a result.

    structure   hypothesis                                    statistic (unit: the representation's own spread)
    star        centre to leaf < leaf to leaf                 mean(spoke) - mean(leaf,leaf)
    path3       adjacent < end to end                         mean(spoke) - d(end,end)
    cycle4      adjacent < opposite                           mean(adjacent) - mean(opposite)
    diamond     the missing pair > the five edges             mean(edges) - d(missing)
    triangle    the three nodes form a tight group             mean inside - degree-matched random set
    K4          the four nodes form a tight group              mean inside - degree-matched random set
    path4       one hop < two hops < three hops                hop means strictly increasing
    triangle_tail  inside the triangle < tail to the core      mean(internal) - mean(tail,core)

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


def enumerate_cliques(n: int, adjacency: list[set[int]]) -> list[tuple[int, ...]]:
    """Every four-clique in the graph, once, as an ascending tuple.

    The first version keyed the result on the tuple as produced, and the pair order depended on the
    iteration order of a set, so 2,341,056 tuples covered only 1,570,609 distinct cliques. The
    ascending form is what the count and the downstream sampling both need.
    """
    cliques: set[tuple[int, ...]] = set()
    for u in range(n):
        above = sorted(w for w in adjacency[u] if w > u)
        for index, v in enumerate(above):
            for w in above[index + 1:]:
                if w not in adjacency[v]:
                    continue
                for x in adjacency[v]:
                    if x > w and x in adjacency[u] and x in adjacency[w]:
                        cliques.add((u, v, w, x))
    return sorted(cliques)


def sample_instances(adjacency: list[set[int]], rng: np.random.Generator) -> dict:
    """Distinct instances per structure, each satisfying its own definition strictly."""
    n = len(adjacency)
    neighbours = [sorted(adjacency[i]) for i in range(n)]
    degrees = np.array([len(w) for w in neighbours])
    pools: dict[str, dict[frozenset, dict]] = {name: {} for name in STRUCTURES}

    def pick(values) -> int:
        values = list(values)
        return int(values[rng.integers(len(values))])

    def anchor_at(degree: int) -> int | None:
        candidates = np.flatnonzero(degrees >= degree)
        return None if candidates.size == 0 else int(pick(candidates))

    def keep(name: str, nodes: list[int], block: dict) -> None:
        if len(set(nodes)) != len(nodes):
            raise AssertionError(f"{name}: repeated node in {nodes}")
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
        keep("star4", [a] + leaves, {"closer": [(a, leaf) for leaf in leaves],
                                     "farther": list(combinations(leaves, 2))})

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
        keep("path3", [a, b, c], {"closer": [(b, a), (b, c)], "farther": [(a, c)]})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["cycle4"]) >= MAX_INSTANCES:
            break
        a = anchor_at(2)
        if a is None:
            continue
        b = pick(neighbours[a])
        # c must be a neighbour of b that is NOT adjacent to a: otherwise a-c is a diagonal
        rest_b = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest_b:
            continue
        c = pick(rest_b)
        # d closes the cycle with a and must not be adjacent to b: otherwise b-d is a diagonal
        rest_c = [w for w in neighbours[c]
                  if w not in (a, b) and w in adjacency[a] and w not in adjacency[b]]
        if not rest_c:
            continue
        d = pick(rest_c)
        keep("cycle4", [a, b, c, d], {"closer": [(a, b), (b, c), (c, d), (d, a)],
                                      "farther": [(a, c), (b, d)]})

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
        keep("diamond", [a, b, c, d], {"closer": [(a, b), (a, c), (b, c), (b, d), (c, d)],
                                       "farther": [(a, d)]})

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
        keep("triangle", [a, b, c], {})

    for _ in range(MAX_ATTEMPTS):
        if len(pools["path4"]) >= MAX_INSTANCES:
            break
        a = anchor_at(1)
        if a is None:
            continue
        b = pick(neighbours[a])
        # c must not be adjacent to a, or the path would close a triangle
        rest_b = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest_b:
            continue
        c = pick(rest_b)
        # d must close the path only at c: not adjacent to b (that is a diagonal) nor to a
        rest_c = [w for w in neighbours[c]
                  if w not in (b, a) and w not in adjacency[a] and w not in adjacency[b]]
        if not rest_c:
            continue
        d = pick(rest_c)
        keep("path4", [a, b, c, d], {"hop1": [(a, b), (b, c), (c, d)],
                                     "hop2": [(a, c), (b, d)], "hop3": [(a, d)]})

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
        keep("triangle_tail", [a, b, c, d], {"closer": [(a, b), (a, c), (b, c)],
                                             "farther": [(d, c)]})

    return {name: list(blocks.values()) for name, blocks in pools.items()}


def validate(instances: dict, adjacency: list[set[int]]) -> dict:
    """Re-check every instance against its definition; a non-zero count is a defect, not a result."""
    def edge(left: int, right: int) -> bool:
        return right in adjacency[left]

    def nonedge(left: int, right: int) -> bool:
        return left != right and right not in adjacency[left]

    violations = {name: 0 for name in STRUCTURES}
    for instance in instances.get("star4", []):
        a, leaves = instance["nodes"][0], instance["nodes"][1:]
        if not all(edge(a, leaf) for leaf in leaves):
            violations["star4"] += 1
        elif not all(nonedge(x, y) for x, y in combinations(leaves, 2)):
            violations["star4"] += 1
    for instance in instances.get("path3", []):
        a, b, c = instance["nodes"]
        if not (edge(a, b) and edge(b, c) and nonedge(a, c)):
            violations["path3"] += 1
    for instance in instances.get("triangle", []):
        a, b, c = instance["nodes"]
        if not (edge(a, b) and edge(b, c) and edge(a, c)):
            violations["triangle"] += 1
    for instance in instances.get("cycle4", []):
        a, b, c, d = instance["nodes"]
        if not (edge(a, b) and edge(b, c) and edge(c, d) and edge(d, a)
                and nonedge(a, c) and nonedge(b, d)):
            violations["cycle4"] += 1
    for instance in instances.get("diamond", []):
        a, b, c, d = instance["nodes"]
        if not (edge(a, b) and edge(a, c) and edge(b, c) and edge(b, d) and edge(c, d)
                and nonedge(a, d)):
            violations["diamond"] += 1
    for instance in instances.get("path4", []):
        a, b, c, d = instance["nodes"]
        if not (edge(a, b) and edge(b, c) and edge(c, d)
                and nonedge(a, c) and nonedge(a, d) and nonedge(b, d)):
            violations["path4"] += 1
    for instance in instances.get("triangle_tail", []):
        a, b, c, d = instance["nodes"]
        if not (edge(a, b) and edge(a, c) and edge(b, c) and edge(c, d)
                and nonedge(a, d) and nonedge(b, d)):
            violations["triangle_tail"] += 1
    for instance in instances.get("K4", []):
        if not all(edge(x, y) for x, y in combinations(instance["nodes"], 2)):
            violations["K4"] += 1
    return violations


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

    # cliques are enumerated, never sampled: the earlier "109" was what 60,000 random attempts found
    cliques = enumerate_cliques(n, adjacency)
    cliques.sort()
    if len(cliques) > MAX_INSTANCES:
        chosen = rng.choice(len(cliques), size=MAX_INSTANCES, replace=False)
        instances["K4"] = [{"nodes": list(cliques[int(index)])} for index in sorted(chosen)]
    else:
        instances["K4"] = [{"nodes": list(clique)} for clique in cliques]
    exhaustive = {"K4": {"exhaustive_count": len(cliques),
                         "used": len(instances["K4"]),
                         "sampled": len(cliques) > MAX_INSTANCES}}

    violations = validate(instances, adjacency)

    degrees = np.array([len(adjacency[i]) for i in range(n)])
    order = np.argsort(degrees, kind="mergesort")
    bucket_of = np.empty(n, dtype=np.int64)
    bucket_of[order] = np.minimum((np.arange(n) * 5) // max(n, 1), 4)
    bucket_pool = [np.flatnonzero(bucket_of == b) for b in range(5)]

    # the tightness null is drawn once per instance, with distinct items, and shared by every
    # representation -- otherwise two representations would be compared against different nulls
    matched_sets: dict[str, list[list[tuple[int, ...]]]] = {}
    for name in ("triangle", "K4"):
        blocks: list[list[tuple[int, ...]]] = []
        for instance in instances[name]:
            nodes = set(instance["nodes"])
            sets: list[tuple[int, ...]] = []
            for _ in range(MATCHED_DRAWS):
                drawn: list[int] = []
                for node in instance["nodes"]:
                    pool = bucket_pool[int(bucket_of[node])]
                    for _ in range(64):
                        candidate = int(pool[rng.integers(pool.size)])
                        if (candidate in nodes or candidate in drawn
                                or candidate in adjacency[node]):
                            continue
                        drawn.append(candidate)
                        break
                if len(drawn) == len(instance["nodes"]):
                    sets.append(tuple(drawn))
            blocks.append(sets)
        matched_sets[name] = blocks

    journal: dict = {"seed": SEED, "max_instances": MAX_INSTANCES, "min_instances": MIN_INSTANCES,
                     "matched_draws": MATCHED_DRAWS, "hypotheses": HYPOTHESIS,
                     "exhaustive": exhaustive, "structural_violations": violations,
                     "note": "every structure uses one rule: value = mean(closer) - mean(farther), "
                             "supported iff value < 0; instances sampled without replacement; "
                             "instances re-validated against their definitions",
                     "arms": {}}

    for arm in ARMS:
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)
        if len(q) != n:
            raise SystemExit(f"{arm}: {len(q)} vectors for {n} items")

        pairs = [(a, b) for name in STRUCTURES for instance in instances[name]
                 for group in ("closer", "farther", "hop1", "hop2", "hop3")
                 for a, b in instance.get(group, [])]
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
                for index, instance in enumerate(instances[name]):
                    if name in ("triangle", "K4"):
                        sets = matched_sets[name][index]
                        if not sets:
                            continue
                        inside = float(np.mean([distance(representation, a, b)
                                                for a, b in combinations(instance["nodes"], 2)]))
                        null = float(np.mean([
                            np.mean([distance(representation, a, b)
                                     for a, b in combinations(drawn, 2)]) for drawn in sets]))
                        value = inside - null
                        supported = value < 0
                    elif name == "path4":
                        hop1 = float(np.mean([distance(representation, a, b)
                                              for a, b in instance["hop1"]]))
                        hop2 = float(np.mean([distance(representation, a, b)
                                              for a, b in instance["hop2"]]))
                        hop3 = float(np.mean([distance(representation, a, b)
                                              for a, b in instance["hop3"]]))
                        value = hop3 - hop1
                        hop_first += int(hop1 < hop2)
                        hop_second += int(hop2 < hop3)
                        supported = hop1 < hop2 < hop3
                    else:
                        closer = float(np.mean([distance(representation, a, b)
                                                for a, b in instance["closer"]]))
                        farther = float(np.mean([distance(representation, a, b)
                                                 for a, b in instance["farther"]]))
                        value = closer - farther
                        supported = value < 0
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
    print("structural violations:", violations)
    print("K4:", exhaustive["K4"])
    for arm, block in journal["arms"].items():
        print(f"\n=== {arm}: one distance hypothesis per structure (strict instances) ===")
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
