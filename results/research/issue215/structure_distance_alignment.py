"""Structure distance alignment: is a structure's real edge set closer than the same anchors' non-edges?

The owner's protocol, verbatim: for a star with centre A and neighbours B, C, D, E,

    S_real  = d(A,B) + d(A,C) + d(A,D) + d(A,E)
    S_other = d(A,F) + d(A,G) + d(A,H) + d(A,I)      F..I not neighbours of A, not in the structure

and the comparison succeeds when S_real < S_other.  The alignment rate of a structure is the share of
its instances that succeed.  The strict variant -- the true neighbours are exactly the k nearest
items of the anchor -- is reported alongside, because it asks for much more than a sum comparison.

The same instances, anchors and pair counts are used for every representation, and the reference is
the behaviour graph, never a pre-quantization vector and never a trained edge predictor:

    q          the quantized vector, Euclidean distance
    sid        the SID tree as closeness:  d = 1 - lcp / 3
    sid_rev    the SID tree as difference: d = lcp / 3
    fused      z-scored q + z-scored sid
    fused_rev  z-scored q + z-scored sid_rev

One number this metric does NOT give: it measures distance alignment with the behaviour neighbours,
not that the whole structure has been reconstructed (the owner says so explicitly).

    python3 results/research/issue215/structure_distance_alignment.py [ARM,...]
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
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "structure_distance_alignment.json"
ARMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("TIGER",)
SEED = 42
MAX_INSTANCES = 2000          # per structure
DRAWS = 10                    # random non-neighbour sets per instance
STRICT_ANCHORS = 400          # anchors per structure for the "exactly the k nearest" test
STRUCTURES = ("edge", "path3", "triangle", "path4", "star4", "cycle4", "triangle_tail",
              "diamond", "K4")
REPRESENTATIONS = ("q", "sid", "sid_rev", "fused", "fused_rev")


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
    """Every structure as {nodes, edges, groups}; `edges` are the behaviour edges of the structure."""
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
        nodes = [a, b]
        instances["edge"].append({"nodes": nodes, "edges": [(a, b)], "groups": {"edge": [(a, b)]}})

    for _ in range(MAX_INSTANCES):
        b = anchor_at(2)
        if b is None:
            continue
        a = pick(neighbours[b])
        rest = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest:
            continue
        c = pick(rest)
        instances["path3"].append({"nodes": [a, b, c], "edges": [(b, a), (b, c)],
                                  "groups": {"spoke": [(b, a), (b, c)], "end": [(a, c)]}})

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
                                      "edges": [(a, b), (a, c), (b, c)],
                                      "groups": {"edge": [(a, b), (a, c), (b, c)]}})

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
                                   "edges": [(a, b), (b, c), (c, d)],
                                   "groups": {"hop1": [(a, b), (b, c), (c, d)],
                                              "hop2": [(a, c), (b, d)], "hop3": [(a, d)]}})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        leaves = [neighbours[a][int(index)] for index in picks]
        if any(leaves[i] in adjacency[leaves[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["star4"].append({"nodes": [a] + leaves,
                                   "edges": [(a, leaf) for leaf in leaves],
                                   "groups": {"spoke": [(a, leaf) for leaf in leaves],
                                              "leaf": list(combinations(leaves, 2))}})

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
                                    "edges": [(a, b), (b, c), (c, d), (d, a)],
                                    "groups": {"adjacent": [(a, b), (b, c), (c, d), (d, a)],
                                               "opposite": [(a, c), (b, d)]}})

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
        instances["triangle_tail"].append({
            "nodes": [a, b, c, d],
            "edges": [(a, b), (a, c), (b, c), (c, d)],
            "groups": {"internal": [(a, b), (a, c), (b, c)], "tail": [(d, c)]}})

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
        instances["diamond"].append({
            "nodes": [a, b, c, d],
            "edges": [(a, b), (a, c), (b, c), (b, d), (c, d)],
            "groups": {"edge": [(a, b), (a, c), (b, c), (b, d), (c, d)], "missing": [(a, d)]}})

    for _ in range(MAX_INSTANCES):
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        others = [neighbours[a][int(index)] for index in picks]
        if not all(others[i] in adjacency[others[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        nodes = [a] + others
        instances["K4"].append({"nodes": nodes, "edges": list(combinations(nodes, 2)),
                                "groups": {"edge": list(combinations(nodes, 2))}})

    return {name: block for name, block in instances.items()}


def tree_distance_from(tokens: np.ndarray, anchor: int) -> np.ndarray:
    """1 - lcp/3 for every item against one anchor, vectorised over the whole table."""
    match1 = tokens[:, 0] == tokens[anchor, 0]
    match2 = match1 & (tokens[:, 1] == tokens[anchor, 1])
    match3 = match2 & (tokens[:, 2] == tokens[anchor, 2])
    lcp_len = match1.astype(np.int16) + match2.astype(np.int16) + match3.astype(np.int16)
    return 1.0 - lcp_len / 3.0


def lcp(tokens: np.ndarray, left: int, right: int) -> int:
    count = 0
    for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
        if x != y:
            break
        count += 1
    return count


def main() -> int:
    n, adjacency = load_behaviour()
    rng = np.random.default_rng(SEED)
    instances = build_instances(adjacency, rng)
    journal: dict = {"seed": SEED, "max_instances": MAX_INSTANCES, "draws": DRAWS,
                     "strict_anchors": STRICT_ANCHORS, "arms": {}}

    for arm in ARMS:
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)
        if len(q) != n:
            raise SystemExit(f"{arm}: {len(q)} vectors for {n} items")

        # the scale of each representation over the same pair population
        sampled = [(int(pair[0]), int(pair[1]))
                   for name in STRUCTURES for instance in instances[name]
                   for pair in instance["edges"]]
        q_values = np.array([np.linalg.norm(q[a] - q[b]) for a, b in sampled])
        sid_values = np.array([1.0 - lcp(tokens, a, b) / 3.0 for a, b in sampled])
        q_mean, q_std = float(q_values.mean()), float(q_values.std() or 1.0)
        sid_mean, sid_std = float(sid_values.mean()), float(sid_values.std() or 1.0)

        def distance(representation: str, left: int, right: int) -> float:
            euclid = float(np.linalg.norm(q[left] - q[right]))
            tree = 1.0 - lcp(tokens, left, right) / 3.0
            if representation == "q":
                return euclid
            if representation == "sid":
                return tree
            if representation == "sid_rev":
                return 1.0 - tree
            if representation == "fused":
                return (euclid - q_mean) / q_std + (tree - sid_mean) / sid_std
            if representation == "fused_rev":
                return (euclid - q_mean) / q_std + (1.0 - tree - sid_mean) / sid_std
            raise KeyError(representation)

        block: dict = {}
        for name in STRUCTURES:
            block[name] = {"instances": len(instances[name])}
            for representation in REPRESENTATIONS:
                successes = 0
                ties = 0
                attempts = 0
                strict_hits = 0
                strict_attempts = 0
                all_draws_hits = 0
                rank_percentiles: list[float] = []
                for instance in instances[name]:
                    nodes = set(instance["nodes"])
                    edges = instance["edges"]
                    for _ in range(DRAWS):
                        attempts += 1
                        total_real = 0.0
                        total_other = 0.0
                        drawn: list[tuple[int, int]] = []
                        for left, right in edges:
                            total_real += distance(representation, left, right)
                            partner = -1
                            for _ in range(64):
                                candidate = int(rng.integers(n))
                                if (candidate != left and candidate not in nodes
                                        and candidate not in adjacency[left]):
                                    partner = candidate
                                    break
                            if partner < 0:
                                break
                            drawn.append((left, partner))
                            total_other += distance(representation, left, partner)
                        if len(drawn) != len(edges):
                            continue
                        if total_real < total_other:
                            successes += 1
                        elif total_real == total_other:
                            ties += 1
                    if successes == DRAWS:
                        all_draws_hits += 1
                    # strict: are the structural neighbours exactly the k nearest items of each anchor?
                    if strict_attempts < STRICT_ANCHORS:
                        anchor = edges[0][0]
                        truth = {other for left, right in edges
                                 for other in ((right,) if left == anchor else
                                               (left,) if right == anchor else ())}
                        if truth:
                            tree = tree_distance_from(tokens, anchor)
                            euclid = np.linalg.norm(q - q[anchor], axis=1)
                            if representation == "q":
                                ranks = euclid
                            elif representation == "fused":
                                ranks = ((euclid - q_mean) / q_std
                                         + (tree - sid_mean) / sid_std)
                            elif representation == "fused_rev":
                                ranks = ((euclid - q_mean) / q_std
                                         + (1.0 - tree - sid_mean) / sid_std)
                            else:
                                ranks = tree if representation == "sid" else 1.0 - tree
                            order = np.argsort(ranks, kind="mergesort")
                            k = len(truth)
                            if {int(item) for item in order[:k]} == truth:
                                strict_hits += 1
                            strict_attempts += 1
                            positions = {int(item): index for index, item in enumerate(order)}
                            rank_percentiles.extend(
                                positions[item] / n for item in truth if item in positions)
                entry = {"alignment_rate": successes / max(attempts, 1),
                          "tie_aware_rate": (successes + 0.5 * ties) / max(attempts, 1),
                          "tie_share": ties / max(attempts, 1),
                          "all_draws_rate": all_draws_hits / max(len(instances[name]), 1)}
                if strict_attempts:
                    entry["strict_rate"] = strict_hits / strict_attempts
                    entry["median_rank_percentile"] = float(np.median(rank_percentiles)) \
                        if rank_percentiles else None
                block[name][representation] = entry
        journal["arms"][arm] = block

    JOURNAL.write_text(json.dumps(journal, indent=1, default=float))
    for arm, block in journal["arms"].items():
        print(f"\n=== {arm}: alignment rate (S_real < S_other), {DRAWS} draws per instance ===")
        print(f"{'structure':<14}" + "".join(f"{rep:>12}" for rep in REPRESENTATIONS))
        for name in STRUCTURES:
            line = f"{name:<14}"
            for representation in REPRESENTATIONS:
                line += f"{block[name][representation]['alignment_rate']:>12.3f}"
            print(line)
        print("tie-aware rate (a tie scores 0.5) and tie share:")
        print(f"{'structure':<14}" + "".join(f"{rep:>12}" for rep in REPRESENTATIONS))
        for name in STRUCTURES:
            line = f"{name:<14}"
            for representation in REPRESENTATIONS:
                entry = block[name][representation]
                line += f"{entry['tie_aware_rate']:>7.3f}/{entry['tie_share']:<4.2f}"
            print(line)
        print("strict rate (structural neighbours are exactly the k nearest):")
        print(f"{'structure':<14}" + "".join(f"{rep:>12}" for rep in REPRESENTATIONS))
        for name in STRUCTURES:
            line = f"{name:<14}"
            for representation in REPRESENTATIONS:
                value = block[name][representation].get("strict_rate")
                line += f"{value:>12.4f}" if value is not None else f"{'n/a':>12}"
            print(line)
        print("median rank percentile of a true neighbour:")
        print(f"{'structure':<14}" + "".join(f"{rep:>12}" for rep in REPRESENTATIONS))
        for name in STRUCTURES:
            line = f"{name:<14}"
            for representation in REPRESENTATIONS:
                value = block[name][representation].get("median_rank_percentile")
                line += f"{value:>12.3f}" if value is not None else f"{'n/a':>12}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
