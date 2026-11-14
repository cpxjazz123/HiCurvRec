"""Relative distance patterns inside the nine local structures.

The question the owner actually wants answered is not "can a model predict whether A and B share a
behaviour edge".  It is: for one local structure of the real behaviour graph, does the quantized
vector and the SID tree express the **relative distances between the items inside that structure**
-- and which pattern do they express.  Nothing here is a classification task; the behaviour graph
only says which items form a structure and what their positions inside it are.

Two goals, never mixed into one number:

    A  relative-distance preservation.  The reference is the pre-quantization vector z: inside one
       structure instance, does a representation keep the ordering of the z-distances?  Measured as
       a per-instance Kendall tau over the instance's pairs, averaged per structure.

    B  structural semantic pattern.  Each structure has its own signature statistic (the ring
       statistic of a four-cycle, the spoke-versus-leaf statistic of a star, ...).  The behaviour
       graph does not say which sign is "correct" -- a four-cycle does not have to put the opposite
       nodes closer -- so the statistic is reported with its sign, its magnitude and the share of
       instances that carry that sign, for every representation.  Agreement between representations
       is a result, not an assumption.

Representations, all on the same structures, the same instances and the same pair labels:

    z          the pre-quantization vector (the reference for goal A, not a candidate)
    q          the quantized vector, Euclidean distance
    sid        the SID tree as closeness:  d = 1 - lcp / 3   (a deeper shared prefix is closer)
    sid_rev    the SID tree as difference: d = lcp / 3       (a deeper shared prefix is farther)
    fused      z-scored q + z-scored sid
    fused_rev  z-scored q + z-scored sid_rev

The statistics (all on distances, so "smaller" is always "closer"):

    path3         R = mean(centre to ends) - d(end to end)
    triangle      CV = std of the three edge distances / their mean
    path4         mean distance at hop 1, 2 and 3, and whether they increase with the hop
    star4         R = mean(centre to leaf) - mean(leaf to leaf)
    cycle4        R = mean(adjacent) - mean(opposite)
    triangle_tail R = mean(inside the triangle) - mean(tail to the triangle)
    diamond       R = d(the one missing pair) - mean(the five edges)
    K4            CV of the six distances
    edge          not applicable: two nodes have one distance and no internal comparison

    python3 results/research/issue215/relative_distance_patterns.py [ARM,...]
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
JOURNAL = OUT / "relative_distance_patterns.json"
#: owner 2026-11-14: the work continues on TIGER only; the list stays a parameter so a different
#: arm can be read without touching this file
ARMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("TIGER",)
SEED = 42
MAX_INSTANCES = 4000          # per structure
STRUCTURES = ("edge", "path3", "triangle", "path4", "star4", "cycle4", "triangle_tail",
              "diamond", "K4")
REPRESENTATIONS = ("z", "q", "sid", "sid_rev", "fused", "fused_rev")


def load_behaviour() -> tuple[int, list[set[int]], np.ndarray]:
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return n, adjacency, edges


def build_instances(n: int, adjacency: list[set[int]], rng: np.random.Generator) -> dict:
    """Every structure type as a list of instances, each a dict of pair groups.

    The groups are the pair labels the structure itself defines; the graph supplies only the items
    and their positions, never a "should be closer" answer.
    """
    neighbours = [sorted(adjacency[i]) for i in range(n)]
    instances: dict[str, list[dict]] = {name: [] for name in STRUCTURES}

    def pick(values: list[int]) -> int:
        return int(values[rng.integers(len(values))])

    def anchor_at(degree: int) -> int | None:
        candidates = np.flatnonzero(np.array([len(w) for w in neighbours]) >= degree)
        return None if candidates.size == 0 else int(pick(candidates.tolist()))

    # edge: two nodes, no internal comparison possible
    for _ in range(MAX_INSTANCES):
        a = anchor_at(1)
        if a is None:
            continue
        b = pick(neighbours[a])
        instances["edge"].append({"groups": {"edge": [(a, b)]}})

    # path3: b in the middle, a and c not adjacent
    for _ in range(MAX_INSTANCES):
        b = anchor_at(2)
        if b is None:
            continue
        a = pick(neighbours[b])
        rest = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest:
            continue
        c = pick(rest)
        instances["path3"].append({"groups": {"spoke": [(b, a), (b, c)], "end": [(a, c)]}})

    # triangle
    for _ in range(MAX_INSTANCES):
        a = anchor_at(2)
        if a is None:
            continue
        b = pick(neighbours[a])
        common = [w for w in neighbours[a] if w != b and w in adjacency[b]]
        if not common:
            continue
        c = pick(common)
        instances["triangle"].append({"groups": {"edge": [(a, b), (a, c), (b, c)]}})

    # path4: a - b - c - d, with a and d not adjacent
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
        instances["path4"].append({"groups": {"hop1": [(a, b), (b, c), (c, d)],
                                              "hop2": [(a, c), (b, d)],
                                              "hop3": [(a, d)]}})

    # star4: one centre, three leaves that are mutually unconnected
    for _ in range(MAX_INSTANCES):
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        leaves = [neighbours[a][int(index)] for index in picks]
        if any(leaves[i] in adjacency[leaves[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["star4"].append({"groups": {
            "spoke": [(a, leaf) for leaf in leaves],
            "leaf": list(combinations(leaves, 2))}})

    # cycle4: a - b - c - d - a
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
        instances["cycle4"].append({"groups": {
            "adjacent": [(a, b), (b, c), (c, d), (d, a)],
            "opposite": [(a, c), (b, d)]}})

    # triangle with a tail: triangle a-b-c, tail d on c, d adjacent to neither a nor b
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
        instances["triangle_tail"].append({"groups": {
            "internal": [(a, b), (a, c), (b, c)],
            "tail": [(d, a), (d, b), (d, c)]}})

    # diamond: the complete graph on four nodes minus exactly one edge (a and d are the missing pair)
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
        instances["diamond"].append({"groups": {
            "edge": [(a, b), (a, c), (b, c), (b, d), (c, d)],
            "missing": [(a, d)]}})

    # K4: six edges among four nodes
    for _ in range(MAX_INSTANCES):
        a = anchor_at(3)
        if a is None:
            continue
        picks = rng.choice(len(neighbours[a]), size=3, replace=False)
        others = [neighbours[a][int(index)] for index in picks]
        if not all(others[i] in adjacency[others[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["K4"].append({"groups": {"edge": list(combinations([a] + others, 2))}})

    return {name: block[:MAX_INSTANCES] for name, block in instances.items()}


def pair_distance(vectors: np.ndarray, pair: tuple[int, int]) -> float:
    left, right = pair
    return float(np.linalg.norm(vectors[left] - vectors[right]))


def lcp(tokens: np.ndarray, pair: tuple[int, int]) -> int:
    left, right = tokens[pair[0]], tokens[pair[1]]
    count = 0
    for x, y in zip(left.tolist(), right.tolist()):
        if x != y:
            break
        count += 1
    return count


def distance_over(instance: dict, how) -> dict[str, list[float]]:
    return {label: [how(pair) for pair in pairs] for label, pairs in instance["groups"].items()}


def statistic(structure: str, distances: dict[str, list[float]]) -> float | None:
    """The structure's signature statistic, in distance units; None where the structure has none."""
    mean = lambda values: float(np.mean(values))  # noqa: E731
    if structure == "edge":
        return None
    if structure == "path3":
        return mean(distances["spoke"]) - mean(distances["end"])
    if structure == "triangle":
        return float(np.std(distances["edge"]))
    if structure == "path4":
        return None                      # reported as the three hop means, not as one number
    if structure == "star4":
        return mean(distances["spoke"]) - mean(distances["leaf"])
    if structure == "cycle4":
        return mean(distances["adjacent"]) - mean(distances["opposite"])
    if structure == "triangle_tail":
        return mean(distances["internal"]) - mean(distances["tail"])
    if structure == "diamond":
        return mean(distances["missing"]) - mean(distances["edge"])
    if structure == "K4":
        return float(np.std(distances["edge"]))
    raise KeyError(structure)


def kendall(first: list[float], second: list[float]) -> float:
    """Kendall tau over the pairs of one instance: how much of the z ordering a representation keeps."""
    concordant = discordant = 0
    for i, j in combinations(range(len(first)), 2):
        left = first[i] - first[j]
        right = second[i] - second[j]
        product = left * right
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    total = concordant + discordant
    return 0.0 if total == 0 else (concordant - discordant) / total


def main() -> int:
    n, adjacency, _ = load_behaviour()
    rng = np.random.default_rng(SEED)
    instances = build_instances(n, adjacency, rng)

    journal: dict = {"seed": SEED, "max_instances": MAX_INSTANCES,
                     "structures": {}, "arms": {}}

    for arm in ARMS:
        z = np.load(VECTORS / f"{arm}_z.npy").astype(np.float64)
        q = np.load(VECTORS / f"{arm}_q.npy").astype(np.float64)
        tokens = np.load(VECTORS / f"{arm}_tokens.npy").astype(np.int64)
        if not (len(z) == len(q) == len(tokens)):
            raise SystemExit(f"{arm}: vectors and tokens disagree on the item count")

        # standardise the two distance scales over the sampled pairs before fusing them
        q_scale: list[float] = []
        sid_scale: list[float] = []
        for name in STRUCTURES:
            for instance in instances[name]:
                for pairs in instance["groups"].values():
                    for pair in pairs:
                        q_scale.append(pair_distance(q, pair))
                        sid_scale.append(1.0 - lcp(tokens, pair) / 3.0)
        q_mean, q_std = float(np.mean(q_scale)), float(np.std(q_scale) or 1.0)
        sid_mean, sid_std = float(np.mean(sid_scale)), float(np.std(sid_scale) or 1.0)

        def how(representation: str):
            if representation == "z":
                return lambda pair: pair_distance(z, pair)
            if representation == "q":
                return lambda pair: pair_distance(q, pair)
            if representation == "sid":
                return lambda pair: 1.0 - lcp(tokens, pair) / 3.0
            if representation == "sid_rev":
                return lambda pair: lcp(tokens, pair) / 3.0
            if representation == "fused":
                return lambda pair: ((pair_distance(q, pair) - q_mean) / q_std
                                     + (1.0 - lcp(tokens, pair) / 3.0 - sid_mean) / sid_std)
            if representation == "fused_rev":
                return lambda pair: ((pair_distance(q, pair) - q_mean) / q_std
                                     + (lcp(tokens, pair) / 3.0 - sid_mean) / sid_std)
            raise KeyError(representation)

        scales: dict[str, tuple[float, float]] = {}
        for representation in REPRESENTATIONS:
            how_distance = how(representation)
            values = [how_distance(pair)
                      for name in STRUCTURES for instance in instances[name]
                      for pairs in instance["groups"].values() for pair in pairs]
            scales[representation] = (float(np.mean(values)), float(np.std(values) or 1.0))

        block: dict = {}
        for name in STRUCTURES:
            block[name] = {"instances": len(instances[name])}
            for representation in REPRESENTATIONS:
                how_distance = how(representation)
                values: list[float] = []
                taus: list[float] = []
                hops: dict[str, list[float]] = {key: [] for key in ("hop1", "hop2", "hop3")}
                for instance in instances[name]:
                    distances = distance_over(instance, how_distance)
                    value = statistic(name, distances)
                    if value is not None:
                        values.append(value)
                    if name == "path4":
                        for key in hops:
                            hops[key].append(float(np.mean(distances[key])))
                    if representation != "z":
                        z_distances = distance_over(instance, how("z"))
                        flat = [value for label in instance["groups"] for value in distances[label]]
                        flat_z = [value for label in instance["groups"] for value in z_distances[label]]
                        taus.append(kendall(flat_z, flat))
                scale_mean, scale_std = scales[representation]
                entry: dict = {"tau_vs_z": float(np.mean(taus)) if taus else None,
                               "scale_mean": scale_mean, "scale_std": scale_std}
                if values:
                    entry["statistic"] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "share_positive": float(np.mean([value > 0 for value in values])),
                    }
                    if name in ("triangle", "K4"):
                        # dispersion relative to the representation's own spread
                        entry["normalised"] = float(np.mean(values) / scale_std)
                    else:
                        # the difference in distance units, expressed in the representation's own spread
                        entry["normalised"] = float(np.mean(values) / scale_std)
                if name == "path4":
                    entry["hop_means"] = {key: float(np.mean(values)) for key, values in hops.items()}
                    entry["hop_ratios"] = {key: (float(np.mean(values)) - scale_mean) / scale_std
                                            for key, values in hops.items()}
                    entry["hop_increasing"] = float(np.mean([
                        (hop1 < hop2) and (hop2 < hop3)
                        for hop1, hop2, hop3 in zip(hops["hop1"], hops["hop2"], hops["hop3"])]))
                block[name][representation] = entry
        journal["arms"][arm] = block

    JOURNAL.write_text(json.dumps(journal, indent=1, default=float))
    for arm, block in journal["arms"].items():
        print(f"\n=== {arm} ===")
        for name in STRUCTURES:
            if name == "edge":
                print(f"{name:<14} n/a (two nodes, one distance)")
                continue
            line = f"{name:<14}"
            for representation in ("q", "sid", "sid_rev", "fused", "fused_rev"):
                entry = block[name][representation]
                if name == "path4":
                    ratios = entry["hop_ratios"]
                    line += (f"  {representation}: {ratios['hop1']:+.2f}/{ratios['hop2']:+.2f}/"
                             f"{ratios['hop3']:+.2f}sd inc={entry['hop_increasing']:.2f}")
                elif name in ("triangle", "K4"):
                    line += (f"  {representation}: disp={entry['normalised']:.3f}"
                             f"(>0 n/a)")
                else:
                    stat = entry["statistic"]
                    line += (f"  {representation}: {entry['normalised']:+.3f}sd"
                             f" (>0 {stat['share_positive']:.2f})")
            print(line)
        print("tau vs z (goal A):")
        for name in STRUCTURES:
            if name == "edge":
                continue
            row = "  ".join(f"{representation}={block[name][representation]['tau_vs_z']:+.3f}"
                            for representation in ("q", "sid", "sid_rev", "fused", "fused_rev"))
            print(f"  {name:<14} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
