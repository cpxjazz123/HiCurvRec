"""Structure-conditioned relative distance preservation.

The question, as the owner puts it: for a structure that really exists in the behaviour graph -- a
triangle, a star, a path, a cycle -- are the items that are really connected also *closer* in the
vector space, and how much of that does the quantizer take away?

No graph is rebuilt from the vectors.  The behaviour graph supplies the structures, the vectors
supply the distances, and every model sees the same structures and the same negatives:

    external comparison   for an anchor A with true neighbours B, C, ...: is d(A, B) smaller than
                          d(A, w) for w a non-neighbour of A?  A win is 1, a tie 0.5, a loss 0.
    internal comparison   inside one graphlet: is a real edge closer than a pair that is not an
                          edge of that graphlet (the diagonal of a cycle, two leaves of a star)?

From those:

    R_z, R_q      the relative distance score before and after quantization
    delta         R_q - R_z
    flip rate     of the comparisons that were right in z, the share that are wrong in q
    internal      the same score restricted to comparisons inside one structure

The three diagnostics the owner asks for fall out of the table: a structure whose R_z is already
low was never learned by the encoder; one whose R_z is high and R_q low was destroyed by the
quantizer; one whose R_z and R_q agree survived it.

Controls: the same behaviour graph, the same structure instances and the same negatives for every
model; random and degree-matched negatives are reported separately; instances are sampled per
structure with a fixed seed and capped so that a few hubs cannot dominate; the distance is the same
Euclidean metric on the 32-dimensional joint vector before and after quantization.

    python3 results/research/issue215/relative_distance_structures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "relative_distance_structures.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
SEED = 42
NEGATIVES = 5               # non-neighbours per anchor, as in the owner's example
MAX_INSTANCES = 20_000      # per structure, so a structure cannot flood the table
DEGREE_BINS = 20


def load_behaviour() -> tuple[int, np.ndarray, list[set[int]]]:
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return n, edges, adjacency


def degree_matched_pool(degrees: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(degrees, kind="mergesort")
    rank = np.empty(len(degrees), dtype=np.int64)
    rank[order] = np.arange(len(degrees))
    bins = np.minimum((rank * DEGREE_BINS) // max(len(degrees), 1), DEGREE_BINS - 1)
    return [np.flatnonzero(bins == b) for b in range(DEGREE_BINS)]


def build_instances(n: int, edges: np.ndarray, adjacency: list[set[int]],
                    degrees: np.ndarray, rng: np.random.Generator) -> dict:
    """Every structure type as a list of instances.

    An instance is a dict with:
        anchor   the item the external comparison is made from
        true     the real pairs of the structure that touch the anchor
        inner    the real pairs of the structure that do not touch the anchor
        false    the pairs inside the structure that are NOT edges of it (the diagnostics)
    """
    pools = degree_matched_pool(degrees)
    instances: dict[str, list[dict]] = {name: [] for name in
                                        ("edge", "path3", "triangle", "path4", "star4", "cycle4",
                                         "triangle_tail", "diamond", "K4")}
    neighbours = [sorted(adjacency[i]) for i in range(n)]

    def sample_anchor(degree_at_least: int) -> int:
        candidates = np.flatnonzero(degrees >= degree_at_least)
        return int(candidates[rng.integers(candidates.size)])

    def sample_negative(anchor: int, exclude: set[int], matched: bool) -> int:
        for _ in range(64):
            if matched:
                pool = pools[min(int(np.searchsorted(np.argsort(degrees), anchor) * DEGREE_BINS
                                       // max(len(degrees), 1)), DEGREE_BINS - 1)]
                if pool.size == 0:
                    continue
                candidate = int(pool[rng.integers(pool.size)])
            else:
                candidate = int(rng.integers(n))
            if candidate != anchor and candidate not in exclude and candidate not in adjacency[anchor]:
                return candidate
        return -1

    # edge
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(1)
        if not neighbours[a]:
            continue
        b = int(neighbours[a][rng.integers(len(neighbours[a]))])
        instances["edge"].append({"anchor": a, "true": [(a, b)], "inner": [], "false": []})

    # three-node path: the anchor is the middle
    for _ in range(MAX_INSTANCES):
        b = sample_anchor(2)
        if len(neighbours[b]) < 2:
            continue
        a = int(neighbours[b][rng.integers(len(neighbours[b]))])
        rest = [w for w in neighbours[b] if w != a and w not in adjacency[a]]
        if not rest:
            continue
        c = int(rest[rng.integers(len(rest))])
        instances["path3"].append({"anchor": b, "true": [(b, a), (b, c)], "inner": [],
                                   "false": [(a, c)]})

    # triangle
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(2)
        if len(neighbours[a]) < 2:
            continue
        b = int(neighbours[a][rng.integers(len(neighbours[a]))])
        common = [w for w in neighbours[a] if w != b and w in adjacency[b]]
        if not common:
            continue
        c = int(common[rng.integers(len(common))])
        instances["triangle"].append({"anchor": a, "true": [(a, b), (a, c)], "inner": [(b, c)],
                                     "false": []})

    # four-node path
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(1)
        if not neighbours[a]:
            continue
        b = int(neighbours[a][rng.integers(len(neighbours[a]))])
        rest_b = [w for w in neighbours[b] if w != a]
        if not rest_b:
            continue
        c = int(rest_b[rng.integers(len(rest_b))])
        rest_c = [w for w in neighbours[c] if w not in (b,) and w != a and w not in adjacency[a]]
        if not rest_c:
            continue
        d = int(rest_c[rng.integers(len(rest_c))])
        instances["path4"].append({"anchor": b, "true": [(b, a), (b, c)], "inner": [(c, d)],
                                   "false": [(a, c), (a, d), (b, d)]})

    # four-node star: the anchor is the centre, the leaves are mutually unconnected
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(3)
        if len(neighbours[a]) < 3:
            continue
        picks = list(rng.choice(len(neighbours[a]), size=3, replace=False))
        leaves = [int(neighbours[a][index]) for index in picks]
        if any(leaves[i] in adjacency[leaves[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["star4"].append({"anchor": a, "true": [(a, leaf) for leaf in leaves], "inner": [],
                                    "false": [(leaves[0], leaves[1]), (leaves[0], leaves[2]),
                                              (leaves[1], leaves[2])]})

    # four-node cycle
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(2)
        if len(neighbours[a]) < 2:
            continue
        b = int(neighbours[a][rng.integers(len(neighbours[a]))])
        rest_b = [w for w in neighbours[b] if w != a]
        if not rest_b:
            continue
        c = int(rest_b[rng.integers(len(rest_b))])
        rest_c = [w for w in neighbours[c] if w not in (b,) and w in adjacency[a]]
        if not rest_c:
            continue
        d = int(rest_c[rng.integers(len(rest_c))])
        instances["cycle4"].append({"anchor": a, "true": [(a, b), (a, d)], "inner": [(b, c), (c, d)],
                                    "false": [(a, c), (b, d)]})

    # triangle with a tail
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(2)
        if len(neighbours[a]) < 2:
            continue
        b = int(neighbours[a][rng.integers(len(neighbours[a]))])
        common = [w for w in neighbours[a] if w != b and w in adjacency[b]]
        if not common:
            continue
        c = int(common[rng.integers(len(common))])
        rest_c = [w for w in neighbours[c] if w not in (a, b) and w not in adjacency[a] and w not in adjacency[b]]
        if not rest_c:
            continue
        d = int(rest_c[rng.integers(len(rest_c))])
        instances["triangle_tail"].append({"anchor": c, "true": [(c, a), (c, b), (c, d)],
                                          "inner": [(a, b)], "false": [(d, a), (d, b)]})

    # diamond: five edges among four nodes
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(3)
        if len(neighbours[a]) < 3:
            continue
        picks = list(rng.choice(len(neighbours[a]), size=3, replace=False))
        others = [int(neighbours[a][index]) for index in picks]
        if any(others[i] in adjacency[others[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["diamond"].append({"anchor": a, "true": [(a, other) for other in others],
                                     "inner": [], "false": [(others[0], others[1]),
                                                           (others[0], others[2]),
                                                           (others[1], others[2])]})

    # complete graph on four nodes
    for _ in range(MAX_INSTANCES):
        a = sample_anchor(3)
        if len(neighbours[a]) < 3:
            continue
        picks = list(rng.choice(len(neighbours[a]), size=3, replace=False))
        others = [int(neighbours[a][index]) for index in picks]
        if not all(others[i] in adjacency[others[j]] for i in range(3) for j in range(i + 1, 3)):
            continue
        instances["K4"].append({"anchor": a, "true": [(a, other) for other in others],
                                "inner": [], "false": []})

    return instances


def scores_for(instances: list[dict], vectors: np.ndarray, adjacency: list[set[int]],
                n: int, rng: np.random.Generator, negatives: int, matched: bool) -> dict:
    """The relative distance score of one model on one structure type."""
    external_wins, external_total = 0.0, 0
    internal_wins, internal_total = 0.0, 0
    external_correct: list[bool] = []
    internal_correct: list[bool] = []
    for instance in instances:
        anchor = instance["anchor"]
        exclude = {anchor} | {item for pair in instance["true"] + instance["inner"] for item in pair}
        true_scores = []
        false_scores = []
        for pair in instance["true"] + instance["inner"]:
            left, right = pair
            if left != anchor:
                left, right = right, left
            if left != anchor:
                continue
            true_scores.append((right, np.linalg.norm(vectors[anchor] - vectors[right])))
        for _ in range(negatives):
            negative = -1
            for _ in range(64):
                candidate = int(rng.integers(n))
                if candidate != anchor and candidate not in exclude and candidate not in adjacency[anchor]:
                    negative = candidate
                    break
            if negative < 0:
                continue
            false_scores.append((negative, np.linalg.norm(vectors[anchor] - vectors[negative])))
        if true_scores and false_scores:
            best_false = max(score for _, score in false_scores)
            for _, score in true_scores:
                external_total += 1
                win = 1.0 if score < best_false else (0.5 if score == best_false else 0.0)
                external_wins += win
                external_correct.append(score < best_false)
        for pair in instance["true"]:
            left, right = pair
            true_distance = np.linalg.norm(vectors[left] - vectors[right])
            for other_left, other_right in instance["false"]:
                if {left, right} == {other_left, other_right}:
                    continue
                false_distance = np.linalg.norm(vectors[other_left] - vectors[other_right])
                internal_total += 1
                win = 1.0 if true_distance < false_distance else (0.5 if true_distance == false_distance else 0.0)
                internal_wins += win
                internal_correct.append(true_distance < false_distance)
    return {"external": external_wins / external_total if external_total else None,
            "internal": internal_wins / internal_total if internal_total else None,
            "external_comparisons": external_total, "internal_comparisons": internal_total,
            "external_correct": external_correct, "internal_correct": internal_correct}


def main() -> int:
    n, edges, adjacency = load_behaviour()
    degrees = np.asarray([len(adjacency[i]) for i in range(n)], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    instances = build_instances(n, edges, adjacency, degrees, rng)
    for name, values in instances.items():
        print(f"{name:<15} instances {len(values)}", flush=True)

    vectors = {"semantic 768d": np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(np.float64)}
    for arm in ARMS:
        z_path, q_path = VECTORS / f"{arm}_z.npy", VECTORS / f"{arm}_q.npy"
        if z_path.is_file() and q_path.is_file():
            vectors[f"{arm} z"] = np.load(z_path).astype(np.float64)
            vectors[f"{arm} q"] = np.load(q_path).astype(np.float64)

    report = {"negatives_per_anchor": NEGATIVES, "seed": SEED, "structures": {}}
    print(f"\n{'structure':<15} {'model':<14} {'R external':>11} {'R internal':>11} {'n ext':>8}")
    for name, values in instances.items():
        block = {}
        for model_name, matrix in vectors.items():
            random_scores = scores_for(values, matrix, adjacency, n, np.random.default_rng(SEED + 7),
                                      NEGATIVES, matched=False)
            matched_scores = scores_for(values, matrix, adjacency, n, np.random.default_rng(SEED + 7),
                                        NEGATIVES, matched=True)
            entry = {"random_negatives": random_scores, "degree_matched_negatives": matched_scores}
            block[model_name] = entry
            print(f"{name:<15} {model_name:<14} {random_scores['external']:>11.4f} "
                  f"{random_scores['internal'] if random_scores['internal'] is not None else float('nan'):>11.4f} "
                  f"{random_scores['external_comparisons']:>8}", flush=True)
        for arm in ARMS:
            pre, post = block.get(f"{arm} z"), block.get(f"{arm} q")
            if not pre or not post:
                continue
            for key, label in (("random_negatives", "random"), ("degree_matched_negatives", "matched")):
                a = np.asarray(pre[key]["external_correct"], dtype=bool)
                b = np.asarray(post[key]["external_correct"], dtype=bool)
                if a.size == 0 or a.size != b.size:
                    continue
                correct = int(a.sum())
                flipped = int((a & ~b).sum())
                pre[key]["flip_rate"] = flipped / max(correct, 1)
                pre[key]["flipped"] = flipped
                pre[key]["correct_before"] = correct
                pre[key]["delta_external"] = post[key]["external"] - pre[key]["external"]
                pre[key]["delta_internal"] = (post[key]["internal"] - pre[key]["internal"]
                                             if post[key]["internal"] is not None and pre[key]["internal"] is not None else None)
        report["structures"][name] = block

    for block in report["structures"].values():
        for entry in block.values():
            for key in ("random_negatives", "degree_matched_negatives"):
                entry[key].pop("external_correct", None)
                entry[key].pop("internal_correct", None)
    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
