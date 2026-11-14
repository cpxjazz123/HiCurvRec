"""Structural recoverability: does the SID tree help recover local structure?

The quantizer's own vectors were measured against the real behaviour graph's local structures and the
internal scores came out weak (0.54 to 0.60 against a chance of 0.5): the vector says "these two are
related" but not reliably "this one is the edge".  The SID tree carries a different kind of
information -- how many leading levels two items share -- and the owner's hypothesis is that the two
together recover more than either alone.

Nothing is retrained.  For every arm its exported SID table and its quantized readout are frozen, and
the relation score is

    S(a, b) = -distance(a, b) + lambda * shared_prefix_levels(a, b)

with the distance normalised and lambda chosen on a validation split of the comparisons.  Four arms of
the comparison, all on the same structure instances and the same splits:

    A   the quantized vectors alone
    B   the SID tree alone
    C   the two combined
    D   the vectors plus a SHUFFLED SID tree -- the control for "more input is more capacity"

The hard negatives are the ones one step away: an item adjacent to the true neighbour but not to the
anchor.  Random negatives make everything look easy.

    python3 results/research/issue215/sid_tree_recoverability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "sid_tree_recoverability.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
SEED = 42
VALIDATION = 0.2
#: lambda may be negative: if the tree's information points the other way, a positive-only
#: grid would forbid the model from using it at all.
LAMBDA_GRID = np.round(np.arange(-10.0, 10.01, 0.05), 3)
STRUCTURES = ("path3", "path4", "star4", "cycle4", "triangle_tail", "diamond")

sys.path.insert(0, str(OUT))


#: where each arm's exported identifier table lives
SID_PATHS = {
    "TIGER": REPO / "analysis_sid_recall" / "r2" / "24" / "item_sids.json",
    "F0": REPO / "results" / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0" / "item_sids.json",
    "B": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_global_rq" / "item_sids.json",
    "C": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_rq" / "item_sids.json",
    "D": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_shuffled_rq" / "item_sids.json",
}


def load_sid(arm: str) -> dict[int, tuple[int, int, int]]:
    """The arm's own three-token identifier for every item."""
    path = SID_PATHS[arm]
    payload = json.loads(path.read_text())
    return {int(key): tuple(int(value) for value in row[:3]) for key, row in payload.items()}


def shared_levels(sid: dict[int, tuple], a: int, b: int) -> int:
    left = sid.get(a)
    right = sid.get(b)
    if left is None or right is None:
        return 0
    count = 0
    for x, y in zip(left, right):
        if x != y:
            break
        count += 1
    return count


def shuffled_sid(sid: dict[int, tuple], rng: np.random.Generator) -> dict[int, tuple]:
    """The same tree with the items' positions permuted: the code histogram is kept, the structure is not."""
    items = sorted(sid)
    codes = [sid[item] for item in items]
    order = rng.permutation(len(items))
    return {item: codes[order[index]] for index, item in enumerate(items)}


def build_comparisons() -> dict[str, list[tuple]]:
    """Internal comparisons per structure: a real edge against a non-edge of the same graphlet."""
    from relative_distance_structures import build_instances, load_behaviour

    n, edges, adjacency = load_behaviour()
    degrees = np.asarray([len(adjacency[i]) for i in range(n)], dtype=np.int64)
    instances = build_instances(n, edges, adjacency, degrees, np.random.default_rng(SEED))
    comparisons: dict[str, list[tuple]] = {}
    for name in STRUCTURES:
        pairs = []
        for instance in instances[name]:
            trues = list(instance["true"]) + list(instance["inner"])
            falses = list(instance["false"])
            for true_pair in trues:
                for false_pair in falses:
                    if set(true_pair) == set(false_pair):
                        continue
                    # the owner's test 2 compares a real edge with the non-edges that touch the
                    # same item (leaf to centre against leaf to the other leaves), so the two pairs
                    # must share an endpoint
                    if not (set(true_pair) & set(false_pair)):
                        continue
                    pairs.append((tuple(true_pair), tuple(false_pair)))
        comparisons[name] = pairs
        print(f"{name:<15} internal comparisons {len(pairs)}", flush=True)
    return comparisons


def external_comparisons() -> dict[str, list[tuple]]:
    """External comparisons against the hard negatives: one step away from the true neighbour."""
    from relation_scores import local_negatives
    from relative_distance_structures import load_behaviour

    n, edges, adjacency = load_behaviour()
    degrees = np.asarray([len(adjacency[i]) for i in range(n)], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    negatives = local_negatives(edges, adjacency, n, rng)
    negative_set = {(int(a), int(b)) for a, b in negatives.tolist()}
    pairs = []
    for a, b in edges.tolist():
        a, b = int(a), int(b)
        for item, other in ((a, b), (b, a)):
            for w in list(adjacency[other])[:4]:
                if w == item or w in adjacency[item]:
                    continue
                key = (min(item, w), max(item, w))
                if key not in negative_set:
                    continue
                pairs.append(((item, other), key))
    print(f"external comparisons against hard negatives: {len(pairs)}", flush=True)
    return {"edge": pairs}


def distances(vectors: np.ndarray, pairs: list[tuple]) -> np.ndarray:
    left = vectors[[a for a, _ in pairs]]
    right = vectors[[b for _, b in pairs]]
    return np.linalg.norm(left - right, axis=1)


def tree_levels(sid: dict[int, tuple], pairs: list[tuple]) -> np.ndarray:
    return np.asarray([shared_levels(sid, a, b) for a, b in pairs], dtype=np.float64)


def accuracy(true_score: np.ndarray, false_score: np.ndarray) -> float:
    """A win is 1, a tie 0.5, a loss 0 -- the owner's rule."""
    return float(np.mean(np.where(true_score > false_score, 1.0,
                                np.where(true_score == false_score, 0.5, 0.0))))


def evaluate(pairs: list[tuple], vectors: np.ndarray, sid: dict[int, tuple],
             shuffled: dict[int, tuple], validation: np.ndarray, test: np.ndarray) -> dict:
    # the comparison list is a set of (true pair, false pair) pairs; the two scores are per comparison
    true_pairs = [pair[0] for pair in pairs]
    false_pairs = [pair[1] for pair in pairs]
    true_distance = distances(vectors, true_pairs)
    false_distance = distances(vectors, false_pairs)
    true_levels = tree_levels(sid, true_pairs)
    false_levels = tree_levels(sid, false_pairs)
    true_shuffled = tree_levels(shuffled, true_pairs)
    false_shuffled = tree_levels(shuffled, false_pairs)

    def z(values: np.ndarray) -> np.ndarray:
        spread = float(values.std())
        return (values - float(values.mean())) / (spread if spread > 1e-12 else 1.0)

    true_d, false_d = z(true_distance), z(false_distance)
    true_l, false_l = z(true_levels), z(false_levels)
    true_s, false_s = z(true_shuffled), z(false_shuffled)

    best_lambda, best_score = 0.0, -1.0
    for lambda_value in LAMBDA_GRID:
        score = accuracy((-true_d + lambda_value * true_l)[validation],
                         (-false_d + lambda_value * false_l)[validation])
        if score > best_score:
            best_lambda, best_score = float(lambda_value), score

    tree_wins = float(np.mean(true_l > false_l))
    tree_ties = float(np.mean(true_l == false_l))
    tree_losses = float(np.mean(true_l < false_l))
    return {
        "tree_true_larger": tree_wins,
        "tree_tied": tree_ties,
        "tree_false_larger": tree_losses,
        "A vectors only": accuracy(-true_d[test], -false_d[test]),
        "B tree only": accuracy(true_l[test], false_l[test]),
        "C vectors + tree": accuracy((-true_d + best_lambda * true_l)[test],
                                      (-false_d + best_lambda * false_l)[test]),
        "D vectors + shuffled tree": accuracy((-true_d + best_lambda * true_s)[test],
                                              (-false_d + best_lambda * false_s)[test]),
        "lambda": best_lambda,
        "validation_score": best_score,
        "n_test": int(test.sum()),
    }


def main() -> int:
    comparisons = build_comparisons()
    external = external_comparisons()
    report = {"structures": {}}
    for arm in ARMS:
        vector_path = VECTORS / f"{arm}_q.npy"
        if not vector_path.is_file():
            continue
        vectors = np.load(vector_path).astype(np.float64)
        sid = load_sid(arm)
        shuffled = shuffled_sid(sid, np.random.default_rng(SEED + 3))
        block = {}
        for name, pairs in list(comparisons.items()) + list(external.items()):
            rng = np.random.default_rng(SEED + 5)
            order = rng.permutation(len(pairs))
            cut = int((1 - VALIDATION) * len(pairs))
            validation = order[:cut]
            test = order[cut:]
            block[name] = evaluate(pairs, vectors, sid, shuffled, validation, test)
            scores = block[name]
            print(f"{arm:<6} {name:<15} A {scores['A vectors only']:.4f} | B {scores['B tree only']:.4f} | "
                  f"C {scores['C vectors + tree']:.4f} | D {scores['D vectors + shuffled tree']:.4f} "
                  f"(lambda {scores['lambda']:+.2f}) | tree: true> {scores['tree_true_larger']:.3f} "
                  f"= {scores['tree_tied']:.3f} < {scores['tree_false_larger']:.3f}", flush=True)
        report["structures"][arm] = block

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
