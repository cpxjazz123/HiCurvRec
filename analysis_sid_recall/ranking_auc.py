"""Neighbour vs non-neighbour ranking AUC after quantization.

The mechanism under test (issue #144) asks for one thing: an observed behaviour
edge should rank closer than a non-edge.  With only SIDs available, closeness is
the length of the shared prefix (0-3 tokens), and negatives are sampled to match
the positive's degree so popularity cannot win the comparison.

    python3 analysis_sid_recall/ranking_auc.py --arms 24,48
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
MIRROR = Path(__file__).resolve().parent / "r2"
OUT = Path(__file__).resolve().parent / "ranking_auc.json"
SEED = 42
PAIRS = 40_000


def edges_with_triangles() -> tuple[np.ndarray, np.ndarray]:
    """Edges plus the number of triangles sitting on each of them."""
    from scipy import sparse
    pairs, triangles = edges_with_triangles()
    nodes, inverse = np.unique(pairs, return_inverse=True)
    index = inverse.reshape(-1, 2)
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(len(nodes), len(nodes))).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    triangles = np.asarray((A @ A)[index[:, 0], index[:, 1]]).ravel()
    return pairs, triangles


def edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def prefix_length(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    length = 0
    for left, right in zip(a, b):
        if left != right:
            break
        length += 1
    return length


def auc(pairs: np.ndarray, table: dict, rng: np.random.Generator,
        subset: np.ndarray | None = None) -> dict:
    degree: dict[int, int] = {}
    for left, right in pairs:
        degree[int(left)] = degree.get(int(left), 0) + 1
        degree[int(right)] = degree.get(int(right), 0) + 1
    by_degree: dict[int, list[int]] = {}
    for node, value in degree.items():
        by_degree.setdefault(value // 5, []).append(node)   # degree buckets of five

    pool = np.arange(len(pairs)) if subset is None else np.flatnonzero(subset)
    picks = rng.choice(pool, min(PAIRS, len(pool)), replace=False)
    scores_pos, scores_neg = [], []
    for position in picks:
        left, right = int(pairs[position, 0]), int(pairs[position, 1])
        if left not in table or right not in table:
            continue
        bucket = by_degree.get(degree.get(left, 0) // 5, [])
        if len(bucket) < 2:
            continue
        other = bucket[rng.integers(len(bucket))]
        if other == left or other in ():
            continue
        if prefix_length(tuple(table[left]), tuple(table[right])) == prefix_length(
                tuple(table[left]), tuple(table[other])) and right == other:
            continue
        scores_pos.append(prefix_length(tuple(table[left]), tuple(table[right])))
        scores_neg.append(prefix_length(tuple(table[left]), tuple(table[other])))
    pos = np.asarray(scores_pos)
    neg = np.asarray(scores_neg)
    wins = (pos[:, None] > neg[None, :]).mean() if len(pos) and len(neg) else None
    ties = (pos[:, None] == neg[None, :]).mean() if len(pos) and len(neg) else None
    return {"pairs": int(min(len(pos), len(neg))),
            "mean_prefix_positive": float(pos.mean()) if len(pos) else None,
            "mean_prefix_negative": float(neg.mean()) if len(neg) else None,
            "auc": float(wins + 0.5 * ties) if wins is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="24")
    parser.add_argument("--sid", default="", help="an item_sids.json outside the mirror")
    parser.add_argument("--strata", action="store_true",
                        help="report triangle-free and triangle edges separately")
    args = parser.parse_args()
    pairs, triangles = edges_with_triangles()
    report = {}
    if args.sid:
        table = {int(k): v for k, v in json.loads(Path(args.sid).read_text()).items()}
        block = auc(pairs, table, np.random.default_rng(SEED))
        block["triangle_free"] = auc(pairs, table, np.random.default_rng(SEED), subset=triangles == 0)
        block["in_triangle"] = auc(pairs, table, np.random.default_rng(SEED), subset=triangles > 0)
        report["sid"] = block
        print("sid %s: AUC %.3f | positive %.2f vs negative %.2f"
              % (Path(args.sid).parent.name, block["auc"],
                 block["mean_prefix_positive"], block["mean_prefix_negative"]))
    for number in [n.strip() for n in args.arms.split(",") if n.strip()]:
        path = MIRROR / number / "item_sids.json"
        if not path.is_file():
            continue
        table = {int(k): v for k, v in json.loads(path.read_text()).items()}
        rng = np.random.default_rng(SEED)
        report[number] = auc(pairs, table, rng)
        if args.strata:
            report[number]["triangle_free"] = auc(pairs, table, rng, subset=triangles == 0)
            report[number]["in_triangle"] = auc(pairs, table, rng, subset=triangles > 0)
        block = report[number]
        print("arm %s: AUC %.3f | positive prefix %.2f vs negative %.2f | pairs %d"
              % (number, block["auc"], block["mean_prefix_positive"],
                 block["mean_prefix_negative"], block["pairs"]))
        if args.strata:
            print("      triangle-free AUC %.3f | in-triangle AUC %.3f"
                  % (block["triangle_free"]["auc"], block["in_triangle"]["auc"]))
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
