"""Issue #241 stage 1: the directed item-transition graph and its three-way split.

Only ``results/stage0_build_parquet/train.parquet`` is read.  The transitions are the
canonical ones the repo already uses (``analysis_sid_recall/build_behavior_graph.py``):
one sequence per user = that user's longest history plus its target, consecutive items
joined, self-loops dropped.

Three disjoint pieces, so nothing the teacher is asked to predict is visible to it:

    users   ->  visible 90% / eval 10%          (seeded, per user, never mixed)
    visible ->  ego graph            90% of edges
            ->  teacher targets      10% of edges, removed from the ego graph
    eval    ->  independent edges    all of the eval users' transitions

``graph/metadata.json`` records the split seeds, self-loop policy, counts and the
sha256 of every emitted file.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = REPO / "results" / "research" / "issue241" / "graph"
USER_SPLIT_SEED = 241001
EDGE_HOLDOUT_SEED = 241002
EVAL_USER_FRACTION = 0.10
TARGET_EDGE_FRACTION = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequences() -> dict[int, list[int]]:
    frame = pd.read_parquet(TRAIN)
    longest: dict[int, list[int]] = {}
    for user, history in zip(frame["user"].to_numpy(), frame["history"]):
        key = int(user)
        if key not in longest or len(history) > len(longest[key]):
            longest[key] = list(history)
    by_user: dict[int, list[int]] = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["history"], frame["target"]
    ):
        key = int(user)
        if len(history) == len(longest[key]):
            by_user[key] = [int(item) for item in history] + [int(target)]
    return {user: by_user[user] for user in sorted(by_user)}


def transitions(seqs: dict[int, list[int]], users) -> Counter:
    counts: Counter = Counter()
    for user in users:
        sequence = seqs[user]
        for left, right in zip(sequence[:-1], sequence[1:]):
            if left == right:
                continue  # self-loops dropped, recorded in the metadata
            counts[(left, right)] += 1
    return counts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sequences()
    users = np.array(sorted(seqs), dtype=np.int64)
    rng = np.random.default_rng(USER_SPLIT_SEED)
    order = rng.permutation(len(users))
    cut = int(round(len(users) * (1.0 - EVAL_USER_FRACTION)))
    visible_users = users[np.sort(order[:cut])]
    eval_users = users[np.sort(order[cut:])]

    visible_counts = transitions(seqs, visible_users)
    eval_counts = transitions(seqs, eval_users)

    keys = np.array(sorted(visible_counts), dtype=np.int64)
    weights = np.array([visible_counts[tuple(k)] for k in keys], dtype=np.int64)
    rng_edge = np.random.default_rng(EDGE_HOLDOUT_SEED)
    held = rng_edge.random(len(keys)) < TARGET_EDGE_FRACTION
    target_edges = keys[held]
    ego_edges = keys[~held]
    ego_weights = weights[~held]

    n_items = int(max(max(k) for k in seqs.values()) + 1)
    rows = ego_edges[:, 0]
    cols = ego_edges[:, 1]
    ego = sparse.coo_matrix((ego_weights, (rows, cols)), shape=(n_items, n_items)).tocsr()
    ego.sort_indices()

    eval_keys = np.array(sorted(eval_counts), dtype=np.int64)
    eval_weights = np.array([eval_counts[tuple(k)] for k in eval_keys], dtype=np.int64)
    self_loops = sum(
        1 for user in users for left, right in zip(seqs[user][:-1], seqs[user][1:])
        if left == right
    )

    np.savez_compressed(
        OUT / "ego_csr.npz",
        indptr=ego.indptr, indices=ego.indices, data=ego.data,
        shape=np.array(ego.shape, dtype=np.int64),
    )
    np.save(OUT / "target_edges.npy", target_edges)
    np.save(OUT / "target_weights.npy", weights[held])
    np.save(OUT / "eval_edges.npy", eval_keys)
    np.save(OUT / "eval_weights.npy", eval_weights)
    np.save(OUT / "visible_users.npy", visible_users)
    np.save(OUT / "eval_users.npy", eval_users)

    metadata = {
        "schema": "genrec.directed_graph.issue241.v1",
        "source": {
            "path": str(TRAIN.relative_to(REPO)),
            "sha256": sha256(TRAIN),
        },
        "definition": "i -> j for consecutive items in one user's longest history + target; self-loops dropped",
        "mapping": "node id equals the stage-0 item id; no remapping",
        "seeds": {
            "user_split": USER_SPLIT_SEED,
            "edge_holdout": EDGE_HOLDOUT_SEED,
            "eval_user_fraction": EVAL_USER_FRACTION,
            "target_edge_fraction": TARGET_EDGE_FRACTION,
        },
        "self_loops_dropped": int(self_loops),
        "counts": {
            "users": int(len(users)),
            "visible_users": int(len(visible_users)),
            "eval_users": int(len(eval_users)),
            "items": n_items,
            "visible_directed_edges": int(len(keys)),
            "ego_edges": int(len(ego_edges)),
            "target_edges": int(len(target_edges)),
            "eval_directed_edges": int(len(eval_keys)),
            "ego_items_with_out_edge": int((np.diff(ego.indptr) > 0).sum()),
        },
        "files": {},
    }
    for name in (
        "ego_csr.npz", "target_edges.npy", "target_weights.npy",
        "eval_edges.npy", "eval_weights.npy", "visible_users.npy", "eval_users.npy",
    ):
        metadata["files"][name] = sha256(OUT / name)
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
