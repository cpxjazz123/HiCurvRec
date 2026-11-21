#!/usr/bin/env python3
"""Does directed multi-step topology carry the direction the embeddings miss?

Directed RRWP (relative random-walk probabilities) is computed from the training graph
only, then asked one question: on held-out one-way edges, do walk probabilities predict
which way the edge points?

    P   = D_out^-1 A                      (one step, training edges only)
    P^2 = P @ P                           (two steps)

For a held-out pair (a, b) both orders are scored, so the features are
[P[a,b], P[b,a], P2[a,b], P2[b,a], out_deg(a), out_deg(b)].  A one-step feature cannot
answer the question for a held-out edge (that edge is absent from the training graph, so
P[a,b] = 0); only the two-step feature can, through shared neighbours -- which is exactly
the "multi-hop topology" the proposal claims the embeddings lack.

Two probes, same split, same seed: embeddings only, and embeddings + RRWP features.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import augment_gnn as base  # noqa: E402

OUT = HERE / "rrwp_direction.json"
STEPS = 1500
BATCH = 2048


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rrwp(rows: np.ndarray, cols: np.ndarray, n_items: int):
    """One and two step random-walk matrices from the training edges."""
    data = np.ones(len(rows), dtype=np.float32)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.where(out_degree > 0, 1.0 / np.maximum(out_degree, 1e-9), 0.0)
    transition = sp.diags(inverse) @ adjacency
    two_step = (transition @ transition).tocsr()
    return transition.tocsr(), two_step, out_degree


def gather(matrix: sp.csr_matrix, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[a, b]).ravel()


def probe(x: np.ndarray, rrwp_features: np.ndarray | None, held_pairs: np.ndarray,
          device, holdout: float = 0.3, steps: int = STEPS) -> dict:
    import torch
    import torch.nn as nn
    n = len(held_pairs)
    both = np.concatenate([held_pairs, held_pairs[:, ::-1]], axis=0)
    labels = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    rng = np.random.default_rng(base.SEED + 7)
    items = np.unique(both.ravel())
    rng.shuffle(items)
    cut = int((1 - holdout) * len(items))
    train_items = set(items[:cut].tolist())
    eval_items = set(items[cut:].tolist())
    tr = [i for i, (a, b) in enumerate(both.tolist())
          if a in train_items and b in train_items]
    ev = [i for i, (a, b) in enumerate(both.tolist())
          if a in eval_items and b in eval_items]
    if rrwp_features is not None:
        # the RRWP block is already aligned with `both` (the same pair order), so it is
        # concatenated per pair and never re-indexed
        feat = np.concatenate([x[both[:, 0]], x[both[:, 1]], x[both[:, 0]] * x[both[:, 1]],
                               rrwp_features], axis=1)
    else:
        feat = np.concatenate([x[both[:, 0]], x[both[:, 1]], x[both[:, 0]] * x[both[:, 1]]],
                              axis=1)
    in_dim = feat.shape[1]
    torch.manual_seed(base.SEED)
    head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 256), nn.GELU(),
                         nn.Linear(256, 2)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr = torch.from_numpy(feat[tr].astype(np.float32)).to(device)
    xev = torch.from_numpy(feat[ev].astype(np.float32)).to(device)
    ytr = torch.from_numpy(labels[tr]).to(device)
    yev = torch.from_numpy(labels[ev]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(base.SEED)
    for _ in range(steps):
        pick = torch.randint(0, len(xtr), (min(BATCH, len(xtr)),), generator=generator)
        loss = torch.nn.functional.cross_entropy(head(xtr[pick]), ytr[pick])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        guess = head(xev).argmax(-1)
    return {"direction_accuracy": float((guess == yev).float().mean()),
            "pairs_eval": int(len(ev)), "pairs_train": int(len(tr))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    graph = np.load(base.FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = base.split_edges(
        rows, cols)
    transition, two_step, out_degree = rrwp(msg_rows, msg_cols, n_items)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    held_directed = set(map(tuple, held_pairs.tolist()))
    one_way = held_pairs[np.array([(b, a) not in held_directed
                                   for a, b in held_pairs.tolist()])]
    a, b = one_way[:, 0], one_way[:, 1]
    both = np.concatenate([one_way, one_way[:, ::-1]], axis=0)
    features = np.stack([
        gather(transition, both[:, 0], both[:, 1]),
        gather(transition, both[:, 1], both[:, 0]),
        gather(two_step, both[:, 0], both[:, 1]),
        gather(two_step, both[:, 1], both[:, 0]),
        np.log1p(out_degree[both[:, 0]]),
        np.log1p(out_degree[both[:, 1]]),
    ], axis=1).astype(np.float32)

    states = {"z": np.load(REPO / "results" / "research" / "directed_graphlet_teacher" /
                           "fidelity" / "fidelity_states.npz")["z"],
              "semantics": np.load(base.SEMANTICS)}
    report = {"inputs": {"full_graph": sha256(base.FULL_GRAPH),
                         "semantics": sha256(base.SEMANTICS)},
              "split": {"message_edges": int(len(msg_rows)), "held_edges": int(len(held_rows)),
                        "held_one_way": int(len(one_way))},
              "rrwp": {"steps": [1, 2], "source": "training edges only",
                       "features": "P[a,b], P[b,a], P2[a,b], P2[b,a], log out-degree of both"},
              "probe": {"steps": STEPS, "batch": BATCH, "item_disjoint": True,
                        "note": "one-step features are zero for the held-out edge itself; "
                                "only two-step can answer"},
              "arms": {}}
    for name, x in states.items():
        report["arms"][name] = probe(x, None, one_way, device)
        report["arms"][f"{name}_plus_rrwp"] = probe(x, features, one_way, device)
    report["arms"]["rrwp_only"] = probe(np.zeros((n_items, 1), dtype=np.float32), features,
                                        one_way, device)
    se = 0.5 / math.sqrt(report["arms"]["z"]["pairs_eval"])
    report["readings"] = {
        "se": round(se, 4),
        "z_plus_rrwp_minus_z_sigma": round(
            (report["arms"]["z_plus_rrwp"]["direction_accuracy"]
             - report["arms"]["z"]["direction_accuracy"]) / se, 2),
        "rrwp_only_minus_chance_sigma": round(
            (report["arms"]["rrwp_only"]["direction_accuracy"] - 0.5) / se, 2)}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for key, value in report["arms"].items():
        print(f"  {key}: {value['direction_accuracy']:.4f} (n={value['pairs_eval']})")
    print("written", args.out)


if __name__ == "__main__":
    main()
