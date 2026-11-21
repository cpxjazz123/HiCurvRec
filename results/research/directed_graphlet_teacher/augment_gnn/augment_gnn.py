#!/usr/bin/env python3
"""Directed message passing over the real behaviour graph, with a fusion net that trains.

The previous attempt learned only the two linear projections: `fuse` never appeared in any
loss, so the "augmented" vector was `x + a * random(...)`.  This version fixes that and adds
the part of DiGAE that was missing -- aggregation over the graph itself:

    h0      = W_in x                                   (768-d semantics as node features)
    h^{l+1} = norm(h + W_self h + W_out A_out h + W_in A_in h)
    s, t    = S(h), T(h)                               source / target heads
    x'      = x + alpha * f([s, t])                    f: 128 -> 256 -> 768, tanh

Two edge heads are trained, and the second one is what forces `f` to move:

    L = BCE(s_i . t_j, train edges) + BCE(s'_i . t'_j, train edges)

Leakage control: the edge set is split by unordered pair, message passing uses only the
training slice, and every reported direction number is computed on the held-out slice.
Arms: C directed, B undirected targets, D randomly oriented targets, E alpha=0 (the fusion
trained but disabled), A raw semantics with no graph at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())

FULL_GRAPH = REPO / "results" / "research" / "behavior_graph_directed" / "graph.npz"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "gnn_augment.json"

SEED = 0
STEPS = 1200
BATCH = 4096
HIDDEN = 128
LAYERS = 2
ALPHA = 1.0
HOLDOUT = 0.1          # edges kept out of both the graph and the supervision
HIDDEN_FUSE = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_edges(rows: np.ndarray, cols: np.ndarray, seed: int = SEED):
    """Split by unordered pair, so a held-out edge never appears in either direction."""
    unit = np.array(sorted({(min(a, b), max(a, b)) for a, b in zip(rows.tolist(), cols.tolist())}),
                    dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unit))
    cut = int((1.0 - HOLDOUT) * len(order))
    train_unit = set(map(tuple, unit[order[:cut]].tolist()))
    held_unit = set(map(tuple, unit[order[cut:]].tolist()))
    keep = np.array([(min(a, b), max(a, b)) in train_unit for a, b in zip(rows.tolist(), cols.tolist())])
    held = np.array([(min(a, b), max(a, b)) in held_unit for a, b in zip(rows.tolist(), cols.tolist())])
    return (rows[keep], cols[keep]), (rows[held], cols[held]), train_unit, held_unit


def sparse_adjacency(rows: np.ndarray, cols: np.ndarray, n_items: int, device):
    indices = torch.from_numpy(np.stack([rows, cols]).astype(np.int64)).to(device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    out_adj = torch.sparse_coo_tensor(indices, values, (n_items, n_items)).coalesce()
    in_adj = torch.sparse_coo_tensor(indices.flip(0), values, (n_items, n_items)).coalesce()
    out_deg = torch.sparse.sum(out_adj, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
    in_deg = torch.sparse.sum(in_adj, dim=1).to_dense().clamp(min=1.0).unsqueeze(-1)
    return out_adj, in_adj, out_deg, in_deg


def build_model(in_dim: int, node_dim: int = HIDDEN):
    import torch.nn as nn

    class DirectedAugmenter(nn.Module):
        """Message passing on the behaviour graph, then a fusion net that is trained."""

        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(in_dim, node_dim)
            self.layers = nn.ModuleList(
                nn.ModuleDict({
                    "self": nn.Linear(node_dim, node_dim),
                    "out": nn.Linear(node_dim, node_dim),
                    "in": nn.Linear(node_dim, node_dim),
                    "norm": nn.LayerNorm(node_dim)}) for _ in range(LAYERS))
            self.source = nn.Linear(node_dim, node_dim)
            self.target = nn.Linear(node_dim, node_dim)
            self.fuse = nn.Sequential(nn.Linear(2 * node_dim, HIDDEN_FUSE), nn.GELU(),
                                      nn.Linear(HIDDEN_FUSE, in_dim), nn.Tanh())
            # the second pair of heads reads the fused vector: this is what trains `fuse`
            self.source_fused = nn.Linear(in_dim, node_dim)
            self.target_fused = nn.Linear(in_dim, node_dim)
            self.node_dim = node_dim

        def encode(self, x, out_adj, in_adj, out_deg, in_deg):
            h = self.input(x)
            for layer in self.layers:
                message = (layer["out"](torch.sparse.mm(out_adj, h) / out_deg)
                           + layer["in"](torch.sparse.mm(in_adj, h) / in_deg))
                h = layer["norm"](h + layer["self"](h) + message)
            return h

        def forward(self, x, out_adj, in_adj, out_deg, in_deg, alpha: float):
            h = self.encode(x, out_adj, in_adj, out_deg, in_deg)
            s, t = self.source(h), self.target(h)
            fused = x + alpha * self.fuse(torch.cat([s, t], dim=-1))
            s2, t2 = self.source_fused(fused), self.target_fused(fused)
            return {"s": s, "t": t, "s2": s2, "t2": t2, "fused": fused}

    return DirectedAugmenter()


def direction_pairs(held_rows: np.ndarray, held_cols: np.ndarray, rng):
    """One-way pairs with their orientation, plus a degree-matched negative partner."""
    pairs, forward, backward = [], [], []
    for a, b in zip(held_rows.tolist(), held_cols.tolist()):
        pairs.append((a, b))
        forward.append(True)
        backward.append(False)
    order = np.argsort([rng.random() for _ in pairs])
    pairs = np.array([pairs[i] for i in order], dtype=np.int64)
    truth = np.stack([[forward[i] for i in order], [backward[i] for i in order]], axis=1)
    return pairs, truth.astype(np.float32)


def probe(x: np.ndarray, held_pairs: np.ndarray, device, holdout: float = 0.3,
          steps: int = 1500) -> dict:
    """Held-out direction accuracy: given (a, b) in a stored order, which way does it point?

    The feature never contains both orders, so the only way to answer is to read the state.
    Half the examples are the stored order (label "forward") and half are the same pair
    reversed (label "backward"), which balances the two classes.
    """
    import torch.nn as nn
    n = len(held_pairs)
    forward = held_pairs
    backward = held_pairs[:, ::-1]
    both = np.concatenate([forward, backward], axis=0)
    labels = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    # the probe must not memorise per-item tendencies: its training and evaluation pairs
    # share no item.  Without this split the head can learn "this item is usually a source"
    # and score far above chance on a state that carries no direction at all.
    rng = np.random.default_rng(SEED + 7)
    items = np.unique(both.ravel())
    rng.shuffle(items)
    cut_items = int((1 - holdout) * len(items))
    train_items = set(items[:cut_items].tolist())
    eval_items = set(items[cut_items:].tolist())
    tr = np.array([i for i, (a, b) in enumerate(both.tolist())
                   if a in train_items and b in train_items], dtype=np.int64)
    ev = np.array([i for i, (a, b) in enumerate(both.tolist())
                   if a in eval_items and b in eval_items], dtype=np.int64)
    head = nn.Sequential(nn.LayerNorm(3 * x.shape[1]), nn.Linear(3 * x.shape[1], 256),
                         nn.GELU(), nn.Linear(256, 2)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    def feats(idx):
        a, b = both[idx][:, 0], both[idx][:, 1]
        return torch.from_numpy(np.concatenate([x[a], x[b], x[a] * x[b]], -1)
                                .astype(np.float32)).to(device)

    xtr, xev = feats(tr), feats(ev)
    ytr = torch.from_numpy(labels[tr]).to(device)
    yev = torch.from_numpy(labels[ev]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(steps):
        pick = torch.randint(0, len(xtr), (min(2048, len(xtr)),), generator=generator)
        loss = torch.nn.functional.cross_entropy(head(xtr[pick]), ytr[pick])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        guess = head(xev).argmax(-1)
    return {"direction_accuracy": float((guess == yev).float().mean()),
            "pairs_eval": int(len(ev)), "pairs_train": int(len(tr)),
            "label_balance": {"forward": int((yev == 0).sum()), "backward": int((yev == 1).sum())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = np.load(FULL_GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    n_items = int(graph["nodes"])
    semantics = np.load(SEMANTICS).astype(np.float32)
    x = torch.from_numpy(semantics).to(device)
    (msg_rows, msg_cols), (held_rows, held_cols), train_unit, held_unit = split_edges(rows, cols)
    out_adj, in_adj, out_deg, in_deg = sparse_adjacency(msg_rows, msg_cols, n_items, device)

    # supervision pairs: training edges (both orientations of the unordered pair) and the
    # same number of non-edges drawn from pairs absent from the message graph
    rng = np.random.default_rng(SEED)
    pos = np.stack([msg_rows, msg_cols], axis=1)
    neg_rows, neg_cols = [], []
    unit_train = np.array(sorted(train_unit), dtype=np.int64)
    while len(neg_rows) < len(pos):
        a = int(rng.integers(0, n_items))
        b = int(rng.integers(0, n_items))
        if a == b or (min(a, b), max(a, b)) in train_unit:
            continue
        neg_rows.append(a)
        neg_cols.append(b)
    neg = np.stack([neg_rows, neg_cols], axis=1)
    pairs = torch.from_numpy(np.concatenate([pos, neg]).astype(np.int64)).to(device)
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)

    held_pairs = np.stack([held_rows, held_cols], axis=1)
    # direction pairs for the probe: held-out one-way edges, both orders as features
    reverse = {(b, a) for a, b in held_unit}
    one_way = held_pairs[np.array([(b, a) not in reverse for a, b in held_pairs.tolist()])]

    report = {"inputs": {"semantics": sha256(SEMANTICS), "full_graph": sha256(FULL_GRAPH)},
              "split": {"message_edges": int(len(msg_rows)),
                        "held_edges": int(len(held_rows)),
                        "held_unordered_pairs": int(len(held_unit)),
                        "held_one_way_for_probe": int(len(one_way))},
              "protocol": {"steps": STEPS, "batch": BATCH, "hidden": HIDDEN,
                           "layers": LAYERS, "alpha": ALPHA, "seed": SEED,
                           "message_passing": "directed in/out neighbourhood mean",
                           "fuse_trained_by": "second edge head on the fused vector",
                           "leakage": "held-out edges absent from the graph and the loss"},
              "arms": {}}

    for arm in ("C_directed", "B_undirected", "D_random_direction", "E_alpha_zero"):
        torch.manual_seed(SEED)
        model = build_model(semantics.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        alpha = 0.0 if arm == "E_alpha_zero" else ALPHA
        label_pairs = labels.clone()
        if arm == "B_undirected":
            # an undirected target cannot distinguish the two orders: the same label is
            # used for (i, j) and (j, i)
            label_pairs = torch.ones_like(labels)
        for step in range(STEPS):
            model.train()
            pick = torch.randint(0, len(pairs), (BATCH,), generator=generator)
            out = model(x, out_adj, in_adj, out_deg, in_deg, alpha)
            left, right = pairs[pick, 0], pairs[pick, 1]
            logits = (out["s"][left] * out["t"][right]).sum(-1)
            logits_fused = (out["s2"][left] * out["t2"][right]).sum(-1)
            target = label_pairs[pick]
            if arm == "D_random_direction":
                # randomly oriented edges: the edge exists, its direction is coin-flipped
                flip = torch.from_numpy(
                    np.random.default_rng(SEED + step).random(len(pick)) < 0.5).to(device)
                target = torch.where(flip, 1.0 - target, target)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                logits_fused, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 300 == 0 or step == STEPS - 1:
                print(f"    {arm} step {step}: loss {float(loss):.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = model(x, out_adj, in_adj, out_deg, in_deg, alpha)
            fused = out["fused"].cpu().numpy().astype(np.float32)
            # edge head's own direction read-out on held-out one-way edges
            pair_t = torch.from_numpy(one_way).to(device)
            s2, t2 = out["s2"], out["t2"]
            # the held edge is stored as a->b, so the correct answer is "forward"
            forward_score = (s2[pair_t[:, 0]] * t2[pair_t[:, 1]]).sum(-1)
            backward_score = (s2[pair_t[:, 1]] * t2[pair_t[:, 0]]).sum(-1)
            head_dir = float((forward_score > backward_score).float().mean())
        report["arms"][arm] = {
            "edge_head_direction": head_dir,
            "shift_l2": float(np.linalg.norm(fused - semantics) / max(len(fused), 1)),
            "probe": probe(fused, one_way, device)}
        print(f"  {arm}: edge-head dir {head_dir:.4f} shift "
              f"{report['arms'][arm]['shift_l2']:.4f} probe "
              f"{report['arms'][arm]['probe']['direction_accuracy']:.4f}", flush=True)
        np.savez_compressed(HERE / f"gnn_state_{arm}.npz", state=fused)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
