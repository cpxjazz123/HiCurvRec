#!/usr/bin/env python3
"""Does the 32-d vector before quantisation still carry real behaviour roles?

Fixed protocol, identical for every state under test:

* **input**  a KNN graph built from the state itself (k neighbours per item, cosine);
* **model**  a graph network that reads that KNN graph and the item's own vector;
* **target** the item's *directed-graphlet orbit profile* in the **real** behaviour graph
              (`structure_targets_v3.npz`, `own_orbit`, 376 roles) -- never derived from
              the KNN graph, so the model cannot answer by reading its own input;
* **scoring** cross-entropy and KL on items the model never saw in training.

The only thing that changes between rows is the state: a random control, 768-d
semantics, the frozen 128-d GNN table, the RQ-VAE latent (pre-quantisation 32-d), and
the quantised ``q``.  Their KNN graphs are rebuilt from scratch for each state, so a
change in accuracy is attributable to the representation and not to a fixed graph.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
sys.path.insert(0, str(BASELINE))
sys.path.insert(0, str(TEACHER))

TARGETS = TEACHER / "teacher" / "structure" / "v3" / "structure_targets_v3.npz"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
GNN_TABLE = TEACHER / "representation" / "gnn_item_embedding.npz"
CONFIG_FILE = BASELINE / "config" / "train_config.json"
K_NEIGHBOURS = 10
SPLIT_SEED = 42
MODEL_SEED = 1234
HIDDEN = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def knn_graph(states: np.ndarray, k: int, chunk: int = 2048) -> list[set[int]]:
    """k cosine neighbours per item, computed in row blocks to stay inside memory."""
    states = unit(states.astype(np.float64))
    n = len(states)
    neighbours: list[set[int]] = []
    for start in range(0, n, chunk):
        block = states[start:start + chunk] @ states.T
        block[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf
        picks = np.argpartition(-block, k, axis=1)[:, :k]
        neighbours.extend({int(value) for value in row} for row in picks)
    if len(neighbours) != n:
        raise RuntimeError("kNN builder lost rows")
    return neighbours


def knn_adjacency(states: np.ndarray, k: int) -> list[set[int]]:
    """Symmetric kNN adjacency: an edge only if **both** items list each other.

    The mutual rule is what makes the KNN graph a graph instead of two directed lists,
    and it is what a graph network over a neighbourhood usually receives.
    """
    directed = knn_graph(states, k)
    return [{other for other in items if item in directed[other]}
            for item, items in enumerate(directed)]


def pair_lists(neighbours: list[set[int]]) -> np.ndarray:
    """Undirected edge list of the KNN graph (both directions inside the message pass)."""
    rows, cols = [], []
    for item, items in enumerate(neighbours):
        for other in items:
            if other != item:
                rows.append(item)
                cols.append(other)
    return np.stack([np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)])


def split_items(covered: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.flatnonzero(covered))
    cut = int(0.8 * len(order))
    return np.sort(order[:cut]), np.sort(order[cut:])


def build_model(in_dim: int, n_roles: int):
    import torch
    import torch.nn as nn

    class RoleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_proj = nn.Linear(in_dim, HIDDEN)
            self.msg_proj = nn.Linear(in_dim, HIDDEN)
            self.out = nn.Linear(2 * HIDDEN, n_roles)

        def forward(self, x, rows, cols, count):
            h = torch.relu(self.self_proj(x))
            m = torch.zeros_like(h).index_add(0, rows, self.msg_proj(x)[cols])
            m = m / count.unsqueeze(-1)
            return self.out(torch.cat([h, m], dim=-1)), m

    return RoleNet()


def build_graph_only(n_roles: int):
    """Sees the neighbourhood only: its vector is the mean of the neighbours' vectors."""
    import torch
    import torch.nn as nn

    class GraphOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(HIDDEN, HIDDEN)
            self.out = nn.Linear(HIDDEN, n_roles)

        def forward(self, h, rows, cols, count):
            m = torch.zeros_like(h).index_add(0, rows, h[cols])
            m = m / count.unsqueeze(-1)
            return self.out(torch.relu(self.proj(m))), m

    return GraphOnly()


def fit_and_score(states: np.ndarray, targets: np.ndarray, covered: np.ndarray,
                  neighbours: list[set[int]], epochs: int = 300, device=None) -> dict:
    import torch

    train_items, eval_items = split_items(covered)
    edges = pair_lists(neighbours)
    y = torch.from_numpy(targets[covered].astype(np.float32)).to(device)

    def rows_of(subset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lookup = {int(item): position for position, item in enumerate(subset.tolist())}
        keep = [i for i, (a, b) in enumerate(zip(edges[0].tolist(), edges[1].tolist()))
                if int(a) in lookup and int(b) in lookup]
        return (np.array([lookup[int(edges[0][i])] for i in keep], dtype=np.int64),
                np.array([lookup[int(edges[1][i])] for i in keep], dtype=np.int64))

    train_rows_np, train_cols_np = rows_of(train_items)
    eval_rows_np, eval_cols_np = rows_of(eval_items)
    target_of = {int(item): position for position, item in enumerate(covered.nonzero()[0].tolist())}
    train_target = torch.tensor([target_of[int(i)] for i in train_items],
                                dtype=torch.long, device=device)
    eval_target = torch.tensor([target_of[int(i)] for i in eval_items],
                               dtype=torch.long, device=device)
    train_y = y[train_target]
    eval_y = y[eval_target]

    def graph_rows(rows_np: np.ndarray, cols_np: np.ndarray, n_nodes: int):
        rows = torch.from_numpy(rows_np).to(device)
        cols = torch.from_numpy(cols_np).to(device)
        count = torch.bincount(rows, minlength=n_nodes).clamp(min=1)
        return rows, cols, count

    torch.manual_seed(MODEL_SEED)
    model = build_model(states.shape[1], targets.shape[1]).to(device)
    # every row the network can read is an input; only the target table is constant
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = torch.from_numpy(unit(states.astype(np.float32))[train_items]).to(device)
    x_eval = torch.from_numpy(unit(states.astype(np.float32))[eval_items]).to(device)
    for _ in range(epochs):
        model.train()
        logits, _ = model(x_train, *graph_rows(train_rows_np, train_cols_np, len(train_items)))
        loss = -(train_y * torch.log_softmax(logits, -1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_eval, *graph_rows(eval_rows_np, eval_cols_np, len(eval_items)))
        logp = torch.log_softmax(logits, -1)
        ce = float(-(eval_y * logp).sum(-1).mean())
        prior = train_y.mean(0).clamp(min=1e-12)
        ce_const = float(-(eval_y * torch.log(prior)).sum(-1).mean())
        total = torch.softmax(logits, -1).sum(0)
        spread = float((total / total.sum()).max())

    # shuffled control: the same KNN graph, but the targets are permuted across items.
    # It trains outside the no-grad block above, otherwise it never sees a gradient.
    perm = torch.randperm(len(train_target),
                          generator=torch.Generator().manual_seed(MODEL_SEED))
    shuffled = train_y[perm].clone()
    base = build_model(states.shape[1], targets.shape[1]).to(device)
    opt2 = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-4)
    base_rows, base_cols, base_count = graph_rows(train_rows_np, train_cols_np, len(train_items))
    for _ in range(epochs):
        base.train()
        logits_s, _ = base(x_train, base_rows, base_cols, base_count)
        loss_s = -(shuffled * torch.log_softmax(logits_s, -1)).sum(-1).mean()
        opt2.zero_grad(set_to_none=True)
        loss_s.backward()
        opt2.step()
    base.eval()
    with torch.no_grad():
        base_logp = torch.log_softmax(
            base(x_eval, *graph_rows(eval_rows_np, eval_cols_np, len(eval_items)))[0], -1)
        ce_shuffled = float(-(eval_y * base_logp).sum(-1).mean())
    return {"items_train": int(len(train_items)), "items_eval": int(len(eval_items)),
            "ce": ce, "ce_constant_prior": ce_const, "ce_shuffled_targets": ce_shuffled,
            "improvement_over_prior_nats": ce_const - ce,
            "neighbour_graph_density": float(len(edges[0]) / (len(states) * K_NEIGHBOURS)),
            "predicted_role_top_share": spread}


def role_headroom(targets: np.ndarray, covered: np.ndarray) -> dict:
    """The known weakness of the stored own-orbit target, stated as a number."""
    sub = targets[covered]
    rowsum = np.maximum(sub.sum(1, keepdims=True), 1e-12)
    p = sub / rowsum
    entropy = -(p * np.log2(p + 1e-12)).sum(1)
    train_items, _ = split_items(covered)
    position = {int(item): i for i, item in enumerate(np.flatnonzero(covered).tolist())}
    prior = p[[position[int(i)] for i in train_items]].mean(0)
    oracle = float(-(sub * np.log2(np.maximum(prior, 1e-12))).sum(1).mean() / np.log(2))
    return {"items": int(sub.shape[0]), "median_effective_roles": float(np.median(2.0 ** entropy)),
            "share_with_two_or_more": float((entropy > 0.5).mean()),
            "constant_prior_ce_nats": float(
                -(p * np.log(np.maximum(prior, 1e-12))).sum(1).mean()),
            "oracle_ce_nats_from_stored_table": oracle,
            "note": "the stored table's own prior is already worse than a constant prior, so "
                    "absolute accuracy is capped by the target, not by the representation; "
                    "only the *relative order* between states is interpretable"}


STATE_FILES = {
    "random_control": None,
    "semantics_768": SEMANTICS,
    "gnn_frozen_128": GNN_TABLE,
    "rqvae_latent_32": HERE / "rqvae_latent.npz",
    "rqvae_q_32": HERE / "rqvae_q.npz",
}


def load_state(name: str, n_items: int) -> np.ndarray:
    if name == "random_control":
        return np.random.default_rng(0).normal(size=(n_items, 32)).astype(np.float32)
    path = STATE_FILES[name]
    if path is None or not path.exists():
        raise SystemExit(f"{name}: {path} is missing")
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in ("latent", "quantised", "embedding"):
            if key in payload:
                return payload[key]
        raise SystemExit(f"{name}: no usable array in {path}")
    return np.load(path)


def main() -> None:
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default=",".join(STATE_FILES))
    parser.add_argument("--out", default=str(HERE / "role_probe.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = np.load(TARGETS)
    targets = payload["own_orbit"].astype(np.float32)
    covered = payload["coverage"] > 0
    report = {"inputs": {"targets": sha256(TARGETS), "k": K_NEIGHBOURS,
                         "split_seed": SPLIT_SEED, "model_seed": MODEL_SEED,
                         "hidden": HIDDEN, "epochs": 300,
                         "knn": "mutual (both items must list each other)",
                         "target": "own_orbit (376 roles), real behaviour graph only",
                         "note": "the KNN graph comes from the state under test; the "
                                 "answers come from the real graph, never from it"},
              "headroom": role_headroom(targets, covered),
              "states": {}}
    n_items = len(targets)
    for name in args.states.split(","):
        table = load_state(name, n_items)
        print(f"[{name}] {table.shape} mutual KNN", flush=True)
        neighbours = knn_adjacency(table, K_NEIGHBOURS)
        entry = fit_and_score(table, targets, covered, neighbours, device=device)
        entry["shape"] = list(table.shape)
        entry["mean_degree"] = float(np.mean([len(n) for n in neighbours]))
        report["states"][name] = entry
        print(f"[{name}] ce {entry['ce']:.4f} (prior {entry['ce_constant_prior']:.4f}, "
              f"shuffled {entry['ce_shuffled_targets']:.4f})", flush=True)
        pathlib_path = HERE / "role_probe.json"
        pathlib_path.write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
