"""Issue #241 stage 1: the frozen directed-topology teacher.

For every item i the teacher encodes i's 1-hop and 2-hop directed ego subgraph into a
32-dim vector T_i, then decodes two distributions over the fixed item vocabulary:

    P_out(i, .)   the successor distribution
    P_in(i, .)    the predecessor distribution

Those distributions are the *only* thing the pair-strength stage is allowed to use: the
issue forbids using T_i . T_j similarity directly, because two unrelated hubs can share a
role without sharing purchase context.

Supervision is the held-out real transitions only (``graph/target_edges.npy``), which the
ego graph never contains.  Node features are structural (role, hop, degree, edge weight);
the item embedding is deliberately not an input, so the score cannot degenerate into an ID
lookup.  Item identity appears only as the fixed decoding vocabulary, which the issue
explicitly allows.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
GRAPH = REPO / "results" / "research" / "issue241" / "graph"
OUT = REPO / "results" / "research" / "issue241" / "teacher"
SEED = 241003
K1_OUT, K1_IN, K2_PER, K2_CAP = 8, 8, 4, 32
FEATURE_DIM = 8
HIDDEN = 64
T_DIM = 32
NEGATIVES = 64
BATCH = 256
EPOCHS = 40
LR = 3e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_graph():
    blob = np.load(GRAPH / "ego_csr.npz")
    ego = sp.csr_matrix(
        (blob["data"], blob["indices"], blob["indptr"]), shape=tuple(blob["shape"])
    )
    out_csr = ego
    in_csr = ego.T.tocsr()
    out_csr.sort_indices()
    in_csr.sort_indices()
    return out_csr, in_csr


def neighbours(csr, node: int, k: int, rng) -> np.ndarray:
    start, end = csr.indptr[node], csr.indptr[node + 1]
    if start == end:
        return np.empty(0, dtype=np.int64)
    cols = csr.indices[start:end]
    if len(cols) <= k:
        return cols.astype(np.int64)
    # highest-weight neighbours first, then a seeded random subset of them
    weights = csr.data[start:end]
    order = np.argsort(-weights, kind="stable")[: max(k * 3, k)]
    pick = rng.choice(order, size=k, replace=False)
    return cols[pick].astype(np.int64)


def ego_sample(root: int, out_csr, in_csr, rng) -> dict:
    """Sample one directed ego subgraph and return arrays ready for the GNN."""
    hop1_out = neighbours(out_csr, root, K1_OUT, rng)
    hop1_in = neighbours(in_csr, root, K1_IN, rng)
    nodes = [root]
    edges = []          # (src_pos, dst_pos)
    role = np.zeros(1, dtype=np.int64)      # 0 root, 1 hop1, 2 hop2
    direction = np.zeros(1, dtype=np.int64)  # 0 root, 1 out-edge, 2 in-edge

    def add(node: int, src: int, dir_code: int, hop: int, forward: bool):
        nodes.append(int(node))
        pos = len(nodes) - 1
        role_list.append(hop)
        direction_list.append(dir_code)
        if forward:
            edges.append((src, pos))
        else:
            edges.append((pos, src))

    role_list = [0]
    direction_list = [0]
    for node in hop1_out:
        add(node, 0, 1, 1, forward=True)
    for node in hop1_in:
        add(node, 0, 2, 1, forward=False)
    hop1_count = len(nodes)
    for src in range(1, hop1_count):
        if len(nodes) >= K2_CAP + 1:
            break
        source_node = nodes[src]
        for csr, dir_code, forward in ((out_csr, 1, True), (in_csr, 2, False)):
            for node in neighbours(csr, source_node, K2_PER, rng):
                if len(nodes) >= K2_CAP + 1:
                    break
                if int(node) in nodes:
                    continue
                add(int(node), src, dir_code, 2, forward=forward)
    return {
        "nodes": np.array(nodes, dtype=np.int64),
        "role": np.array(role_list, dtype=np.int64),
        "direction": np.array(direction_list, dtype=np.int64),
        "edges": np.array(edges, dtype=np.int64).reshape(-1, 2),
    }


def node_features(nodes: np.ndarray, role: np.ndarray, direction: np.ndarray,
                  out_csr, in_csr) -> torch.Tensor:
    out_deg = np.diff(out_csr.indptr)[nodes].astype(np.float64)
    in_deg = np.diff(in_csr.indptr)[nodes].astype(np.float64)
    features = np.stack([
        (role == 0).astype(np.float64),
        (role == 1).astype(np.float64),
        (role == 2).astype(np.float64),
        (direction == 1).astype(np.float64),
        (direction == 2).astype(np.float64),
        np.log1p(in_deg),
        np.log1p(out_deg),
        np.log1p(in_deg * out_deg),
    ], axis=1)
    features = features / np.maximum(features.std(axis=0, ddof=0), 1e-6)
    return torch.tensor(features, dtype=torch.float32, device=DEVICE)


class DirectedSAGE(nn.Module):
    """Directed message passing over a batch of disjoint ego subgraphs.

    All samples in a batch are concatenated into one graph (edges are built per sample,
    so nothing crosses), which turns 256 tiny forwards into one.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = HIDDEN, t_dim: int = T_DIM):
        super().__init__()
        self.self_1 = nn.Linear(feature_dim, hidden)
        self.in_1 = nn.Linear(feature_dim, hidden)
        self.out_1 = nn.Linear(feature_dim, hidden)
        self.self_2 = nn.Linear(hidden, hidden)
        self.in_2 = nn.Linear(hidden, hidden)
        self.out_2 = nn.Linear(hidden, hidden)
        self.pool = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, t_dim),
        )

    @staticmethod
    def _segment_mean(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
        if values.shape[0] == 0:
            return values.new_zeros(count, values.shape[1])
        out = values.new_zeros(count, values.shape[1])
        weight = values.new_zeros(count, 1)
        out.index_add_(0, index, values)
        weight.index_add_(0, index, values.new_ones(values.shape[0], 1))
        return out / weight.clamp(min=1.0)

    def forward(self, features, edge_index, sample_id, role, root_index):
        size = features.shape[0]
        batch = root_index.shape[0]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        h = features
        h = F.relu(self.self_1(h)
                   + self._segment_mean(self.in_1(h[src]), dst, size)
                   + self._segment_mean(self.out_1(h[dst]), src, size))
        h = F.relu(self.self_2(h)
                   + self._segment_mean(self.in_2(h[src]), dst, size)
                   + self._segment_mean(self.out_2(h[dst]), src, size))
        root = h[root_index]
        hop1 = self._segment_mean(h[role == 1], sample_id[role == 1], batch)
        hop2 = self._segment_mean(h[role == 2], sample_id[role == 2], batch)
        return self.pool(torch.cat([root, hop1, hop2], dim=1))


class Teacher(nn.Module):
    def __init__(self, n_items: int):
        super().__init__()
        self.encoder = DirectedSAGE()
        self.head_out = nn.Linear(T_DIM, n_items, bias=True)
        self.head_in = nn.Linear(T_DIM, n_items, bias=True)

    def encode(self, batch, out_csr, in_csr):
        """Concatenate the batch's ego subgraphs into one graph and encode them together.

        Degrees are gathered once for the whole batch instead of once per sample; the
        per-sample numpy path was 2 ms/sample and dominated the epoch.
        """
        nodes_all, roles_all, dirs_all = [], [], []
        edge_rows, sample_ids, roots = [], [], []
        offset = 0
        for index, sample in enumerate(batch):
            count = len(sample["nodes"])
            nodes_all.append(sample["nodes"])
            roles_all.append(sample["role"])
            dirs_all.append(sample["direction"])
            if sample["edges"].size:
                edge_rows.append(sample["edges"] + offset)
            sample_ids.append(np.full(count, index, dtype=np.int64))
            roots.append(offset)
            offset += count
        nodes_all = np.concatenate(nodes_all)
        roles_all = np.concatenate(roles_all)
        dirs_all = np.concatenate(dirs_all)
        in_deg = np.diff(in_csr.indptr)[nodes_all].astype(np.float32)
        out_deg = np.diff(out_csr.indptr)[nodes_all].astype(np.float32)
        features = np.stack([
            (roles_all == 0).astype(np.float32),
            (roles_all == 1).astype(np.float32),
            (roles_all == 2).astype(np.float32),
            (dirs_all == 1).astype(np.float32),
            (dirs_all == 2).astype(np.float32),
            np.log1p(in_deg),
            np.log1p(out_deg),
            np.log1p(in_deg * out_deg),
        ], axis=1)
        features = features / np.maximum(features.std(axis=0, ddof=0), 1e-6)
        edge_index = (torch.tensor(np.concatenate(edge_rows, axis=0), device=DEVICE)
                      if edge_rows else torch.zeros(0, 2, dtype=torch.long, device=DEVICE))
        return self.encoder(
            torch.tensor(features, device=DEVICE),
            edge_index,
            torch.tensor(np.concatenate(sample_ids), device=DEVICE),
            torch.tensor(roles_all, device=DEVICE),
            torch.tensor(roots, dtype=torch.long, device=DEVICE),
        )


def sampled_softmax_loss(vector: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over one positive plus sampled negatives, without a full-vocabulary matmul."""
    batch = vector.shape[0]
    pos_logit = (vector * weight[positives]).sum(-1, keepdim=True) + bias[positives]
    neg_logit = torch.einsum("bd,bnd->bn", vector, weight[negatives]) + bias[negatives]
    logits = torch.cat([pos_logit, neg_logit], dim=1)
    return F.cross_entropy(logits, torch.zeros(batch, dtype=torch.long, device=vector.device))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out_csr, in_csr = load_graph()
    n_items = int(out_csr.shape[0])
    targets = np.load(GRAPH / "target_edges.npy")
    target_weights = np.load(GRAPH / "target_weights.npy")
    eval_edges = np.load(GRAPH / "eval_edges.npy")
    eval_weights = np.load(GRAPH / "eval_weights.npy")

    # popularity for negative sampling (visible graph out-degree mass)
    popularity = np.asarray(out_csr.sum(axis=1)).ravel()
    popularity = (popularity + 1.0) ** 0.75
    popularity = popularity / popularity.sum()

    model = Teacher(n_items).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    t0 = time.time()
    roots = targets[:, 0]
    order = np.arange(len(targets))
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(order)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), BATCH):
            chunk = order[start:start + BATCH]
            batch = [ego_sample(int(roots[k]), out_csr, in_csr, rng) for k in chunk]
            vectors = model.encode(batch, out_csr, in_csr)
            positives_out = torch.tensor(targets[chunk, 1], device=DEVICE)
            positives_in = torch.tensor(targets[chunk, 0], device=DEVICE)
            negatives = torch.tensor(
                rng.choice(n_items, size=(len(chunk), NEGATIVES), p=popularity),
                device=DEVICE,
            )
            loss_out = sampled_softmax_loss(vectors, model.head_out.weight, model.head_out.bias,
                                            positives_out, negatives)
            loss_in = sampled_softmax_loss(vectors, model.head_in.weight, model.head_in.bias,
                                           positives_in, negatives)
            loss = 0.5 * (loss_out + loss_in)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss)
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        losses.append(mean_loss)
        print(f"[teacher] epoch={epoch} loss={mean_loss:.5f} elapsed={time.time() - t0:.0f}s", flush=True)

    torch.save({"state_dict": model.state_dict(), "n_items": n_items,
                "feature_dim": FEATURE_DIM, "hidden": HIDDEN, "t_dim": T_DIM}, OUT / "teacher.pt")
    (OUT / "train_log.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "negatives": NEGATIVES,
        "lr": LR, "losses": losses, "wall_time_s": round(time.time() - t0, 1),
        "target_edges": int(len(targets)),
        "ego_budget": {"k1_out": K1_OUT, "k1_in": K1_IN, "k2_per": K2_PER, "k2_cap": K2_CAP},
    }, indent=2), encoding="utf-8")
    print("teacher saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
