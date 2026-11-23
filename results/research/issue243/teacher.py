"""Issue #242 step 2-4: local topology gating + five curvature experts (GraphMoRE style).

Two separate things, deliberately not mixed:

* ``T_i`` -- a 128-dim descriptor of i's 1-hop and 2-hop directed ego subgraph, computed by a
  direction-aware two-layer GNN over the 768-dim semantic features plus structural flags.
  T_i only feeds the gate; it is never the relation label.
* ``h_e(i)`` -- expert e's 16-dim embedding of item i, produced by a full-graph directed GNN
  trained only through the graph-distance distortion below.

    W_i        = softmax(gate(T_i)),           W_ij = softmax(W_i * W_j)
    D_geo^2    = sum_e W_ij,e * d_e(i,j)^2
    L_D        = mean |(D_geo / g)^2 - 1|,     g = unweighted shortest path, undirected view

Curvatures are -3, -1, 0, +1, +3.  Negative curvature uses the Poincare ball of radius
1/sqrt|c|, zero is Euclidean, positive curvature uses a sphere of radius 1/sqrt(c).  All
projections clamp inside the open ball/sphere so the geodesics stay finite.

There is no next-item classification, no link-prediction gate and no positive/negative
classification loss -- the issue removed all three.
"""
from __future__ import annotations

import json
import math

import kappa_gcn
import manifold
import topology
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
GRAPH = REPO / "results" / "research" / "issue242" / "graph"  # reused, hashes verified
OUT = REPO / "results" / "research" / "issue243" / "teacher"
FEATURES = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
SEED = 243002
CURVATURES = (-3.0, -1.0, 0.0, 1.0, 3.0)
NODE_OUT = 16
HIDDEN = 64
T_DIM = 128
K1, K2_PER, K2_CAP = 8, 4, 24
PAIR_BATCH = 192
STEPS = 400
LR = 1e-3
EPS = 1e-7
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- manifolds
def load_all():
    blob = np.load(GRAPH / "ego_csr.npz")
    ego = sp.csr_matrix((blob["data"], blob["indices"], blob["indptr"]),
                        shape=tuple(blob["shape"]))
    ego.sort_indices()
    embeddings = np.load(FEATURES).astype(np.float32)
    n = ego.shape[0]
    out_deg = np.diff(ego.indptr).astype(np.float32)
    in_csr = ego.T.tocsr()
    in_csr.sort_indices()
    in_deg = np.diff(in_csr.indptr).astype(np.float32)
    structural = np.stack([
        np.log1p(in_deg), np.log1p(out_deg), np.log1p(in_deg * out_deg),
        (in_deg == 0).astype(np.float32), (out_deg == 0).astype(np.float32),
        np.ones(n, dtype=np.float32),
    ], axis=1)
    return ego, in_csr, embeddings, structural


def edge_index(csr: sp.csr_matrix):
    coo = csr.tocoo()
    return torch.tensor(np.stack([coo.row, coo.col], axis=1), dtype=torch.long, device=DEVICE)


def neighbours(csr, node: int, k: int, rng) -> np.ndarray:
    start, end = csr.indptr[node], csr.indptr[node + 1]
    if start == end:
        return np.empty(0, dtype=np.int64)
    cols = csr.indices[start:end]
    if len(cols) <= k:
        return cols.astype(np.int64)
    pick = rng.choice(len(cols), size=k, replace=False)
    return cols[pick].astype(np.int64)


def bfs_distances(csr: sp.csr_matrix, undirected: sp.csr_matrix, root: int, depth: int):
    """Unweighted BFS on the undirected projection, returning {node: hop}."""
    seen = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if seen[node] >= depth:
            continue
        start, end = undirected.indptr[node], undirected.indptr[node + 1]
        for nxt in undirected.indices[start:end]:
            nxt = int(nxt)
            if nxt not in seen:
                seen[nxt] = seen[node] + 1
                queue.append(nxt)
    return seen


# --------------------------------------------------------------------------- model
class DirectedLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.in_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(in_dim, out_dim, bias=False)

    @staticmethod
    def scatter(values, index, size):
        out = values.new_zeros(size, values.shape[1])
        weight = values.new_zeros(size, 1)
        out.index_add_(0, index, values)
        weight.index_add_(0, index, values.new_ones(values.shape[0], 1))
        return out / weight.clamp(min=1.0)

    def forward(self, x, edges, size):
        src, dst = edges[:, 0], edges[:, 1]
        agg_in = self.scatter(self.in_proj(x[src]), dst, size)
        agg_out = self.scatter(self.out_proj(x[dst]), src, size)
        return F.relu(self.self_proj(x) + agg_in + agg_out)


class TopologyDescriptor(nn.Module):
    """T_i: 128 dims = two scales x (root, mean_in, mean_out, mean_all) of 16."""

    def __init__(self, semantic_dim: int, structural_dim: int):
        super().__init__()
        width = semantic_dim + structural_dim
        self.layer1 = DirectedLayer(width, HIDDEN)
        self.layer2 = DirectedLayer(HIDDEN, NODE_OUT)

    def encode(self, samples, semantic, structural):
        """One scale: 64 dims = root, mean_in, mean_out, mean_all, each 16.

        ``semantic`` / ``structural`` are numpy arrays; ego indices are gathered on the host.
        """
        rows_sem, rows_struct = [], []
        edges_all, sample_id, directions, roots = [], [], [], []
        offset = 0
        for index, sample in enumerate(samples):
            nodes = sample["nodes"]
            rows_sem.append(semantic[nodes])
            rows_struct.append(structural[nodes])
            if sample["edges"].size:
                edges_all.append(sample["edges"] + offset)
            sample_id.append(np.full(len(nodes), index, dtype=np.int64))
            directions.append(sample["direction"])
            roots.append(offset)
            offset += len(nodes)
        x = torch.cat([
            torch.tensor(np.concatenate(rows_sem), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.concatenate(rows_struct), dtype=torch.float32, device=DEVICE),
        ], dim=1)
        edges = (torch.tensor(np.concatenate(edges_all), dtype=torch.long, device=DEVICE)
                 if edges_all else torch.zeros(0, 2, dtype=torch.long, device=DEVICE))
        sample_id = torch.tensor(np.concatenate(sample_id), device=DEVICE)
        directions = torch.tensor(np.concatenate(directions), device=DEVICE)
        roots = torch.tensor(roots, dtype=torch.long, device=DEVICE)
        size = x.shape[0]
        batch = roots.shape[0]
        h = self.layer1(x, edges, size)
        h = self.layer2(h, edges, size)
        root = h[roots]
        mean_in = DirectedLayer.scatter(h[directions == 2], sample_id[directions == 2], batch)
        mean_out = DirectedLayer.scatter(h[directions == 1], sample_id[directions == 1], batch)
        mean_all = DirectedLayer.scatter(h[directions != 0], sample_id[directions != 0], batch)
        return torch.cat([root, mean_in, mean_out, mean_all], dim=1)  # (B, 4*NODE_OUT)

    def encode_two_scales(self, roots, out_csr, in_csr, rng, semantic, structural):
        """T_i is the concatenation of the 1-hop and 2-hop descriptors: 2 x 64 = 128.

        The sampler is the induced-edge one from ``topology`` (issue #244 §2): decide the
        node set first, then keep every directed edge among those nodes.
        """
        one_hop = [topology.ego_induced(int(r), out_csr, in_csr, rng, radius=1) for r in roots]
        two_hop = [topology.ego_induced(int(r), out_csr, in_csr, rng, radius=2) for r in roots]
        return torch.cat([
            self.encode(one_hop, semantic, structural),
            self.encode(two_hop, semantic, structural),
        ], dim=1)


class Gate(nn.Module):
    def __init__(self, n_experts: int = len(CURVATURES)):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * 4 * NODE_OUT, 64), nn.ReLU(), nn.Linear(64, n_experts))

    def forward(self, descriptors):
        return torch.softmax(self.mlp(descriptors), dim=-1)


class Expert(nn.Module):
    """One curvature expert, now a real kappa-GCN stack (CONTRACT §2)."""

    def __init__(self, semantic_dim: int, structural_dim: int, curvature: float):
        super().__init__()
        self.curvature = float(curvature)
        self.stack = kappa_gcn.KappaExpert(semantic_dim + structural_dim, HIDDEN, NODE_OUT, curvature)

    def forward(self, x, in_edges, out_edges):
        return self.stack(x, in_edges, out_edges)


class MixedGeometryTeacher(nn.Module):
    def __init__(self, semantic_dim: int, structural_dim: int):
        super().__init__()
        self.descriptor = TopologyDescriptor(semantic_dim, structural_dim)
        self.gate = Gate()
        self.experts = nn.ModuleList([
            Expert(semantic_dim, structural_dim, c) for c in CURVATURES
        ])

    def weights_for(self, descriptors):
        return self.gate(descriptors)

    def embeddings(self, x, edges):
        # (neighbour, target) pairs: the edge itself feeds the in-weight, the reversed
        # edge feeds the out-weight, so direction actually selects a different parameter
        in_edges = edges
        out_edges = edges[:, [1, 0]]
        return [expert(x, in_edges, out_edges) for expert in self.experts]

    def distance(self, embeddings, w_i, w_j, index_i, index_j):
        """D_geo and the per-expert distances for the selected node pairs."""
        pair_weights = torch.softmax(w_i * w_j, dim=-1)
        squared = []
        for expert_index, (expert, embedding) in enumerate(zip(self.experts, embeddings)):
            d = manifold.dist(embedding[index_i], embedding[index_j], expert.curvature)
            squared.append(d.pow(2))
        stacked = torch.stack(squared, dim=1)
        d_squared = (pair_weights * stacked).sum(dim=1)
        return d_squared.clamp(min=0.0).sqrt(), stacked.sqrt(), pair_weights


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    """Rank correlation without pulling in scipy: Pearson on the ranks."""
    a = torch.argsort(torch.argsort(left.detach())).float()
    b = torch.argsort(torch.argsort(right.detach())).float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out_csr, in_csr, embeddings, structural = load_all()
    n = embeddings.shape[0]
    semantic = torch.tensor(embeddings, device=DEVICE)
    structural_t = torch.tensor(structural, device=DEVICE)
    full_x = torch.cat([semantic, structural_t], dim=1)
    out_edges = edge_index(out_csr)
    undirected = (out_csr + out_csr.T).tocsr()
    undirected.data[:] = 1

    degree = np.diff(out_csr.indptr)
    # degree-balanced root pool: five degree quintiles, equal mass each
    active = np.where(degree > 0)[0]
    quintile_edges = np.quantile(degree[active], [0.2, 0.4, 0.6, 0.8])
    strata = [active[np.digitize(degree[active], quintile_edges) == q] for q in range(5)]
    roots = np.concatenate([np.repeat(s, max(1, len(active) // (5 * len(s))))
                            for s in strata if len(s)])

    model = MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    history = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        # --- sampled pairs, distance-bucketed -----------------------------
        picks = rng.choice(len(roots), size=PAIR_BATCH, replace=False)
        nodes_i = np.unique(roots[picks])[: PAIR_BATCH // 2]
        # One C-level BFS for every root in the step: the pure-Python deque version was
        # ~0.2 s per root and dominated the whole step.
        distance_matrix = csgraph.shortest_path(
            undirected, method="D", unweighted=True, indices=nodes_i)
        pairs = []
        for row, node_i in enumerate(nodes_i):
            by_hop = {}
            for node, hop in enumerate(distance_matrix[row]):
                if 1 <= hop <= 4:
                    by_hop.setdefault(int(hop), []).append(int(node))
            if not by_hop:
                continue
            # distance bucket balanced inside the root: draw a hop uniformly, then a node
            hops = sorted(by_hop)
            for _ in range(2):
                hop = int(hops[rng.integers(0, len(hops))])
                node_j = int(by_hop[hop][rng.integers(0, len(by_hop[hop]))])
                pairs.append((int(node_i), node_j, hop))
        if len(pairs) < 8:
            continue
        pairs = pairs[:PAIR_BATCH]
        index_i = torch.tensor([p[0] for p in pairs], device=DEVICE)
        index_j = torch.tensor([p[1] for p in pairs], device=DEVICE)
        gap = torch.tensor([float(p[2]) for p in pairs], device=DEVICE)

        unique = sorted({int(x) for p in pairs for x in (p[0], p[1])})
        position = {node: k for k, node in enumerate(unique)}
        descriptors = model.descriptor.encode_two_scales(
            unique, out_csr, in_csr, rng, embeddings, structural)
        weights = model.weights_for(descriptors)
        local_i = torch.tensor([position[int(x)] for x in index_i.tolist()], device=DEVICE)
        local_j = torch.tensor([position[int(x)] for x in index_j.tolist()], device=DEVICE)
        expert_emb = model.embeddings(full_x, out_edges)
        d_geo, per_expert, pair_w = model.distance(
            expert_emb, weights[local_i], weights[local_j], index_i, index_j)
        ratio = d_geo / gap.clamp(min=1.0)
        loss = (ratio.pow(2) - 1.0).abs().mean()
        optimiser.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimiser.step()
        if step % 20 == 0 or step == 1:
            gate_entropy = float(-(weights * torch.log(weights + 1e-9)).sum(-1).mean())
            history.append({
                "step": step, "loss": float(loss.detach()), "grad_norm": grad_norm,
                "d_geo_mean": float(d_geo.mean()), "gap_mean": float(gap.mean()),
                "spearman_batch": spearman(d_geo, gap),
                "gate_entropy": gate_entropy,
                "gate_mean": [float(v) for v in weights.mean(0)],
                "expert_d_mean": [float(v.mean()) for v in per_expert.T],
                "elapsed": round(time.time() - t0, 1),
            })
            print(json.dumps(history[-1]), flush=True)

    torch.save({"state_dict": model.state_dict(), "curvatures": list(CURVATURES),
                "node_out": NODE_OUT}, OUT / "teacher.pt")
    (OUT / "train_log.json").write_text(json.dumps({
        "seed": SEED, "steps": STEPS, "pair_batch": PAIR_BATCH, "lr": LR,
        "budget_note": "STEPS/PAIR_BATCH fixed before the run by wall-clock budget, not by any observed metric",
        "sampling": "root: uniform over five out-degree quintiles; partner: hop drawn uniformly from {1,2,3,4} then uniform within the hop",
        "curvatures": list(CURVATURES), "history": history,
        "wall_time_s": round(time.time() - t0, 1),
        "features_sha256": sha256(FEATURES),
        "graph_metadata_sha256": sha256(GRAPH / "metadata.json"),
    }, indent=2), encoding="utf-8")
    print("teacher saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
