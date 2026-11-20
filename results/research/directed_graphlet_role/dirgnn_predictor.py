"""Direction-aware graph network as the structure-aware predictor (#221).

The head so far reads item vectors only.  This trains a message-passing encoder
over the behaviour graph -- in-neighbours and out-neighbours aggregated with
separate weights, the Dir-GNN rule -- and asks whether the resulting per-node
representations carry the directed pattern of a held-out instance.

Fairness:
  * the graph excludes every edge inside a held-out instance (same reduced graph
    the DGDV uses), so a held-out instance's own edges never reach its nodes;
  * the train/held-out split is by item; a test instance is made only of items
    the model never trained on;
  * the same head, initialisation and optimiser run for every graph variant.

Variants: directed (in/out separate), undirected (in/out collapsed -- the
control that says whether *direction* is doing the work), shuffled (the same
edge set with directions randomised), and no-graph (item vector only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dgdv_features import heldout_edge_pairs, reduce_graph, shuffle_directions  # noqa: E402
from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402

REPO = HERE.parents[2]
FAMILY = "P3"
LAYERS = 2
HIDDEN = 64
STEPS = 600
HELD_FRACTION = 0.3
SEED = 20261120


class DirLayer(nn.Module):
    def __init__(self, dim: int, hidden: int, directional: bool = True):
        super().__init__()
        self.self_proj = nn.Linear(dim, hidden)
        self.in_proj = nn.Linear(dim, hidden)
        self.out_proj = nn.Linear(dim, hidden) if directional else None

    def forward(self, h, in_index, out_index, n):
        total_in = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, in_index[1], h[in_index[0]])
        degree = torch.zeros(n, 1, device=h.device).index_add_(
            0, in_index[1], torch.ones(len(in_index[1]), 1, device=h.device))
        aggregated = self.in_proj(total_in / degree.clamp(min=1))
        if self.out_proj is None:
            return F.gelu(self.self_proj(h) + aggregated)
        total_out = torch.zeros(n, h.shape[1], device=h.device).index_add_(
            0, out_index[1], h[out_index[0]])
        degree_out = torch.zeros(n, 1, device=h.device).index_add_(
            0, out_index[1], torch.ones(len(out_index[1]), 1, device=h.device))
        return F.gelu(self.self_proj(h) + aggregated
                      + self.out_proj(total_out / degree_out.clamp(min=1)))


class DirGNN(nn.Module):
    def __init__(self, dim: int, hidden: int = HIDDEN, layers: int = LAYERS,
                 directional: bool = True):
        super().__init__()
        self.input = nn.Linear(dim, hidden)
        self.layers = nn.ModuleList([DirLayer(hidden, hidden, directional)
                                     for _ in range(layers)])

    def forward(self, x, in_index, out_index):
        h = F.gelu(self.input(x))
        for layer in self.layers:
            h = layer(h, in_index, out_index, x.shape[0])
        return h


class PooledHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x):
        h = self.phi(x)
        return self.head(torch.cat([h.mean(1), h.amax(1)], -1))


def edge_indices(matrix, device):
    """(source, destination) index arrays for message passing."""
    rows, cols = matrix.nonzero()
    return (torch.from_numpy(rows.astype(np.int64)).to(device),
            torch.from_numpy(cols.astype(np.int64)).to(device))


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    full = load_adjacency()
    n = full.shape[0]
    fit = reduce_graph(full, heldout_edge_pairs())
    rng_np = np.random.default_rng(SEED)
    shuffled = shuffle_directions(fit, rng_np)
    undirected = (fit + fit.T).astype(bool).tocsr()
    undirected.data[:] = True
    graphs = {"directed": (fit, True), "undirected": (undirected, False),
              "shuffled": (shuffled, True)}

    semantic = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    z = torch.from_numpy(semantic.astype("float32")).to(device)

    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=60000)
    held_families = load_pool("heldout", vocab, max_instances=20000)
    all_items = np.unique(np.concatenate(
        [nodes.reshape(-1) for nodes, _o, _s in train_families.values()]
        + [nodes.reshape(-1) for nodes, _o, _s in held_families.values()]))
    held_items = rng_np.permutation(all_items)[: int(HELD_FRACTION * len(all_items))]
    held_set = np.zeros(int(all_items.max()) + 1, dtype=bool)
    held_set[held_items] = True

    def partition(families):
        out = {}
        for family, (nodes, orbits, subtypes) in families.items():
            inside = held_set[nodes]
            out[family] = ((nodes[~inside.any(1)], subtypes[~inside.any(1)]),
                           (nodes[inside.all(1)], subtypes[inside.all(1)]))
        return out

    parts = partition(train_families)
    held_parts = partition(held_families)
    nodes, subtypes = parts[FAMILY][0]
    counts = np.bincount(subtypes)
    best_two = np.argsort(counts)[::-1][:2]
    mapping = {int(old): new for new, old in enumerate(best_two)}

    def balanced(nodes, subtypes, cap):
        picks = []
        for label in best_two:
            index = np.flatnonzero(subtypes == label)
            picks.append(rng_np.choice(index, min(cap, len(index)), replace=False))
        chosen = np.concatenate(picks)
        rng_np.shuffle(chosen)
        return nodes[chosen], np.array([mapping[int(s)] for s in subtypes[chosen]])

    train_nodes, train_y = balanced(nodes, subtypes, 6000)
    # Held-out instances come from the held-out pool, whose edges are removed from
    # the graph: the model must predict a triple's pattern without ever seeing the
    # triple's own edges.  The item-disjoint subset is reported alongside as the
    # stricter reading (no item on both sides either).
    held_pool_nodes, held_pool_y = balanced(held_families[FAMILY][0],
                                            held_families[FAMILY][2], 6000)
    disjoint_mask = held_set[held_pool_nodes].all(1)
    held_nodes, held_y = held_pool_nodes, held_pool_y
    print(f"train {train_nodes.shape} heldout {held_nodes.shape} "
          f"classes {best_two.tolist()}", flush=True)
    x_train = torch.from_numpy(train_nodes).long().to(device)
    x_held = torch.from_numpy(held_nodes).long().to(device)
    disjoint = torch.from_numpy(disjoint_mask).to(device)
    y_train = torch.from_numpy(train_y).long().to(device)
    y_held = torch.from_numpy(held_y).long().to(device)
    no_edges = (torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.int64, device=device))
    empty_pair = (no_edges, no_edges)

    results = {"family": FAMILY, "subtypes": [int(v) for v in best_two],
               "layers": LAYERS, "hidden": HIDDEN, "steps": STEPS,
               "items": int(n), "fit_edges": int(fit.nnz), "runs": []}
    for name, (matrix, directional) in graphs.items():
        rows, cols = edge_indices(matrix, device)
        in_index = (rows, cols)          # message u -> v : u = rows, v = cols
        out_index = (cols, rows)         # message v -> w : w = rows... swapped
        for tag, use_graph in ((name, True), (f"{name}_no_graph", False)):
            torch.manual_seed(7)
            gnn = DirGNN(z.shape[1], directional=directional).to(device)
            head = PooledHead(HIDDEN).to(device)
            params = list(gnn.parameters()) + list(head.parameters())
            optimizer = torch.optim.AdamW(params, lr=1e-3)
            in_use, out_use = (in_index, out_index) if use_graph else empty_pair
            for step in range(1, STEPS + 1):
                h = gnn(z, in_use, out_use)
                logits = head(h[x_train])
                loss = F.cross_entropy(logits, y_train)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                h = gnn(z, in_use, out_use)
                train_acc = float((head(h[x_train]).argmax(-1) == y_train).float().mean())
                predictions = head(h[x_held]).argmax(-1)
                held_acc = float((predictions == y_held).float().mean())
                if int(disjoint.sum()) > 20:
                    disjoint_acc = float((predictions[disjoint] == y_held[disjoint]).float().mean())
                else:
                    disjoint_acc = None
            row = {"variant": tag, "graph": use_graph, "directional": directional,
                   "train_accuracy": train_acc, "heldout_accuracy": held_acc,
                   "heldout_instances": int(len(held_y)),
                   "item_disjoint_accuracy": disjoint_acc,
                   "item_disjoint_instances": int(disjoint.sum()),
                   "loss": float(loss.detach()), "chance": 0.5}
            results["runs"].append(row)
            print(json.dumps(row), flush=True)
    (HERE / "dirgnn_predictor.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
