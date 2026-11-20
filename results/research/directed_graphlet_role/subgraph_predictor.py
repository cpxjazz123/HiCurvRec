"""Component test: can a subgraph encoder read a directed pattern when it is given?

The instance predictors so far read item vectors only, and the Dir-GNN control
removed the instance's own edges, so both were asked to *infer* structure that
was not in their input.  This is the complementary test: feed the induced
directed adjacency of the instance itself and check whether subtype and orbit
are recovered on an independent (held-out pool) test set.

Controls, same encoder, same init, same data:
  directed   the induced directed adjacency (what the user asked to test)
  no_graph   every edge zeroed -- the readout is the bare item vectors
  shuffled   the same undirected edges with each direction flipped at p=1/2

If `directed` recovers subtype/orbit and the two controls do not, the encoder is
adequate and the earlier failure is an input problem, not a model problem.
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

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import (FAMILIES, instance_targets, load_pool,  # noqa: E402
                           load_vocabulary)

REPO = HERE.parents[2]
TRAIN_CAP = 20000
TEST_CAP = 5000
HIDDEN = 128
LAYERS = 2
STEPS = 1500
BATCH = 512
SEED = 20261120


class SubgraphEncoder(nn.Module):
    """In/out separated message passing on the instance's induced subgraph."""

    def __init__(self, dim: int, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.self_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.in_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.out_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for layer in range(len(self.self_proj)):
            degree_in = adjacency.sum(1).clamp(min=1e-6)            # (B, k)
            degree_out = adjacency.sum(2).clamp(min=1e-6)
            in_message = torch.einsum("bji,bjh->bih", adjacency, h) / degree_in.unsqueeze(-1)
            out_message = torch.einsum("bij,bjh->bih", adjacency, h) / degree_out.unsqueeze(-1)
            h = F.gelu(self.self_proj[layer](h) + self.in_proj[layer](in_message)
                       + self.out_proj[layer](out_message))
        return h


class Readout(nn.Module):
    def __init__(self, dim: int, n_subtypes: int, n_orbits: int, hidden: int = 128):
        super().__init__()
        self.subtype = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, n_subtypes))
        self.context = nn.Linear(2 * dim, dim)
        self.orbit = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, n_orbits))

    def forward(self, h: torch.Tensor):
        pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        context = self.context(pooled).unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.subtype(pooled), self.orbit(torch.cat([h, context, h * context], -1))


def adjacency_tensor(nodes: np.ndarray, edge_keys: np.ndarray, n_items: int) -> np.ndarray:
    count, width = nodes.shape
    out = np.zeros((count, width, width), dtype=np.float32)
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
            out[:, i, j] = np.isin(keys, edge_keys).astype(np.float32)
    return out


def shuffle_directions(a: np.ndarray, seed: int) -> np.ndarray:
    """Reverse the direction of each unordered pair with probability 1/2.

    The decision is made once per unordered pair, not once per matrix cell: an
    elementwise mask would let (i,j) flip while (j,i) does not, which creates a
    reciprocal pair out of a single edge, or removes an edge when both cells
    flip.  Here the undirected skeleton -- every pair's edge count -- is
    preserved exactly, so the control changes direction and nothing else.
    """
    rng = np.random.default_rng(seed)
    out = a.copy()
    for i in range(a.shape[1]):
        for j in range(i + 1, a.shape[2]):
            flip = rng.random(len(a)) < 0.5
            out[:, i, j] = np.where(flip, a[:, j, i], a[:, i, j])
            out[:, j, i] = np.where(flip, a[:, i, j], a[:, j, i])
    return out


def macro_f1(prediction: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        tp = int(((prediction == label) & (target == label)).sum())
        fp = int(((prediction == label) & (target != label)).sum())
        fn = int(((prediction != label) & (target == label)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edge_keys = rows.astype(np.int64) * n_items + cols
    semantic = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                        "sentence_t5.npy").astype("float32")).to(device)
    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=TRAIN_CAP)
    test_families = load_pool("heldout", vocab, max_instances=TEST_CAP)

    prepared = {}
    for family in FAMILIES:
        train_nodes, train_orbits, train_subtypes = train_families[family]
        test_nodes, test_orbits, test_subtypes = test_families[family]
        prepared[family] = {
            "train": (train_nodes, train_orbits, train_subtypes,
                      instance_targets(family, train_subtypes, train_orbits, vocab)),
            "test": (test_nodes, test_orbits, test_subtypes,
                     instance_targets(family, test_subtypes, test_orbits, vocab)),
        }
        print(f"{family}: train {len(train_nodes)} test {len(test_nodes)}", flush=True)

    def make_adjacency(nodes, mode):
        if mode == "no_graph":
            return np.zeros((len(nodes), nodes.shape[1], nodes.shape[1]), dtype=np.float32)
        a = adjacency_tensor(nodes, edge_keys, n_items)
        return shuffle_directions(a, SEED) if mode == "shuffled" else a

    adjacency_cache = {}
    for family in FAMILIES:
        for split in ("train", "test"):
            nodes = prepared[family][split][0]
            for mode in ("directed", "no_graph", "shuffled"):
                adjacency_cache[(family, split, mode)] = torch.from_numpy(
                    make_adjacency(nodes, mode)).to(device)

    results = {}
    for mode in ("directed", "no_graph", "shuffled"):
        torch.manual_seed(7)
        encoder = SubgraphEncoder(semantic.shape[1]).to(device)
        readout = Readout(HIDDEN, vocab["n_subtypes"], vocab["n_orbits"]).to(device)
        params = list(encoder.parameters()) + list(readout.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        allowed = vocab["allowed"].to(device)

        def batches(split):
            for family in FAMILIES:
                nodes, orbits, subtypes, targets = prepared[family][split]
                adjacency = adjacency_cache[(family, split, mode)]
                valid = targets["subtype"] >= 0
                index = torch.from_numpy(nodes).long().to(device)
                kept = np.flatnonzero(valid)
                yield (family, index, adjacency, targets, kept)

        data = list(batches("train"))
        generator = torch.Generator().manual_seed(SEED)
        for step in range(1, STEPS + 1):
            family, index, adjacency, targets, kept = data[step % len(data)]
            draw = torch.randint(0, len(kept), (min(BATCH, len(kept)),), generator=generator).numpy()
            pick = torch.from_numpy(kept[draw]).to(device)
            h = encoder(semantic[index[pick]], adjacency[pick])
            type_logits, role_logits = readout(h)
            target_sub = torch.from_numpy(targets["subtype"]).to(device)[pick]
            target_orbit = torch.from_numpy(targets["orbit"]).to(device)[pick]
            loss_type = F.cross_entropy(type_logits, target_sub.clamp(min=0))
            # Unconditional orbit CE: the head is NOT told the true subtype, so the
            # number that comes out is end-to-end orbit recognition, not a readout
            # conditioned on the answer.
            loss_role = F.cross_entropy(role_logits.reshape(-1, role_logits.shape[-1]),
                                        target_orbit.reshape(-1))
            loss = loss_type + loss_role
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 300 == 0:
                print(f"  {mode} step {step} loss {float(loss):.4f}", flush=True)

        row = {"variant": mode, "per_family": {}}
        with torch.no_grad():
            for family, index, adjacency, targets, kept in batches("test"):
                h = encoder(semantic[index], adjacency)
                type_logits, role_logits = readout(h)
                prediction = type_logits.argmax(-1).cpu().numpy()
                target_sub = targets["subtype"]
                subtype_ids = torch.from_numpy(target_sub).to(device).clamp(min=0)
                mask = allowed[subtype_ids].unsqueeze(1).expand(-1, role_logits.shape[1], -1).clone()
                mask[torch.from_numpy(target_sub < 0).to(device)] = True
                unconditional = role_logits.argmax(-1).cpu().numpy()
                masked = role_logits.masked_fill(~mask, float("-inf"))
                conditioned = masked.argmax(-1).cpu().numpy()
                orbit_target = targets["orbit"]
                keep = kept
                subtype_f1 = macro_f1(prediction[keep], target_sub[keep], vocab["n_subtypes"])
                orbit_f1 = macro_f1(unconditional[keep].reshape(-1),
                                    orbit_target[keep].reshape(-1), vocab["n_orbits"])
                orbit_f1_conditioned = macro_f1(conditioned[keep].reshape(-1),
                                                orbit_target[keep].reshape(-1), vocab["n_orbits"])
                row["per_family"][family] = {
                    "subtype_macro_f1": subtype_f1,
                    "orbit_macro_f1": orbit_f1,
                    "orbit_macro_f1_conditioned": orbit_f1_conditioned,
                    "instances": int(len(keep)),
                    "majority_subtype": float(np.bincount(target_sub[keep]).max() / len(keep)),
                    "majority_orbit": float(np.bincount(orbit_target[keep].reshape(-1)).max()
                                            / orbit_target[keep].size),
                }
        row["subtype_macro_f1"] = float(np.mean([v["subtype_macro_f1"] for v in row["per_family"].values()]))
        row["orbit_macro_f1"] = float(np.mean([v["orbit_macro_f1"] for v in row["per_family"].values()]))
        results[mode] = row
        print(json.dumps({"variant": mode, "subtype": row["subtype_macro_f1"],
                          "orbit": row["orbit_macro_f1"]}), flush=True)
    (HERE / "subgraph_predictor.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
