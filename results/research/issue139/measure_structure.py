"""Measure structural fidelity on one footing for every arm of issue #139.

The core hypothesis is not "the auxiliary numbers improved": it is that the arm
whose geometry weight AF3 chooses keeps more of the training graph's local
structure than the arm that fixes the weight. That is a comparison of the
structural distortion itself,

    D_AF3(G, final)  <  D_AF3(F0, final),

on the same graph, the same support, the same shared readout and the same
normalisation. This script produces exactly that number, and it also splits the
distortion into the part the encoder already carries and the part quantization
adds:

* ``D_continuous`` reads the encoder's own tangent output as the readout -- the
  representation the quantizer receives;
* ``D_discrete`` reads the reconstruction of the complete hard SID -- what Stage 3
  actually consumes;
* the difference is what quantization costs on top of the encoder's own error.

Cross-checks that do not replace the distortion: the adjacency Brier on the
support, the soft-degree error, and how much of each item's graph neighbourhood
survives as its nearest neighbours in the SID readout.

Usage:
    python3 results/research/issue139/measure_structure.py F0 G [--k 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
#: The arm whose curvature the measurement builds its model with. The E arm
#: runs at curvature zero and its checkpoint carries that in its extra state, so
#: a model built from another arm's tree refuses to load it -- correctly.
ARM_CURVATURE = {"F0": 1.0, "G": 1.0, "L": 1.0, "E": 0.0, "A": 1.0}
TREE = REPO / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0"
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(REPO))

from model import af3_structure as structure  # noqa: E402
from model.mixed_rq import quantized_readout, split  # noqa: E402

AF3_CACHE = (REPO / "results" / "research" / "issue121" / "community_curvature"
             / "train_edge_af3.npz")
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
READOUT = REPO / "results" / "research" / "issue139" / "readout.json"
SUPPORT_CACHE = REPO / "results" / "research" / "issue139" / "support.npz"
OUT = REPO / "results" / "research" / "issue139" / "structural_fidelity.json"


def support_for(graph: dict, kappa: dict, embeddings: np.ndarray) -> dict:
    """The frozen support, cached so every arm is measured on the same pairs."""
    if SUPPORT_CACHE.is_file():
        payload = np.load(SUPPORT_CACHE, allow_pickle=True)
        return {"pairs": payload["pairs"], "is_edge": payload["is_edge"],
                "edges": int(payload["edges"]), "unobserved": int(payload["unobserved"])}
    support = structure.build_support(graph, kappa, embeddings, k=16, seed=42)
    np.savez_compressed(SUPPORT_CACHE, pairs=support["pairs"], is_edge=support["is_edge"],
                        edges=support["edges"], unobserved=support["unobserved"])
    return support


def per_pair_targets(kappa: dict, pairs: np.ndarray) -> np.ndarray:
    lookup = {(int(i), int(j)): float(value)
              for i, j, value in zip(kappa["rows"], kappa["cols"], kappa["kappa"])}
    return np.array([lookup.get((int(i), int(j)), 0.0) for i, j in pairs])


def distortion(phi: torch.Tensor, support: dict, kappa: dict, targets: np.ndarray,
               tau: float, intercept: float, n_nodes: int) -> dict:
    """``D_AF3`` with the frozen readout, on the whole support."""
    pairs = support["pairs"]
    index = torch.as_tensor(pairs.T, dtype=torch.long)
    left = phi[index[0]]
    right = phi[index[1]]
    denominator = (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
    delta = 1.0 - (left * right).sum(-1) / denominator
    probability = torch.sigmoid(intercept - delta.double() / max(tau, 1e-6))
    p = probability.detach().cpu().numpy()
    term = structure.uncertainty_term(pairs, p, n_nodes)
    predicted = p * term
    is_edge = support["is_edge"]
    scale = float(max(1.0, float(np.subtract(*np.quantile(
        targets[is_edge], [0.75, 0.25])))))
    edge_half = float(structure.huber((predicted[is_edge] - targets[is_edge]) / scale).mean())
    other_half = float(structure.huber(predicted[~is_edge] / scale).mean())
    return {"distortion": 0.5 * edge_half + 0.5 * other_half,
            "edge_half": edge_half, "unobserved_half": other_half, "scale": scale,
            # The adjacency Brier is over the readout probability, which is a
            # probability; mu_hat = p * term carries the degree scale and a
            # Brier over it is not a Brier at all.
            "brier": float(np.mean((p - is_edge.astype(np.float64)) ** 2)),
            "probability": p, "predicted": predicted, "delta": delta}


def neighbour_retention(phi: torch.Tensor, graph: dict, k: int) -> dict:
    """How much of an item's graph neighbourhood survives as its nearest SID neighbours."""
    adjacency = structure._neighbours(graph["rows"], graph["cols"], graph["n_nodes"])
    unit = phi.detach().cpu().float()
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # The readout covers every item; the graph covers the training items, which
    # are its first rows.
    items = min(unit.shape[0], len(adjacency))
    hits = 0
    total = 0
    chunk = 2048
    for start in range(0, items, chunk):
        block = unit[start:start + chunk] @ unit.T
        block[torch.arange(block.shape[0]), torch.arange(start, start + block.shape[0])] = -2.0
        nearest = torch.topk(block, k, dim=1).indices.numpy()
        for offset in range(nearest.shape[0]):
            item = start + offset
            # All of the item's graph neighbours, not the first k by index: the
            # adjacency is sorted by node id, so a prefix of it is an arbitrary
            # subset and a retention over it measures nothing.
            truth = set(int(v) for v in adjacency[item])
            if not truth:
                continue
            hits += len(truth & set(int(v) for v in nearest[offset]))
            total += len(truth)
    return {"k": k, "retention": hits / total if total else None, "pairs": total}


def arm_phi(arm: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """The continuous readout and the discrete readout of one arm."""
    result = REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm)
    checkpoint = torch.load(result / "rqvae_best.pth", map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    import train_rqvae as trainer
    config = trainer._tokenizer_config()
    from model.model import MixedRQVAE
    model = MixedRQVAE(config, in_dim=768, curvature=ARM_CURVATURE.get(arm, 1.0))
    model.load_state_dict(state, strict=True)
    model.eval()

    sid = json.loads((result / "item_sids.json").read_text())
    items = sid["items"] if isinstance(sid, dict) and "items" in sid else sid
    keys = sorted(items, key=lambda key: int(key))
    tokens = torch.tensor([[(items[key]["sid"] if isinstance(items[key], dict) else items[key])[level]
                            for level in range(3)] for key in keys], dtype=torch.long)

    embeddings = torch.from_numpy(np.load(EMBEDDINGS).astype(np.float32))
    with torch.no_grad():
        continuous = model.encoder(embeddings)
        codebooks = [layer.embed.weight for layer in model.rq.levels]
        euclidean, curved = [], []
        for level, book in enumerate(codebooks):
            picked = book[tokens[:, level]]
            left, right = split(picked)
            euclidean.append(left)
            curved.append(torch.tanh(right) * 0.0 + right if model.rq.curvature == 0 else
                          __import__("model.poincare_contract", fromlist=["expmap0"]).expmap0(
                              right, model.rq.curvature))
        discrete = quantized_readout(euclidean, curved, model.rq.curvature)
    return continuous, discrete, {"tokens": int(tokens.shape[0]),
                                  "codebooks": [int(book.shape[0]) for book in codebooks]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", default=["F0", "G"])
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    graph = structure.load_graph(AF3_CACHE)
    kappa = structure.target_kappa(graph)
    # The support needs the TRAIN rows only, in graph order: the reviewed
    # builder refuses an array whose rows are not exactly the graph's nodes.
    import pandas as pd
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    embeddings = np.load(EMBEDDINGS)[train_ids]
    support = support_for(graph, kappa, embeddings)
    targets = per_pair_targets(kappa, support["pairs"])
    readout = json.loads(READOUT.read_text())
    print("support %d pairs (%d edges) | tau %.6f intercept %.6f"
          % (len(support["pairs"]), support["edges"], readout["tau"], readout["intercept"]), flush=True)

    payload = {"source": "results/research/issue139/readout.json (frozen by F0 at epoch 1500)",
               "support": str(SUPPORT_CACHE), "arms": {}}
    for arm in args.arms:
        continuous, discrete, info = arm_phi(arm)
        # The graph's node ids are training-item indices, while the readout is
        # rowed by item id. Without this gather the distortion and the neighbour
        # check compare unrelated rows -- and they look plausible while doing it.
        index = torch.as_tensor(train_ids, dtype=torch.long)
        continuous = continuous[index]
        discrete = discrete[index]
        cont = distortion(continuous, support, kappa, targets, readout["tau"],
                          readout["intercept"], graph["n_nodes"])
        disc = distortion(discrete, support, kappa, targets, readout["tau"],
                          readout["intercept"], graph["n_nodes"])
        try:
            retention = neighbour_retention(discrete, graph, args.k)
        except Exception as exc:  # noqa: BLE001 - a cross-check must not hide the distortion
            retention = {"retention": None, "pairs": 0, "error": str(exc)[:200]}
        payload["arms"][arm] = {
            "items": info["tokens"],
            "D_continuous": cont["distortion"], "D_discrete": disc["distortion"],
            "quantization_cost": disc["distortion"] - cont["distortion"],
            "edge_half_continuous": cont["edge_half"], "edge_half_discrete": disc["edge_half"],
            "unobserved_half_continuous": cont["unobserved_half"],
            "unobserved_half_discrete": disc["unobserved_half"],
            "brier_continuous": cont["brier"], "brier_discrete": disc["brier"],
            "neighbour_retention": retention["retention"], "neighbour_pairs": retention["pairs"],
            "scale": disc["scale"],
        }
        print("%-3s D_cont %.6f  D_disc %.6f  cost %+.6f  Brier %.4f  retention %.4f"
              % (arm, cont["distortion"], disc["distortion"],
                 disc["distortion"] - cont["distortion"], disc["brier"],
                 retention["retention"] or 0.0), flush=True)

    arms = payload["arms"]
    if "F0" in arms and "G" in arms:
        payload["hypothesis"] = {
            "statement": "D_AF3(G, final) < D_AF3(F0, final)",
            "G_minus_F0_discrete": arms["G"]["D_discrete"] - arms["F0"]["D_discrete"],
            "G_minus_F0_continuous": arms["G"]["D_continuous"] - arms["F0"]["D_continuous"],
            "supported": arms["G"]["D_discrete"] < arms["F0"]["D_discrete"],
        }
        print("HYPOTHESIS: D(G) - D(F0) = %+.6f  -> %s"
              % (payload["hypothesis"]["G_minus_F0_discrete"],
                 "supported" if payload["hypothesis"]["supported"] else "NOT supported"))
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
