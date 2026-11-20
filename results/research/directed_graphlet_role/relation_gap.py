"""How much relation survives quantisation: z vs q on behaviour edges (#221).

Passing the instance adjacency to the subgraph encoder measures the adjacency,
not the item vector -- the adjacency already contains the answer.  This reads
the item vectors alone: build a k=10 nearest-neighbour graph from z and from q
and ask how many TRAIN behaviour edges land inside it.  The gap between the two
recovery rates is the quantisation cost on relational structure.

z = encoder output (32-d), q = codebook sum of the hard three-token code (32-d).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

from directed_graphlets import load_adjacency  # noqa: E402
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
K = 10
CHUNK = 4096
ARMS = {
    "base": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth",
    "D1": REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "rqvae_best.pth",
    "U": REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_U" / "rqvae_best.pth",
    "S": REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_S" / "rqvae_best.pth",
}


def build_rqvae(config, in_dim, checkpoint, device):
    model_config = type("C", (), dict(hidden_sizes=tuple(config.HIDDEN_SIZES),
                                      codebook_num=config.CODEBOOK_NUM,
                                      codebook_size=tuple(config.CODEBOOK_SIZE),
                                      codebook_dim=config.CODEBOOK_DIM, dropout=0.0,
                                      beta=config.BETA, vq_type=config.VQ_TYPE,
                                      ema_decay=config.EMA_DECAY, fix_code_embs=False,
                                      sk_epsilon=config.SK_EPSILON, sk_iters=config.SK_ITERS))()
    torch.manual_seed(42)
    model = RQVAE(model_config, in_dim=in_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def item_vectors(model, semantic, chunk=CHUNK):
    z_parts, q_parts = [], []
    for start in range(0, len(semantic), chunk):
        encoded = model.encoder(semantic[start:start + chunk])
        quantized, _p, _u, _t = model.rq(encoded)
        z_parts.append(encoded.cpu())
        q_parts.append(quantized.cpu())
    return torch.cat(z_parts), torch.cat(q_parts)


@torch.no_grad()
def knn_recovery(vectors: torch.Tensor, edges: np.ndarray, k: int = K,
                 chunk: int = CHUNK) -> dict:
    """Share of directed behaviour edges that land in the symmetric kNN graph."""
    table = torch.nn.functional.normalize(vectors.float(), dim=-1)
    n = len(table)
    sources = torch.from_numpy(edges[:, 0].astype(np.int64))
    targets = torch.from_numpy(edges[:, 1].astype(np.int64))
    neighbour_sets = [None] * n
    for start in range(0, n, chunk):
        block = table[start:start + chunk] @ table.T
        block[torch.arange(len(block)), torch.arange(start, start + len(block))] = -2.0
        values, indices = block.topk(k, dim=-1, largest=True)
        for row in range(len(block)):
            neighbour_sets[start + row] = set(indices[row].tolist())
    hits = 0
    for u, v in zip(sources.tolist(), targets.tolist()):
        if v in neighbour_sets[u] or u in neighbour_sets[v]:
            hits += 1
    return {"recovery": hits / len(edges), "k": k, "edges": int(len(edges))}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_train_config(TEMPLATE / "config" / "train_config.json")
    semantic = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                        "sentence_t5.npy").astype("float32")).to(device)
    graph = load_adjacency()
    rows, cols = graph.nonzero()
    edges = np.stack([rows, cols], axis=1).astype(np.int64)
    results = {"edges": int(len(edges)), "k": K, "arms": {}}
    for arm, checkpoint in ARMS.items():
        model = build_rqvae(config, semantic.shape[1], checkpoint, device)
        z, q = item_vectors(model, semantic)
        row = {"z": knn_recovery(z, edges), "q": knn_recovery(q, edges)}
        row["q_over_z"] = row["q"]["recovery"] / max(row["z"]["recovery"], 1e-12)
        results["arms"][arm] = row
        print(f"{arm}: z {row['z']['recovery']:.5f} q {row['q']['recovery']:.5f} "
              f"ratio {row['q_over_z']:.3f}", flush=True)
    (HERE / "relation_gap.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
