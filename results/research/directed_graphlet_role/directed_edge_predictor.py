"""Asymmetric directed-edge readout: does the representation separate A->B from B->A? (#221)

The kNN recovery used so far is undirected: it says whether two items stay
near, not whether the transition direction survives.  This trains an
asymmetric bilinear scorer

    s(u, v) = h_u^T W h_v + a.h_u + b.h_v + c        (W not symmetric)

on a fit half of the TRAIN behaviour edges and evaluates on the other half:

  auc          held-out positive edges vs random unconnected pairs
  direction    share of held-out edges where s(u,v) > s(v,u); chance 0.5

Representations compared: z (encoder output), q (codebook sum, i.e. the hard
three-token SID reconstructed), and the SID tokens as a 3x256 one-hot.
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
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

from directed_graphlets import load_adjacency  # noqa: E402
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
SEED = 20261120
STEPS = 1500
BATCH = 4096
ARMS = {
    "base": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth",
    "D1": REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "rqvae_best.pth",
}


class Bilinear(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.src = nn.Parameter(torch.zeros(hidden))
        self.dst = nn.Parameter(torch.zeros(hidden))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, h, u, v):
        hu = self.proj(h[u])
        hv = self.proj(h[v])
        return ((hu @ self.weight) * hv).sum(-1) + (hu * self.src).sum(-1) \
            + (hv * self.dst).sum(-1) + self.bias


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
def item_vectors(model, semantic, chunk=4096):
    z_parts, q_parts, t_parts = [], [], []
    for start in range(0, len(semantic), chunk):
        encoded = model.encoder(semantic[start:start + chunk])
        quantized, _p, _u, tokens = model.rq(encoded)
        z_parts.append(encoded.cpu())
        q_parts.append(quantized.cpu())
        t_parts.append(tokens.cpu())
    return torch.cat(z_parts), torch.cat(q_parts), torch.cat(t_parts)


def sid_onehot(tokens: torch.Tensor, levels: int = 3, width: int = 256) -> torch.Tensor:
    out = torch.zeros(len(tokens), levels * width)
    for level in range(levels):
        out[torch.arange(len(tokens)), level * width + tokens[:, level].long()] = 1.0
    return out


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    pooled = np.concatenate([positive, negative])
    order = np.argsort(pooled)
    ranks = np.empty(len(pooled), dtype=np.float64)
    ranks[order] = np.arange(1, len(pooled) + 1)
    positive_rank = ranks[: len(positive)].sum()
    return float((positive_rank - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    config = load_train_config(TEMPLATE / "config" / "train_config.json")
    semantic = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                        "sentence_t5.npy").astype("float32")).to(device)
    graph = load_adjacency()
    n = graph.shape[0]
    rows, cols = graph.nonzero()
    edges = np.stack([rows, cols], axis=1).astype(np.int64)
    order = rng.permutation(len(edges))
    cut = int(0.8 * len(order))
    fit_edges, test_edges = edges[order[:cut]], edges[order[cut:]]

    edge_set = set(map(tuple, edges.tolist()))
    negatives = []
    while len(negatives) < 200000:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            negatives.append((u, v))
    negatives = np.array(negatives, dtype=np.int64)
    test_negatives = negatives[:100000]

    results = {"edges": int(len(edges)), "fit": int(len(fit_edges)), "test": int(len(test_edges)),
               "arms": {}}
    for arm, checkpoint in ARMS.items():
        model = build_rqvae(config, semantic.shape[1], checkpoint, device)
        z, q, tokens = item_vectors(model, semantic)
        sources = {"z": z, "q": q, "sid_onehot": sid_onehot(tokens)}
        results["arms"][arm] = {}
        for name, table in sources.items():
            h = table.to(device)
            scorer = Bilinear(h.shape[1]).to(device)
            optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-3)
            generator = torch.Generator().manual_seed(SEED)
            for step in range(1, STEPS + 1):
                pick = torch.randint(0, len(fit_edges), (BATCH,), generator=generator)
                positive = torch.from_numpy(fit_edges[pick.numpy()]).to(device)
                neg_pick = torch.randint(0, len(negatives), (BATCH,), generator=generator)
                negative = torch.from_numpy(negatives[neg_pick.numpy()]).to(device)
                score = torch.cat([scorer(h, positive[:, 0], positive[:, 1]),
                                   scorer(h, negative[:, 0], negative[:, 1])])
                label = torch.cat([torch.ones(BATCH, device=device),
                                   torch.zeros(BATCH, device=device)])
                loss = F.binary_cross_entropy_with_logits(score, label)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                test = torch.from_numpy(test_edges).to(device)
                neg = torch.from_numpy(test_negatives).to(device)
                positive_score = scorer(h, test[:, 0], test[:, 1]).cpu().numpy()
                negative_score = scorer(h, neg[:, 0], neg[:, 1]).cpu().numpy()
                # direction: the observed direction must outscore the reverse
                reverse_score = scorer(h, test[:, 1], test[:, 0]).cpu().numpy()
            results["arms"][arm][name] = {
                "auc": auc(positive_score, negative_score),
                "direction_accuracy": float((positive_score > reverse_score).mean()),
                "direction_chance": 0.5,
            }
            row = results["arms"][arm][name]
            print(f"{arm} {name}: auc {row['auc']:.4f} direction {row['direction_accuracy']:.4f}",
                  flush=True)
    (HERE / "directed_edge_predictor.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
