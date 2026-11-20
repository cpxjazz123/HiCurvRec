"""Can the representation tell A->B from B->A when it is actually trained to?

The first version trained only "is there an edge" (BCE on observed versus
unconnected pairs), then read the direction off post hoc -- the antisymmetric
part of the scorer was never under pressure, so ~50 % said nothing about the
representation.  This version adds the missing objective

    L = BCE(edge) + lambda * ( -log sigmoid( s(u,v) - s(v,u) ) )

and restricts both training and evaluation to ONE-WAY edges (a pair whose
reverse also exists carries no direction to learn), with the degree and
popularity baselines the comparison needs.

Representations: z (encoder output), q (codebook sum = hard SID reconstructed),
sid_onehot (3 x 256 tokens).  Baselines: the same scorer on 8 degree features,
and the rule "the higher in-degree endpoint is the destination".
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
STEPS = 2000
BATCH = 4096
LAMBDAS = (1.0, 10.0)
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
    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def train_scorer(h, fit_edges, negatives, device, lambda_dir: float = 1.0):
    scorer = Bilinear(h.shape[1]).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(SEED)
    final_direction = None
    for step in range(1, STEPS + 1):
        pick = torch.randint(0, len(fit_edges), (BATCH,), generator=generator).numpy()
        positive = torch.from_numpy(fit_edges[pick]).to(device)
        neg_pick = torch.randint(0, len(negatives), (BATCH,), generator=generator).numpy()
        negative = torch.from_numpy(negatives[neg_pick]).to(device)
        forward_score = scorer(h, positive[:, 0], positive[:, 1])
        backward_score = scorer(h, positive[:, 1], positive[:, 0])
        negative_score = scorer(h, negative[:, 0], negative[:, 1])
        edge_loss = F.binary_cross_entropy_with_logits(
            torch.cat([forward_score, negative_score]),
            torch.cat([torch.ones(len(forward_score), device=device),
                       torch.zeros(len(negative_score), device=device)]))
        direction_loss = -F.logsigmoid(forward_score - backward_score).mean()
        loss = edge_loss + lambda_dir * direction_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_direction = float(direction_loss.detach())
    scorer.final_direction_loss = final_direction
    scorer.final_edge_loss = float(edge_loss.detach())
    return scorer


def evaluate(scorer, h, test_edges, test_negatives, device) -> dict:
    with torch.no_grad():
        test = torch.from_numpy(test_edges).to(device)
        neg = torch.from_numpy(test_negatives).to(device)
        forward_score = scorer(h, test[:, 0], test[:, 1]).cpu().numpy()
        backward_score = scorer(h, test[:, 1], test[:, 0]).cpu().numpy()
        negative_score = scorer(h, neg[:, 0], neg[:, 1]).cpu().numpy()
    return {"auc": auc(forward_score, negative_score),
            "direction_accuracy": float((forward_score > backward_score).mean()),
            "final_direction_loss": getattr(scorer, "final_direction_loss", None),
            "final_edge_loss": getattr(scorer, "final_edge_loss", None),
            "direction_loss_floor": float(np.log(2.0))}


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
    edge_set = set(map(tuple, edges.tolist()))
    one_way = np.array([(u, v) for u, v in edges.tolist() if (v, u) not in edge_set],
                       dtype=np.int64)
    order = rng.permutation(len(one_way))
    cut = int(0.8 * len(order))
    fit_edges, test_edges = one_way[order[:cut]], one_way[order[cut:]]

    # Degree-stratified negatives: a random unconnected pair is almost always
    # low-degree on both ends, so plain AUC mostly measures "is this item popular".
    # Each negative keeps the positive's source out-degree decile and destination
    # in-degree decile, which removes that shortcut.
    def deciles(values: np.ndarray) -> np.ndarray:
        bins = np.quantile(values, np.linspace(0, 1, 11)[1:-1])
        return np.digitize(values, bins)

    degree_in = np.asarray(graph.sum(axis=0)).ravel().astype(np.float64)
    degree_out = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
    out_bin, in_bin = deciles(degree_out), deciles(degree_in)
    buckets_out = [np.flatnonzero(out_bin == b) for b in range(10)]
    buckets_in = [np.flatnonzero(in_bin == b) for b in range(10)]
    negatives = []
    wanted = 200000
    while len(negatives) < wanted:
        source = int(rng.integers(0, len(one_way)))
        u0, v0 = int(one_way[source, 0]), int(one_way[source, 1])
        u = int(rng.choice(buckets_out[out_bin[u0]]))
        v = int(rng.choice(buckets_in[in_bin[v0]]))
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            negatives.append((u, v))
    negatives = np.array(negatives, dtype=np.int64)

    mutual = np.asarray(graph.multiply(graph.T).sum(axis=1)).ravel()
    degrees = torch.from_numpy(np.stack([degree_in, degree_out, mutual,
                                         degree_in + degree_out], axis=1).astype("float32")).to(device)

    results = {"edges": int(len(edges)), "one_way_edges": int(len(one_way)),
               "fit": int(len(fit_edges)), "test": int(len(test_edges)),
               "loss": "BCE(edge) + 1.0 * -log sigmoid(s(u,v) - s(v,u))",
               "arms": {}, "baselines": {}}

    # rule baseline: the higher in-degree endpoint is the destination
    rule = ((degree_in[test_edges[:, 1]] > degree_in[test_edges[:, 0]]).mean()
            + 0.5 * (degree_in[test_edges[:, 1]] == degree_in[test_edges[:, 0]]).mean())
    results["baselines"]["in_degree_rule"] = {"direction_accuracy": float(rule),
                                              "direction_chance": 0.5}

    # learned degree baseline: the same scorer on 8 degree features
    torch.manual_seed(SEED)
    degree_scorer = train_scorer(degrees, fit_edges, negatives, device, lambda_dir=10.0)
    results["baselines"]["degree_scorer"] = evaluate(degree_scorer, degrees, test_edges,
                                                     negatives[:100000], device)
    print("degree:", results["baselines"]["degree_scorer"], flush=True)

    for arm, checkpoint in ARMS.items():
        model = build_rqvae(config, semantic.shape[1], checkpoint, device)
        z, q, tokens = item_vectors(model, semantic)
        sources = {"z": z, "q": q, "sid_onehot": sid_onehot(tokens)}
        results["arms"][arm] = {}
        for name, table in sources.items():
            h = table.to(device)
            results["arms"][arm][name] = {}
            for lam in LAMBDAS:
                torch.manual_seed(SEED)
                scorer = train_scorer(h, fit_edges, negatives, device, lambda_dir=lam)
                row = evaluate(scorer, h, test_edges, negatives[:100000], device)
                row["lambda"] = lam
                results["arms"][arm][name][f"lambda_{lam}"] = row
                print(f"{arm} {name} lambda={lam}: auc {row['auc']:.4f} "
                      f"direction {row['direction_accuracy']:.4f} "
                      f"dir_loss {row['final_direction_loss']:.3f}", flush=True)
    (HERE / "directed_edge_predictor.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
