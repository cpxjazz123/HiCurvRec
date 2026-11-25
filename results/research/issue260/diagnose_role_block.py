#!/usr/bin/env python3.9
"""Local check: role-partitioned L2 cost on the frozen R05 checkpoint (no training)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
ARM = REPO / "stage2_RQ-VAE" / "TIGER_role_conditioned_l2_rq"
RUN = REPO / "results/stage2_RQ-VAE/_runs/issue256_r05_formal/TIGER_role_conditioned_l2_rq/result"
OUT = REPO / "results/research/issue260/diagnosis"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ARM))
from train_config import load_train_config
from distances import build_distance
from model import RQVAE
from model.selection import select


def codes_for(model, encoded, weights, role_rows, block=None):
    residual = encoded; quantized = torch.zeros_like(encoded); out = []
    for level, layer in enumerate(model.rq.vq_layers):
        lw = model.level_weights(weights, level, role_rows)
        cost = model.assignment_cost(residual, layer.embed.weight, lw)
        if block is not None and level == 1:
            n = layer.embed.weight.shape[0]
            roles = role_rows.detach().cpu().numpy()
            allowed = np.zeros((len(roles), n), dtype=bool)
            for i, role in enumerate(roles):
                if 0 <= int(role) < 8:
                    allowed[i, int(role) * block:(int(role) + 1) * block] = True
                else:
                    allowed[i, :] = True
            allowed_t = torch.from_numpy(allowed).to(cost.device)
            penalty = cost.detach().abs().amax(dim=1, keepdim=True) + cost.detach().amax(dim=1, keepdim=True) + 1e-6
            cost = torch.where(allowed_t, cost, penalty.expand_as(cost))
        hard_ids = layer._indices(cost, False)
        selected = select(cost, layer.embed.weight, balanced=False,
                          epsilon=layer.sk_epsilon, iterations=layer.sk_iters,
                          training=False, hard_ids=hard_ids)
        out.append(selected["ids"])
        residual = residual - selected["q"]
    return torch.stack(out, dim=1).detach().cpu().numpy()


def rates(codes):
    p = np.load(REPO / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz")
    a, pos, neg = p["anchors"], p["positives"], p["negatives"]
    return {l: {"positive": float((codes[a, :l] == codes[pos, :l]).all(1).mean()),
                "negative": float((codes[a, None, :l] == codes[neg, :l]).all(2).mean())}
            for l in (1, 2, 3)}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_train_config(ARM / "config" / "train_config.json")
    sizes = list(cfg.CODEBOOK_SIZE) * (cfg.CODEBOOK_NUM if len(cfg.CODEBOOK_SIZE) == 1 else 1)
    tokenizer = SimpleNamespace(hidden_sizes=cfg.HIDDEN_SIZES, codebook_num=cfg.CODEBOOK_NUM,
                                codebook_size=tuple(sizes), codebook_dim=cfg.CODEBOOK_DIM,
                                dropout=0.0, beta=cfg.BETA, vq_type=cfg.VQ_TYPE,
                                ema_decay=cfg.EMA_DECAY, fix_code_embs=False,
                                sk_epsilon=cfg.SK_EPSILON, sk_iters=cfg.SK_ITERS)
    model = RQVAE(tokenizer, in_dim=768, distance=build_distance(cfg)).to(device)
    model.load_state_dict(torch.load(RUN / "rqvae_best.pth", map_location="cpu")["state_dict"], strict=True)
    model.eval()
    emb = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")).float().to(device)
    ids = np.arange(len(emb), dtype=np.int64)
    role_table = np.asarray(json.loads((ARM / "role_table.json").read_text())["table"], dtype=np.int64)
    role_rows = model.role_mix.rows_for(ids, device)
    with torch.no_grad():
        encoded = model.encoder(emb)
        _, weights = model.structure.gates_for_items(model.store, ids, device, chunk=1024, behavior_on=True)
        base = codes_for(model, encoded, weights, role_rows)
        block = codes_for(model, encoded, weights, role_rows, block=32)
    payload = {"schema": "genrec.issue260.role_block_probe/1",
               "block": 32, "base": {"rates": rates(base),
                                     "prefix2": len({tuple(v[:2]) for v in base}),
                                     "triples": len({tuple(v) for v in base}),
                                     "changed": None},
               "role_block": {"rates": rates(block),
                              "prefix2": len({tuple(v[:2]) for v in block}),
                              "triples": len({tuple(v) for v in block}),
                              "changed": {l: float((base[:, i] != block[:, i]).mean()) for i, l in enumerate((1, 2, 3))}}}
    np.savez_compressed(OUT / "role_block_codes.npz", base=base, role_block=block)
    (OUT / "role_block_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
