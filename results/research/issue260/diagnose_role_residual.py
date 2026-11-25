#!/usr/bin/env python3.9
"""Local check: add a fixed role prototype to the level-2 residual target (no training)."""
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


def quantize(model, encoded, weights, role_rows, residual_shift=None):
    residual = encoded; out = []
    for level, layer in enumerate(model.rq.vq_layers):
        target = residual
        if level == 1 and residual_shift is not None:
            target = target + residual_shift
        lw = model.level_weights(weights, level, role_rows)
        cost = model.assignment_cost(target, layer.embed.weight, lw)
        hard_ids = layer._indices(cost, False)
        selected = select(cost, layer.embed.weight, balanced=False,
                          epsilon=layer.sk_epsilon, iterations=layer.sk_iters,
                          training=False, hard_ids=hard_ids)
        out.append(selected["ids"])
        residual = residual - selected["q"]
    return torch.stack(out, dim=1).detach().cpu().numpy()


def rates(c):
    p = np.load(REPO / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz")
    a, pos, neg = p["anchors"], p["positives"], p["negatives"]
    return {l: {"positive": float((c[a, :l] == c[pos, :l]).all(1).mean()),
                "negative": float((c[a, None, :l] == c[neg, :l]).all(2).mean())} for l in (1, 2, 3)}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_train_config(ARM / "config/train_config.json")
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
    roles = np.asarray(json.loads((ARM / "role_table.json").read_text())["table"], dtype=np.int64)
    role_rows = model.role_mix.rows_for(ids, device)
    with torch.no_grad():
        encoded = model.encoder(emb)
        _, weights = model.structure.gates_for_items(model.store, ids, device, chunk=1024, behavior_on=True)
        base = quantize(model, encoded, weights, role_rows)
        # The level-2 residual entering that level, then per-role mean shift.
        residual = encoded
        for level, layer in enumerate(model.rq.vq_layers):
            if level == 1:
                break
            lw = model.level_weights(weights, level, role_rows)
            cost = model.assignment_cost(residual, layer.embed.weight, lw)
            selected = select(cost, layer.embed.weight, balanced=False,
                              epsilon=layer.sk_epsilon, iterations=layer.sk_iters,
                              training=False, hard_ids=layer._indices(cost, False))
            residual = residual - selected["q"]
        means = torch.zeros(9, residual.shape[1], device=device)
        for role in range(8):
            mask = torch.from_numpy(roles == role).to(device)
            if bool(mask.any()):
                means[role] = residual[mask].mean(dim=0)
        means[8] = residual.mean(dim=0)
        shift = means[role_rows] - means[8]
        shifted = quantize(model, encoded, weights, role_rows, residual_shift=shift)
    payload = {"schema": "genrec.issue260.role_residual_probe/1",
               "base": {"rates": rates(base), "prefix2": len({tuple(v[:2]) for v in base}),
                        "triples": len({tuple(v) for v in base})},
               "role_residual": {"rates": rates(shifted), "prefix2": len({tuple(v[:2]) for v in shifted}),
                                 "triples": len({tuple(v) for v in shifted}),
                                 "changed": {l: float((base[:, i] != shifted[:, i]).mean())
                                             for i, l in enumerate((1, 2, 3))},
                                 "shift_norm_mean": float(shift.norm(dim=1).mean())}}
    np.savez_compressed(OUT / "role_residual_codes.npz", base=base, role_residual=shifted)
    (OUT / "role_residual_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
