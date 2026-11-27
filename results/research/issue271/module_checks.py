"""Issue #271 real-batch module check: local and global gradients on the actual loss."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "stage2_RQ-VAE/TIGER_global_j_rq"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TREE))
from model.mechanism import RQVAE
from model import global_constraint as gc
from train_config import load_train_config

K269 = REPO / "results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result"


def group_norm(grads, params):
    return float(torch.linalg.vector_norm(torch.stack([
        torch.zeros((), device=p.device) if g is None else torch.linalg.vector_norm(g)
        for g, p in zip(grads, params)])))


def term_grads(term, params):
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def main() -> None:
    cfg = load_train_config(TREE / "config/train_config.json")
    sizes = list(cfg.CODEBOOK_SIZE)
    if len(sizes) == 1:
        sizes *= cfg.CODEBOOK_NUM
    from types import SimpleNamespace
    config = SimpleNamespace(
        hidden_sizes=tuple(cfg.HIDDEN_SIZES), codebook_num=cfg.CODEBOOK_NUM,
        codebook_size=tuple(sizes), codebook_dim=cfg.CODEBOOK_DIM, dropout=0.0,
        beta=cfg.BETA, vq_type=cfg.VQ_TYPE, ema_decay=cfg.EMA_DECAY,
        fix_code_embs=False, sk_epsilon=cfg.SK_EPSILON, sk_iters=cfg.SK_ITERS)
    emb = np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy").astype(np.float32)
    frame = pd.read_parquet(REPO / "results/stage0_build_parquet/train.parquet")
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:4096]
    catalogue = torch.from_numpy(emb).cuda()
    batch = catalogue[torch.from_numpy(train_ids).cuda()]
    model = RQVAE(config, in_dim=emb.shape[1]).cuda()
    payload = torch.load(K269 / "rqvae_best.pth", map_location="cpu", weights_only=False)
    missing = model.load_state_dict(payload["state_dict"], strict=False)
    model.global_j_max.fill_(float(gc.J_MAX))
    model.global_mu.fill_(0.1)
    model.train()
    table, receipt = gc.load_pair_table(REPO / "results/research/issue271/geometry")
    global_sample = gc.sample_pairs(table, count=gc.TRAIN_PAIRS_PER_STEP,
                                    rng=np.random.default_rng(271))
    structural = model.structural_sampler().draw(0, 0, 1, track=False)
    out = model(batch, torch.from_numpy(train_ids).cuda(), catalogue_embeddings=catalogue,
                structural_sample=structural, global_sample=global_sample)
    curvature = out[4]
    original, recon = model.compute_loss(batch, out[0], out[1])
    original_term, local_term, constraint_term = model.loss_terms(original, curvature)
    encoder = list(model.encoder.parameters())
    codebooks = [layer.embed.weight for layer in model.rq.vq_layers]
    decoder = list(model.decoder.parameters())
    pg = term_grads(local_term, encoder + codebooks)
    gg = term_grads(constraint_term, encoder + codebooks)
    dg = term_grads(constraint_term, decoder)
    dot = sum((a * b).sum() for a, b in zip(pg, gg))
    local_norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in pg])))
    global_norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gg])))
    cosine = float(dot / (local_norm * global_norm)) if local_norm > 0 and global_norm > 0 else None
    geometry = float(model.last_global_j.detach())
    payload = {
        "schema": "genrec.issue271.module_checks/1",
        "checkpoint": str(K269 / "rqvae_best.pth"),
        "batch_items": int(len(train_ids)),
        "missing_keys": list(missing.missing_keys),
        "lambda": float(model.curvature_weight),
        "mu": float(model.global_mu),
        "j_max": float(model.global_j_max),
        "global_j": geometry,
        "pair_table": receipt,
        "local_gradient_norm": local_norm,
        "global_gradient_norm": global_norm,
        "local_global_cosine": cosine,
        "encoder_local": group_norm(pg[:len(encoder)], encoder),
        "encoder_global": group_norm(gg[:len(encoder)], encoder),
        "codebooks_local": group_norm(pg[len(encoder):], codebooks),
        "codebooks_global": group_norm(gg[len(encoder):], codebooks),
        "decoder_global": group_norm(dg, decoder),
        "hard_q_gradient": bool(global_norm > 0),
        "assets_require_grad": [name for name, _ in model.named_parameters()
                                 if "kappa" in name or "scale" in name],
    }
    # μ=0 removes the global term from the total loss but not from this probe.
    model.global_mu.fill_(0.0)
    zero = model.loss_terms(original, curvature)
    payload["mu_zero_constraint_term"] = float(zero[2])
    (REPO / "results/research/issue271/MODULE_CHECKS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
