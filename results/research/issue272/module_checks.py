"""Issue #272 real-batch probe: local/global gradients and dual direction at the tight bound."""
from __future__ import annotations
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / 'stage2_RQ-VAE/TIGER_active_j_rq'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(TREE))
from model.mechanism import RQVAE
from model import global_constraint as gc
from train_config import load_train_config

K271 = REPO / 'results/stage2_RQ-VAE/_runs/stage2_cc9a31a9325b439f98c8b1779aaa32aa/TIGER_global_j_rq/result'
PAIRS = K271 / 'global_pairs'
K269 = REPO / 'results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result'


def group_norm(grads, params):
    return float(torch.linalg.vector_norm(torch.stack([
        torch.zeros((), device=p.device) if g is None else torch.linalg.vector_norm(g)
        for g, p in zip(grads, params)])))


def term_grads(term, params):
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def main():
    cfg = load_train_config(TREE / 'config/train_config.json')
    sizes = list(cfg.CODEBOOK_SIZE)
    if len(sizes) == 1:
        sizes *= cfg.CODEBOOK_NUM
    config = SimpleNamespace(hidden_sizes=tuple(cfg.HIDDEN_SIZES), codebook_num=cfg.CODEBOOK_NUM,
        codebook_size=tuple(sizes), codebook_dim=cfg.CODEBOOK_DIM, dropout=0.0, beta=cfg.BETA,
        vq_type=cfg.VQ_TYPE, ema_decay=cfg.EMA_DECAY, fix_code_embs=False,
        sk_epsilon=cfg.SK_EPSILON, sk_iters=cfg.SK_ITERS)
    emb = np.load(REPO / 'stage1_GeneEmbedding/output/sentence_t5.npy').astype(np.float32)
    frame = pd.read_parquet(REPO / 'results/stage0_build_parquet/train.parquet')
    train_ids = np.unique(frame['target'].to_numpy(dtype=np.int64))[:4096]
    catalogue = torch.from_numpy(emb).cuda()
    batch = catalogue[torch.from_numpy(train_ids).cuda()]
    model = RQVAE(config, in_dim=emb.shape[1]).cuda()
    payload = torch.load(K271 / 'rqvae_best.pth', map_location='cpu', weights_only=False)
    missing = model.load_state_dict(payload['state_dict'], strict=False)
    model.global_j_max.fill_(gc.J_MAX)
    model.global_mu.fill_(0.1)
    model.train()
    table, receipt = gc.load_pair_table(PAIRS)
    sample = gc.sample_pairs(table, count=gc.TRAIN_PAIRS_PER_STEP, rng=np.random.default_rng(272))
    structural = model.structural_sampler().draw(0, 0, 1, track=False)
    out = model(batch, torch.from_numpy(train_ids).cuda(), catalogue_embeddings=catalogue,
                structural_sample=structural, global_sample=sample)
    curvature = out[4]
    original, recon = model.compute_loss(batch, out[0], out[1])
    original_term, local_term, constraint_term = model.loss_terms(original, curvature)
    encoder = list(model.encoder.parameters())
    codebooks = [layer.embed.weight for layer in model.rq.vq_layers]
    decoder = list(model.decoder.parameters())
    lg = term_grads(local_term, encoder + codebooks)
    cg = term_grads(constraint_term, encoder + codebooks)
    dg = term_grads(constraint_term, decoder)
    ln = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in lg])))
    gn = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in cg])))
    dot = sum((a * b).sum() for a, b in zip(lg, cg))
    cosine = float(dot / (ln * gn)) if ln > 0 and gn > 0 else None
    j_real = float(model.last_global_j.detach())
    mu_after_violation = model.update_global_multiplier(j_real)
    # satisfied case: a scaled #269 q on the same real pair table
    q269 = gc.load_hard_q(K269 / 'item_sids.json', K269 / 'rqvae_best.pth')
    j_scaled = {s: float(gc.weighted_j(torch.as_tensor(q269 * s, dtype=torch.float64), table).item())
                for s in (0.9, 0.95, 1.0, 1.05)}
    satisfied = next(((s, j) for s, j in sorted(j_scaled.items(), key=lambda kv: kv[1]) if j <= gc.J_MAX), None)
    mu_satisfied = gc.dual_step(0.1, satisfied[1] if satisfied else 0.30, j_max=gc.J_MAX)
    payload = {
        'schema': 'genrec.issue272.module_checks/1',
        'checkpoint': str(K271 / 'rqvae_best.pth'),
        'batch_items': int(len(train_ids)), 'missing_keys': list(missing.missing_keys),
        'lambda': float(model.curvature_weight), 'mu_probe': 0.1,
        'j_max_train': float(gc.J_MAX), 'j_real_estimate': j_real,
        'constraint_violated': bool(j_real > gc.J_MAX),
        'mu_after_violation': mu_after_violation,
        'satisfied_candidate_scale_j': j_scaled,
        'satisfied_case': satisfied,
        'mu_after_satisfied': mu_satisfied,
        'local_gradient_norm': ln, 'global_gradient_norm': gn, 'local_global_cosine': cosine,
        'encoder_local': group_norm(lg[:len(encoder)], encoder),
        'encoder_global': group_norm(cg[:len(encoder)], encoder),
        'codebooks_local': group_norm(lg[len(encoder):], codebooks),
        'codebooks_global': group_norm(cg[len(encoder):], codebooks),
        'decoder_global': group_norm(dg, decoder),
        'hard_q_gradient': bool(gn > 0),
        'pair_table': receipt,
    }
    (REPO / 'results/research/issue272/MODULE_CHECKS.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: payload[k] for k in ('j_real_estimate', 'constraint_violated',
        'mu_after_violation', 'satisfied_case', 'mu_after_satisfied', 'local_gradient_norm',
        'global_gradient_norm', 'local_global_cosine', 'encoder_global', 'codebooks_global')}, indent=1))


if __name__ == '__main__':
    main()
