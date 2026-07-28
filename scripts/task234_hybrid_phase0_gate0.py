#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #234 — Issue #9 Hybrid Phase 0 Gate 0: per-layer assignment composition.

Issue #9 提议 hybrid 分配规则:
  L0 = Per-Codeword κ c_k ~ Uniform(1, 5)
  L1 = Gromov product argmax weight_l = 0.5
  L2 = Gromov product argmax weight_l = 0.5

Goal: 验证 H1 — hybrid 在串行 3-layer forward-pass 中三层是否仍 OPEN 60-90%.
Phase 0 ckpt = HG-Rec baseline (#84). CPU only.

复用 task231 / task232 helpers (load_baseline_ckpt / encode_all / euc_dist /
per_codeword_kappa_stereographic / gromov_score_matrix).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task234_hybrid_gate0")

sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/HG-Rec')


# ---------- helpers (与 task231/232 复用) ----------

def load_baseline_ckpt(ckpt_path, device='cpu'):
    from model.hrqvae import HRQVAE
    from model.utils import EmbDataset

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    args = ckpt['args']
    log.info(f"  num_emb_list={args.num_emb_list}, e_dim={args.e_dim}, "
             f"curvature={getattr(args, 'curvature', 1.0)}, "
             f"product_manifold={getattr(args, 'product_manifold', False)}")

    data = EmbDataset(args.data_path)
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=getattr(args, 'layers', [512, 256, 128, args.e_dim]),
        dropout_prob=getattr(args, 'dropout_prob', 0.0),
        bn=getattr(args, 'bn', False),
        loss_type=args.loss_type,
        quant_loss_weight=getattr(args, 'quant_loss_weight', 1.0),
        beta=args.beta,
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()
    return model, data, args


@torch.no_grad()
def encode_all(model, data, batch_size=256, device='cpu'):
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    latents = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encoder(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def euc_dist(z, e):
    z_sq = (z ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e ** 2).sum(dim=-1, keepdim=True).t()
    ze = z @ e.t()
    return z_sq + e_sq - 2 * ze


def per_codeword_kappa_stereographic(z, e, c_k):
    """Per-codeword κ-Stereographic distance (N, K)."""
    eps = 1e-5
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    z_sq = (z_in ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e_in ** 2).sum(dim=-1, keepdim=True).t()
    ze = z_in @ e_in.t()
    eucl_d_sq = z_sq + e_sq - 2 * ze

    c_k_z_sq = c_k.view(1, -1) * z_sq
    c_k_e_sq = c_k.view(1, -1) * e_sq
    one_minus_cz = (1 - c_k_z_sq).clamp(min=1e-6)
    one_minus_ce = (1 - c_k_e_sq).clamp(min=1e-6)

    inner = 1 + 2 * c_k.view(1, -1) * eucl_d_sq / (one_minus_cz * one_minus_ce)
    inner = inner.clamp(min=1.0 + 1e-12)
    inv_sqrt_c = 1.0 / c_k.sqrt().clamp(min=1e-6)
    score = inv_sqrt_c.view(1, -1) * torch.acosh(inner)
    return score


def poincare_dist_matrix(z, e, c=1.0):
    """Poincaré 距离矩阵 (N, K)."""
    eps = 1e-5
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    z_sq = (z_in ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e_in ** 2).sum(dim=-1, keepdim=True)
    ze = z_in @ e_in.t()
    eucl_d_sq = z_sq + e_sq.t() - 2 * ze
    denom = (1 - z_sq) * (1 - e_sq.t())
    x = 1 + 2 * eucl_d_sq / denom.clamp(min=1e-12)
    x = x.clamp(min=1.0 + 1e-12)
    return torch.acosh(x)


def gromov_score_weight(z, e, c=1.0, weight_l=0.5):
    """Gromov variant (A): score = ρ_e * weight_l - d(z,e), argmax.

    Issue #9 提议 L1/L2 用 Gromov weight=0.5.
    """
    eps = 1e-5
    z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
    e_in = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12) * e_norm

    rho_e = 2 * torch.atanh(e_in.norm(dim=-1).clamp(max=1 - 1e-6))  # (K,)
    d_ze = poincare_dist_matrix(z_in, e_in, c=c)
    return weight_l * rho_e.view(1, -1) - d_ze


def get_layer_inputs(model, data, z_all, layer_idx):
    """Return z seen by `layer_idx` (after Euclidean residual subtraction)."""
    eps = 1e-5
    if layer_idx == 0:
        z_norm = z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
        z_in = z_all / z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm
        return z_in
    # Otherwise: subtract argmin Euclidean quant at all earlier layers
    cur = z_all.clone()
    for lj in range(layer_idx):
        cb = model.hrq.vq_layers[lj].embeddings.weight.detach().cpu()
        d = euc_dist(cur, cb)
        idx = d.argmin(dim=-1)
        z_q = cb[idx]
        cur = cur - z_q
    cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    z_in = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm
    return z_in


# ---------- core: per-layer assignment, two modes ----------

def assign_per_layer_independent(z_layer, e_layer, c_k_l0, weight_l_gromov):
    """Per-layer-independent assignment (issue #6 + #7 separately).

    L0: argmin PC κ with c_k = c_k_l0 (random per seed)
    L1/L2: argmax Gromov weight with weight_l_gromov
    Return (argmin_pc_k for L0, argmax_gromov for L1/L2) + per-layer agreement vs euc.
    """
    d_euc = euc_dist(z_layer, e_layer)
    argmin_euc = d_euc.argmin(dim=-1)

    score_pc = per_codeword_kappa_stereographic(z_layer, e_layer, c_k_l0)
    argmin_pc = score_pc.argmin(dim=-1)
    agreement_pc = (argmin_pc == argmin_euc).float().mean().item()

    score_gromov = gromov_score_weight(z_layer, e_layer, weight_l=weight_l_gromov)
    argmax_gromov = score_gromov.argmax(dim=-1)
    agreement_gromov = (argmax_gromov == argmin_euc).float().mean().item()

    return {
        'argmin_euc': argmin_euc,
        'argmin_pc': argmin_pc,
        'argmax_gromov': argmax_gromov,
        'agreement_pc': agreement_pc,
        'agreement_gromov': agreement_gromov,
    }


def assign_composed_hybrid(model, data, z_all, layer_idx, e_layer, c_k_l0, weight_l_gromov,
                              seed=42, c_low=1.0, c_high=5.0):
    """Composed 3-layer forward-pass with hybrid rule.

    L0: argmin PC κ (uses original z)
    L1: argmax Gromov w=0.5 (uses residual after L0 PC κ quantization)
    L2: argmax Gromov w=0.5 (uses residual after L0 PC κ + L1 Gromov quantization)

    Note: c_k_l0 only used at L0 (size 64). For composed residual at L1/L2,
    we sample a fresh L0-sized c_k (K=64) internally so PC κ L0 has correct size.
    """
    eps = 1e-5
    K_l0 = model.hrq.vq_layers[0].embeddings.weight.shape[0]

    if layer_idx == 0:
        z_norm = z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
        z_in = z_all / z_all.norm(dim=-1, keepdim=True).clamp(min=1e-12) * z_norm

        d_euc = euc_dist(z_in, e_layer)
        argmin_euc = d_euc.argmin(dim=-1)
        score_pc = per_codeword_kappa_stereographic(z_in, e_layer, c_k_l0)
        argmin_pc = score_pc.argmin(dim=-1)
        return (argmin_pc == argmin_euc).float().mean().item()

    cur = z_all.clone()
    cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    cur_proj = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm

    cb0 = model.hrq.vq_layers[0].embeddings.weight.detach().cpu()
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed + 1000 + layer_idx)
    c_k_composed_l0 = torch.from_numpy(
        np.random.uniform(c_low, c_high, size=K_l0)
    ).float()
    torch.set_rng_state(rng_state)
    score0 = per_codeword_kappa_stereographic(cur_proj, cb0, c_k_composed_l0)
    idx0 = score0.argmin(dim=-1)
    z_q0 = cb0[idx0]
    cur = cur - z_q0
    cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    cur_proj = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm

    if layer_idx == 1:
        d_euc = euc_dist(cur_proj, e_layer)
        argmin_euc = d_euc.argmin(dim=-1)
        score_g = gromov_score_weight(cur_proj, e_layer, weight_l=weight_l_gromov)
        argmax_g = score_g.argmax(dim=-1)
        return (argmax_g == argmin_euc).float().mean().item()

    cb1 = model.hrq.vq_layers[1].embeddings.weight.detach().cpu()
    score1 = gromov_score_weight(cur_proj, cb1, weight_l=weight_l_gromov)
    idx1 = score1.argmax(dim=-1)
    z_q1 = cb1[idx1]
    cur = cur - z_q1
    cur_norm = cur.norm(dim=-1, keepdim=True).clamp(min=1e-12, max=1 - eps)
    cur_proj = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-12) * cur_norm

    d_euc = euc_dist(cur_proj, e_layer)
    argmin_euc = d_euc.argmin(dim=-1)
    score_g = gromov_score_weight(cur_proj, e_layer, weight_l=weight_l_gromov)
    argmax_g = score_g.argmax(dim=-1)
    return (argmax_g == argmin_euc).float().mean().item()




# ---------- main Gate 0 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="随机 c_k 抽样次数")
    parser.add_argument("--c_low", type=float, default=1.0)
    parser.add_argument("--c_high", type=float, default=5.0)
    parser.add_argument("--weight_l_gromov", type=float, default=0.5,
                        help="Issue #9 提议 L1/L2 Gromov weight = 0.5")
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task234_hybrid_gate0_results.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #234 — Issue #9 Hybrid Phase 0 Gate 0: per-layer composition")
    log.info("=" * 70)
    log.info(f"  Hybrid rule: L0 = PC κ c_k ~ U({args.c_low},{args.c_high}), "
             f"L1/L2 = Gromov weight_l={args.weight_l_gromov}")
    log.info(f"  n_seeds = {args.n_seeds}")

    model, data, ckpt_args = load_baseline_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = ckpt_args.num_emb_list
    log.info(f"\n=== Per-layer hybrid composition (num_emb_list={n_e_list}) ===")

    all_results = {
        "task": "task234_hybrid_phase0_gate0",
        "ckpt": args.ckpt_path,
        "n_items": int(z_all.shape[0]),
        "e_dim": int(z_all.shape[1]),
        "n_e_list": list(n_e_list),
        "hybrid_rule": {
            "L0": f"PC_kappa c_k ~ Uniform({args.c_low},{args.c_high})",
            "L1_L2": f"Gromov weight_l = {args.weight_l_gromov}",
        },
        "n_seeds": args.n_seeds,
        "c_k_range": [args.c_low, args.c_high],
        "weight_l_gromov": args.weight_l_gromov,
        "per_layer": {},
    }

    PASS_BAND = (0.60, 0.95)
    CONSISTENCY_TOL_PP = 3.0  # ±3pp composed vs independent

    for li in range(len(n_e_list)):
        K = n_e_list[li]
        e = model.hrq.vq_layers[li].embeddings.weight.detach().cpu()
        z_layer = get_layer_inputs(model, data, z_all, li)

        log.info(f"\n--- Layer {li} (K={K}) ---")
        log.info(f"  z_layer norm range: [{z_layer.norm(dim=-1).min().item():.3f}, "
                 f"{z_layer.norm(dim=-1).max().item():.3f}], mean={z_layer.norm(dim=-1).mean().item():.3f}")

        indep_pc_agreements = []
        indep_gromov_agreements = []
        composed_agreements = []
        per_seed_details = []

        for s in range(args.n_seeds):
            seed = 42 + s
            torch.manual_seed(seed)
            np.random.seed(seed)
            c_k = torch.from_numpy(
                np.random.uniform(args.c_low, args.c_high, size=K)
            ).float()

            # Independent (issue #6 / #7 separately)
            indep = assign_per_layer_independent(
                z_layer, e, c_k, args.weight_l_gromov
            )
            indep_pc_agreements.append(indep['agreement_pc'])
            indep_gromov_agreements.append(indep['agreement_gromov'])

            # Composed (Issue #9 hybrid)
            composed = assign_composed_hybrid(
                model, data, z_all, li, e, c_k, args.weight_l_gromov,
                seed=seed, c_low=args.c_low, c_high=args.c_high,
            )
            composed_agreements.append(composed)

            if li == 0:
                indep_str = f"PC={indep['agreement_pc']*100:5.2f}%"
            else:
                indep_str = f"Gromov={indep['agreement_gromov']*100:5.2f}%"

            log.info(f"  Seed {seed}: independent {indep_str}, composed={composed*100:5.2f}%")
            per_seed_details.append({
                'seed': seed,
                'c_k_min': float(c_k.min()),
                'c_k_max': float(c_k.max()),
                'c_k_mean': float(c_k.mean()),
                'independent_agreement': indep['agreement_pc'] if li == 0 else indep['agreement_gromov'],
                'composed_agreement': composed,
            })

        if li == 0:
            indep_means = float(np.mean(indep_pc_agreements))
            indep_rule = "PC κ c_k ~ U(1,5)"
        else:
            indep_means = float(np.mean(indep_gromov_agreements))
            indep_rule = f"Gromov weight={args.weight_l_gromov}"

        composed_means = float(np.mean(composed_agreements))
        delta_pp = abs(composed_means - indep_means) * 100

        if composed_means > 0.95:
            composed_verdict = "FAIL (>95% 一致, hybrid 影响被淹没)"
            pass_open = False
        elif composed_means > 0.60:
            composed_verdict = "OPEN (60-95%)"
            pass_open = True
        else:
            composed_verdict = "TOO_STRONG (<60% 一致)"
            pass_open = True

        pass_consistency = delta_pp <= CONSISTENCY_TOL_PP
        gate0_layer_pass = pass_open and pass_consistency

        log.info(f"  >>> Layer {li} summary:")
        log.info(f"      Independent mean ({indep_rule}): {indep_means*100:5.2f}%")
        log.info(f"      Composed   mean (hybrid rule):  {composed_means*100:5.2f}%")
        log.info(f"      |Δ| = {delta_pp:.2f}pp (阈值 ≤{CONSISTENCY_TOL_PP}pp)")
        log.info(f"      composed verdict: {composed_verdict}")
        log.info(f"      Gate 0 layer pass: {gate0_layer_pass}")

        all_results["per_layer"][str(li)] = {
            "K": K,
            "independent_rule": indep_rule,
            "independent_agreement_mean": indep_means,
            "independent_agreement_std": float(np.std(indep_pc_agreements if li == 0 else indep_gromov_agreements)),
            "composed_agreement_mean": composed_means,
            "composed_agreement_std": float(np.std(composed_agreements)),
            "delta_pp": delta_pp,
            "consistency_tol_pp": CONSISTENCY_TOL_PP,
            "pass_consistency": pass_consistency,
            "composed_verdict": composed_verdict,
            "gate0_layer_pass": gate0_layer_pass,
            "per_seed": per_seed_details,
        }

    layer_passes = [all_results["per_layer"][str(li)]["gate0_layer_pass"] for li in range(len(n_e_list))]
    all_pass = all(layer_passes)
    log.info("\n" + "=" * 70)
    log.info("汇总 (Gate 0 Hybrid Phase 0 per-layer composition):")
    log.info("=" * 70)
    log.info(f"{'Layer':<6} {'K':<5} {'Independent':<14} {'Composed':<10} {'|Δ|pp':<8} {'verdict':<35} {'pass':<5}")
    for li in range(len(n_e_list)):
        r = all_results["per_layer"][str(li)]
        verdict_short = "OPEN" if "OPEN" in r["composed_verdict"] else ("FAIL" if "FAIL" in r["composed_verdict"] else "TOO_STRONG")
        log.info(f"{li:<6} {r['K']:<5} {r['independent_agreement_mean']*100:5.2f}%       "
                 f"{r['composed_agreement_mean']*100:5.2f}%   "
                 f"{r['delta_pp']:5.2f}  {r['composed_verdict']:<35} "
                 f"{'✓' if r['gate0_layer_pass'] else '✗'}")

    all_results["gate0_overall_pass"] = all_pass
    all_results["gate0_layer_pass_count"] = int(sum(layer_passes))
    all_results["gate0_layer_count"] = len(layer_passes)
    log.info(f"\n>>> Gate 0 overall: {sum(layer_passes)}/{len(layer_passes)} layers pass")
    log.info(f">>> Gate 0 verdict: {'PASS → proceed to Gate 1 (Stage 1)' if all_pass else 'PARTIAL/FAIL → 不进 Gate 1'}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()
