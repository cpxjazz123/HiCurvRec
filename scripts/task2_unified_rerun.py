#!/usr/bin/env python3
"""task16 unified rerun: A/B/C three groups with multiple f_radial definitions + cos θ.

Computes for each algorithm and each layer:
- cos θ mean / median / std (NEW diagnostic)
- f_radial_historical: aggregate(|‖r‖ - ‖q‖|² / ‖e‖²)   [matches old task16 JSON]
- f_radial_defA_agg:    aggregate((‖r‖ - ‖q‖)² / ‖e‖²)   [user's definition, no abs]
- f_radial_defB_agg:    aggregate((‖r‖ - ‖q‖cosθ)² / ‖e‖²) [orthogonal projection]
- f_radial_defB_mean:   per-item mean of (‖r‖ - ‖q‖cosθ)² / ‖e‖²

For B group, requires ckpt to be re-trained (current task16 rerun).
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

# IMPORTANT: pre-import src.utils.decorators to break circular import chain
# triggered by Lightning ckpt unpickling (interfaces.py ↔ iterators.py ↔ decorators.py)
import src.utils.decorators  # noqa: F401

import json
import os
import numpy as np
import torch


def forward_residual(x, codebooks, gains=None):
    """Run RKMeans forward pass to extract (r_l, q_l, idx_l) per layer."""
    r_lst = [x.clone()]
    q_lst, idx_lst = [], []
    L = len(codebooks)
    for l in range(L):
        r = r_lst[-1]
        C = codebooks[l]
        if gains is not None and gains[l] is not None:
            eff_C = C * gains[l].unsqueeze(-1)
        else:
            eff_C = C
        # Euclidean NN via squared distance trick
        r_norm2 = (r ** 2).sum(-1, keepdim=True)
        C_norm2 = (eff_C ** 2).sum(-1)
        d2 = r_norm2 - 2 * (r @ eff_C.T) + C_norm2
        idx = d2.argmin(dim=1)
        q = eff_C[idx]
        q_lst.append(q)
        idx_lst.append(idx)
        r_lst.append(r - q)
    return r_lst, q_lst, idx_lst


def compute_diagnostics(r_lst, q_lst, idx_lst):
    """Compute all f_radial variants + cos θ per layer."""
    L = len(r_lst) - 1
    out = []
    for l in range(L):
        r = r_lst[l]
        q = q_lst[l]
        idx = idx_lst[l]
        r_norm = r.norm(dim=-1)
        q_norm = q.norm(dim=-1)
        e = r - q
        e_sq = (e ** 2).sum(dim=-1)            # per-item total squared error
        e_sq_total = e_sq.sum().clamp(min=1e-12)
        cos_theta = ((r * q).sum(dim=-1) /
                     (r_norm * q_norm + 1e-12).clamp(min=1e-12))
        cos_theta = cos_theta.clamp(-1, 1)
        # e_r components
        radial_abs_sq = ((r_norm - q_norm).abs() ** 2)        # |‖r‖ - ‖q‖|²  (historical)
        radial_signed_sq = (r_norm - q_norm) ** 2              # (‖r‖ - ‖q‖)²   (definition A)
        e_r_hat = r_norm - q_norm * cos_theta                   # axial component (signed)
        axial_sq = e_r_hat ** 2                                 # (‖r‖ - ‖q‖cosθ)²  (definition B)
        tangential_sq = (q_norm * (1 - cos_theta ** 2).clamp(min=0)).sqrt() ** 2  # ‖q‖²sin²θ
        out.append({
            'l': l + 1,
            'r_norm_mean': r_norm.mean().item(),
            'r_norm_std':  r_norm.std().item(),
            'q_norm_mean': q_norm.mean().item(),
            'q_norm_std':  q_norm.std().item(),
            'cos_theta_mean':   cos_theta.mean().item(),
            'cos_theta_median': cos_theta.median().item(),
            'cos_theta_std':    cos_theta.std().item(),
            'cos_theta_p25':    cos_theta.quantile(0.25).item(),
            'cos_theta_p75':    cos_theta.quantile(0.75).item(),
            # historical (matches task4_batch1.py: |r|-|q| squared, aggregate)
            'f_radial_historical_agg': (radial_abs_sq.sum() / e_sq_total).item(),
            # definition A: signed difference squared, aggregate
            'f_radial_defA_agg':      (radial_signed_sq.sum() / e_sq_total).item(),
            # definition A: per-item mean
            'f_radial_defA_mean':       (radial_signed_sq / e_sq.clamp(min=1e-12)).mean().item(),
            # definition B (orthogonal projection onto r̂), aggregate
            'f_radial_defB_agg':       (axial_sq.sum() / e_sq_total).item(),
            # definition B, per-item mean
            'f_radial_defB_mean':      (axial_sq / e_sq.clamp(min=1e-12)).mean().item(),
            # geometric sanity
            'max_decomp_residual':     ((axial_sq + tangential_sq - e_sq).abs().max()).item(),
        })
    return out


def load_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    L = sum(1 for k in sd.keys()
            if k.startswith('quantization_layer_list.') and k.endswith('.centroids'))
    codebooks = [sd[f'quantization_layer_list.{l}.centroids'].float() for l in range(L)]
    gains = []
    has_gains = False
    for l in range(L):
        gk = f'quantization_layer_list.{l}.cluster_gains'
        if gk in sd:
            has_gains = True
            gains.append(sd[gk].float())
        else:
            gains.append(None)
    return codebooks, gains, has_gains


def main(algorithms, emb_path, out_dir):
    print(f"=== Loading embedding: {emb_path}")
    X = torch.load(emb_path, map_location='cpu', weights_only=False)
    print(f"  shape={tuple(X.shape)}, dtype={X.dtype}")

    all_results = {}
    for name, ckpt_path in algorithms.items():
        print(f"\n=== {name}: {ckpt_path} ===")
        if not os.path.exists(ckpt_path):
            print(f"  ckpt NOT FOUND, skipping")
            all_results[name] = {'error': f'ckpt missing: {ckpt_path}'}
            continue
        codebooks, gains, has_gains = load_codebooks(ckpt_path)
        print(f"  has_gains={has_gains}, codebook shapes={[list(c.shape) for c in codebooks]}")
        if has_gains:
            print(f"  cluster_gains means: {[g.mean().item() for g in gains]}")
        r_lst, q_lst, idx_lst = forward_residual(X, codebooks, gains if has_gains else None)
        layers = compute_diagnostics(r_lst, q_lst, idx_lst)
        result = {
            'algorithm': name,
            'ckpt': ckpt_path,
            'has_gains': has_gains,
            'layers': layers,
        }
        all_results[name] = result
        print(f"  per-layer summary:")
        print(f"  {'L':<3} {'||r||':<8} {'||q||':<8} {'cosθ':<8} {'f_rad_hist':<11} {'f_rad_defA_agg':<13} {'f_rad_defB_agg':<13}")
        for lyr in layers:
            print(f"  {lyr['l']:<3} {lyr['r_norm_mean']:<8.4f} {lyr['q_norm_mean']:<8.4f} "
                  f"{lyr['cos_theta_mean']:<8.4f} {lyr['f_radial_historical_agg']:<11.4f} "
                  f"{lyr['f_radial_defA_agg']:<13.4f} {lyr['f_radial_defB_agg']:<13.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'task4_unified_rerun.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Saved → {out_path} ===")


if __name__ == '__main__':
    EMB = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
    OUT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task14_unified_rerun'

    ALGOS = {
        'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
        'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-11/13-15-28/checkpoints/checkpoint_000_003000.ckpt',
        'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
    }

    main(ALGOS, EMB, OUT)