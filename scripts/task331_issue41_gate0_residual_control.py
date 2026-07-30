#!/usr/bin/env python3
"""Task #331 / Issue #41 Gate 0 (b) — residual/codebook space control (sanity check).

零 GPU 纯计算 (CPU only).
复用 Task #80/82/89 的 `compute_true_distortion` + `load_hrqvae` + `extract_per_layer_residuals`.
期望: residual 空间 κ_opt ≈ 0 (复现 Task #80 既有结论, sanity check).

硬停止: 若 κ_opt 显著偏离 0 → 实现差异, STOP 排查.

Usage:
  python3 scripts/task331_issue41_gate0_residual_control.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from task80_stage1c_true_distortion import compute_true_distortion
from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def load_hrqvae_no_curvature(ckpt_path, device):
    """Inline loader bypassing task116.load_hrqvae curvature_list kwarg bug.

    task116 passes curvature_list= to HRQVAE.__init__, but current HRQVAE
    does NOT accept curvature_list (verified hrqvae.py:10-23). This loader
    uses Task #84's args schema (basic, no curvature_list).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
        train_args = ckpt.get('args', None)
    else:
        state = ckpt
        train_args = None

    if train_args is None:
        raise RuntimeError(f"❌ No 'args' in ckpt {ckpt_path}")

    data_path = getattr(train_args, 'data_path', None)
    if data_path is None:
        raise RuntimeError(f"❌ No data_path in ckpt args")
    data = EmbDataset(data_path)

    # NO curvature_list — current HRQVAE.__init__ doesn't accept it
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        beta=train_args.beta,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_eps=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, train_args


def extract_per_layer_residuals_inline(model, embeddings_np, device, batch_size, use_sk=False):
    """Inline per-layer residuals extractor (mirrors task116 but standalone)."""
    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    N = embeddings.shape[0]
    with torch.no_grad():
        z = model.encoder(embeddings)
        residuals = [z.cpu().clone()]
        codewords = []
        assignments = []
        residual = z
        for i, quantizer in enumerate(model.hrq.vq_layers):
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            residuals.append(residual.cpu().clone())
            codewords.append(x_res.cpu().clone())
            assignments.append(indices.cpu())
    return residuals, assignments, codewords


# Task #84 baseline Stage 1 ckpt (实际有 subdir)
DEFAULT_CKPT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments'
ITEM_EMB = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'

KAPPA_GRID = [
    0.0, -0.05, -0.10, -0.15, -0.20, -0.30, -0.50, -1.00, -1.50, -2.00,
]

LAYER_LABELS = ['L0_raw_encoded', 'L1_after_Q0', 'L2_after_Q0+Q1', 'L3_final']


def find_latest_ckpt(ckpt_dir):
    """Find latest epoch_*_model.pth in ckpt_dir/<sub>/."""
    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"ckpt_dir does not exist: {ckpt_dir}")
    # Find latest subdir (date-folder), then latest ckpt
    subdirs = sorted([d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))])
    if not subdirs:
        raise RuntimeError(f"No subdir in {ckpt_dir}")
    latest_sub = subdirs[-1]
    full_sub = os.path.join(ckpt_dir, latest_sub)
    ckpts = sorted(
        [os.path.join(full_sub, f) for f in os.listdir(full_sub) if f.startswith('epoch_') and f.endswith('_model.pth')],
        key=os.path.getmtime,
    )
    if not ckpts:
        raise RuntimeError(f"No epoch_*_model.pth in {full_sub}")
    return ckpts[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None, help='Path to .pth (auto-detect if None)')
    p.add_argument('--ckpt_dir', default=DEFAULT_CKPT_DIR)
    p.add_argument('--emb', default=ITEM_EMB)
    p.add_argument('--n_subset', type=int, default=500)
    p.add_argument('--n_iter', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--d', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task331_issue41_gate0_residual_control.json')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Task #331 / Issue #41 Gate 0 (b)] residual control (sanity check)")

    # Resolve ckpt
    ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_dir)
    print(f"  ckpt: {ckpt_path}")

    # Load HRQ-VAE + extract per-layer residuals (inline loader, no curvature_list)
    device = torch.device(args.device)
    model, train_args = load_hrqvae_no_curvature(ckpt_path, device)
    print(f"  Loaded model: in_dim={model.in_dim}, e_dim={model.e_dim}, num_emb_list={model.num_emb_list}")
    import pandas as pd
    df = pd.read_parquet(args.emb)
    if 'embedding' in df.columns:
        embs_full = np.stack([np.asarray(e, dtype=np.float32) for e in df['embedding'].values])
    else:
        raise RuntimeError(f"No 'embedding' column in {args.emb}")
    rng = np.random.default_rng(args.seed)
    if args.n_subset < embs_full.shape[0]:
        idx = rng.choice(embs_full.shape[0], size=args.n_subset, replace=False)
        embs_subset = embs_full[idx]
    else:
        embs_subset = embs_full
    print(f"  Subset: {embs_subset.shape}")

    residuals, _, _ = extract_per_layer_residuals_inline(
        model, embs_subset, device, batch_size=min(256, args.n_subset), use_sk=False,
    )
    print(f"  Residuals: {len(residuals)} layers")
    for i, r in enumerate(residuals):
        print(f"    L{i} ({LAYER_LABELS[i]}): shape={r.shape}, dim={r.shape[1]}")

    # (b) Compute Kruskal stress for each κ × layer
    all_results = {}
    all_elapsed = {}
    for i, r_t in enumerate(residuals):
        r_np = r_t.numpy().astype(np.float32) if hasattr(r_t, 'numpy') else np.asarray(r_t, dtype=np.float32)
        print(f"\n  === L{i} ({LAYER_LABELS[i]}, dim={r_np.shape[1]}) ===")
        print(f"  {'κ':>8} | {'stress':>10} | {'elapsed (s)':>12}")
        layer_results = {}
        layer_elapsed = {}
        for k in KAPPA_GRID:
            t0 = time.time()
            stress = compute_true_distortion(
                r_np, kappa=k, n_subset=min(args.n_subset, r_np.shape[0]),
                n_iter=args.n_iter, lr=args.lr, d=args.d,
                device=args.device, seed=args.seed,
            )
            elapsed = time.time() - t0
            print(f"  {k:>+8.2f} | {stress:>10.4f} | {elapsed:>12.1f}", flush=True)
            layer_results[str(k)] = float(stress)
            layer_elapsed[str(k)] = round(elapsed, 2)
        all_results[f'L{i}_{LAYER_LABELS[i]}'] = layer_results
        all_elapsed[f'L{i}_{LAYER_LABELS[i]}'] = layer_elapsed

    # Best κ per layer
    best_kappas = {}
    for layer_key, layer_results in all_results.items():
        best_k, best_stress = min(layer_results.items(), key=lambda x: x[1])
        best_kappas[layer_key] = {
            'kappa': float(best_k),
            'stress': float(best_stress),
            'abs_kappa': abs(float(best_k)),
        }

    # Sanity check: all layers best |κ| should be ≤ 0.10 (Task #80 结论)
    sanity_pass = all(b['abs_kappa'] <= 0.10 for b in best_kappas.values())

    summary = {
        'task': 'task331_issue41_gate0_residual_control',
        'purpose': 'sanity check that residual space optimal κ ≈ 0 (reproduce Task #80)',
        'method': 'Sala 2018 h-MDS (Kruskal stress-1) on 4 layers of Task #84 HRQ-VAE residuals',
        'ckpt': ckpt_path,
        'kappa_grid': KAPPA_GRID,
        'all_results': all_results,
        'all_elapsed': all_elapsed,
        'best_kappa_per_layer': best_kappas,
        'sanity_pass': bool(sanity_pass),
        'sanity_pass_condition': 'all layers |best κ| ≤ 0.10 (Task #80 既有结论)',
        'task80_reference': 'Task #80 verdict: residual 空间 κ=0 optimal (Idea 1 否证)',
        'gate_0_b_pass': bool(sanity_pass),
        'hard_stop_on_fail': '若 (b) sanity check FAIL, STOP 排查实现差异, 不继续 Gate 1',
    }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"\n  === Sanity check summary ===")
    for layer_key, b in best_kappas.items():
        print(f"  {layer_key}: κ*={b['kappa']:+.2f}, |κ|={b['abs_kappa']:.2f}, stress={b['stress']:.4f}")
    print(f"  Sanity PASS: {sanity_pass}")
    print(f"  Gate 0 (b) PASS: {summary['gate_0_b_pass']}")


if __name__ == '__main__':
    main()