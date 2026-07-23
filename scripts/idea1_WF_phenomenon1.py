#!/usr/bin/env python3
"""Idea 1 现象 1: water-filling 比特分配

输入: Stage 1 embedding x ∈ R^{N × D} (N=11924, D=2048)
1. 算样本协方差 Σ = (1/N) X^T X
2. 特征分解 Σ = U Λ U^T, λ_1 ≥ ... ≥ λ_D
3. 给定总码率 R_total = L * log2(K) (AQ 用 K=256, L=3 → R_total=24 bits)
4. 解 water-filling: R_total = Σ_i max(0, log2(λ_i/θ*))
5. 把 eigenvalue 按降序切给各层:
   - L1: top-K_1 维的 bit 总和 → R_1
   - L2: next K_2 维 → R_2
   - L3: next K_3 维 → R_3
6. 比较理论分配 (R_1/R_total) vs 平权 (1/L=33%)

判定:
- 理论某层份额偏离 1/L > 15% → Idea 1 值得做
- 理论接近均匀 → AQ 平权已经接近最优, kill
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

# AQ 配置: L=3 层, K=256
L_AQ = 3
K_AQ = 256
R_TOTAL = L_AQ * np.log2(K_AQ)  # 24 bits


def water_filling(eigvals, R_total, max_iter=1000, tol=1e-4):
    """Solve R_total = Σ_i max(0, log2(λ_i/θ)) for θ.
    eigvals: sorted descending.
    Returns θ*.
    """
    # Bisection: θ ranges from very small (R_total = log2(λ_1/θ) for all) to λ_min (no assignment)
    log2_total_max = np.log2(eigvals[0] / (eigvals.min() + 1e-12))
    if R_total > log2_total_max:
        # Cap: assign log2(λ_i/eigvals.min()) to all
        return eigvals.min()
    # Bisection
    lo = eigvals.min() * 0.5
    hi = eigvals[0] * 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        R_mid = np.sum(np.maximum(0, np.log2(eigvals / mid)))
        if R_mid < R_total:
            hi = mid  # increase mid → smaller max(0, log(λ/θ))? Actually larger θ → smaller log(λ/θ)
            # Wait: larger θ → log(λ/θ) smaller → R_mid smaller → need smaller θ
        else:
            lo = mid
        if abs(R_mid - R_total) / R_total < tol:
            break
    return mid


def assign_bits_to_layers(eigvals, K_layers, theta):
    """Top-K_1 dims → layer 1, next K_2 → layer 2, etc.
    For each layer, sum max(0, log2(λ_i/θ)) over its assigned dims.
    """
    R_per_layer = []
    assigned_so_far = 0
    for l, K_l in enumerate(K_layers):
        # Slice of eigenvalues for this layer
        sli = eigvals[assigned_so_far:assigned_so_far + K_l]
        R_l = float(np.sum(np.maximum(0, np.log2(sli / theta + 1e-12))))
        R_per_layer.append(R_l)
        assigned_so_far += K_l
    return R_per_layer


def main():
    print('=' * 70)
    print('Idea 1 现象 1: water-filling 比特分配')
    print('=' * 70)
    print(f'  R_total = L × log2(K) = {L_AQ} × log2({K_AQ}) = {R_TOTAL} bits')
    print()

    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    print(f'  x shape: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}')

    # Sample covariance
    print('\nComputing sample covariance Σ = (1/N) X^T X...')
    x_c = x - x.mean(axis=0, keepdims=True)
    Sigma = (x_c.T @ x_c) / x_c.shape[0]
    print(f'  Σ shape: {Sigma.shape}, trace={np.trace(Sigma):.4f}')

    # Eigendecomposition
    print('\nEigendecomposition...')
    eigvals = np.linalg.eigvalsh(Sigma)[::-1]  # descending
    eigvals = np.maximum(eigvals, 1e-12)  # numerical safety
    print(f'  eigvals: top 5 = {eigvals[:5]}')
    print(f'  eigvals: bottom 5 = {eigvals[-5:]}')
    print(f'  eigvals ratio λ_1/λ_D = {eigvals[0]/eigvals[-1]:.2e}')
    print(f'  cumulative variance top-256: {eigvals[:256].sum()/eigvals.sum():.4f}')
    print(f'  cumulative variance top-512: {eigvals[:512].sum()/eigvals.sum():.4f}')
    print(f'  cumulative variance top-768: {eigvals[:768].sum()/eigvals.sum():.4f}')

    # Water-filling solve
    print(f'\nSolving water-filling for R_total={R_TOTAL} bits...')
    theta = water_filling(eigvals, R_TOTAL)
    print(f'  θ* = {theta:.4e}')
    # Active dimensions: those with λ_i > θ*
    n_active = int((eigvals > theta).sum())
    print(f'  Active dims (λ_i > θ*): {n_active} / {len(eigvals)}')
    # Sum of bits assigned
    bits_assigned = float(np.sum(np.maximum(0, np.log2(eigvals / theta))))
    print(f'  Bits assigned: {bits_assigned:.2f} (target: {R_TOTAL})')

    # === Allocation schemes ===
    # Scheme 1: 平权 (AQ baseline)
    equal_share = 1.0 / L_AQ  # 33.33%
    print(f'\n=== Scheme 1: 平权 (AQ) ===')
    print(f'  Per-layer share: {equal_share:.4f} = {equal_share*100:.2f}%')

    # Scheme 2: water-filling K=256 切片
    print(f'\n=== Scheme 2: water-filling K={K_AQ} 切片分配 ===')
    K_layers = [K_AQ, K_AQ, K_AQ]  # 256 + 256 + 256 = 768 dims used
    R_per_layer = assign_bits_to_layers(eigvals, K_layers, theta)
    R_per_layer = np.array(R_per_layer)
    R_total_computed = R_per_layer.sum()
    shares = R_per_layer / R_total_computed
    print(f'  R_per_layer (bits): {R_per_layer}')
    print(f'  R_total: {R_total_computed:.4f}')
    print(f'  Share: {shares}  → {[f"{s*100:.1f}%" for s in shares]}')
    for l in range(L_AQ):
        diff_pp = (shares[l] - equal_share) * 100
        print(f'    L{l+1}: share={shares[l]*100:.1f}%  (Δ vs 1/L = {diff_pp:+.1f} pp)')

    # Scheme 3: greedy round-robin (最差情况)
    # Scheme 4: water-filling alternative K split
    print(f'\n=== Scheme 3: water-filling K=[512, 384, 256] (按方差递减) ===')
    K_layers_alt = [512, 256, 256]
    R_alt = assign_bits_to_layers(eigvals, K_layers_alt, theta)
    shares_alt = np.array(R_alt) / np.sum(R_alt)
    print(f'  R_per_layer: {R_alt}')
    print(f'  Share: {[f"{s*100:.1f}%" for s in shares_alt]}')

    # === Verdict ===
    print('\n' + '=' * 70)
    print('VERDICT — Idea 1 现象 1')
    print('=' * 70)
    max_diff_pp = max(abs(shares[l] - equal_share) * 100 for l in range(L_AQ))
    print(f'  最大 |理论份额 - 1/L|: {max_diff_pp:.1f} pp')
    print(f'  Kill 线: max_diff_pp < 15 pp → 均匀')
    if max_diff_pp < 15:
        verdict = '✗ kill (理论接近均匀, AQ 平权已近最优)'
    else:
        verdict = f'✓ Idea 1 值得做 (理论分配偏离平权 {max_diff_pp:.1f} pp)'
    print(f'  Verdict: {verdict}')

    # Save
    out = {
        'R_total': float(R_TOTAL),
        'K_layers_scheme2': K_layers,
        'theta_star': float(theta),
        'n_active_dims': n_active,
        'R_per_layer_scheme2': R_per_layer.tolist(),
        'share_scheme2': shares.tolist(),
        'equal_share': equal_share,
        'max_diff_pp_vs_equal': float(max_diff_pp),
        'K_layers_scheme3': K_layers_alt,
        'R_per_layer_scheme3': R_alt,
        'share_scheme3': shares_alt.tolist(),
        'cumvar_top256': float(eigvals[:256].sum()/eigvals.sum()),
        'cumvar_top512': float(eigvals[:512].sum()/eigvals.sum()),
        'eigvals_top10': eigvals[:10].tolist(),
        'eigvals_bot10': eigvals[-10:].tolist(),
        'kill_verdict': verdict,
    }
    out_path = os.path.join(OUT_DIR, 'idea1_WF_phenomenon1.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()