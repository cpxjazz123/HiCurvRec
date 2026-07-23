#!/usr/bin/env python3
"""Idea 4 补测 — Simplified QINCo 训练 + Δ_1 / erank_2 对比

设计:
  L1 固定 (用 A_baseline L1 codebook + L1 索引)
  L2 动态: per-L1-cluster affine
    c2_eff[k, i] = (1 + α_k) ⊙ base_c2[i] + β_k
  α_k, β_k 是可学习参数 (K1 × D = 256 × 2048 = 524288 params)
  Loss: ||r1 - c2_eff[idx1, idx2]||² + λ * (||α||² + ||β||²)

指标:
  Δ_1^{QINCo} = [QErr(r_{L1_input}; C_{L1_AQ}) - QErr(r_{L1_input}; C_{L1_QINCo})] / QErr
  erank_2^{QINCo} = exp(H(s/sum)) on SVD of L2 effective codebook
  baseline: Δ_1=15.9, erank_2=82.5

Kill:
  Δ_1^{QINCo} ≈ 15.9 (no improvement) AND erank_2 explodes → QINCo 没价值
  Δ_1^{QINCo} ≪ 15.9 (e.g. <10) OR erank_2 stable → QINCo 通过
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea4_QINCo_aux'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
A_BASELINE_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def qerr(r, C):
    r_norm2 = (r ** 2).sum(-1, keepdims=True)
    c_norm2 = (C ** 2).sum(-1)
    cross = r @ C.T
    d2 = np.maximum(r_norm2 + c_norm2[None, :] - 2 * cross, 0)
    nearest_idx = d2.argmin(axis=-1)
    nearest_c = C[nearest_idx]
    err = ((r - nearest_c) ** 2).sum(axis=-1)
    return float(err.mean()), nearest_idx


def erank(C):
    """Effective rank via SVD entropy."""
    s = torch.linalg.svdvals(C.float())
    p = s / (s.sum() + 1e-12)
    H = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(H))


class SimplifiedQINCo(nn.Module):
    """L2 dynamic: per-L1-cluster affine on a fixed base codebook."""
    def __init__(self, c1, base_c2, K1, K2, D):
        super().__init__()
        # α: (K1, D), β: (K1, D)  with small init
        self.alpha = nn.Parameter(torch.zeros(K1, D) * 0.01)
        self.beta = nn.Parameter(torch.zeros(K1, D) * 0.01)
        self.register_buffer('c1', c1)
        self.register_buffer('base_c2', base_c2)
        self.K1 = K1
        self.K2 = K2
        self.D = D

    def get_c2_eff(self):
        """Returns (K1, K2, D) effective L2 codebook."""
        # (1 + α_k) ⊙ base_c2[i] + β_k
        return (1 + self.alpha.unsqueeze(1)) * self.base_c2.unsqueeze(0) + self.beta.unsqueeze(1)

    def forward(self, r1, idx1):
        """r1: (N, D), idx1: (N,)
        Returns: r2, idx2, c2_eff_full"""
        c2_eff = self.get_c2_eff()  # (K1, K2, D)
        # Get per-sample c2_eff: (N, K2, D)
        c2_per_sample = c2_eff[idx1]  # (N, K2, D)
        # NN assignment
        r_norm2 = (r1 ** 2).sum(-1, keepdim=True)  # (N, 1)
        c_norm2 = (c2_per_sample ** 2).sum(-1)  # (N, K2)
        cross = torch.einsum('nd,nkd->nk', r1, c2_per_sample)  # (N, K2)
        d2 = (r_norm2 + c_norm2 - 2 * cross).clamp(min=0)
        idx2 = d2.argmin(dim=-1)  # (N,)
        c2_selected = c2_per_sample[torch.arange(r1.shape[0]), idx2]  # (N, D)
        r2 = r1 - c2_selected
        return r2, idx2


def main():
    print('=' * 70)
    print('Idea 4 补测 — Simplified QINCo 训练 + Δ_1 / erank_2')
    print('=' * 70)

    # === Load data ===
    print('\n[1] 加载数据...')
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    bundle = torch.load(A_BASELINE_PATH, map_location='cpu', weights_only=False)
    c1_AQ = bundle['codebooks'][0].float()  # (K1, D) = (256, 2048)
    base_c2 = bundle['codebooks'][1].float()  # (256, 2048) — A_baseline L2 codebook as base
    idx1_AQ = bundle['idx_lst'][0]  # (N,) L1 AQ indices
    K1, K2, D = c1_AQ.shape[0], base_c2.shape[0], c1_AQ.shape[1]
    N = x.shape[0]
    print(f'  N={N}, K1={K1}, K2={K2}, D={D}')

    # r1_AQ = residual after AQ L1 quantization
    c1_selected = c1_AQ[idx1_AQ]  # (N, D)
    r1_AQ = x - c1_selected
    print(f'  r1_AQ norm: {(r1_AQ**2).sum(-1).mean().sqrt():.4f}')

    # === Baseline Δ_1 calculation ===
    print('\n[2] AQ baseline Δ_1 (复现) — 用 AQ L1 codebook vs (x) QErr')
    err_x_with_L1, _ = qerr(x, c1_AQ.numpy())
    err_r1_with_L1, _ = qerr(r1_AQ.numpy(), c1_AQ.numpy())
    print(f'  QErr(x; C1_AQ)={err_x_with_L1:.4f}, '
          f'QErr(r1_AQ; C1_AQ)={err_r1_with_L1:.4f}')
    delta_1_baseline = (err_r1_with_L1 - err_x_with_L1) / (err_x_with_L1 + 1e-12)
    # Note: user's formula is reverse — but this Δ_1 number is large (≈15.9) meaning
    # L2 residual is much larger when L1 is reused. This is the "AQ L1 is insufficient" measure.
    # Our convention: positive = L1 insufficient → QINCo should reduce this.
    # Alternative: use A_baseline qerr_baseline values from existing diag

    # Use Δ_l definition from idea1 script:
    # Δ_l = [QErr(r_{l+1}; C_l) - QErr(r_{l+1}; C_{l+1})] / QErr(r_{l+1}; C_{l+1})
    # For baseline AQ: r_1 input = x, r_1_with_C_0 doesn't exist
    # Actually in our code: r_lst[0] = x, r_lst[1] = residual after L1
    # So: QErr(r_1; C_1) = how well does C_1 quantize the residual?
    # We want: how much does C_2 (current layer) beat C_1 (previous layer) on r_2?
    # This is what user formula says.

    # Let me re-derive: from idea1_delta_L_diagnostic.py
    # Δ_l = [QErr(r_{l+1}; C_l) - QErr(r_{l+1}; C_{l+1})] / QErr(r_{l+1}; C_{l+1})
    # for l=1, r_2 = residual after L1 = input to L2
    # QErr(r_2; C_1) = "if we reuse L1 codebook to quantize L2 input"
    # QErr(r_2; C_2) = "if we use L2 codebook to quantize L2 input" (actual)
    # Δ_1 = (QErr_reuse - QErr_actual) / QErr_actual → if positive, C_2 is better than C_1
    # Δ_1 = 15.9 means L2 contribution is 16x more than if we just used L1 codebook
    # That's the "AQ L2 absorbs a lot" measure
    print(f'  → Δ_1 baseline = {delta_1_baseline:.4f}')

    # Use baseline C2 from A_baseline as c2 for erank baseline
    c2_baseline = bundle['codebooks'][1].float()
    erank_2_baseline = erank(c2_baseline)
    print(f'  erank_2 baseline (AQ L2) = {erank_2_baseline:.2f}')

    # === Train QINCo ===
    print('\n[3] 训练 Simplified QINCo...')
    model = SimplifiedQINCo(c1_AQ, base_c2, K1, K2, D).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  QINCo trainable params: {n_params:,} ({n_params/1e6:.2f}M)')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_reg = 1e-3  # regularize α, β

    # Move data
    r1_train = r1_AQ.to(device)
    idx1_train = idx1_AQ.to(device)

    BS = 512
    n_steps = 500
    log_every = 100
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    for step in range(n_steps):
        # Random batch
        batch_idx = rng.choice(N, size=BS, replace=False)
        r1_b = r1_train[batch_idx]
        idx1_b = idx1_train[batch_idx]
        optimizer.zero_grad()
        r2_b, idx2_b = model(r1_b, idx1_b)
        # Reconstruction loss (only L2 part)
        loss_recon = (r2_b ** 2).sum(dim=-1).mean()
        # Regularizer on α, β (prevent huge drift from base)
        loss_reg = lambda_reg * ((model.alpha ** 2).sum() + (model.beta ** 2).sum())
        loss = loss_recon + loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0 or step == 0:
            print(f'    step {step+1:>4d}: loss={float(loss.item()):.6f}, '
                  f'recon={float(loss_recon.item()):.6f}, reg={float(loss_reg.item()):.6f}, '
                  f'alpha_mean_abs={float(model.alpha.abs().mean()):.4f}, '
                  f'beta_mean_abs={float(model.beta.abs().mean()):.4f}')
    train_time = time.time() - t0
    print(f'  Total train time: {train_time:.1f}s')

    # === Evaluate QINCo on full dataset ===
    print('\n[4] Evaluate QINCo on full N=11924...')
    model.eval()
    with torch.no_grad():
        r1_full = r1_AQ.to(device)
        idx1_full = idx1_AQ.to(device)
        # Batch to avoid OOM
        r2_full = []
        idx2_full = []
        for i in range(0, N, BS):
            j = min(i + BS, N)
            r2_b, idx2_b = model(r1_full[i:j], idx1_full[i:j])
            r2_full.append(r2_b.cpu())
            idx2_full.append(idx2_b.cpu())
        r2_qinco = torch.cat(r2_full, dim=0)
        idx2_qinco = torch.cat(idx2_full, dim=0)

    # === Δ_1^{QINCo}: compute Δ_1 with QINCo's L2 effective codebook ===
    # IMPORTANT: use the SAME formula as AQ Δ_1 = (QErr(r_2; C_1) - QErr(r_2; C_2)) / QErr(r_2; C_2)
    # This is the "previous layer codebook vs current layer codebook" measure
    print('\n[5] Δ_1^{QINCo} 计算 (用与 AQ 一致的公式: 前层码本 vs 本层码本)...')
    with torch.no_grad():
        c2_eff_full = model.get_c2_eff().cpu()  # (K1, K2, D)
        # For Δ_1 comparison with AQ, we need a single C_2 to compare C_1 against.
        # QINCo's C_2 is per-cluster, so we compute QErr(r_2; C_2_QINCo_avg)
        # AND also the actual per-sample QErr to capture the full QINCo effect.
        c2_eff_avg = c2_eff_full.mean(dim=0)  # (K2, D) — averaged effective codebook
        actual_qerr = float((r2_qinco ** 2).sum(dim=-1).mean())  # per-sample QErr
        err_r1_with_L1, _ = qerr(r1_AQ.numpy(), c1_AQ.numpy())
        # Δ_1 using per-sample QINCo QErr (the real effect of dynamics):
        delta_1_qinco_persample = (err_r1_with_L1 - actual_qerr) / (actual_qerr + 1e-12)
        # Δ_1 using averaged effective codebook (no dynamics, just structural change):
        err_r1_with_L2_avg, _ = qerr(r1_AQ.numpy(), c2_eff_avg.numpy())
        delta_1_qinco_avgcodebook = (err_r1_with_L1 - err_r1_with_L2_avg) / (err_r1_with_L2_avg + 1e-12)
        # AQ baseline Δ_1 (for reference):
        delta_1_aq = (err_r1_with_L1 - 0.0711) / 0.0711
        print(f'  AQ Δ_1 (baseline):              {delta_1_aq:.4f}')
        print(f'  QINCo Δ_1 (per-sample, actual): {delta_1_qinco_persample:.4f}')
        print(f'  QINCo Δ_1 (avg c2_eff, no dyn): {delta_1_qinco_avgcodebook:.4f}')
        # QINCo Δ_1 with avg codebook measures "structural change to L2 (no per-cluster dynamics)"
        # QINCo Δ_1 with per-sample measures "full QINCo effect"
        delta_1_qinco = delta_1_qinco_persample
        print(f'  → use per-sample as primary: {delta_1_qinco:.4f}')
        print(f'  Δ Δ_1 (AQ - QINCo per-sample) = {delta_1_aq - delta_1_qinco:+.4f}')
        print(f'  Δ Δ_1 (AQ - QINCo avg cb)    = {delta_1_aq - delta_1_qinco_avgcodebook:+.4f}')

    # === erank_2^{QINCo} ===
    print('\n[6] erank_2^{QINCo} 计算...')
    erank_2_qinco_per_l1 = []
    for k in range(K1):
        e = erank(c2_eff_full[k])
        erank_2_qinco_per_l1.append(e)
    erank_2_qinco_mean = float(np.mean(erank_2_qinco_per_l1))
    erank_2_qinco_max = float(np.max(erank_2_qinco_per_l1))
    erank_2_qinco_min = float(np.min(erank_2_qinco_per_l1))
    print(f'  erank_2 baseline (AQ L2):    {erank_2_baseline:.2f}')
    print(f'  erank_2 QINCo (avg over L1): {erank_2_qinco_mean:.2f}')
    print(f'  erank_2 QINCo (max over L1): {erank_2_qinco_max:.2f}')
    print(f'  erank_2 QINCo (min over L1): {erank_2_qinco_min:.2f}')

    # === QErr compare ===
    print('\n[7] QErr 对比...')
    # AQ: r1_AQ - base_c2[idx2_AQ]
    idx2_AQ = bundle['idx_lst'][1]
    r2_AQ = r1_AQ - base_c2[idx2_AQ]
    actual_qerr_aq = float((r2_AQ ** 2).sum(dim=-1).mean())
    qinco_r1_norm = float((r2_qinco ** 2).sum(dim=-1).mean())
    print(f'  AQ QErr on r1:      {actual_qerr_aq:.4f}')
    print(f'  QINCo QErr on r1:   {qinco_r1_norm:.4f}')
    qerr_drop_7 = actual_qerr_aq - qinco_r1_norm
    print(f'  Δ QErr (AQ - QINCo)={qerr_drop_7:+.4f} '
          f'({qerr_drop_7/actual_qerr_aq*100:+.2f}%)')

    # === Final verdict ===
    print('\n' + '=' * 70)
    print('VERDICT — Idea 4 Simplified QINCo')
    print('=' * 70)
    print(f'  Δ_1 baseline (AQ)       = 15.9')
    print(f'  Δ_1 QINCo              = {delta_1_qinco:.4f}')
    print(f'  Δ Δ_1 (drop)           = {15.9 - delta_1_qinco:+.4f}')
    print(f'  erank_2 baseline (AQ)  = {erank_2_baseline:.2f}')
    print(f'  erank_2 QINCo (avg)    = {erank_2_qinco_mean:.2f}')
    print(f'  erank_2 QINCo (max)    = {erank_2_qinco_max:.2f}')
    print()

    # Kill判定 per user
    delta_1_drop = 15.9 - delta_1_qinco
    drop_significant = delta_1_drop > 5.0  # dropped > 5 from baseline 15.9
    erank_not_exploded = erank_2_qinco_max < 500  # not more than 6x baseline 82.5

    print(f'  Kill line 1: Δ_1 drop > 5 from 15.9 → {drop_significant} '
          f'(actual drop = {delta_1_drop:+.4f})')
    print(f'  Kill line 2: erank_2 max < 500 → {erank_not_exploded} '
          f'(actual max = {erank_2_qinco_max:.2f})')

    if drop_significant and erank_not_exploded:
        verdict = '✓ PASS — QINCo 让 L1 吸收更多 (Δ_1 下降) 且 L2 容量不爆炸'
    elif delta_1_drop < 1.0:
        verdict = '✗ KILL — Δ_1 几乎没变, QINCo 没起到动态码本作用'
    elif erank_2_qinco_max > 1000:
        verdict = '✗ KILL — erank_2 爆炸, QINCo 退化成"per-cluster 独立码本"'
    else:
        verdict = '△ BORDERLINE — 部分指标改善, 部分退化'
    print(f'  VERDICT: {verdict}')

    # Actual QErr comparison is the smoking gun
    qerr_drop = actual_qerr_aq - qinco_r1_norm
    print(f'\n  QErr comparison (forward pass): AQ={actual_qerr_aq:.4f}, '
          f'QINCo={qinco_r1_norm:.4f}, Δ={qerr_drop:+.4f} ({qerr_drop/actual_qerr_aq*100:+.2f}%)')
    if qerr_drop < -0.005:
        verdict += ' [NOTE: QINCo 实际 QErr 反向变差 > 0.5%]'

    # Save
    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        return o

    out_path = os.path.join(OUT_DIR, 'idea4_QINCo_simplified.json')
    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'algo': 'QINCo_simplified',
            'K1': K1,
            'K2': K2,
            'D': D,
            'N': N,
            'n_train_steps': n_steps,
            'train_time_s': train_time,
            'lambda_reg': lambda_reg,
            'lr': 1e-3,
            'final_loss': losses[-1] if losses else None,
            'losses_curve': losses[::10] + [losses[-1]] if losses else [],
            'delta_1_baseline_AQ': float(delta_1_aq),
            'delta_1_QINCo_per_sample': float(delta_1_qinco_persample),
            'delta_1_QINCo_avg_codebook': float(delta_1_qinco_avgcodebook),
            'delta_1_drop': float(delta_1_aq - delta_1_qinco_persample),
            'delta_1_drop_avg_cb': float(delta_1_aq - delta_1_qinco_avgcodebook),
            'erank_2_baseline_AQ': float(erank_2_baseline),
            'erank_2_QINCo_mean': float(erank_2_qinco_mean),
            'erank_2_QINCo_max': float(erank_2_qinco_max),
            'erank_2_QINCo_min': float(erank_2_qinco_min),
            'erank_2_QINCo_per_L1_cluster': erank_2_qinco_per_l1,
            'QErr_AQ_L2_on_r1': float(actual_qerr_aq),
            'QErr_QINCo_L2_on_r1': float(qinco_r1_norm),
            'QErr_improvement_pct': float((actual_qerr_aq - qinco_r1_norm) / actual_qerr_aq * 100),
            'verdict': verdict,
        }), f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()