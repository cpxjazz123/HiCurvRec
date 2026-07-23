#!/usr/bin/env python3
"""战线二 L1 — 后处理修复 + 2×2 消融 (Task #46)

4 个 post-processing 修复 + 2×2 whitening 消融 → 测 SCR 是否降到 <1.5x (G1 门控)

L1 (a) log1p norm 压缩: e' = e / ||e|| · log(1 + ||e||)
L1 (b) 分位数归一化: ||e||' = F_target^-1(F_emp(||e||))
L1 (c) per-subspace whitening 2×2 消融: {raw, log1p} × {no-whiten, whiten}
L1 (d) GSRQ Gain-Shape 复活: gain 标量量化 + shape 球面 RQ

判定: 任一 L1 变体把 SCR 从 4.22x 压到 < 1.5x → GO-浅, 否则 → GO-深
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUT_DIR = ROOT / 'products/task154_l1_fixes'
OUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda:0'
SEED = 42
N_ITEMS_TOY = 10000
K_CODEBOOK = 64
N_LAYERS = 3
N_STEPS = 2000
BATCH_SIZE = 512
LR = 1e-3


def load_mckg():
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False, map_location='cpu')
    sub = mckg['subspace_item'].numpy()  # (3, 11924, 64)
    fused = mckg['fused_item'].numpy()  # (11924, 64)
    return sub, fused


def log1p_compress(e):
    """L1 (a) log1p norm compression."""
    n = np.linalg.norm(e, axis=-1, keepdims=True)
    n_safe = n.clip(min=1e-6)
    return e / n_safe * np.log1p(n_safe)


def quantile_normalize(e, target='uniform_0.5_1.5'):
    """L1 (b) 分位数归一化 (norm 层面)."""
    n = np.linalg.norm(e, axis=-1)
    # 经验 CDF
    sorted_n = np.sort(n)
    ranks = np.searchsorted(sorted_n, n)
    emp_cdf = ranks / len(n)  # [0, 1)
    # 目标 CDF
    if target == 'uniform_0.5_1.5':
        target_cdf = lambda u: 0.5 + u * 1.0  # U[0.5, 1.5]
    else:  # half-normal
        target_cdf = lambda u: 0.5 + 0.5 * (1 + np.tanh(2 * u - 0))  # 简化
    n_new = target_cdf(emp_cdf)
    # 重塑
    n_new = n_new.reshape(-1)
    return e / np.linalg.norm(e, axis=-1, keepdims=True).clip(min=1e-6) * n_new.reshape(-1, 1)


def whiten(e):
    """L1 (c) whitening: 减均值 + 对协方差矩阵求逆 sqrt."""
    mu = e.mean(axis=0, keepdims=True)
    e_centered = e - mu
    cov = np.cov(e_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals.clip(min=1e-6)
    # Whitening: W = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return e_centered @ W


class StandardRQ(nn.Module):
    def __init__(self, K=64, dim=64, n_layers=3):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers
        self.cb_list = nn.ParameterList(
            [nn.Parameter(torch.randn(K, dim) * 0.01) for _ in range(n_layers)])

    def forward(self, x):
        all_idx = []
        residual = x
        for layer in range(self.n_layers):
            d = torch.cdist(residual, self.cb_list[layer], p=2)
            idx = d.argmin(dim=-1)
            c = self.cb_list[layer][idx]
            residual = residual - c
            all_idx.append(idx)
        return all_idx, residual


class Flat3Seg(nn.Module):
    def __init__(self, K=64, dim=64, n_layers=3):
        super().__init__()
        self.rq1 = StandardRQ(K, dim, n_layers)
        self.rq2 = StandardRQ(K, dim, n_layers)
        self.rq3 = StandardRQ(K, dim, n_layers)

    def forward(self, x1, x2, x3):
        _, r1 = self.rq1(x1)
        _, r2 = self.rq2(x2)
        _, r3 = self.rq3(x3)
        return r1, r2, r3


def train_standard(model, x, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x.shape[0]
    trajectory = {'step': [], 'loss': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = x[idx]
        _, residual = model(batch)
        loss = (residual ** 2).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['loss'].append(loss.item())
            if step == n_steps - 1:
                print(f'  [{tag}] step {step}: loss={loss.item():.4f}')
    return trajectory


def train_flat3seg(model, x1, x2, x3, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x1.shape[0]
    trajectory = {'step': [], 'total': [], 's': [], 'e': [], 'h': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        b1 = x1[idx]
        b2 = x2[idx]
        b3 = x3[idx]
        r1, r2, r3 = model(b1, b2, b3)
        l1 = (r1 ** 2).sum(dim=-1).mean()
        l2 = (r2 ** 2).sum(dim=-1).mean()
        l3 = (r3 ** 2).sum(dim=-1).mean()
        loss = l1 + l2 + l3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['s'].append(l1.item())
            trajectory['e'].append(l2.item())
            trajectory['h'].append(l3.item())
            trajectory['total'].append(loss.item())
            if step == n_steps - 1:
                print(f'  [{tag}] step {step}: s={l1.item():.4f}, e={l2.item():.4f}, '
                      f'h={l3.item():.4f}, total={loss.item():.4f}')
    return trajectory


def compute_metrics(x, name=''):
    """QMP M1 + M3."""
    n = np.linalg.norm(x, axis=-1)
    rho_max = n.max() / (n.mean() + 1e-9)
    cv = n.std() / (n.mean() + 1e-9)
    print(f'  [{name}] norm: mean={n.mean():.3f}, max={n.max():.2f}, ρ_max={rho_max:.2f}, CV={cv:.3f}')
    return {'rho_max': float(rho_max), 'cv': float(cv), 'mean': float(n.mean()), 'max': float(n.max())}


def main():
    print('========== 战线二 L1 — 后处理修复 ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sub_raw, fused_raw = load_mckg()
    print(f'fused shape: {fused_raw.shape}')
    print(f'sub shape: {sub_raw.shape}')
    print()
    print('=== 原始 QMP M1 ===')
    metrics_raw = {
        'fused_raw': compute_metrics(fused_raw, 'fused_raw'),
        'sub_s_raw': compute_metrics(sub_raw[0], 'sub_s_raw'),
        'sub_e_raw': compute_metrics(sub_raw[1], 'sub_e_raw'),
        'sub_h_raw': compute_metrics(sub_raw[2], 'sub_h_raw'),
    }

    # 准备 8 种 fused 输入 (2x2 + 2 baseline)
    # baseline 1: fused_raw (无处理)
    # baseline 2: fused_L2 norm
    fused_l2 = fused_raw / np.linalg.norm(fused_raw, axis=-1, keepdims=True).clip(min=1e-6)

    # 修复版 1: log1p
    fused_log1p = log1p_compress(fused_raw)

    # 修复版 2: quantile (uniform [0.5, 1.5])
    fused_quantile = quantile_normalize(fused_raw, target='uniform_0.5_1.5')

    # 修复版 3: whiten + raw
    fused_whiten = whiten(fused_raw)

    # 修复版 4: whiten + log1p
    fused_whiten_log1p = log1p_compress(fused_whiten)

    print()
    print('=== 修复后 QMP M1 ===')
    variants = {
        'fused_L2_baseline': fused_l2,
        'fused_raw_baseline': fused_raw,
        'fused_log1p': fused_log1p,
        'fused_quantile': fused_quantile,
        'fused_whiten': fused_whiten,
        'fused_whiten_log1p': fused_whiten_log1p,
    }
    metrics_post = {}
    for name, v in variants.items():
        metrics_post[name] = compute_metrics(v, name)

    # 2×2 消融 (whiten × log1p)
    print()
    print('=== 2×2 消融 (whiten × log1p) 测 SCR ===')

    results = {}
    N = N_ITEMS_TOY
    for name, v in variants.items():
        v_dev = torch.from_numpy(v[:N]).float().to(DEVICE)
        torch.manual_seed(SEED)
        m = StandardRQ(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
        traj = train_standard(m, v_dev, tag=name)
        results[name] = {
            'final_loss': traj['loss'][-1],
            'trajectory': traj,
        }

    # SCR 计算 (vs fused_L2 baseline)
    print()
    print('=== SCR (vs fused_L2 baseline) ===')
    b_l2 = results['fused_L2_baseline']['final_loss']
    print(f'  B-L2 (baseline)        = {b_l2:.4f}')
    for name, r in results.items():
        if name == 'fused_L2_baseline':
            continue
        scr = r['final_loss'] / b_l2
        flag = '✅ GO-浅' if scr < 1.5 else ('⚠️ 边界' if scr < 3.0 else '❌ GO-深')
        print(f'  {name:30s} = {r["final_loss"]:.4f}, SCR = {scr:.2f}x  {flag}')

    # 现在做 3 段拆分对比 (Flat-3seg) 看 log1p / quantile 是否也帮助 3 段
    print()
    print('=== 3 段拆分对照: log1p 是否帮助 3 段? ===')
    sub_l2 = []
    for i in range(3):
        sub_l2.append(sub_raw[i] / np.linalg.norm(sub_raw[i], axis=-1, keepdims=True).clip(min=1e-6))
    sub_log1p = [log1p_compress(sub_raw[i]) for i in range(3)]
    sub_l2_dev = [torch.from_numpy(s[:N]).float().to(DEVICE) for s in sub_l2]
    sub_log1p_dev = [torch.from_numpy(s[:N]).float().to(DEVICE) for s in sub_log1p]

    # Flat-3seg on raw clip (已有 E2 = 1.020, reference)
    print('  E2 reference (raw clip): 1.020')

    torch.manual_seed(SEED)
    m_f3_l2 = Flat3Seg(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_f3_l2 = train_flat3seg(m_f3_l2, sub_l2_dev[0], sub_l2_dev[1], sub_l2_dev[2], tag='F3-L2')
    f3_l2 = traj_f3_l2['total'][-1]

    torch.manual_seed(SEED)
    m_f3_log1p = Flat3Seg(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_f3_log1p = train_flat3seg(m_f3_log1p, sub_log1p_dev[0], sub_log1p_dev[1], sub_log1p_dev[2],
                                    tag='F3-log1p')
    f3_log1p = traj_f3_log1p['total'][-1]

    print(f'  F3-L2 (sub L2 norm):       {f3_l2:.4f}')
    print(f'  F3-log1p (sub log1p):      {f3_log1p:.4f}')

    summary = {
        'task': 'Task #46 战线二 L1 后处理修复',
        'metrics_raw': metrics_raw,
        'metrics_post': metrics_post,
        'final_losses_fused': {n: r['final_loss'] for n, r in results.items()},
        'scr_vs_baseline': {n: r['final_loss'] / b_l2 for n, r in results.items()},
        'f3_seg': {
            'raw_clip_ref': 1.020,
            'L2': f3_l2,
            'log1p': f3_log1p,
        },
        'g1_gate': {
            'best_scr': min(r['final_loss'] / b_l2 for n, r in results.items() if n != 'fused_L2_baseline'),
            'passes_g1_shallow': min(r['final_loss'] / b_l2 for n, r in results.items() if n != 'fused_L2_baseline') < 1.5,
        },
    }

    out = OUT_DIR / 'task154_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()