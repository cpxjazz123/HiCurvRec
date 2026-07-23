#!/usr/bin/env python3
"""Task #44 Phase 1 V4 — Flat-3seg 对照矩阵 (终止前最后检验).

用户怀疑: C/D 差是因为 "原始子空间比 fused 脏" + "norm long tail", 不是因为几何机制.
设计对照矩阵分离 4 个变量:
  变量 X1: 几何机制 (StandardRQ=欧氏 / PMRQ=曲率)
  变量 X2: 输入归一化 (L2 norm / NORM_CAP=5.0 clip / raw)
  变量 X3: 段数 (1=fused / 3=独立子空间)
  变量 X4: 维度 (concat 192d)

实验矩阵 (5 组, 全在 toy toy 设定 K=64, 10K items, 3 层, 2000 steps):
  ref1. B-L2norm       (fused, L2 norm)             — 已有 baseline (B = 0.275)
  E1.   B-rawnorm      (fused, raw, max=381)        — 验证 fused 本身的 norm 鲁棒
  E2.   Flat-3seg-raw  (3 sub, NORM_CAP=5.0)        — 与 C/D 同样输入, 只换欧氏
  E3.   Flat-3seg-L2   (3 sub, 各自 L2 norm)        — 隔离 norm 归一化的贡献
  E4.   Flat-3seg-cat  (concat 3 sub → 192d, L2)    — 隔离 "段数 vs 维度" 的贡献

判定:
  - 如果 Flat-3seg-L2 ≈ 0.275: 几何机制是 root cause (假设 A: PM-RQ 否证成立, 可终止)
  - 如果 Flat-3seg-L2 ≫ 0.275: 数据脏是 root cause (假设 B: PM-RQ 未否证, 需先修数据管线)
  - 如果 Flat-3seg-raw ≫ Flat-3seg-L2: 确认 norm long tail 是具体原因
  - 如果 Flat-3seg-cat ≈ 0.275: 段数无关, 维度/容量是核心
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUT_DIR = ROOT / 'products/task44_pmrq_phase1'
OUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda:0'
SEED = 42
N_ITEMS_TOY = 10000
K_CODEBOOK = 64
N_LAYERS = 3
N_STEPS = 2000
BATCH_SIZE = 512
LR = 1e-3
NORM_CAP = 5.0


def load_mckg_data():
    """5 种输入准备的对照."""
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False, map_location='cpu')
    sub = mckg['subspace_item'].numpy()
    fused = mckg['fused_item'].numpy()

    # fused L2 normalized (B 用)
    fused_l2 = fused / np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-6)
    # fused raw
    fused_raw = fused.copy()

    # 3 sub NORM_CAP clip (C/D 用)
    sub_clipped = []
    for i in range(3):
        norms = np.linalg.norm(sub[i], axis=1, keepdims=True)
        clip_factor = np.minimum(NORM_CAP / norms.clip(min=1e-6), 1.0)
        sub_clipped.append(sub[i] * clip_factor)

    # 3 sub L2 normalized each
    sub_l2 = []
    for i in range(3):
        v = sub[i] / np.linalg.norm(sub[i], axis=1, keepdims=True).clip(min=1e-6)
        sub_l2.append(v)

    # 3 sub concat → 192d, then L2 norm
    sub_cat = np.concatenate(sub_l2, axis=1)  # (N, 192)
    sub_cat_l2 = sub_cat / np.linalg.norm(sub_cat, axis=1, keepdims=True).clip(min=1e-6)

    return {
        'fused_l2': fused_l2.astype(np.float32),
        'fused_raw': fused_raw.astype(np.float32),
        'sub_clipped': [s.astype(np.float32) for s in sub_clipped],
        'sub_l2': [s.astype(np.float32) for s in sub_l2],
        'sub_cat_l2': sub_cat_l2.astype(np.float32),
    }


class StandardRQ(nn.Module):
    """Standard RQ-VAE: K codebook per layer, L2 nearest."""
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
    """3 个独立 StandardRQ, loss = sum of 3 seg final residual norm²."""
    def __init__(self, K=64, dim=64, n_layers=3):
        super().__init__()
        self.rq_s = StandardRQ(K, dim, n_layers)
        self.rq_e = StandardRQ(K, dim, n_layers)
        self.rq_h = StandardRQ(K, dim, n_layers)

    def forward(self, x_s, x_e, x_h):
        _, r_s = self.rq_s(x_s)
        _, r_e = self.rq_e(x_e)
        _, r_h = self.rq_h(x_h)
        return (r_s, r_e, r_h)


def train_standard(model, x, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x.shape[0]
    trajectory = {'step': [], 'recon_mse': []}
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
            trajectory['recon_mse'].append(loss.item())
            print(f'  [{tag}] step {step}: loss={loss.item():.4f}')
    return trajectory


def train_flat3seg(model, x_s, x_e, x_h, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_s.shape[0]
    trajectory = {'step': [], 's_loss': [], 'e_loss': [], 'h_loss': [], 'total': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        bs = x_s[idx]
        be = x_e[idx]
        bh = x_h[idx]
        r_s, r_e, r_h = model(bs, be, bh)
        ls = (r_s ** 2).sum(dim=-1).mean()
        le = (r_e ** 2).sum(dim=-1).mean()
        lh = (r_h ** 2).sum(dim=-1).mean()
        loss = ls + le + lh
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['s_loss'].append(ls.item())
            trajectory['e_loss'].append(le.item())
            trajectory['h_loss'].append(lh.item())
            trajectory['total'].append(loss.item())
            print(f'  [{tag}] step {step}: s={ls.item():.4f}, e={le.item():.4f}, '
                  f'h={lh.item():.4f}, total={loss.item():.4f}')
    return trajectory


def main():
    print('========== Task #44 V4: Flat-3seg 对照矩阵 ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = load_mckg_data()
    fused_l2 = torch.from_numpy(data['fused_l2'][:N_ITEMS_TOY]).to(DEVICE)
    fused_raw = torch.from_numpy(data['fused_raw'][:N_ITEMS_TOY]).to(DEVICE)
    sub_clipped = [torch.from_numpy(s[:N_ITEMS_TOY]).to(DEVICE) for s in data['sub_clipped']]
    sub_l2 = [torch.from_numpy(s[:N_ITEMS_TOY]).to(DEVICE) for s in data['sub_l2']]
    sub_cat_l2 = torch.from_numpy(data['sub_cat_l2'][:N_ITEMS_TOY]).to(DEVICE)

    print(f'fused_l2: {fused_l2.shape}, mean norm {fused_l2.norm(dim=-1).mean():.3f}')
    print(f'fused_raw: {fused_raw.shape}, mean norm {fused_raw.norm(dim=-1).mean():.3f}, max {fused_raw.norm(dim=-1).max():.1f}')
    for i, n in enumerate(['sphere', 'euclid', 'hyperbolic']):
        print(f'sub_clipped[{i}] {n}: mean norm {sub_clipped[i].norm(dim=-1).mean():.3f}, '
              f'max {sub_clipped[i].norm(dim=-1).max():.2f}')
        print(f'sub_l2[{i}]       {n}: mean norm {sub_l2[i].norm(dim=-1).mean():.3f}, '
              f'max {sub_l2[i].norm(dim=-1).max():.3f}')
    print(f'sub_cat_l2: {sub_cat_l2.shape}, mean norm {sub_cat_l2.norm(dim=-1).mean():.3f}')
    print()

    results = {}

    # E1: B-rawnorm
    print('========== E1: B-rawnorm (fused raw, 不归一化) ==========')
    torch.manual_seed(SEED)
    m_e1 = StandardRQ(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_e1 = train_standard(m_e1, fused_raw, tag='E1-Braw')
    results['E1_B_rawnorm'] = {'trajectory': traj_e1, 'final_loss': traj_e1['recon_mse'][-1]}

    # E2: Flat-3seg-raw (与 C/D 同样输入, 只换欧氏)
    print('\n========== E2: Flat-3seg-raw (3 sub NORM_CAP=5.0, 独立欧氏 RQ) ==========')
    torch.manual_seed(SEED)
    m_e2 = Flat3Seg(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_e2 = train_flat3seg(m_e2, sub_clipped[0], sub_clipped[1], sub_clipped[2], tag='E2-F3raw')
    results['E2_flat3seg_raw'] = {'trajectory': traj_e2, 'final_loss': traj_e2['total'][-1]}

    # E3: Flat-3seg-L2
    print('\n========== E3: Flat-3seg-L2 (3 sub 各自 L2 norm, 独立欧氏 RQ) ==========')
    torch.manual_seed(SEED)
    m_e3 = Flat3Seg(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_e3 = train_flat3seg(m_e3, sub_l2[0], sub_l2[1], sub_l2[2], tag='E3-F3L2')
    results['E3_flat3seg_L2'] = {'trajectory': traj_e3, 'final_loss': traj_e3['total'][-1]}

    # E4: Flat-3seg-cat (concat 192d)
    print('\n========== E4: Flat-3seg-cat (concat 192d, L2 norm) ==========')
    torch.manual_seed(SEED)
    m_e4 = StandardRQ(K=K_CODEBOOK, dim=192, n_layers=N_LAYERS).to(DEVICE)
    traj_e4 = train_standard(m_e4, sub_cat_l2, tag='E4-cat192')
    results['E4_concat192'] = {'trajectory': traj_e4, 'final_loss': traj_e4['recon_mse'][-1]}

    # 总结
    print('\n========== 对照矩阵总结 ==========')
    b_l2 = 0.2751  # 已记录的 baseline
    c_pmrq_raw = 3.5117  # V3 PM-RQ 3段, 同样 NORM_CAP 输入
    print(f'baseline (B-L2norm, fused normalized):  final_loss = {b_l2:.4f}  ← Reference')
    print(f'E1 (B-rawnorm, fused raw max=381):     final_loss = {results["E1_B_rawnorm"]["final_loss"]:.4f}')
    print(f'E2 (Flat-3seg-raw, 3 sub clipped):     final_loss = {results["E2_flat3seg_raw"]["final_loss"]:.4f}  ← KEY')
    print(f'E3 (Flat-3seg-L2, 3 sub L2 norm):      final_loss = {results["E3_flat3seg_L2"]["final_loss"]:.4f}  ← KEY')
    print(f'E4 (Flat-3seg-cat 192d, L2 norm):      final_loss = {results["E4_concat192"]["final_loss"]:.4f}')
    print(f'(PM-RQ 3 段 V3, same clip input):       final_loss = {c_pmrq_raw:.4f}  ← 已有, 几何机制组')

    # 关键判定
    e2 = results['E2_flat3seg_raw']['final_loss']
    e3 = results['E3_flat3seg_L2']['final_loss']
    e4 = results['E4_concat192']['final_loss']
    e1 = results['E1_B_rawnorm']['final_loss']

    print()
    print('=== 关键判定 ===')
    print(f'(1) E3 (Flat-L2) vs B-L2 baseline = {e3:.4f} vs {b_l2:.4f}')
    print(f'    差异 = {(e3 - b_l2) / b_l2 * 100:+.1f}%')
    if e3 < b_l2 * 1.15:
        print(f'    → Flat-3seg-L2 接近 baseline, 数据本身没问题, **几何机制 (PM-RQ) 是 root cause**')
    else:
        print(f'    → Flat-3seg-L2 显著高于 baseline ({((e3 - b_l2) / b_l2 * 100):.1f}%), '
              f'**数据/管线有问题 (norm/clip/维度)**, 不能简单归罪 PM-RQ')
    print()
    print(f'(2) E2 (Flat-raw) vs E3 (Flat-L2) = {e2:.4f} vs {e3:.4f}')
    print(f'    差异 = {(e2 - e3) / max(e3, 1e-6) * 100:+.1f}%')
    if e2 > e3 * 1.5:
        print(f'    → L2 norm 改善显著, **norm long tail 是具体原因**')
    else:
        print(f'    → L2 norm 改善不显著, norm 不是 root cause')
    print()
    print(f'(3) E4 (concat 192d) vs E3 (Flat-3seg 3*64d) = {e4:.4f} vs {e3:.4f}')
    print(f'    差异 = {(e4 - e3) / max(e3, 1e-6) * 100:+.1f}%')
    print()
    print(f'(4) E1 (B-rawnorm fused raw) vs B-L2 = {e1:.4f} vs {b_l2:.4f}')
    print(f'    差异 = {(e1 - b_l2) / b_l2 * 100:+.1f}%')
    if e1 > b_l2 * 1.2:
        print(f'    → fused 原始数据也有 norm 问题, 但 B 用 L2 norm 屏蔽了')
    else:
        print(f'    → fused raw 不退化, baseline 用 L2 norm 是更严苛的对照')
    print()
    print(f'(5) Flat-3seg-raw (欧氏) vs PM-RQ 3段 (曲率) = {e2:.4f} vs {c_pmrq_raw:.4f}')
    print(f'    差异 = {(c_pmrq_raw - e2) / max(e2, 1e-6) * 100:+.1f}%')
    if c_pmrq_raw > e2 * 1.5:
        print(f'    → 即使同样脏输入, PM-RQ 曲率机制让结果更差, **PM-RQ 本身有害**')
    else:
        print(f'    → 曲率机制没有让结果更差, 主要是数据问题')

    summary = {
        'task': 'Task #44 Phase 1 V4 Flat-3seg 对照矩阵',
        'config': {
            'N_ITEMS': N_ITEMS_TOY, 'K': K_CODEBOOK, 'N_LAYERS': N_LAYERS,
            'N_STEPS': N_STEPS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'NORM_CAP': NORM_CAP,
        },
        'final_losses': {
            'B_L2norm_baseline (已有)': b_l2,
            'E1_B_rawnorm_fused_raw': e1,
            'E2_Flat3seg_raw_clip': e2,
            'E3_Flat3seg_L2_norm': e3,
            'E4_concat192_L2_norm': e4,
            'PMRQ_3seg_V3_对照': c_pmrq_raw,
        },
        'trajectories': {
            'E1': results['E1_B_rawnorm']['trajectory'],
            'E2': results['E2_flat3seg_raw']['trajectory'],
            'E3': results['E3_flat3seg_L2']['trajectory'],
            'E4': results['E4_concat192']['trajectory'],
        },
        'judgment': {
            'data_normalized_close_to_baseline': bool(e3 < b_l2 * 1.15),
            'L2_norm_significant_improvement': bool(e2 > e3 * 1.5),
            'PMRQ_worse_than_flat_even_with_dirty_data': bool(c_pmrq_raw > e2 * 1.5),
            'fused_raw_degrades': bool(e1 > b_l2 * 1.2),
        },
    }

    out = OUT_DIR / 'task44_phase1_v4_ablation.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()