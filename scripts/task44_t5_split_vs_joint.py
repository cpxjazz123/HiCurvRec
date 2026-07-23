#!/usr/bin/env python3
"""Task #44 V5 — T5 (sentence-t5-base) embedding 上跑拆分 vs 拼接对照.

用户怀疑: MCKG 是 margin ranking loss 训出来的, 只保证排序, 不保证度量结构.
           Toy 3 段 vs 拼接 1.75x 的差距可能不是 "架构拆分" 的固有问题,
           而是 MCKG embedding 本身的度量结构缺陷.

对照设计 (T5 sentence-t5-base 768d, norm 健康 max=1.12):
  T1. T5-L2-768     T5 768d L2 norm, StandardRQ 768d          (joint baseline)
  T2. T5-L2-3seg    T5 768d L2 norm → 3×256d, Flat-3seg 3×256d (split)
  T3. T5-Raw-3seg   T5 768d raw         → 3×256d, Flat-3seg 3×256d (split + no norm)
  T4. T5-L2-192     T5 768d L2 norm → 3×256d → concat 768d, StandardRQ 768d  (control, vs T1)

判定:
  - T2 / T1 = 如果 ≈ 3.0x → "拆分独立" 是欧氏 RQ 固有代价 (理论下界)
              如果 ≈ 4.2x → "拆分代价" 在 MCKG 上被放大
              如果 ≈ 1.0x → 拆分反而更好, 彻底打破理论
  - T3 / T2 = L2 norm 在 T5 拆分下是否有帮助 (对比 MCKG 的 E2 vs E3 反向)

数据来源: logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt
         (sentence-t5-base, shape (11924, 768), 已存在)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
T5_PATH = ROOT / 'logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt'
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


def load_t5():
    """T5 sentence-t5-base, shape (11924, 768)."""
    t = torch.load(T5_PATH, weights_only=False, map_location='cpu')
    if torch.is_tensor(t):
        return t.float()
    raise RuntimeError(f'Unexpected type: {type(t)}')


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
            print(f'  [{tag}] step {step}: loss={loss.item():.4f}')
    return trajectory


def train_flat3seg(model, x1, x2, x3, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x1.shape[0]
    trajectory = {'step': [], 'loss1': [], 'loss2': [], 'loss3': [], 'total': []}
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
            trajectory['loss1'].append(l1.item())
            trajectory['loss2'].append(l2.item())
            trajectory['loss3'].append(l3.item())
            trajectory['total'].append(loss.item())
            print(f'  [{tag}] step {step}: seg1={l1.item():.4f}, seg2={l2.item():.4f}, '
                  f'seg3={l3.item():.4f}, total={loss.item():.4f}')
    return trajectory


def main():
    print('========== Task #44 V5: T5 (sentence-t5-base) 拆分 vs 拼接对照 ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t5_full = load_t5()
    print(f'T5 shape: {t5_full.shape}')
    N = t5_full.shape[0]
    print(f'T5 norm: mean={t5_full.norm(dim=-1).mean():.3f}, '
          f'max={t5_full.norm(dim=-1).max():.3f}, '
          f'p99={np.percentile(t5_full.norm(dim=-1).numpy(), 99):.3f}')

    # 5 种输入准备
    t5_l2 = t5_full / t5_full.norm(dim=-1, keepdim=True).clip(min=1e-6)  # unit sphere
    # 拆分到 3 段 (按 dim 顺序分)
    t5_l2_split = [t5_l2[:, i*256:(i+1)*256].contiguous() for i in range(3)]
    t5_raw_split = [t5_full[:, i*256:(i+1)*256].contiguous() for i in range(3)]

    print(f'T5-L2 norm: {t5_l2.norm(dim=-1).mean():.4f} (应该≈1.0)')
    for i, s in enumerate(t5_l2_split):
        print(f'T5-L2-split[{i}] shape: {s.shape}, norm: {s.norm(dim=-1).mean():.3f}')
    for i, s in enumerate(t5_raw_split):
        print(f'T5-Raw-split[{i}] shape: {s.shape}, norm: {s.norm(dim=-1).mean():.3f}, '
              f'max: {s.norm(dim=-1).max():.3f}')

    # 移到 device
    t5_l2_dev = t5_l2[:N_ITEMS_TOY].to(DEVICE)
    t5_l2_split_dev = [s[:N_ITEMS_TOY].to(DEVICE) for s in t5_l2_split]
    t5_raw_split_dev = [s[:N_ITEMS_TOY].to(DEVICE) for s in t5_raw_split]

    results = {}

    # T1: T5-L2-768 (joint StandardRQ)
    print('\n========== T1: T5-L2-768 (joint StandardRQ 768d) ==========')
    torch.manual_seed(SEED)
    m_t1 = StandardRQ(K=K_CODEBOOK, dim=768, n_layers=N_LAYERS).to(DEVICE)
    traj_t1 = train_standard(m_t1, t5_l2_dev, tag='T1-768')
    results['T1_L2_768_joint'] = {'trajectory': traj_t1, 'final_loss': traj_t1['loss'][-1]}

    # T2: T5-L2-3seg (split 3×256d, Flat-3seg)
    print('\n========== T2: T5-L2-3seg (split 3×256d, Flat-3seg) ==========')
    torch.manual_seed(SEED)
    m_t2 = Flat3Seg(K=K_CODEBOOK, dim=256, n_layers=N_LAYERS).to(DEVICE)
    traj_t2 = train_flat3seg(m_t2, t5_l2_split_dev[0], t5_l2_split_dev[1], t5_l2_split_dev[2],
                              tag='T2-3seg-L2')
    results['T2_L2_3seg_split'] = {'trajectory': traj_t2, 'final_loss': traj_t2['total'][-1]}

    # T3: T5-Raw-3seg (split 3×256d, Flat-3seg, raw 不归一化)
    print('\n========== T3: T5-Raw-3seg (split 3×256d, Flat-3seg, raw) ==========')
    torch.manual_seed(SEED)
    m_t3 = Flat3Seg(K=K_CODEBOOK, dim=256, n_layers=N_LAYERS).to(DEVICE)
    traj_t3 = train_flat3seg(m_t3, t5_raw_split_dev[0], t5_raw_split_dev[1], t5_raw_split_dev[2],
                              tag='T3-3seg-Raw')
    results['T3_Raw_3seg_split'] = {'trajectory': traj_t3, 'final_loss': traj_t3['total'][-1]}

    # 总结
    print('\n========== T5 对照矩阵总结 ==========')
    t1 = results['T1_L2_768_joint']['final_loss']
    t2 = results['T2_L2_3seg_split']['final_loss']
    t3 = results['T3_Raw_3seg_split']['final_loss']
    print(f'T1 (T5-L2-768,  joint 768d):     final_loss = {t1:.4f}')
    print(f'T2 (T5-L2-3seg, split 3×256d):   final_loss = {t2:.4f}')
    print(f'T3 (T5-Raw-3seg, split 3×256d):  final_loss = {t3:.4f}')

    # MCKG V4 对照 (已有数据)
    b_mckg = 0.275
    e3_mckg = 1.160
    print()
    print('=== MCKG V4 对照 (已有) ===')
    print(f'  MCKG baseline (B-L2 fused):   {b_mckg:.4f}')
    print(f'  MCKG Flat-3seg-L2 (E3):       {e3_mckg:.4f}')
    print(f'  MCKG 拆分/拼接 ratio = {e3_mckg / b_mckg:.2f}x')

    print()
    print('=== T5 对照 (本次) ===')
    print(f'  T5 拆分/拼接 ratio = {t2 / t1:.2f}x')
    print(f'  T5 norm 影响 (T3/T2) = {t3 / t2:.2f}x')

    print()
    print('=== 关键判定 ===')
    mckg_split_ratio = e3_mckg / b_mckg
    t5_split_ratio = t2 / t1
    print(f'(1) MCKG 拆分/拼接 = {mckg_split_ratio:.2f}x, T5 拆分/拼接 = {t5_split_ratio:.2f}x')
    if abs(t5_split_ratio - mckg_split_ratio) < 0.5:
        print(f'    → 两个 ratio 接近, **"拆分代价"是欧氏 RQ 固有, 不是 MCKG 特有**')
    else:
        print(f'    → 两个 ratio 差距大, **MCKG 上"拆分代价"被放大 (margin ranking 数据问题)**')

    print()
    t5_norm_impact = t3 / t2
    mckg_norm_impact_e2_e3 = 1.020 / 1.160
    print(f'(2) T5 norm 影响 = {t5_norm_impact:.2f}x, MCKG norm 影响 (E2/E3) = {mckg_norm_impact_e2_e3:.2f}x')
    print(f'    (T5 已经 norm 健康, 所以 L2 norm 应该帮助小; MCKG 长尾严重, 但 V4 发现 E2/E3 反向)')

    summary = {
        'task': 'Task #44 V5 T5 split-vs-joint 对照',
        'config': {
            'N_ITEMS': N_ITEMS_TOY, 'K': K_CODEBOOK, 'N_LAYERS': N_LAYERS,
            'N_STEPS': N_STEPS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
        },
        't5_norm_stats': {
            'mean': float(t5_full.norm(dim=-1).mean()),
            'max': float(t5_full.norm(dim=-1).max()),
            'p99': float(np.percentile(t5_full.norm(dim=-1).numpy(), 99)),
        },
        'final_losses': {
            'T1_L2_768_joint': t1,
            'T2_L2_3seg_split': t2,
            'T3_Raw_3seg_split': t3,
        },
        'reference_mckg': {
            'B_L2_baseline': b_mckg,
            'E3_flat3seg_L2': e3_mckg,
            'split_ratio_mckg': mckg_split_ratio,
        },
        'judgment': {
            'split_ratio_t5': t5_split_ratio,
            'norm_impact_t5': t5_norm_impact,
            'split_ratio_difference': abs(t5_split_ratio - mckg_split_ratio),
            'split_cost_amplified_in_mckg': abs(t5_split_ratio - mckg_split_ratio) > 0.5,
        },
        'trajectories': {
            'T1': traj_t1,
            'T2': traj_t2,
            'T3': traj_t3,
        },
    }

    out = OUT_DIR / 'task44_phase1_v5_t5_ablation.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()