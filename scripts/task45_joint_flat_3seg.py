#!/usr/bin/env python3
"""战线三 3.1 — Joint-Flat-3seg (Task #45)

设计: 3 个 unfused subspace (NORM_CAP=5.0, 与 V4 E2 对齐), 无曲率机制.
  Loss = ||x_concat - x_hat_concat||^2 (192d 全局 norm²)
  产品距离 d²(r, (c_1, c_2, c_3)) = Σ ||r_s - c_s||²
  联合 argmin 等于逐段独立 argmin (数学恒等), 但训练耦合:
    - 单一全局重构损失
    - 共享 commitment loss (基于 d_total)
    - 联合梯度/EMA 更新

判定:
  - 接近 E4 (≈1.75x): 架构罪在训练解耦
  - 停在 E2/E3 (≈3.7-4.2x): 分段产品结构本身相对统一空间有本质劣势
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
OUT_DIR = ROOT / 'products/task153_joint_flat3seg'
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
STE_TEMP = 5.0
COMMIT_W = 1.0


def load_mckg():
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False, map_location='cpu')
    sub = mckg['subspace_item'].numpy()  # (3, 11924, 64)
    sub_clipped = []
    for i in range(3):
        norms = np.linalg.norm(sub[i], axis=1, keepdims=True)
        clip_factor = np.minimum(NORM_CAP / norms.clip(min=1e-6), 1.0)
        sub_clipped.append(sub[i] * clip_factor)
    return sub_clipped


class JointFlat3Seg(nn.Module):
    """3 段独立 codebook, 训练耦合: 全局 192d norm² + 共享 commitment."""
    def __init__(self, K=64, dim=64, n_layers=3):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers
        # 3 段独立 codebook (per layer)
        self.cb_s = nn.ParameterList([nn.Parameter(torch.randn(K, dim) * 0.01) for _ in range(n_layers)])
        self.cb_e = nn.ParameterList([nn.Parameter(torch.randn(K, dim) * 0.01) for _ in range(n_layers)])
        self.cb_h = nn.ParameterList([nn.Parameter(torch.randn(K, dim) * 0.01) for _ in range(n_layers)])

    def forward_layer(self, r_s, r_e, r_h):
        # 算各段距离 (raw, 欧氏)
        d_s = torch.cdist(r_s, self.cb_s[0], p=2)
        d_e = torch.cdist(r_e, self.cb_e[0], p=2)
        d_h = torch.cdist(r_h, self.cb_h[0], p=2)
        # 产品距离 = 各段距离之和 (数学恒等: argmin 等价于各段独立 argmin)
        d_total = d_s + d_e + d_h
        # STE 量化
        soft_s = F.softmax(-d_s * STE_TEMP, dim=-1)
        soft_e = F.softmax(-d_e * STE_TEMP, dim=-1)
        soft_h = F.softmax(-d_h * STE_TEMP, dim=-1)
        # 选 idx
        idx_s = d_s.argmin(dim=-1)
        idx_e = d_e.argmin(dim=-1)
        idx_h = d_h.argmin(dim=-1)
        # 硬码字 (forward)
        c_s_hard = self.cb_s[0][idx_s]
        c_e_hard = self.cb_e[0][idx_e]
        c_h_hard = self.cb_h[0][idx_h]
        # soft 码字 (backward)
        c_s_ste = soft_s @ self.cb_s[0]
        c_e_ste = soft_e @ self.cb_e[0]
        c_h_ste = soft_h @ self.cb_h[0]
        # STE: hard + (soft - soft.detach())
        c_s = c_s_hard + (c_s_ste - c_s_ste.detach())
        c_e = c_e_hard + (c_e_ste - c_e_ste.detach())
        c_h = c_h_hard + (c_h_ste - c_h_ste.detach())
        # 残差
        r_s_next = r_s - c_s
        r_e_next = r_e - c_e
        r_h_next = r_h - c_h
        return (idx_s, idx_e, idx_h), (c_s, c_e, c_h), (r_s_next, r_e_next, r_h_next), d_total

    def forward(self, r_s, r_e, r_h):
        all_idx_s, all_idx_e, all_idx_h, all_d_total = [], [], [], []
        for layer in range(self.n_layers):
            idx, codes, residuals, d_total = self.forward_layer(r_s, r_e, r_h)
            idx_s, idx_e, idx_h = idx
            all_idx_s.append(idx_s)
            all_idx_e.append(idx_e)
            all_idx_h.append(idx_h)
            all_d_total.append(d_total)
            r_s, r_e, r_h = residuals
        return (all_idx_s, all_idx_e, all_idx_h), (r_s, r_e, r_h), all_d_total


def train_joint(model, sub_s, sub_e, sub_h, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = sub_s.shape[0]
    trajectory = {'step': [], 'recon_global': [], 'recon_seg': [],
                  'commit': [], 'total': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        r_s = sub_s[idx]
        r_e = sub_e[idx]
        r_h = sub_h[idx]

        all_idx, residuals, all_d_total = model(r_s, r_e, r_h)
        all_idx_s, all_idx_e, all_idx_h = all_idx
        r_s_final, r_e_final, r_h_final = residuals

        # 全局 192d 重构损失
        cat_final = torch.cat([r_s_final, r_e_final, r_h_final], dim=-1)
        recon_global = (cat_final ** 2).sum(dim=-1).mean()

        # 每段损失 (用于诊断, 不直接贡献 gradient)
        recon_seg = (r_s_final ** 2).sum(dim=-1).mean() + \
                    (r_e_final ** 2).sum(dim=-1).mean() + \
                    (r_h_final ** 2).sum(dim=-1).mean()

        # commitment loss (each segment × each layer)
        commit = 0.0
        for layer_idx, (idx_s, idx_e, idx_h, dt) in enumerate(
                zip(all_idx_s, all_idx_e, all_idx_h, all_d_total)):
            commit = commit + dt.gather(1, idx_s.unsqueeze(1)).squeeze(1).mean() \
                             + dt.gather(1, idx_e.unsqueeze(1)).squeeze(1).mean() \
                             + dt.gather(1, idx_h.unsqueeze(1)).squeeze(1).mean()

        loss = recon_global + COMMIT_W * commit

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['recon_global'].append(recon_global.item())
            trajectory['recon_seg'].append(recon_seg.item())
            trajectory['commit'].append(commit.item())
            trajectory['total'].append(loss.item())
            print(f'  step {step}: recon_global={recon_global.item():.4f}, '
                  f'recon_seg={recon_seg.item():.4f}, commit={commit.item():.4f}, '
                  f'total={loss.item():.4f}')
    return trajectory


def main():
    print('========== 战线三 3.1 Joint-Flat-3seg ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sub_clipped = load_mckg()
    sub_s = torch.from_numpy(sub_clipped[0][:N_ITEMS_TOY]).float().to(DEVICE)
    sub_e = torch.from_numpy(sub_clipped[1][:N_ITEMS_TOY]).float().to(DEVICE)
    sub_h = torch.from_numpy(sub_clipped[2][:N_ITEMS_TOY]).float().to(DEVICE)
    print(f'sub_s: {sub_s.shape}, mean norm {sub_s.norm(dim=-1).mean():.3f}')
    print(f'sub_e: {sub_e.shape}, mean norm {sub_e.norm(dim=-1).mean():.3f}')
    print(f'sub_h: {sub_h.shape}, mean norm {sub_h.norm(dim=-1).mean():.3f}')

    model = JointFlat3Seg(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj = train_joint(model, sub_s, sub_e, sub_h, n_steps=N_STEPS)

    # 附加: log1p 修复后对比
    print()
    print('=== 附加: log1p 修复后 Joint-Flat-3seg (campaign G1 pass) ===')
    def log1p_norm(t):
        n = t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return t / n * torch.log1p(n)
    sub_s_l1 = log1p_norm(sub_s)
    sub_e_l1 = log1p_norm(sub_e)
    sub_h_l1 = log1p_norm(sub_h)
    print(f'log1p sub_s norm: {sub_s_l1.norm(dim=-1).mean():.3f}, max {sub_s_l1.norm(dim=-1).max():.3f}')
    print(f'log1p sub_e norm: {sub_e_l1.norm(dim=-1).mean():.3f}, max {sub_e_l1.norm(dim=-1).max():.3f}')
    print(f'log1p sub_h norm: {sub_h_l1.norm(dim=-1).mean():.3f}, max {sub_h_l1.norm(dim=-1).max():.3f}')

    torch.manual_seed(SEED)
    model_l1 = JointFlat3Seg(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_l1 = train_joint(model_l1, sub_s_l1, sub_e_l1, sub_h_l1, n_steps=N_STEPS)
    final_recon_global_l1 = traj_l1['recon_global'][-1]
    final_recon_seg_l1 = traj_l1['recon_seg'][-1]

    # 评估: 跟 E2/E3/E4 对比
    final_recon_global = traj['recon_global'][-1]
    final_recon_seg = traj['recon_seg'][-1]
    print()
    print('=== Joint-Flat-3seg 结果 ===')
    print(f'  recon_global (192d norm²) = {final_recon_global:.4f}  ← 与 E4 concat 192d 同口径')
    print(f'  recon_seg (3×64d 各自 norm² 求和) = {final_recon_seg:.4f}  ← 与 E2/E3 同口径')
    print()
    print('=== V4 对照 ===')
    print(f'  B-L2 baseline (fused):         0.275')
    print(f'  E2 Flat-3seg-raw (3×64d 独立): 1.020')
    print(f'  E3 Flat-3seg-L2  (3×64d 独立): 1.160')
    print(f'  E4 concat-192d L2 (joint):     0.482')

    print()
    print('=== 判定 ===')
    print(f'  recon_seg (Joint-Flat-3seg) vs E2/E3 (Flat-3seg-raw/L2): {final_recon_seg:.4f} vs 1.020/1.160')
    print(f'  recon_global vs E4 (concat-192d): {final_recon_global:.4f} vs 0.482')
    print()

    if final_recon_seg < 1.5:
        print('→ 接近 E4 (1.75x): 架构罪在训练解耦, 修训练即可 → PM-RQ 曲率机制获公平审判资格')
    else:
        print('→ 停在 E2/E3 水平 (3.7-4.2x): 分段产品结构本身相对统一空间有本质劣势')

    summary = {
        'task': 'Task #45 Joint-Flat-3seg (战线三 3.1)',
        'config': {
            'N_ITEMS': N_ITEMS_TOY, 'K': K_CODEBOOK, 'N_LAYERS': N_LAYERS,
            'N_STEPS': N_STEPS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
            'NORM_CAP': NORM_CAP, 'STE_TEMP': STE_TEMP, 'COMMIT_W': COMMIT_W,
        },
        'final_losses': {
            'recon_global_192d': final_recon_global,
            'recon_seg_3x64d': final_recon_seg,
            'log1p_recon_global_192d': final_recon_global_l1,
            'log1p_recon_seg_3x64d': final_recon_seg_l1,
        },
        'reference_v4': {
            'B_L2_baseline': 0.275,
            'E2_flat3seg_raw': 1.020,
            'E3_flat3seg_L2': 1.160,
            'E4_concat192_L2': 0.482,
            'F3_log1p_from_task154': 0.3639,
        },
        'judgment': {
            'close_to_E4': bool(final_recon_seg < 1.5),
            'stuck_at_E2_E3': bool(final_recon_seg > 2.5),
            'log1p_below_F3_log1p': bool(final_recon_seg_l1 < 0.3639),
        },
        'trajectory': traj,
        'trajectory_log1p': traj_l1,
    }
    out = OUT_DIR / 'task153_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()