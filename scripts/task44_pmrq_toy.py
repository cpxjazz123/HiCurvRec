#!/usr/bin/env python3
"""Task #44 — PM-RQ Toy 实验 (Phase 1, 第二轮: STE 量化器 + in-batch distance normalize).

第一轮诊断:
  - C 组 PM-RQ 3 段 final_loss=1.926 (vs B=0.275), 7× 退化
  - κ 和 fusion_logits 训练全程不变
  - Root cause: argmin 阻断梯度 + 距离量级不匹配 (sphere=0.34, euclid=1.0, hyperbolic=2.7)

第二轮修复:
  - STE 量化器: soft assignment 让梯度能传到 fusion_logits 和 κ
  - In-batch per-segment distance normalize (per batch 重新计算 mean)
  - 提高 κ 的 lr (×100), 让它能动

实验组:
  B. MCKG-standard (fused_64d normalized, 标准 RQ-VAE)
  C. PM-RQ 3 段 (核心方法, STE 修复版)
  D. PM-RQ 2 段消融 (sphere + euclid, 去 hyperbolic)
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
STE_TEMP = 5.0  # soft assignment temperature


def load_mckg_subspaces():
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False, map_location='cpu')
    sub = mckg['subspace_item'].numpy()
    fused = mckg['fused_item'].numpy()
    d0_kappas = [+0.8446, 0.0, -1.0586]

    sub_clipped = []
    for i, (name, k) in enumerate(zip(['sphere', 'euclid', 'hyperbolic'], d0_kappas)):
        norms = np.linalg.norm(sub[i], axis=1, keepdims=True)
        clip_factor = np.minimum(NORM_CAP / norms.clip(min=1e-6), 1.0)
        sub_clipped.append(sub[i] * clip_factor)

    fused_norm = fused / np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-6)

    return {
        'sub_sphere': torch.from_numpy(sub_clipped[0]).float(),
        'sub_euclid': torch.from_numpy(sub_clipped[1]).float(),
        'sub_hyperbolic': torch.from_numpy(sub_clipped[2]).float(),
        'fused_normalized': torch.from_numpy(fused_norm).float(),
        'd0_kappas': d0_kappas,
    }


def stereographic_dist(x, c, kappa):
    from task_artifacts.scripts.mckg_model.stereographic import dist_kappa
    B, D = x.shape
    K = c.shape[0]
    x_e = x.unsqueeze(1).expand(B, K, D)
    c_e = c.unsqueeze(0).expand(B, K, D)
    return dist_kappa(x_e.reshape(-1, D), c_e.reshape(-1, D), kappa).reshape(B, K)


# ===================== PM-RQ v2 (STE 版) =====================

class PMRQv2(nn.Module):
    def __init__(self, K=64, dim=64, n_layers=3,
                 init_kappa_sphere=0.8446, init_kappa_hyperbolic=-1.0586,
                 use_hyperbolic=True):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers
        self.use_hyperbolic = use_hyperbolic

        self.kappa_sphere = nn.Parameter(torch.tensor(float(init_kappa_sphere)))
        self.kappa_hyperbolic = nn.Parameter(torch.tensor(float(init_kappa_hyperbolic)))

        n_logits = 3 if use_hyperbolic else 2
        self.fusion_logits = nn.Parameter(torch.ones(n_logits))

        self.cb_sphere = nn.Parameter(torch.randn(K, dim) * 0.01)
        self.cb_euclid = nn.Parameter(torch.randn(K, dim) * 0.01)
        if use_hyperbolic:
            self.cb_hyperbolic = nn.Parameter(torch.randn(K, dim) * 0.01)

    def forward_layer(self, r_s, r_e, r_h=None):
        """STE + in-batch normalize."""
        # 1. 算各段距离 (raw)
        d_s = stereographic_dist(r_s, self.cb_sphere, self.kappa_sphere)
        d_e = torch.cdist(r_e, self.cb_euclid, p=2)
        if self.use_hyperbolic:
            d_h = stereographic_dist(r_h, self.cb_hyperbolic, self.kappa_hyperbolic)
            d_list = [d_s, d_e, d_h]
        else:
            d_list = [d_s, d_e]

        # 2. Per-segment normalize (in-batch mean, 不传梯度)
        d_list_norm = [di / (di.mean().detach() + 1e-6) for di in d_list]

        # 3. fusion (weighted average)
        w = F.softmax(self.fusion_logits, dim=0)
        d_total = sum(wi * di for wi, di in zip(w, d_list_norm))

        # 4. argmin 选 idx
        idx = d_total.argmin(dim=-1)

        # 5. STE 码字 (各段独立)
        soft_s = F.softmax(-d_s * STE_TEMP, dim=-1)
        soft_e = F.softmax(-d_e * STE_TEMP, dim=-1)
        c_s_hard = self.cb_sphere[idx]
        c_e_hard = self.cb_euclid[idx]
        c_s_ste = soft_s @ self.cb_sphere
        c_e_ste = soft_e @ self.cb_euclid
        c_s = c_s_hard + (c_s_ste - c_s_ste.detach())
        c_e = c_e_hard + (c_e_ste - c_e_ste.detach())
        if self.use_hyperbolic:
            soft_h = F.softmax(-d_h * STE_TEMP, dim=-1)
            c_h_hard = self.cb_hyperbolic[idx]
            c_h_ste = soft_h @ self.cb_hyperbolic
            c_h = c_h_hard + (c_h_ste - c_h_ste.detach())

        # 6. 残差 (欧氏减法, 简化版)
        r_s_next = r_s - c_s
        r_e_next = r_e - c_e
        if self.use_hyperbolic:
            r_h_next = r_h - c_h
        return idx, (c_s, c_e, c_h if self.use_hyperbolic else None), (r_s_next, r_e_next, r_h_next if self.use_hyperbolic else None)

    def forward(self, r_s, r_e, r_h=None):
        all_idx, all_codes = [], []
        for layer in range(self.n_layers):
            idx, codes, residuals = self.forward_layer(r_s, r_e, r_h)
            all_idx.append(idx)
            all_codes.append(codes)
            r_s, r_e, r_h = residuals
        return all_idx, all_codes, (r_s, r_e, r_h)


class StandardRQ(nn.Module):
    def __init__(self, K=64, dim=64, n_layers=3):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers
        self.cb_list = nn.ParameterList([nn.Parameter(torch.randn(K, dim) * 0.01) for _ in range(n_layers)])

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


def train_pmrq(model, sub_s, sub_e, sub_h, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR, use_hyperbolic=True):
    # κ 高 lr, 其他正常 lr
    kappa_params = [model.kappa_sphere]
    if use_hyperbolic:
        kappa_params.append(model.kappa_hyperbolic)
    other_params = [p for n, p in model.named_parameters() if 'kappa' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': kappa_params, 'lr': lr * 50},
    ])

    N = sub_s.shape[0]
    trajectory = {'step': [], 'kappa_s': [], 'kappa_h': [],
                  'w_s': [], 'w_e': [], 'w_h': [], 'recon_mse': []}

    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        r_s = sub_s[idx]
        r_e = sub_e[idx]
        r_h = sub_h[idx] if use_hyperbolic else None

        all_idx, all_codes, residuals = model(r_s, r_e, r_h)
        # recon loss: sum of ||final residual||^2 per segment
        r_s_final, r_e_final, r_h_final = residuals
        loss = (r_s_final ** 2).sum(dim=-1).mean() + (r_e_final ** 2).sum(dim=-1).mean()
        if use_hyperbolic:
            loss = loss + (r_h_final ** 2).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == n_steps - 1:
            with torch.no_grad():
                w = F.softmax(model.fusion_logits, dim=0).cpu().tolist()
            trajectory['step'].append(step)
            trajectory['kappa_s'].append(model.kappa_sphere.item())
            trajectory['kappa_h'].append(model.kappa_hyperbolic.item() if use_hyperbolic else 0.0)
            trajectory['w_s'].append(w[0])
            trajectory['w_e'].append(w[1])
            trajectory['w_h'].append(w[2] if use_hyperbolic else 0.0)
            trajectory['recon_mse'].append(loss.item())
            print(f'  step {step}: loss={loss.item():.4f}, κ_s={model.kappa_sphere.item():+.4f}, κ_h={model.kappa_hyperbolic.item() if use_hyperbolic else 0.0:+.4f}, w=[{w[0]:.3f}, {w[1]:.3f}, {w[2] if use_hyperbolic else 0.0:.3f}]')
    return trajectory


def train_standard(model, x, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = x.shape[0]
    trajectory = {'step': [], 'recon_mse': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = x[idx]
        all_idx, residual = model(batch)
        loss = (residual ** 2).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['recon_mse'].append(loss.item())
            print(f'  step {step}: loss={loss.item():.4f}')
    return trajectory


def main():
    print('========== Task #44 Phase 1 PM-RQ Toy (第二轮: STE) ==========')
    print(f'GPU: {DEVICE}')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = load_mckg_subspaces()
    sub_s = data['sub_sphere'][:N_ITEMS_TOY].to(DEVICE)
    sub_e = data['sub_euclid'][:N_ITEMS_TOY].to(DEVICE)
    sub_h = data['sub_hyperbolic'][:N_ITEMS_TOY].to(DEVICE)
    fused = data['fused_normalized'][:N_ITEMS_TOY].to(DEVICE)

    print(f'sub_s: {sub_s.shape}, mean norm {sub_s.norm(dim=-1).mean():.3f}')
    print(f'sub_e: {sub_e.shape}, mean norm {sub_e.norm(dim=-1).mean():.3f}')
    print(f'sub_h: {sub_h.shape}, mean norm {sub_h.norm(dim=-1).mean():.3f}')
    print(f'fused: {fused.shape}, mean norm {fused.norm(dim=-1).mean():.3f}')

    results = {}

    print('\n========== B 组: MCKG-standard ==========')
    model_b = StandardRQ(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS).to(DEVICE)
    traj_b = train_standard(model_b, fused, n_steps=N_STEPS)
    results['B_standard_fused'] = {'trajectory': traj_b, 'final_loss': traj_b['recon_mse'][-1]}
    torch.save(model_b.state_dict(), OUT_DIR / 'model_B_standard_v2.pt')

    print('\n========== C 组: PM-RQ 3 段 (STE) ==========')
    model_c = PMRQv2(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS, use_hyperbolic=True).to(DEVICE)
    print(f'  init κ_s={model_c.kappa_sphere.item():+.4f}, κ_h={model_c.kappa_hyperbolic.item():+.4f}, fusion_logits={model_c.fusion_logits.cpu().tolist()}')
    traj_c = train_pmrq(model_c, sub_s, sub_e, sub_h, n_steps=N_STEPS, use_hyperbolic=True)
    results['C_pmrq_3seg'] = {'trajectory': traj_c, 'final_loss': traj_c['recon_mse'][-1]}
    torch.save(model_c.state_dict(), OUT_DIR / 'model_C_pmrq_3seg_v2.pt')

    print('\n========== D 组: PM-RQ 2 段消融 ==========')
    model_d = PMRQv2(K=K_CODEBOOK, dim=64, n_layers=N_LAYERS, use_hyperbolic=False).to(DEVICE)
    print(f'  init κ_s={model_d.kappa_sphere.item():+.4f}, fusion_logits={model_d.fusion_logits.cpu().tolist()}')
    traj_d = train_pmrq(model_d, sub_s, sub_e, None, n_steps=N_STEPS, use_hyperbolic=False)
    results['D_pmrq_2seg'] = {'trajectory': traj_d, 'final_loss': traj_d['recon_mse'][-1]}
    torch.save(model_d.state_dict(), OUT_DIR / 'model_D_pmrq_2seg_v2.pt')

    print('\n========== 总结 ==========')
    print(f'B (MCKG-standard): final_loss = {results["B_standard_fused"]["final_loss"]:.4f}')
    print(f'C (PM-RQ 3 段):    final_loss = {results["C_pmrq_3seg"]["final_loss"]:.4f}')
    print(f'D (PM-RQ 2 段):    final_loss = {results["D_pmrq_2seg"]["final_loss"]:.4f}')
    print()
    w_c = F.softmax(model_c.fusion_logits, dim=0).cpu().tolist()
    print(f'C 训练后 fusion 权重: w_s={w_c[0]:.3f}, w_e={w_c[1]:.3f}, w_h={w_c[2]:.3f}')
    print(f'C 训练后 κ: κ_s={model_c.kappa_sphere.item():+.4f}, κ_h={model_c.kappa_hyperbolic.item():+.4f}')
    w_d = F.softmax(model_d.fusion_logits, dim=0).cpu().tolist()
    print(f'D 训练后 fusion 权重: w_s={w_d[0]:.3f}, w_e={w_d[1]:.3f}')

    summary = {
        'task': 'Task #44 Phase 1 PM-RQ Toy v2 (STE)',
        'config': {
            'N_ITEMS': N_ITEMS_TOY, 'K': K_CODEBOOK, 'N_LAYERS': N_LAYERS,
            'N_STEPS': N_STEPS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
            'NORM_CAP': NORM_CAP, 'STE_TEMP': STE_TEMP,
            'D0_KAPPAS': data['d0_kappas'],
        },
        'final_losses': {
            'B': results['B_standard_fused']['final_loss'],
            'C': results['C_pmrq_3seg']['final_loss'],
            'D': results['D_pmrq_2seg']['final_loss'],
        },
        'C_final_weights': w_c,
        'C_final_kappas': [model_c.kappa_sphere.item(), model_c.kappa_hyperbolic.item()],
        'D_final_weights': w_d,
        'D_final_kappa_sphere': model_d.kappa_sphere.item(),
        'B_trajectory': results['B_standard_fused']['trajectory'],
        'C_trajectory': results['C_pmrq_3seg']['trajectory'],
        'D_trajectory': results['D_pmrq_2seg']['trajectory'],
        'phase1_decision': {},
    }
    summary['phase1_decision']['go_to_phase2'] = summary['final_losses']['C'] <= summary['final_losses']['B'] * 1.05
    summary['phase1_decision']['reason'] = f'C {summary["final_losses"]["C"]:.4f} vs B {summary["final_losses"]["B"]:.4f}'
    summary['phase1_decision']['D_vs_C_diff_pct'] = (summary['final_losses']['D'] - summary['final_losses']['C']) / max(summary['final_losses']['C'], 1e-6) * 100
    summary['phase1_decision']['hyperbolic_necessary'] = abs(summary['phase1_decision']['D_vs_C_diff_pct']) > 10

    out = OUT_DIR / 'task44_phase1_v2_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()
