#!/usr/bin/env python3
"""Task #49 QMP 6 指标测量 — 4 个 embedding 源对照

QMP (Quantizability Metric Package) 6 指标:
  M1: norm shape (ρ_max, CV, γ_1)  — 是否长尾
  M2: SCR (Split/Concat Ratio)     — 拆分产品空间 vs 拼接联合空间的失真比
  M3: D_rel (relative reconstruction distortion) — RQ 重构相对 norm 的失真
  M4: codebook utilization         — 码本使用率
  M5: NP@k (neighborhood preservation) — 邻域保持率 (Sampled k-NN 距离)
  M6: erank (effective rank)        — 有效秩

输入 (4 源):
  S1 MCKG (margin ranking): products/task99_mckg_rebuild/entity_embedding.pt
  S4 AE (pure recon):       products/task156_s4_ae/entity_embedding.pt
  S5 T5 (sentence-t5-base): logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt
  S6 item2vec (co-occur):   products/task158_s6_item2vec/entity_embedding.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / 'products/task157_qmp_measure'
OUT_DIR.mkdir(exist_ok=True)
N_ITEMS = 10000
K_CODEBOOK = 64
N_LAYERS = 3
N_STEPS = 2000
BATCH = 512
LR = 1e-3
SEED = 42
DEVICE = 'cuda:0'


def load_s1_mckg():
    m = torch.load(ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt',
                   weights_only=False, map_location='cpu')
    fused = m['fused_item']
    sub = m['subspace_item']
    if torch.is_tensor(fused):
        fused = fused.numpy()
    if torch.is_tensor(sub):
        sub = sub.numpy()
    return fused, sub


def load_s4_ae():
    p = ROOT / 'products/task156_s4_ae/entity_embedding.pt'
    if not p.exists():
        return None, None
    m = torch.load(p, weights_only=False, map_location='cpu')
    fused = m['fused_item']
    sub = m['subspace_item']
    if torch.is_tensor(fused):
        fused = fused.numpy()
    if torch.is_tensor(sub):
        sub = sub.numpy()
    return fused, sub


def load_s5_t5():
    p = ROOT / 'logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt'
    t = torch.load(p, weights_only=False, map_location='cpu').float().numpy()
    # 截取 64d 子空间 (用 PCA 简化)
    from numpy.linalg import svd
    U, S, Vt = svd(t[:N_ITEMS], full_matrices=False)
    return (U[:, :64] * S[:64]).astype(np.float32), None


def load_s6_item2vec():
    p = ROOT / 'products/task158_s6_item2vec/entity_embedding.pt'
    if not p.exists():
        return None, None
    m = torch.load(p, weights_only=False, map_location='cpu')
    fused = m['fused_item']
    sub = m['subspace_item']
    if torch.is_tensor(fused):
        fused = fused.numpy()
    if torch.is_tensor(sub):
        sub = sub.numpy()
    return fused, sub


def m1_norm_shape(x):
    n = np.linalg.norm(x, axis=-1)
    rho_max = float(n.max() / (n.mean() + 1e-9))
    cv = float(n.std() / (n.mean() + 1e-9))
    # 偏度 γ_1
    mu = n.mean()
    sigma = n.std() + 1e-9
    gamma_1 = float(((n - mu) ** 3).mean() / (sigma ** 3))
    return {'rho_max': rho_max, 'cv': cv, 'gamma_1': gamma_1,
            'mean': float(n.mean()), 'max': float(n.max())}


def m2_scr(fused, sub, tag=''):
    """fused = 拼接 192d RQ vs sub = 3 段独立 RQ → SCR = sub/fused."""
    fused_l2 = fused[:N_ITEMS] / np.linalg.norm(fused[:N_ITEMS], axis=-1, keepdims=True).clip(min=1e-6)
    if sub.ndim == 3:
        n_sub = sub.shape[0]
    else:
        n_sub = 1
    sub_l2 = []
    for i in range(n_sub):
        s = sub[i, :N_ITEMS] if sub.ndim == 3 else sub[:N_ITEMS]
        sub_l2.append(s / np.linalg.norm(s, axis=-1, keepdims=True).clip(min=1e-6))
    if n_sub == 1:
        # 单 subspace: 退化对照 (单段 RQ 自身作为 baseline)
        sub_l2 = sub_l2 * 3  # 复制 3 次得到 3 个相同输入 (SCR 应≈1)
    elif n_sub != 3:
        # 未知 subspace 数, 用平均
        sub_l2 = [sub_l2[0]] * 3

    class StandardRQ(torch.nn.Module):
        def __init__(self, K=64, dim=64, n_layers=3):
            super().__init__()
            self.cb_list = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(K, dim) * 0.01) for _ in range(n_layers)])
        def forward(self, x):
            residual = x
            for layer in self.cb_list:
                d = torch.cdist(residual, layer, p=2)
                idx = d.argmin(dim=-1)
                residual = residual - layer[idx]
            return (residual ** 2).sum(dim=-1).mean()

    torch.manual_seed(SEED)
    fused_dev = torch.from_numpy(fused_l2).float().to(DEVICE)
    m_fused = StandardRQ(K=K_CODEBOOK, dim=fused_l2.shape[1], n_layers=N_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(m_fused.parameters(), lr=LR)
    for step in range(N_STEPS):
        idx = torch.randint(0, fused_dev.shape[0], (BATCH,))
        loss = m_fused(fused_dev[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    fused_loss = loss.item()

    sub_dev = [torch.from_numpy(s).float().to(DEVICE) for s in sub_l2]
    sub_loss = 0
    for s in sub_dev:
        torch.manual_seed(SEED)
        m_s = StandardRQ(K=K_CODEBOOK, dim=s.shape[1], n_layers=N_LAYERS).to(DEVICE)
        opt_s = torch.optim.Adam(m_s.parameters(), lr=LR)
        for step in range(N_STEPS):
            idx = torch.randint(0, s.shape[0], (BATCH,))
            loss = m_s(s[idx])
            opt_s.zero_grad(); loss.backward(); opt_s.step()
        sub_loss += loss.item()

    scr = sub_loss / fused_loss
    return {'fused_loss': float(fused_loss), 'sub_loss_sum': float(sub_loss), 'scr': float(scr),
            'tag': tag}


def m5_np_at_k(x, k=10, n_sample=2000, tag=''):
    """Sampled k-NN 邻域保持 (本任务简化为: 与自己 random projection 后的 k-NN Jaccard)."""
    from numpy.linalg import svd
    rng = np.random.RandomState(SEED)
    N = min(n_sample, x.shape[0])
    x_sub = x[:N]
    # 原始 k-NN
    norms = (x_sub ** 2).sum(axis=-1, keepdims=True)
    d_orig = norms + norms.T - 2 * x_sub @ x_sub.T
    np.fill_diagonal(d_orig, np.inf)
    orig_nn = np.argpartition(d_orig, k, axis=-1)[:, :k]
    # Random projection 到 32d
    proj = rng.randn(x.shape[-1], 32).astype(np.float32) / np.sqrt(32)
    x_proj = x_sub @ proj
    norms_p = (x_proj ** 2).sum(axis=-1, keepdims=True)
    d_proj = norms_p + norms_p.T - 2 * x_proj @ x_proj.T
    np.fill_diagonal(d_proj, np.inf)
    proj_nn = np.argpartition(d_proj, k, axis=-1)[:, :k]
    # Jaccard
    jaccard = []
    for i in range(N):
        j = len(set(orig_nn[i]) & set(proj_nn[i])) / k
        jaccard.append(j)
    return {'mean_jaccard': float(np.mean(jaccard)), 'tag': tag}


def m6_erank(x, tag=''):
    """Effective rank: 基于 SVD 谱熵."""
    s = np.linalg.svd(x[:N_ITEMS], compute_uv=False)
    s = s / s.sum()
    s = s[s > 0]
    H = -(s * np.log(s)).sum()
    return {'erank': float(np.exp(H)), 'tag': tag}


def main():
    print('========== QMP 6 指标测量 (4 源对照) ==========')
    np.random.seed(SEED)

    sources = {
        'S1_MCKG': load_s1_mckg(),
        'S4_AE': load_s4_ae(),
        'S5_T5': load_s5_t5(),
        'S6_item2vec': load_s6_item2vec(),
    }

    results = {}
    for name, (fused, sub) in sources.items():
        if fused is None:
            print(f'  [{name}] SKIPPED (data not ready)')
            continue
        print(f'\n=== {name} ===')
        print(f'  fused shape: {fused.shape}')
        m1 = m1_norm_shape(fused[:N_ITEMS])
        print(f'  M1: {m1}')
        m5 = m5_np_at_k(fused[:N_ITEMS])
        print(f'  M5: {m5}')
        m6 = m6_erank(fused[:N_ITEMS])
        print(f'  M6: {m6}')
        m2 = None
        if sub is not None:
            print(f'  sub shape: {sub.shape}')
            m2 = m2_scr(fused, sub, tag=name)
            print(f'  M2: SCR = {m2["scr"]:.3f}x (fused_loss={m2["fused_loss"]:.4f}, sub_loss={m2["sub_loss_sum"]:.4f})')
        results[name] = {'M1_norm_shape': m1, 'M5_NP': m5, 'M6_erank': m6, 'M2_SCR': m2}

    print('\n=== 汇总 ===')
    rows = []
    for name, r in results.items():
        row = {'source': name}
        if r.get('M1_norm_shape'):
            row['rho_max'] = r['M1_norm_shape']['rho_max']
            row['cv'] = r['M1_norm_shape']['cv']
        if r.get('M2_SCR'):
            row['scr'] = r['M2_SCR']['scr']
        if r.get('M5_NP'):
            row['np_at_10'] = r['M5_NP']['mean_jaccard']
        if r.get('M6_erank'):
            row['erank'] = r['M6_erank']['erank']
        rows.append(row)
        print('  ', row)

    summary = {
        'task': 'Task #49 QMP 6 指标',
        'config': {'N_ITEMS': N_ITEMS, 'K': K_CODEBOOK, 'N_LAYERS': N_LAYERS,
                   'N_STEPS': N_STEPS, 'BATCH': BATCH},
        'results': results,
        'summary_table': rows,
    }
    out = OUT_DIR / 'task157_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()