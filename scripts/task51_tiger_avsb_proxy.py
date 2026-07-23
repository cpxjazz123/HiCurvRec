#!/usr/bin/env python3
"""Task #51 战线四端点 — TIGER A vs B 验证 (QMP→Recall 相关性代理)

设计: 用 Stage 2 RQ-VAE 重构损失作为下游 Recall 的代理 (Task #109 已证明 ρ=-0.67)
  A: S5 T5 (QMP 中等, norm 健康)
  B: S1 MCKG (QMP 最差, norm 长尾严重)
  C: S1 MCKG + log1p 修复 (验证 L1 fix 在 TIGER 端是否同样有效)

每个源跑一次 Stage 2 RQ-VAE 训练 + 推断, 测:
  - Stage 2 RQ recon loss (proxy for downstream Recall)
  - Codebook utilization (码本利用率)
  - SID entropy (token 多样性)

判定:
  - T5 < MCKG  → QMP→Recall 假设成立 (用 norm 健康的 embedding 更好)
  - MCKG + log1p < MCKG → L1 fix 端到端有效
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / 'products/task159_tiger_avsb'
OUT_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda:0'
SEED = 42
N_ITEMS = 10000
K = 256  # 与 GRID 默认 codebook 一致
N_LAYERS = 3
N_STEPS = 5000
BATCH = 512
LR = 3e-3


def load_s1_mckg():
    m = torch.load(ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt',
                   weights_only=False, map_location='cpu')
    return m['fused_item'].numpy()


def load_s5_t5():
    p = ROOT / 'logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt'
    t = torch.load(p, weights_only=False, map_location='cpu').float().numpy()
    # PCA 64d
    from numpy.linalg import svd
    U, S, Vt = svd(t[:N_ITEMS], full_matrices=False)
    return (U[:, :64] * S[:64]).astype(np.float32)


def log1p_compress(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n_safe = n.clip(min=1e-6)
    return x / n_safe * np.log1p(n_safe)


class RQ(nn.Module):
    def __init__(self, K=K, dim=64, n_layers=N_LAYERS):
        super().__init__()
        self.K = K
        self.cb_list = nn.ParameterList(
            [nn.Parameter(torch.randn(K, dim) * 0.02) for _ in range(n_layers)])

    def forward(self, x):
        all_idx = []
        residual = x
        for layer in self.cb_list:
            d = torch.cdist(residual, layer, p=2)
            idx = d.argmin(dim=-1)
            c = layer[idx]
            residual = residual - c
            all_idx.append(idx)
        return all_idx, residual


def train_rq(model, x, tag='', n_steps=N_STEPS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    N = x.shape[0]
    trajectory = {'step': [], 'recon': []}
    for step in range(n_steps):
        idx = torch.randint(0, N, (BATCH,))
        batch = x[idx]
        _, residual = model(batch)
        loss = (residual ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == n_steps - 1:
            trajectory['step'].append(step)
            trajectory['recon'].append(loss.item())
            print(f'  [{tag}] step {step}: recon={loss.item():.4f}')
    return trajectory


def eval_rq(model, x, tag=''):
    """Eval mode: 跑全量, 算 RQ recon loss, codebook utilization, SID entropy."""
    model.eval()
    with torch.no_grad():
        all_idx_layers = [[], [], []]
        total_loss = 0.0
        N = x.shape[0]
        for i in range(0, N, BATCH):
            batch = x[i:i+BATCH]
            all_idx, residual = model(batch)
            total_loss += (residual ** 2).sum(dim=-1).sum().item()
            for L in range(3):
                all_idx_layers[L].append(all_idx[L].cpu())
        recon = total_loss / N

        utilizations = []
        entropies = []
        for L in range(3):
            all_idx_L = torch.cat(all_idx_layers[L]).numpy()
            unique, counts = np.unique(all_idx_L, return_counts=True)
            util = len(unique) / K
            p = counts / counts.sum()
            H = -(p * np.log(p + 1e-9)).sum()
            utilizations.append(util)
            entropies.append(H)
    print(f'  [{tag}] recon={recon:.4f}, util={utilizations}, entropy_layer={[f"{h:.2f}" for h in entropies]}')
    return {
        'recon_loss': float(recon),
        'utilization_layer': [float(u) for u in utilizations],
        'entropy_layer': [float(h) for h in entropies],
    }


def main():
    print('========== 战线四端点 — TIGER A vs B 验证 ==========')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sources = {
        'A_S5_T5_raw': load_s5_t5(),
        'B_S1_MCKG_raw': load_s1_mckg()[:N_ITEMS],
        'C_S1_MCKG_log1p': log1p_compress(load_s1_mckg()[:N_ITEMS]),
    }

    results = {}
    for name, e in sources.items():
        print(f'\n=== {name} (shape {e.shape}) ===')
        norm = np.linalg.norm(e, axis=-1)
        print(f'  norm: mean={norm.mean():.3f}, max={norm.max():.3f}')

        x = torch.from_numpy(e).float().to(DEVICE)
        torch.manual_seed(SEED)
        m = RQ(K=K, dim=64, n_layers=N_LAYERS).to(DEVICE)
        traj = train_rq(m, x, tag=name)
        eval_res = eval_rq(m, x, tag=name + '_eval')
        results[name] = {
            'norm_stats': {
                'mean': float(norm.mean()),
                'max': float(norm.max()),
            },
            'trajectory': traj,
            'eval': eval_res,
        }

    # 汇总对比
    print('\n========== A vs B vs C 端点对比 ==========')
    a = results['A_S5_T5_raw']['eval']['recon_loss']
    b = results['B_S1_MCKG_raw']['eval']['recon_loss']
    c = results['C_S1_MCKG_log1p']['eval']['recon_loss']
    print(f'A (T5):   {a:.4f}')
    print(f'B (MCKG): {b:.4f}  ({b/a:.2f}× A)')
    print(f'C (MCKG + log1p): {c:.4f}  ({c/a:.2f}× A)')

    print()
    if a < b:
        print(f'✅ QMP→Recall 假设成立: T5 (norm 健康) RQ recon {a:.4f} < MCKG {b:.4f} ({b/a:.2f}×)')
    if c < b:
        print(f'✅ L1 log1p 修复在 TIGER 端有效: C {c:.4f} < B {b:.4f} ({(b-c)/b*100:.1f}% 改进)')
    else:
        print(f'⚠️ L1 log1p 修复在 TIGER 端未显效: C {c:.4f} ≥ B {b:.4f}')

    print()
    print('Codebook utilization:')
    for name, r in results.items():
        u = r['eval']['utilization_layer']
        print(f'  {name}: {[f"{x:.2f}" for x in u]}')

    print()
    print('SID entropy (token diversity, 高 = 多样性):')
    for name, r in results.items():
        e = r['eval']['entropy_layer']
        print(f'  {name}: {[f"{x:.2f}" for x in e]}')

    summary = {
        'task': 'Task #51 战线四端点 TIGER A vs B 验证',
        'config': {'N_ITEMS': N_ITEMS, 'K': K, 'N_LAYERS': N_LAYERS,
                   'N_STEPS': N_STEPS, 'BATCH': BATCH, 'LR': LR},
        'results': results,
        'comparison': {
            'A_T5': a,
            'B_MCKG': b,
            'C_MCKG_log1p': c,
            'B_over_A': float(b / a),
            'C_over_A': float(c / a),
            'C_improvement_vs_B': float((b - c) / b) if b > 0 else 0,
            'qmp_recall_hypothesis_holds': bool(a < b),
            'l1_fix_effective_end_to_end': bool(c < b),
        },
    }

    out = OUT_DIR / 'task159_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {out}')


if __name__ == '__main__':
    main()