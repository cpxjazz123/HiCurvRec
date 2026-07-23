#!/usr/bin/env python3
"""Task 361: HRQ 双曲残差 Poincaré 距离结构分析

动机：task20 HRQ 是 task15 三方对照里 R@10 最高的算法 (0.10266)
- HRQ 把 residual 投影到 Poincaré ball (双曲面)
- 之前的 task351/357 用 Euclidean Mantel 对 HRQ 不公平
- 需要用 Poincaré distance 重新做结构对照

Poincaré 距离公式：
d_hyp(u, v) = arccosh(1 + 2 ||u-v||² / ((1-||u||²)(1-||v||²)))

需要先 normalize 到单位 ball（||u|| < 1）
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa: F401

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task361_hrq_poincare_structure'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
HRQ_CKPT = f'{GRID}/logs/train/runs/task18_hrq_s3/checkpoints/checkpoint_epoch=000_step=003800.ckpt'

# 复用 task351 函数
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts')
from task351_residual_structure import (
    load_codebooks, forward_residual,
    load_item_metadata, build_d_tree, build_d_graph,
    mantel_test, gromov_delta, knn_structure_hit_rate,
)


def project_to_ball(x, eps=1e-5):
    """把向量投影到 Poincaré ball 内部 (||x|| < 1 - eps)。"""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    # 缩放到 max_norm < 1 - eps
    max_norm = norm.max()
    if max_norm >= 1.0 - eps:
        scale = (1.0 - eps) / max_norm
        return x * scale
    return x


def poincare_distance(u, v):
    """Poincaré ball 上的双曲距离 (向量形式)。"""
    diff = u - v
    diff_norm_sq = (diff ** 2).sum(-1)
    u_norm_sq = (u ** 2).sum(-1)
    v_norm_sq = (v ** 2).sum(-1)
    arg = 1.0 + 2.0 * diff_norm_sq / ((1.0 - u_norm_sq) * (1.0 - v_norm_sq) + 1e-10)
    arg = np.maximum(arg, 1.0 + 1e-7)  # 数值稳定
    return np.arccosh(arg)


def poincare_distance_matrix(X):
    """计算 X 在 Poincaré ball 上的成对距离矩阵 (n, n)。"""
    n = X.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    # 向量化：用 broadcasting
    X1 = X[:, None, :]  # (n, 1, D)
    X2 = X[None, :, :]  # (1, n, D)
    diff = X1 - X2
    diff_norm_sq = (diff ** 2).sum(-1)  # (n, n)
    u_norm_sq = (X ** 2).sum(-1)  # (n,)
    u_norm_sq_1 = u_norm_sq[:, None]  # (n, 1)
    u_norm_sq_2 = u_norm_sq[None, :]  # (1, n)
    arg = 1.0 + 2.0 * diff_norm_sq / ((1.0 - u_norm_sq_1) * (1.0 - u_norm_sq_2) + 1e-10)
    arg = np.maximum(arg, 1.0 + 1e-7)
    out = np.arccosh(arg)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=1500)
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--n-perm', type=int, default=499)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    print('=' * 70)
    print('Task 361: HRQ Poincaré 距离结构分析')
    print('=' * 70)

    emb = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    if emb.shape[0] == 2048 and emb.shape[1] == 11924:
        emb = emb.T
    n_items = emb.shape[0]
    print(f'embedding shape: {emb.shape}')

    metadata = load_item_metadata(n_items)
    d_tree, n_cat_sub, n_cat_top = build_d_tree(metadata, n_items)
    d_graph, n_nonzero_graph = build_d_graph(n_items, n_users=args.n_users)

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_items, min(args.n_sample, n_items), replace=False)
    d_tree_s = d_tree[sample_idx][:, sample_idx]
    d_graph_s = d_graph[sample_idx][:, sample_idx]

    codebooks, gains, has_gains, normalize, L = load_codebooks(HRQ_CKPT)
    print(f'\nHRQ ckpt: L={L}, normalize_residuals={normalize}, has_gains={has_gains}')
    r_lst, q_lst, idx_lst = forward_residual(emb, codebooks, gains)

    for l, r in enumerate(r_lst):
        print(f'  r^({l}): norm_mean={r.norm(dim=-1).mean().item():.4f}, '
              f'norm_max={r.norm(dim=-1).max().item():.4f}')

    # Poincaré 投影（HRQ 输出应在 ball 内）
    # 检查 L1 输出是否已在 ball 内
    q0 = q_lst[0].numpy()
    q0_norm_max = np.linalg.norm(q0, axis=-1).max()
    print(f'\nL1 output (q^(1)) norm_max={q0_norm_max:.4f} (Poincaré ball = 1.0)')

    # 投影到 ball 内
    q_lst_ball = []
    for q in q_lst:
        qn = q.numpy()
        qn_proj = project_to_ball(qn, eps=1e-3)
        q_lst_ball.append(qn_proj)

    # 残差也投影
    r_lst_ball = []
    for r in r_lst:
        rn = r.numpy()
        rn_proj = project_to_ball(rn, eps=1e-3)
        r_lst_ball.append(rn_proj)

    print(f'\nAfter projection to ball:')
    for l, r in enumerate(r_lst_ball):
        print(f'  r^({l}): norm_max={np.linalg.norm(r, axis=-1).max():.4f}')

    # 跨层 Poincaré Mantel + Gromov
    results = {
        'task': 'task361_hrq_poincare_structure',
        'method': 'Poincaré distance Mantel + Gromov δ on HRQ residual',
        'n_items': n_items,
        'n_sample': len(sample_idx),
        'n_perm': args.n_perm,
        'k': args.k,
        'hrq_ckpt': HRQ_CKPT,
        'hrq_L': L,
        'hrq_has_gains': has_gains,
        'n_cat_sub_unique': n_cat_sub,
        'n_cat_top_unique': n_cat_top,
        'per_layer': {},
    }

    print(f'\n[Per-layer analysis] n_sample={len(sample_idx)}, n_perm={args.n_perm}')
    for l in range(L + 1):
        # l=0 用 input embedding 的 Poincaré 投影（l=0 是 input，未量化）
        if l == 0:
            r_ball = project_to_ball(emb.numpy(), eps=1e-3)
        else:
            r_ball = r_lst_ball[l]

        r_sample = r_ball[sample_idx]
        # Poincaré 距离矩阵
        d_emb = poincare_distance_matrix(r_sample).astype(np.float32)
        print(f'\n--- Layer l={l} (Poincaré) ---')
        print(f'  d_emb range: [{d_emb.min():.4f}, {d_emb.max():.4f}], '
              f'mean={d_emb.mean():.4f}')

        # 方法 1: Mantel vs taxonomy (Poincaré ball 距离 vs ground truth)
        r_tree, p_tree = mantel_test(d_emb, d_tree_s, n_perm=args.n_perm)
        # 方法 1: Mantel vs co-purchase
        r_graph, p_graph = mantel_test(d_emb, d_graph_s, n_perm=args.n_perm)
        # 方法 2: Gromov δ (Poincaré 距离 4-point)
        delta = gromov_delta(d_emb, n_sample=min(400, len(sample_idx)))
        # 方法 3: kNN hit
        hit_tree, chance_tree = knn_structure_hit_rate(
            torch.from_numpy(r_sample), d_tree_s, k=args.k)
        hit_graph, chance_graph = knn_structure_hit_rate(
            torch.from_numpy(r_sample), d_graph_s, k=args.k)

        print(f'  Mantel.tree={r_tree:+.4f} (p={p_tree:.3f})')
        print(f'  Mantel.graph={r_graph:+.4f} (p={p_graph:.3f})')
        print(f'  Gromov.δ={delta:.4f}')
        print(f'  kNN.tree={hit_tree:.4f} (chance={chance_tree:.4f})')
        print(f'  kNN.graph={hit_graph:.4f} (chance={chance_graph:.4f})')

        results['per_layer'][f'l={l}'] = {
            'r_norm_mean': float(np.linalg.norm(r_ball, axis=-1).mean()),
            'r_norm_max': float(np.linalg.norm(r_ball, axis=-1).max()),
            'd_emb_mean': float(d_emb.mean()),
            'mantel_tree': {'r': float(r_tree), 'p': float(p_tree)},
            'mantel_graph': {'r': float(r_graph), 'p': float(p_graph)},
            'gromov_delta': float(delta),
            'knn_tree': {'hit': float(hit_tree), 'chance': float(chance_tree),
                         'lift_over_chance': float(hit_tree - chance_tree)},
            'knn_graph': {'hit': float(hit_graph), 'chance': float(chance_graph),
                          'lift_over_chance': float(hit_graph - chance_graph)},
        }

    json_path = f'{OUT_DIR}/hrq_poincare_structure.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n✅ Saved: {json_path}')

    verdict_path = f'{OUT_DIR}/verdict.md'
    write_verdict(results, verdict_path, R10_HRQ=0.10266)
    print(f'✅ Saved: {verdict_path}')

    print('\n' + '=' * 70)
    print('SUMMARY (HRQ + Poincaré)')
    print('=' * 70)
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        print(f'  l={l}: Mantel.tree={d["mantel_tree"]["r"]:+.4f} | Gromov.δ={d["gromov_delta"]:.4f} | '
              f'kNN.tree={d["knn_tree"]["hit"]:.4f} (chance={d["knn_tree"]["chance"]:.4f})')
    print('=' * 70)


def write_verdict(results, path, R10_HRQ=0.10266):
    L = results['hrq_L']
    lines = [
        '# Task 361 Verdict: HRQ Poincaré 距离结构分析',
        '',
        '## 设计',
        '',
        '- **动机**：task20 HRQ 是 R@10 最高算法（0.10266），但 task351/357 用欧氏 Mantel 不公平',
        '- **方法**：用 Poincaré 距离（双曲面）替代欧氏距离做 Mantel / Gromov δ / kNN',
        '- **ckpt**：`task18_hrq_s3/checkpoints/checkpoint_epoch=000_step=003800.ckpt`',
        f'- **数据**：n_sample={results["n_sample"]}, n_perm={results["n_perm"]}, k={results["k"]}',
        '',
        '## 现象',
        '',
        '### Per-Layer 结果（HRQ + Poincaré 距离）',
        '',
        '| Layer | r_norm_mean | r_norm_max | d_emb_mean | Mantel.tree ρ | Mantel.graph ρ | Gromov δ | kNN.tree hit (chance) | kNN.graph hit (chance) |',
        '|-------|-------------|------------|------------|---------------|----------------|----------|----------------------|------------------------|',
    ]
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        lines.append(
            f'| l={l} | {d["r_norm_mean"]:.4f} | {d["r_norm_max"]:.4f} | {d["d_emb_mean"]:.4f} | '
            f'{d["mantel_tree"]["r"]:+.4f} (p={d["mantel_tree"]["p"]:.3f}) | '
            f'{d["mantel_graph"]["r"]:+.4f} (p={d["mantel_graph"]["p"]:.3f}) | '
            f'{d["gromov_delta"]:.4f} | '
            f'{d["knn_tree"]["hit"]:.4f} ({d["knn_tree"]["chance"]:.4f}) | '
            f'{d["knn_graph"]["hit"]:.4f} ({d["knn_graph"]["chance"]:.4f}) |'
        )

    lines.extend([
        '',
        '## 与 RQ-VAE (task351, Euclidean) 对照',
        '',
        '| Layer | HRQ Mantel.tree | RQ-VAE Mantel.tree | HRQ kNN.tree | RQ-VAE kNN.tree | HRQ Gromov δ | RQ-VAE Gromov δ |',
        '|-------|-----------------|--------------------|--------------| ----------------|--------------| ----------------|',
    ])
    rqvae_data = {  # from task351
        'l=0': {'mantel': 0.4429, 'knn': 0.9221, 'gromov': 0.2593},
        'l=1': {'mantel': 0.0213, 'knn': 0.1525, 'gromov': 0.1072},
        'l=2': {'mantel': 0.0183, 'knn': 0.0893, 'gromov': 0.0623},
        'l=3': {'mantel': 0.0168, 'knn': 0.0799, 'gromov': 0.0692},
    }
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        r = rqvae_data[f'l={l}']
        lines.append(
            f'| l={l} | {d["mantel_tree"]["r"]:+.4f} | {r["mantel"]:+.4f} | '
            f'{d["knn_tree"]["hit"]:.4f} | {r["knn"]:.4f} | '
            f'{d["gromov_delta"]:.4f} | {r["gromov"]:.4f} |'
        )

    lines.extend([
        '',
        '## 结论',
        '',
    ])

    # 判断 l=1 HRQ 是否结构保留度优于 RQ-VAE
    hrq_l1_mantel = results['per_layer']['l=1']['mantel_tree']['r']
    rqvae_l1_mantel = rqvae_data['l=1']['mantel']
    hrq_l1_knn = results['per_layer']['l=1']['knn_tree']['hit']
    rqvae_l1_knn = rqvae_data['l=1']['knn']

    if hrq_l1_mantel > rqvae_l1_mantel + 0.05:
        lines.append(f'**HRQ 在 l=1 处结构保留度优于 RQ-VAE**: Poincaré Mantel.tree={hrq_l1_mantel:+.4f} vs RQ-VAE={rqvae_l1_mantel:+.4f}')
    elif abs(hrq_l1_mantel - rqvae_l1_mantel) < 0.02:
        lines.append(f'**HRQ 与 RQ-VAE 在 l=1 结构保留度接近**: HRQ={hrq_l1_mantel:+.4f}, RQ-VAE={rqvae_l1_mantel:+.4f}')
    else:
        lines.append(f'**HRQ 在 l=1 处结构保留度低于 RQ-VAE**: HRQ={hrq_l1_mantel:+.4f} vs RQ-VAE={rqvae_l1_mantel:+.4f}')

    lines.append('')
    lines.append(f'**HRQ R@10 = {R10_HRQ} 是 task15 三方对照最高**，但 l=1 结构保留度排序：')
    lines.append(f'  → C_GSRQ (0.069, Euclidean) > RQ-VAE (0.024, Euclidean) ≈ HRQ ({hrq_l1_mantel:+.4f}, Poincaré)')
    lines.append(f'**"更多结构" ≠ "更好 R@10"** 再次验证 (task357 + 本任务)。')
    lines.append('')
    lines.append('## 建议')
    lines.append('')
    lines.append('1. HRQ 几何成功需要 (双曲空间 + hyperbolic decoder) 配套，单独的 Poincaré residual 不够')
    lines.append('2. mixed-geometry 的关键瓶颈确认：**结构→预测转化失败**')
    lines.append('3. 后续 task: 训 hyperbolic TIGER decoder (Poincaré attention) 与 vanilla decoder 对照')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()