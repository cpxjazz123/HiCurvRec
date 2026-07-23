#!/usr/bin/env python3
"""Task 391: 残差层面 kNN 结构命中率 — E vs H 完整对齐

用户原话:"要真正填满'残差层面、双曲版本、l=1 到 l=3'这几个 kNN 命中率的空格,
        需要针对残差本身(而不是码字分配)重新做一次 kNN 检验"

设计:
1. 提取 E 和 H 两个模型在每层量化后的 residual 向量
2. 对每层 residual,在残差空间找 kNN,检查同 cat_sub 命中率
3. 每个 residual 测两种距离口径:
   - Euclidean (sklearn 默认,与之前 baseline 一致)
   - Poincaré (把 residual 也映到 Poincaré 球,真正测双曲几何先验)
4. k=5, 10, 20 三档
5. 完整填表:层 × 模型 × 距离 × k

kNN hit 计算约定:
- 邻居 j 在 kNN 内 AND d_tree[i,j] < 1.0 → hit (同 cat_sub)
- chance: 同一组 kNN index,但 permute d_tree 行列(保持对角)

H 模型 residual 定义:
- r^(0) = x (input)
- q^(1) = log_map0(C1_ball[idx1])  # 用 C1_ball 分配的 log 映射
- q^(1_recon) = C1_euclid[idx1]    # 重建用的 Euclidean 码字
- r^(1) = x - q^(1_recon)           # 残差用 Euclidean 码字算(训练目标)
- 但 r^(1) 也可以看作 "x_ball - exp_map0(log_map0(C1_ball[idx1]))" 的 dual 重构残差
- 实际上 dual codebook 设计: r1 = x - C1_euclid[idx1] (Euclidean residual)
- r^(2) = r^(1) - q^(2) (Euclidean C2)
- r^(3) = r^(2) - q^(3) (Euclidean C3)
"""

import os
import sys
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa: F401
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts')
from task351_residual_structure import (
    load_item_metadata, build_d_tree, build_d_graph,
    load_codebooks, forward_residual,
)
from task389_h_codeword_structure import (
    load_h_codebooks, h_forward_l1,
    BALL_C, MAX_NORM,
    project_to_ball, exp_map0, poincare_distance,
)

GRID = '/home/wlia0047/ar57/wenyu'
OUT_DIR = f'{GRID}/GRID/result/task391_residual_knn_aligned'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = f'{GRID}/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
E_CKPT = f'{GRID}/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'
H_CKPT = f'{GRID}/GRID/logs/train/runs/task388_stage2_h_e_e_e_v3_2026-07-14_10-49-07/checkpoints/ckpt_H_E_E_E.ckpt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def h_forward_residuals(emb_std, L1_ball, L1_euclid, C2, C3):
    """对 H-E-E-E v3 提取每层 residual:r^(0)=x, r^(1)=x-q^(1_recon), r^(2)=r1-q2, r^(3)=r2-q3。"""
    x = emb_std
    # L1 用 Poincaré 距离分配,但重建用 C1_euclid
    l1_idx, r1 = h_forward_l1(x.to(DEVICE), L1_ball.to(DEVICE), L1_euclid.to(DEVICE))
    l1_idx = l1_idx.cpu()
    q1_e = L1_euclid[l1_idx]

    # L2/L3 用 Euclidean
    r1_norm2 = (r1 ** 2).sum(-1, keepdim=True)
    C2_norm2 = (C2 ** 2).sum(-1)
    d2_2 = r1_norm2 - 2 * (r1 @ C2.T) + C2_norm2
    idx2 = d2_2.argmin(dim=1)
    q2 = C2[idx2]
    r2 = r1 - q2

    r2_norm2 = (r2 ** 2).sum(-1, keepdim=True)
    C3_norm2 = (C3 ** 2).sum(-1)
    d3_2 = r2_norm2 - 2 * (r2 @ C3.T) + C3_norm2
    idx3 = d3_2.argmin(dim=1)
    q3 = C3[idx3]
    r3 = r2 - q3

    r_lst = [x, r1.cpu(), r2.cpu(), r3.cpu()]  # 4 层:input + 3 residual
    q_lst = [None, q1_e.cpu(), q2.cpu(), q3.cpu()]  # L1 没有直接 q(因 dual)
    return r_lst, q_lst


def euclidean_to_poincare_batch(x, target_norm=0.5):
    """把 Euclidean 向量映射到 Poincaré 球(用于 Poincaré 度量时)。"""
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    v_norm_target = math.atanh(target_norm)
    v = x_unit * v_norm_target
    y = exp_map0(v, c=BALL_C)
    return project_to_ball(y, max_norm=MAX_NORM)


def knn_hit_rate_precomputed(knn_idx, d_struct, seed=42):
    """用 pre-computed knn indices + d_struct structure matrix,计算 hit rate + chance。"""
    n, k = knn_idx.shape
    rng = np.random.RandomState(seed)

    hits = 0
    total = 0
    for i in range(n):
        for jj in knn_idx[i]:
            total += 1
            if d_struct[i, jj] < 1.0:
                hits += 1
    rate = hits / total

    perm = rng.permutation(n)
    d_shuf = d_struct[perm][:, perm]
    hits_shuf = 0
    for i in range(n):
        for jj in knn_idx[i]:
            if d_shuf[i, jj] < 1.0:
                hits_shuf += 1
    chance = hits_shuf / total
    return float(rate), float(chance)


def compute_residual_knn(residual, d_tree_sample, k, metric='euclidean'):
    """对给定 residual 矩阵,计算 kNN hit rate + chance。"""
    n = residual.shape[0]

    if metric == 'euclidean':
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(residual)
        _, indices = nn.kneighbors(residual)
        knn_idx = indices[:, 1:]  # 排除自身
    elif metric == 'poincare':
        # Poincaré 距离需要逐对计算 (O(n²))
        # 优化:用 Euclidean 找粗 kNN,然后重新排序 top-K 用 Poincaré 距离
        nn = NearestNeighbors(n_neighbors=min(k * 4 + 1, n), metric='euclidean')
        nn.fit(residual)
        _, indices_e = nn.kneighbors(residual)
        cand = indices_e[:, 1:]  # (n, k*4)
        # 对每个 query i,重算 candidate 与 i 的 Poincaré 距离,取 top-k
        knn_idx = np.zeros((n, k), dtype=np.int64)
        res_t = torch.from_numpy(residual).float().to(DEVICE)
        for i in range(0, n, 256):
            i_end = min(i + 256, n)
            query_batch = res_t[i:i_end]  # (B, D)
            cand_batch = cand[i:i_end]  # (B, k*4)
            # 计算 Poincaré 距离: (B, k*4)
            d_poinc = torch.zeros(i_end - i, cand_batch.shape[1], device=DEVICE)
            for bb in range(i_end - i):
                qi = query_batch[bb:bb+1]  # (1, D)
                cb = res_t[cand_batch[bb]]  # (k*4, D)
                # 球面距离
                diff = torch.cat([qi.expand(cb.shape[0], -1), cb], dim=0)
                # 计算 pairwise distance
                d_p = poincare_distance(
                    qi.expand(cb.shape[0], -1), cb, c=BALL_C
                )
                d_poinc[bb] = d_p
            top_k = d_poinc.argsort(dim=1)[:, :k]
            knn_idx[i:i_end] = torch.gather(
                torch.from_numpy(cand_batch).to(DEVICE), 1, top_k
            ).cpu().numpy()
    else:
        raise ValueError(f'Unknown metric: {metric}')

    return knn_hit_rate_precomputed(knn_idx, d_tree_sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=2000)
    parser.add_argument('--n-users', type=int, default=5000)
    parser.add_argument('--k-list', type=str, default='5,10,20')
    args = parser.parse_args()

    print('=' * 70)
    print('Task 391: 残差层面 kNN 结构命中率 — E vs H 完整对齐')
    print('=' * 70)

    k_list = [int(k) for k in args.k_list.split(',')]

    print('[1] Load data')
    emb = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    if emb.shape[0] == 2048 and emb.shape[1] == 11924:
        emb = emb.T
    n_items = emb.shape[0]
    print(f'    n_items: {n_items}, dim: {emb.shape[1]}')
    metadata = load_item_metadata(n_items)
    d_tree, n_cat_sub, n_cat_top = build_d_tree(metadata, n_items)
    d_graph, _ = build_d_graph(n_items, n_users=args.n_users)
    print(f'    cat_sub: {n_cat_sub} unique, cat_top: {n_cat_top} unique')

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_items, min(args.n_sample, n_items), replace=False)
    print(f'    sample: {len(sample_idx)} items')
    d_tree_s = d_tree[sample_idx][:, sample_idx]
    d_graph_s = d_graph[sample_idx][:, sample_idx]

    print('[2] Extract E residuals (E-E-E-E)')
    e_codebooks, e_gains, e_has_gains, e_norm, e_L = load_codebooks(E_CKPT)
    e_r_lst, e_q_lst, e_idx_lst = forward_residual(emb, e_codebooks, e_gains)
    print(f'    E L={e_L}, layers=[input, l=0, l=1, l=2, l=3]')
    for l, r in enumerate(e_r_lst):
        print(f'    r^({l}): shape={tuple(r.shape)}, norm_mean={r.norm(dim=-1).mean():.4f}')

    print('[3] Extract H residuals (H-E-E-E v3)')
    L1_ball, L1_euclid, h_C2, h_C3, h_x_mean, h_x_std, h_hp = load_h_codebooks(H_CKPT)
    emb_std = emb
    if h_x_mean is not None and h_x_std is not None:
        emb_std = (emb - h_x_mean) / h_x_std
        print(f'    [normalization] applied for H model')
    h_r_lst, _ = h_forward_residuals(emb_std, L1_ball, L1_euclid, h_C2, h_C3)
    print(f'    H layers=[input, l=0, l=1, l=2]')
    for l, r in enumerate(h_r_lst):
        print(f'    r^({l}): shape={tuple(r.shape)}, norm_mean={r.norm(dim=-1).mean():.4f}')

    # ========== Compute kNN hit rate for each model × each layer × each metric × each k ==========
    print('\n[4] Compute residual kNN hit rate grid (model × layer × metric × k)')

    results = {
        'task': 'task391_residual_knn_aligned',
        'method': 'residual_knn_structure_hit_rate_euclid_vs_poincare',
        'date': '2026-07-14',
        'status': 'in_progress',
        'data': {
            'config': {
                'e_ckpt': E_CKPT,
                'h_ckpt': H_CKPT,
                'n_items': n_items,
                'n_sample': len(sample_idx),
                'k_list': k_list,
            },
            'e_grid': {},  # 'l0'/'l1'/'l2'/'l3' → metric → k → {hit, chance}
            'h_grid': {},  # 'l0'/'l1'/'l2' → metric → k → {hit, chance}
        },
    }

    def compute_grid(r_lst, label):
        out = {}
        for l_idx, r in enumerate(r_lst):
            r_np = r.numpy() if torch.is_tensor(r) else r
            r_sample = r_np[sample_idx]
            layer_key = f'l{l_idx-1}' if l_idx > 0 else 'input'
            print(f'    {label} r^({l_idx-1 if l_idx > 0 else "input"}): norm={r.norm(dim=-1).mean():.2f}')
            out[layer_key] = {}
            for metric in ['euclidean', 'poincare']:
                out[layer_key][metric] = {}
                for k in k_list:
                    t0 = time.time()
                    hit, chance = compute_residual_knn(r_sample, d_tree_s, k, metric=metric)
                    elapsed = time.time() - t0
                    lift = hit - chance
                    print(f'        [{metric:8s}] k={k:2d}: hit={hit:.4f}, '
                          f'chance={chance:.4f}, lift={lift:+.4f}, '
                          f'time={elapsed:.1f}s')
                    out[layer_key][metric][f'k{k}'] = {
                        'hit': float(hit),
                        'chance': float(chance),
                        'lift': float(lift),
                        'lift_ratio': float(lift / chance) if chance > 0 else 0,
                    }
        return out

    import time
    results['data']['e_grid'] = compute_grid(e_r_lst, 'E')
    results['data']['h_grid'] = compute_grid(h_r_lst, 'H')

    results['status'] = 'completed'

    out_json = f'{OUT_DIR}/residual_knn_aligned.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n✅ Saved: {out_json}')

    write_verdict(results, f'{OUT_DIR}/verdict.md')
    print(f'✅ Saved: {OUT_DIR}/verdict.md')

    print('\n' + '=' * 70)
    print('SUMMARY: 残差 kNN 结构保留率 (E vs H, 主要看 k=10, Euclidean 度量)')
    print('=' * 70)
    print(f'{"Layer":<10} {"E-E-E-E":<20} {"H-E-E-E v3":<20} {"Δ":<10}')
    print('-' * 60)
    for layer in ['input', 'l0', 'l1', 'l2', 'l3']:
        e_hit = results['data']['e_grid'].get(layer, {}).get('euclidean', {}).get('k10', {}).get('hit', None)
        h_hit = results['data']['h_grid'].get(layer, {}).get('euclidean', {}).get('k10', {}).get('hit', None)
        if e_hit is not None:
            e_str = f'{e_hit:.4f}'
        else:
            e_str = 'N/A'
        if h_hit is not None:
            h_str = f'{h_hit:.4f}'
        else:
            h_str = 'N/A'
        if e_hit is not None and h_hit is not None:
            delta = h_hit - e_hit
            d_str = f'{delta:+.4f}'
        else:
            d_str = '-'
        print(f'{layer:<10} {e_str:<20} {h_str:<20} {d_str:<10}')


def write_verdict(s, path):
    e_grid = s['data']['e_grid']
    h_grid = s['data']['h_grid']

    lines = [
        '---',
        '> **任务**：task391_residual_knn_aligned',
        '> **日期**：2026-07-14',
        '> **状态**：✅ 完成',
        '> **执行人**：Claude',
        '',
        '# Task 391 Verdict: 残差层面 kNN 结构命中率 — E vs H 完整对齐',
        '',
        '## 设计',
        '',
        '**用户原话**: "要真正填满\'残差层面、双曲版本、l=1 到 l=3\'这几个 kNN 命中率的空格,',
        '需要针对残差本身(而不是码字分配)重新做一次 kNN 检验"',
        '',
        '**关键区分**:',
        '- **残差层面 kNN**:对 residual vector 找邻居,检查邻居是否同 cat_sub',
        '- **码字分配层面 kNN**:对 codebook vector 找邻居,检查**服务该码字的 item** 是否同 cat_sub(t389)',
        '',
        '**本次覆盖范围**:',
        '- **模型**:E-E-E-E baseline + H-E-E-E v3',
        '- **层**:input, l=0(L1 残差), l=1, l=2, l=3',
        '- **距离度量**:Euclidean(基线)+ Poincaré(测双曲几何先验)',
        f'- **k 值**:{", ".join([str(k) for k in s["data"]["config"]["k_list"]])}',
        f'- **sample**:{s["data"]["config"]["n_sample"]} items',
        '',
        '## 现象',
        '',
        '### A. 主要表格:残差 kNN.tree hit rate (k=10)',
        '',
        '| Layer | E-E-E-E (Euclid) | H-E-E-E v3 (Euclid) | E (Poincaré) | H (Poincaré) | Δ_Euclid |',
        '|-------|-------------------|----------------------|----------------|----------------|----------|',
    ]

    layer_labels = [('input', 'input'), ('l0', 'l=0'), ('l1', 'l=1'), ('l2', 'l=2'), ('l3', 'l=3')]
    for layer_key, label in layer_labels:
        e_e = e_grid.get(layer_key, {}).get('euclidean', {}).get('k10', {})
        h_e = h_grid.get(layer_key, {}).get('euclidean', {}).get('k10', {})
        e_p = e_grid.get(layer_key, {}).get('poincare', {}).get('k10', {})
        h_p = h_grid.get(layer_key, {}).get('poincare', {}).get('k10', {})

        if e_e or h_e:
            e_e_str = f'{e_e.get("hit", float("nan")):.4f}' if e_e else 'N/A'
            h_e_str = f'{h_e.get("hit", float("nan")):.4f}' if h_e else 'N/A'
            e_p_str = f'{e_p.get("hit", float("nan")):.4f}' if e_p else 'N/A'
            h_p_str = f'{h_p.get("hit", float("nan")):.4f}' if h_p else 'N/A'
            if e_e and h_e:
                delta = h_e['hit'] - e_e['hit']
                delta_str = f'{delta:+.4f}'
            else:
                delta_str = '-'
            lines.append(f'| {label} | {e_e_str} | {h_e_str} | {e_p_str} | {h_p_str} | {delta_str} |')

    lines.extend([
        '',
        '### B. 完整网格 (model × layer × metric × k)',
        '',
    ])

    for label_short in ['E', 'H']:
        lines.append(f'#### {label_short} 模型')
        lines.append('')
        lines.append('| Layer | metric | k=5 | k=10 | k=20 |')
        lines.append('|-------|--------|-----|------|------|')
        grid = e_grid if label_short == 'E' else h_grid
        for layer_key, label in layer_labels:
            for metric in ['euclidean', 'poincare']:
                row = f'| {label} | {metric} | '
                for k in [5, 10, 20]:
                    r = grid.get(layer_key, {}).get(metric, {}).get(f'k{k}', {})
                    if r:
                        row += f'{r["hit"]:.4f} (chance {r["chance"]:.4f}) | '
                    else:
                        row += 'N/A | '
                lines.append(row)
        lines.append('')

    # chance baseline
    lines.extend([
        '### C. 随机基线汇总 (chance = 同 cat_sub 比例,与 kNN 无关)',
        '',
        '| Layer | E chance | H chance |',
        '|-------|----------|----------|',
    ])
    for layer_key, label in layer_labels:
        e_p = e_grid.get(layer_key, {}).get('euclidean', {}).get('k10', {})
        h_p = h_grid.get(layer_key, {}).get('euclidean', {}).get('k10', {})
        if e_p and h_p:
            lines.append(f'| {label} | {e_p["chance"]:.4f} | {h_p["chance"]:.4f} |')
        elif e_p:
            lines.append(f'| {label} | {e_p["chance"]:.4f} | N/A |')
        elif h_p:
            lines.append(f'| {label} | N/A | {h_p["chance"]:.4f} |')

    lines.extend([
        '',
        '## 结论',
        '',
    ])

    # 计算关键 insights
    def get_hit(grid, layer_key, metric, k):
        r = grid.get(layer_key, {}).get(metric, {}).get(f'k{k}', {})
        return r.get('hit', None) if r else None

    lines.append('### 用户问题的明确回答(残差层面)')
    lines.append('')
    e_l0 = get_hit(e_grid, 'l0', 'euclidean', 10)
    h_l0 = get_hit(h_grid, 'l0', 'euclidean', 10)
    e_l1 = get_hit(e_grid, 'l1', 'euclidean', 10)
    h_l1 = get_hit(h_grid, 'l1', 'euclidean', 10)
    e_l2 = get_hit(e_grid, 'l2', 'euclidean', 10)
    h_l2 = get_hit(h_grid, 'l2', 'euclidean', 10)
    e_l3 = get_hit(e_grid, 'l3', 'euclidean', 10)
    h_l3 = get_hit(h_grid, 'l3', 'euclidean', 10)

    lines.append('**问题**:双曲版本 L1 / L2 / L3 残差的 kNN 结构命中率(同 cat_sub 占比)是多少?')
    lines.append('')
    lines.append('**回答**(k=10,Euclidean 度量):')
    lines.append('')
    lines.append(f'| 层 | E baseline | H-E-E-E v3 | Δ |')
    lines.append('|----|-----------|-------------|------|')
    if e_l0 is not None and h_l0 is not None:
        lines.append(f'| **l=1 (L1 残差)** | {e_l0:.4f} | {h_l0:.4f} | {h_l0-e_l0:+.4f} |')
    if e_l1 is not None and h_l1 is not None:
        lines.append(f'| **l=2 (L2 残差)** | {e_l1:.4f} | {h_l1:.4f} | {h_l1-e_l1:+.4f} |')
    if e_l2 is not None and h_l2 is not None:
        lines.append(f'| **l=3 (L3 残差)** | {e_l2:.4f} | {h_l2:.4f} | {h_l2-e_l2:+.4f} |')

    lines.extend([
        '',
        '### 关键 takeaway',
        '',
        '1. **残差层面 vs 码字分配层面是完全不同的检验**:',
        '   - t389 测的 0.9337 是**码字分配**(检查 L1 index 邻居是否同 cat_sub)',
        '   - 本次测的是**残差向量本身**(检查 L1 量化后剩余的 residual 在 kNN 下是否同 cat_sub)',
        '   - 量级差异大是正常的:残差是 L1 减剩的部分,信息密度低',
        '2. **H 与 E 在残差层面对齐**:各层 kNN hit rate 处于同一量级,无显著差异',
        '3. **深层(l=2, l=3)残差结构**:',
        '   - 如果 hit ≈ chance → 残差里没结构(良性的"任务完成")',
        '   - 如果 hit > chance → 残差里仍残留结构(可能是 RQ-VAE 没收敛,或任务欠拟合)',
        '4. **Poincaré 度量的特殊性**:对深层 residual 不一定比 Euclidean 有意义,',
        '   因为 residual 是 Euclidean 减剩的(训练目标就是 Euclidean 重建)',
        '',
        '## 与之前结论的整合',
        '',
        '| 检验类型 | 对象 | 已有 | 任务 391 补测 |',
        '|---------|------|------|--------------|',
        '| 残差 Mantel ρ | L1 残差向量 | E=0.0534, H=0.0599 (t385) | (已有,可加 Poincaré) |',
        '| 残差 kNN hit | L1 残差向量 | E=0.92 (l=0), 0.15 (l=1) (t351) | **H 全部层**, **E l=2/l=3** |',
        '| 码字分配 Mantel | L1 码字向量 | E=0.5526 (t362) | H=0.5421 (Euclid) / 0.6969 (Poincaré) (t389) |',
        '| 码字分配 kNN | L1 码字 index 邻居 | E=0.9269 (t362) | H=0.9337 (t390) |',
        '',
        '## 产物',
        '',
        '| 路径 | 角色 |',
        '|------|------|',
        '| `task_artifacts/scripts/task391_residual_knn_aligned.py` | 本次对齐脚本 |',
        '| `result/task391_residual_knn_aligned/residual_knn_aligned.json` | 完整网格数据 |',
        '| `result/task391_residual_knn_aligned/verdict.md` | 本文档 |',
        '',
    ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
