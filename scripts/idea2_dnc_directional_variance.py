#!/usr/bin/env python3
"""Idea 2 DNC: 方向性方差分解 — 残差 vs 任务相关方向 (cat_sub)

测什么:
- 构造任务相关子空间 U (top-m S_B 特征向量, m=10)
- 逐层算 ρ_l^{task} = E[||U^T r_l||²] / E[||r_l||²]
- 算 ρ^{task}(q_1): L1 量化输出在任务相关方向上的方差占比
- 三种结局:
  (a) ρ^{task}(q_1) > 0.7 → L1 抓的是任务相关方差, 降级叙事升级
  (b) ρ^{task}(q_1) < 0.3 → L1 抓错了对象, 大发现
  (c) 中间 → ρ_l^{task} 曲线本身是诊断量

复用:
- Stage 1 embedding: /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/{A,B,C}_baseline_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt
- toys_metadata.json (cat_sub 24 类)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea2_dnc'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'

RQIDX_PATHS = {
    'A_RQ_VAE':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_MMQ':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_GSRQ':    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF_K256_64_16': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def build_task_subspace(emb, labels_str, m=10):
    """从类标签构造任务相关子空间 U ∈ R^{d×m}

    S_B = Σ_k (n_k/N) (μ_k - μ_G)(μ_k - μ_G)^T
    U = top-m eigenvectors of S_B

    labels_str: list of str (cat_sub)
    """
    label_set = sorted(set(labels_str))
    label_to_id = {l: i for i, l in enumerate(label_set)}
    N = emb.shape[0]
    d = emb.shape[1]
    label_ids = np.array([label_to_id[l] for l in labels_str])

    # 全局均值
    mu_G = emb.mean(dim=0)  # (d,)

    # S_B
    S_B = torch.zeros(d, d)
    for k in label_set:
        mask = torch.tensor([l == k for l in labels_str])
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        mu_k = emb[mask].mean(dim=0)
        diff = (mu_k - mu_G).unsqueeze(1)  # (d, 1)
        S_B += (n_k / N) * (diff @ diff.T)

    # 取 top-m 特征向量
    eigvals, eigvecs = torch.linalg.eigh(S_B)  # 升序
    U = eigvecs[:, -m:]  # (d, m), top-m
    eigvals_top = eigvals[-m:].tolist()

    print(f'[S_B] {len(label_set)} classes, top-{m} eigvals: {[round(v, 2) for v in eigvals_top]}')
    total_var = eigvals.sum().item()
    explained = sum(eigvals_top) / total_var if total_var > 0 else 0.0
    print(f'[S_B] total variance explained by top-{m}: {explained:.4f}')
    return U, label_set, eigvals_top, total_var


def compute_rho_task(residual, U):
    """ρ^{task}(r) = E[||U^T r||²] / E[||r||²]
    residual: (N, d)
    U: (d, m)
    返回 ρ^{task} 标量
    """
    r_proj = residual @ U  # (N, m)
    var_task = (r_proj ** 2).sum(dim=1).mean().item()  # E[||U^T r||²]
    var_total = (residual ** 2).sum(dim=1).mean().item()  # E[||r||²]
    if var_total == 0:
        return 0.0
    return var_task / var_total


def main():
    print('=' * 70)
    print('Idea 2 DNC: 方向性方差分解 — 残差 vs 任务相关方向 (cat_sub)')
    print('=' * 70)

    # 1. 加载 embedding + labels
    print('\n[Step 1] 加载 Stage 1 embedding + toys metadata')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N, d = emb.shape
    print(f'  Embedding: {tuple(emb.shape)}')

    with open(META_PATH) as f:
        md = json.load(f)
    labels_str = [md[str(i)]['cat_sub'] for i in range(N)]
    print(f'  Labels: cat_sub, {len(set(labels_str))} unique')
    print(f'  Top 5 cat_sub: {Counter(labels_str).most_common(5)}')

    # 2. 构造任务相关子空间
    print('\n[Step 2] 构造任务相关子空间 U (top-10 S_B 特征向量)')
    U, label_set, eigvals_top, S_B_total = build_task_subspace(emb, labels_str, m=10)
    label_to_id = {l: i for i, l in enumerate(label_set)}

    # 3. 逐算法、逐层算 ρ_l^{task}
    print('\n[Step 3] 逐算法、逐层 ρ_l^{task} + ρ^{task}(q_1)')
    results = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            print(f'[WARN] {algo_name}: rqidx 不存在')
            continue

        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        r_lst = data['r_lst']  # list of (N, d)
        q_lst = data['q_lst']  # list of (N, d), len = n_layers

        # 算每个 residual 的 ρ^{task}
        rho_per_layer = [compute_rho_task(r, U) for r in r_lst]

        # ρ^{task}(q_1): q_lst[0] 是 L1 量化输出
        rho_q1 = compute_rho_task(q_lst[0], U) if len(q_lst) >= 1 else None

        print(f'\n[{algo_name}]')
        for l, rho in enumerate(rho_per_layer):
            label = f'r_{l}' if l < len(rho_per_layer) - 1 else f'r_{l} (final)'
            print(f'  ρ^{{task}}({label}) = {rho:.4f}')
        if rho_q1 is not None:
            print(f'  ρ^{{task}}(q_1)   = {rho_q1:.4f}')

        results.append({
            'algorithm': algo_name,
            'rho_task_per_layer_residual': rho_per_layer,
            'rho_task_q1': rho_q1,
        })

    # 4. 现象 1 判定
    print('\n[Step 4] 现象 1 判定')
    verdicts = []
    for r in results:
        rho_q1 = r['rho_task_q1']
        if rho_q1 is None:
            continue
        if rho_q1 > 0.7:
            v = '(a) 高: L1 抓的是任务相关方差'
        elif rho_q1 < 0.3:
            v = '(b) 低: L1 抓错了对象 (大发现)'
        else:
            v = '(c) 中间: 曲线为新诊断量'
        verdicts.append({'algorithm': r['algorithm'], 'rho_task_q1': rho_q1, 'verdict': v})
        print(f'  {r["algorithm"]}: ρ^{{task}}(q_1) = {rho_q1:.4f} → {v}')

    # 现象 2: 归一化 (剥范数后重测)
    print('\n[Step 5] 现象 2: 归一化 r̂_l = r_l / ||r_l||')
    results_normalized = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            continue
        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        r_lst = data['r_lst']
        q_lst = data['q_lst']

        rho_n_per_layer = []
        for r in r_lst:
            norms = r.norm(dim=1, keepdim=True).clamp(min=1e-8)
            r_norm = r / norms
            rho_n_per_layer.append(compute_rho_task(r_norm, U))

        # CDNV_l^{dir} = E_k[Var(U^T r | k)] / E_{k≠k'}[||U^T(μ_k - μ_{k'})||²]
        # 简化为 ρ^{task}_within / ρ^{task}_between (类内类间方差比)
        cdnv_per_layer = []
        for r in r_lst:
            r_proj = r @ U  # (N, m)
            label_arr = np.array([label_to_id[l] for l in labels_str])
            var_within, var_between = 0.0, 0.0
            for k in label_set:
                mask = torch.tensor([l == k for l in labels_str])
                n_k = mask.sum().item()
                if n_k < 2:
                    continue
                r_proj_k = r_proj[mask]
                mu_k = r_proj_k.mean(dim=0)
                var_within += ((r_proj_k - mu_k) ** 2).sum(dim=1).mean().item() * (n_k / N)
                mu_global = r_proj.mean(dim=0)
                var_between += (n_k / N) * ((mu_k - mu_global) ** 2).sum().item()
            cdnv = var_within / (var_between + 1e-8)
            cdnv_per_layer.append(cdnv)

        print(f'\n[{algo_name}] (normalized)')
        for l, (rho_n, cdnv) in enumerate(zip(rho_n_per_layer, cdnv_per_layer)):
            print(f'  ρ̂^{{task}}(r_{l}) = {rho_n:.4f}, CDNV = {cdnv:.4f}')

        results_normalized.append({
            'algorithm': algo_name,
            'rho_task_per_layer_residual_normalized': rho_n_per_layer,
            'cdnv_per_layer': cdnv_per_layer,
        })

    # 现象 2 kill 判定
    print('\n[Step 6] 现象 2 kill 判定')
    for r in results_normalized:
        rho_n = r['rho_task_per_layer_residual_normalized']
        cdnv = r['cdnv_per_layer']
        # 标准差
        rho_std = float(np.std(rho_n))
        cdnv_std = float(np.std(cdnv))
        rho_diff = max(rho_n) - min(rho_n)
        cdnv_diff = max(cdnv) - min(cdnv)
        if rho_diff < 0.05 and cdnv_diff < 0.05:
            v = 'KILLED (方向不携带层间差异信息, 范数解释一切)'
        else:
            v = f'ρ_diff={rho_diff:.3f}, CDNV_diff={cdnv_diff:.3f} → 方向有独立贡献'
        print(f'  {r["algorithm"]}: {v}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea2_phen1_per_layer_task_ratio.json')
    with open(out_json, 'w') as f:
        json.dump({
            'task_subspace': {'m': 10, 'eigvals_top': eigvals_top, 'total_var_explained': float(sum(eigvals_top) / S_B_total) if S_B_total > 0 else None},
            'per_algorithm_raw': results,
            'per_algorithm_normalized': results_normalized,
            'verdicts_phen1': verdicts,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task22_directional_neural_collapse.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    return results, results_normalized, verdicts


if __name__ == '__main__':
    main()