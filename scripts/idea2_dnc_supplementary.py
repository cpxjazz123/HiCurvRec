#!/usr/bin/env python3
"""Idea 2 DNC 补测 A: 随机基线对照 + 原始 embedding 对照

补测 A 步骤 1: 随机正交基线 (200 次)
对 q_1 用 200 个随机 m-维子空间算 ρ_rand^(s), 验证 ρ^{task}(q_1)=0.11 是否真的偏离随机
z = (ρ^{task}_actual - μ_rand) / σ_rand
kill: |z| < 2

补测 A 步骤 2: 原始 embedding x 的 ρ^{task}(x) 用同样的真任务子空间
关键对比:
- ρ^{task}(x) ≈ ρ^{task}(q_1) ≈ 0.11 → q_1 忠实反映原始方差结构 → 范数叙事, 不是新发现
- ρ^{task}(x) >> ρ^{task}(q_1) → L1 主动丢弃任务信息 → 真正大发现

复用:
- Stage 1 embedding x
- toys_metadata.json cat_sub
- 各算法 rqidx.q_lst[0]
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
    """从类标签构造任务相关子空间 U ∈ R^{d×m}"""
    label_set = sorted(set(labels_str))
    label_to_id = {l: i for i, l in enumerate(label_set)}
    N, d = emb.shape

    mu_G = emb.mean(dim=0)

    S_B = torch.zeros(d, d)
    for k in label_set:
        mask = torch.tensor([l == k for l in labels_str])
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        mu_k = emb[mask].mean(dim=0)
        diff = (mu_k - mu_G).unsqueeze(1)
        S_B += (n_k / N) * (diff @ diff.T)

    eigvals, eigvecs = torch.linalg.eigh(S_B)
    U = eigvecs[:, -m:]
    eigvals_top = eigvals[-m:].tolist()
    total_var = eigvals.sum().item()
    explained = sum(eigvals_top) / total_var if total_var > 0 else 0.0
    return U, label_set, eigvals_top, total_var


def compute_rho_task(residual, U):
    """ρ^{task}(r) = E[||U^T r||²] / E[||r||²]"""
    r_proj = residual @ U
    var_task = (r_proj ** 2).sum(dim=1).mean().item()
    var_total = (residual ** 2).sum(dim=1).mean().item()
    if var_total == 0:
        return 0.0
    return var_task / var_total


def random_orth_subspace(d, m, generator):
    """生成 d×m 的随机正交子空间 (用 QR 分解)"""
    A = torch.randn(d, m, generator=generator)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :m]


def random_baseline_test(target_vec, d, m, n_trials=200, seed=42):
    """对 target_vec (N, d), 算 n_trials 个随机正交子空间下的 ρ
    target_vec 暂时假定为 (N, d) 或者 (N,) — 此函数处理 (N, d) 形式, 取整组
    """
    g = torch.Generator().manual_seed(seed)
    rhos = []
    for s in range(n_trials):
        U_rand = random_orth_subspace(d, m, g)
        rhos.append(compute_rho_task(target_vec, U_rand))
    rhos = np.array(rhos)
    return rhos


def main():
    print('=' * 70)
    print('Idea 2 DNC 补测 A: 随机基线 + 原始 embedding 对照')
    print('=' * 70)

    # 1. 加载 + 构子空间
    print('\n[Step 1] 加载 embedding + 构任务子空间 U')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N, d = emb.shape
    print(f'  Embedding: {tuple(emb.shape)}')

    with open(META_PATH) as f:
        md = json.load(f)
    labels_str = [md[str(i)]['cat_sub'] for i in range(N)]

    U, label_set, eigvals_top, total_var = build_task_subspace(emb, labels_str, m=10)
    print(f'  Task subspace: m=10, top-10 eigvals explained ratio = {sum(eigvals_top)/total_var:.4f}')

    # 2. ρ^{task}(x) — 原始 embedding 在任务子空间的占比
    print('\n[Step 2] ρ^{task}(x) - 原始 embedding x 在任务子空间的占比')
    rho_x = compute_rho_task(emb, U)
    print(f'  ρ^{{task}}(x) = {rho_x:.4f}  ← 原始数据本身的"任务相关方差占比"')

    # 3. ρ^{task}(emb - μ) — 中心化后的占比 (去掉整体偏移的影响)
    emb_centered = emb - emb.mean(dim=0)
    rho_x_centered = compute_rho_task(emb_centered, U)
    print(f'  ρ^{{task}}(x_中心化) = {rho_x_centered:.4f}')

    # 4. ρ^{task}(q_1) - 量化输出在任务子空间 (上次结论 ~0.11)
    print('\n[Step 3] ρ^{task}(q_1) - 与 q_1 上次的数值核对')
    actual_rhos_q1 = {}
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            continue
        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        q_lst = data['q_lst']
        q1 = q_lst[0]
        rho_q1 = compute_rho_task(q1, U)
        actual_rhos_q1[algo_name] = rho_q1
        print(f'  {algo_name}: ρ^{{task}}(q_1) = {rho_q1:.4f}')

    # 5. 补测 A 步骤 1: 随机正交基线 (200 次) — 算 μ_rand, σ_rand, z-score
    print('\n[Step 4] 补测 A 步骤 1: 随机正交基线 (200 trials)')
    N_TRIALS = 200
    results_z = {}
    for algo_name in RQIDX_PATHS.keys():
        rqidx_path = RQIDX_PATHS[algo_name]
        if not os.path.exists(rqidx_path):
            continue
        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        q_lst = data['q_lst']
        q1 = q_lst[0]

        rhos_rand = random_baseline_test(q1, d=d, m=10, n_trials=N_TRIALS, seed=42)
        mu_rand = float(rhos_rand.mean())
        sigma_rand = float(rhos_rand.std())
        rho_actual = actual_rhos_q1[algo_name]
        z = (rho_actual - mu_rand) / sigma_rand if sigma_rand > 1e-12 else 0.0

        print(f'\n  [{algo_name}]')
        print(f'    ρ^{{task}}_actual(q_1) = {rho_actual:.4f}')
        print(f'    ρ_rand 分布: μ = {mu_rand:.4f}, σ = {sigma_rand:.4f}')
        print(f'    z-score = (ρ_actual - μ_rand) / σ_rand = {z:.3f}')
        print(f'    ρ_rand 5/50/95 百分位: '
              f'{np.percentile(rhos_rand, 5):.4f} / {np.percentile(rhos_rand, 50):.4f} / {np.percentile(rhos_rand, 95):.4f}')

        if abs(z) >= 3:
            verdict = f'STRONG (|z|={abs(z):.2f} >= 3): ρ^{{task}}(q_1) 显著偏离随机基线'
        elif abs(z) >= 2:
            verdict = f'PASS (|z|={abs(z):.2f} >= 2): 统计显著偏离随机'
        else:
            verdict = f'KILL (|z|={abs(z):.2f} < 2): ρ^{{task}}(q_1) 在随机基线范围内, "任务方向"叙事站不住'

        print(f'    → {verdict}')

        results_z[algo_name] = {
            'rho_actual': rho_actual,
            'rho_rand_mean': mu_rand,
            'rho_rand_std': sigma_rand,
            'rho_rand_p5': float(np.percentile(rhos_rand, 5)),
            'rho_rand_p50': float(np.percentile(rhos_rand, 50)),
            'rho_rand_p95': float(np.percentile(rhos_rand, 95)),
            'z_score': float(z),
            'verdict': verdict,
        }

    # 6. 决定性对比: ρ^{task}(x_centered) vs ρ^{task}(q_1)
    # 注意: 必须用中心化的 x, 否则全局均值偏移拉低 ρ 数值, 给出虚假的小 ratio
    print('\n[Step 5] 决定性对比: ρ^{task}(x_centered) vs ρ^{task}(q_1)')
    decisive = {}
    for algo_name in results_z.keys():
        rho_x_val = rho_x_centered  # 关键修正: 用中心化版本
        rho_q1 = results_z[algo_name]['rho_actual']
        ratio = rho_x_val / rho_q1 if rho_q1 > 1e-9 else float('inf')

        if ratio < 1.5:
            decisive_verdict = f'EMBED_DOMINATED: ρ(x_c)={rho_x_val:.4f} ≈ ρ(q_1)={rho_q1:.4f} (ratio={ratio:.2f}); q_1 反映原始方差结构, 非大发现'
        elif ratio < 3:
            decisive_verdict = f'WEAK_DIFFERENCE: ratio={ratio:.2f}, q_1 有部分信息丢失但不算颠覆性'
        else:
            decisive_verdict = f'STRONG_DROP: ρ(x_c)={rho_x_val:.4f} >> ρ(q_1)={rho_q1:.4f} (ratio={ratio:.2f}); L1 主动丢弃任务信息, 真正大发现'

        print(f'  {algo_name}: ρ(x_centered) = {rho_x_val:.4f}, ρ(q_1) = {rho_q1:.4f}, ratio = {ratio:.2f}')
        print(f'    → {decisive_verdict}')
        decisive[algo_name] = {
            'rho_x_centered': rho_x_val,
            'rho_x_raw': rho_x,
            'rho_q1': rho_q1,
            'ratio': ratio,
            'verdict': decisive_verdict,
        }

    # 7. 综合判定
    print('\n[Step 6] 综合 DNC "大发现" 命运判定')
    n_big_discovery = sum(1 for d in decisive.values() if 'STRONG_DROP' in d['verdict'])
    n_embed_dominated = sum(1 for d in decisive.values() if 'EMBED_DOMINATED' in d['verdict'])

    if n_big_discovery >= 3:
        overall = 'BIG_FINDING_CONFIRMED: 至少 3/4 算法 STRONG_DROP, L1 在丢任务信息是真大发现'
    elif n_embed_dominated >= 3:
        overall = 'BIG_FINDING_KILLED: 至少 3/4 算法 ratio < 1.5, q_1 忠实反映原始方差, 不是新发现, 退回范数叙事'
    else:
        overall = 'MIXED: 结果不一致, 需要更细致的子类分析才能定性'

    print(f'  STRONG_DROP 数: {n_big_discovery} / {len(decisive)}')
    print(f'  EMBED_DOMINATED 数: {n_embed_dominated} / {len(decisive)}')
    print(f'  → 综合判定: {overall}')

    # 8. 保存
    out_json = os.path.join(OUT_DIR, 'idea2_supplementary_testA.json')
    with open(out_json, 'w') as f:
        json.dump({
            'task_subspace': {
                'm': 10,
                'eigvals_top': eigvals_top,
                'total_var': total_var,
                'explained_ratio': sum(eigvals_top) / total_var,
            },
            'baseline_rho_x': rho_x,
            'baseline_rho_x_centered': rho_x_centered,
            'random_subspace_test': results_z,
            'decisive_comparison': decisive,
            'overall_verdict': overall,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task22_directional_neural_collapse.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    return overall, results_z, decisive


if __name__ == '__main__':
    main()
