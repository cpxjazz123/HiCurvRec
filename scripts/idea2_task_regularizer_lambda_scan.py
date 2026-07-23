#!/usr/bin/env python3
"""诊断组C 现象1: post-hoc λ 扫描 — 加权最近邻重新分配码字

目的: 测试"任务相关正则项"能否修复 L1 信息丢失 (从 ρ=0.11 提升到 0.20+)

方法:
- 不重新训练码本, 只改变量化时"选哪个码字"的规则
  原: q_1 = argmin_c ||x - c||²
  新: q_1'(λ) = argmin_c [|x - c|² - λ · ||U^T c||²]
- 对 λ ∈ {0, 0.1, 0.5, 1.0, 2.0} 重新指派所有 item
- 测 ρ^{task}(q_1') + 重建误差 E[||x - q_1'||²]

复用:
- Stage 1 embedding: /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
- Toys metadata (cat_sub labels): /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json
- Group A L1 codebook: /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt
- 任务相关子空间 U ∈ R^{2048×10}: 复用 idea2 supplementary 的 build_task_subspace

kill 线:
- λ=2 时 ρ^task 仍未回到 0.20+ → KILL (post-hoc 不可行, 需要真重训)
- 重建误差涨幅 > 50% → KILL (信息丢失 vs 重建质量是硬权衡)
- λ 较小 (如 0.1) 时已能拉回 + 重建增长 < 20% → PASS (进入现象 2 重训)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea2_dnc'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
GROUP_A_RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'

LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
M_TASK = 10  # task subspace dim
SEED = 42


def build_task_subspace(emb, labels_str, m=M_TASK):
    label_set = sorted(set(labels_str))
    N, d = emb.shape
    mu_G = emb.mean(dim=0)
    S_B = torch.zeros(d, d)
    for k in label_set:
        mask = torch.tensor([l == k for l in labels_str])
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        mu_k = emb[mask].mean(dim=0)
        diff = (mu_k - mu_G).unsqueeze(-1)
        S_B += (n_k / N) * (diff @ diff.T)
    eigvals, eigvecs = torch.linalg.eigh(S_B)
    U = eigvecs[:, -m:]
    eigvals_top = eigvals[-m:].tolist()
    total_var = eigvals.sum().item()
    explained = sum(eigvals_top) / total_var if total_var > 0 else 0.0
    return U, label_set, eigvals_top, total_var, explained


def compute_rho_task_vector(v, U):
    """ρ^{task}(v) = E[||U^T v||²] / E[||v||²]"""
    v_proj = v @ U
    var_task = (v_proj ** 2).sum(dim=1).mean().item()
    var_total = (v ** 2).sum(dim=1).mean().item()
    if var_total == 0:
        return 0.0
    return var_task / var_total


def main():
    print('=' * 70)
    print('诊断组C 现象1: post-hoc λ 扫描 — 加权最近邻重新分配')
    print('=' * 70)

    # 1. 加载 embedding + labels
    print('\n[Step 1] 加载 Stage 1 embedding + toys metadata')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N, d = emb.shape
    print(f'  embedding: {tuple(emb.shape)}')
    with open(META_PATH) as f:
        md = json.load(f)
    labels_str = [md[str(i)]['cat_sub'] for i in range(N)]
    n_classes = len(set(labels_str))
    print(f'  labels (cat_sub): {n_classes} unique classes')

    # 2. 构造任务相关子空间 U
    print('\n[Step 2] 构造任务相关子空间 U (top-10 S_B 特征向量)')
    U, label_set, eigvals_top, total_var, explained = build_task_subspace(emb, labels_str, m=M_TASK)
    print(f'  U: {tuple(U.shape)}, top-10 eigvals: {[round(v, 2) for v in eigvals_top]}')
    print(f'  explained_ratio: {explained:.4f}')

    # 3. 加载 Group A L1 codebook
    print('\n[Step 3] 加载 Group A L1 codebook (256×2048)')
    data = torch.load(GROUP_A_RQIDX, weights_only=False, map_location='cpu')
    codebooks = data['codebooks']
    C_1 = codebooks[0].float()  # (256, 2048)
    K = C_1.shape[0]
    print(f'  C_1: {tuple(C_1.shape)}, K={K}')

    # 4. 算 ||x - c||² 矩阵
    print('\n[Step 4] 算 pairwise ||x - c||² 距离矩阵 (N×K)')
    sq_dist = torch.cdist(emb, C_1, p=2).pow(2)  # (N, K)
    print(f'  sq_dist shape: {tuple(sq_dist.shape)}')

    # 5. 算每个码字的 ||U^T c||²
    print('\n[Step 5] 算 ||U^T c||² (K 标量)')
    C_1_proj = C_1 @ U  # (K, m)
    task_score_per_code = (C_1_proj ** 2).sum(dim=1)  # (K,)
    print(f'  task_score: min={task_score_per_code.min():.3f}, '
          f'max={task_score_per_code.max():.3f}, '
          f'mean={task_score_per_code.mean():.3f}')

    # 6. λ 扫描
    print('\n[Step 6] λ 扫描重新分配码字 + 测 ρ^{task} + 重建误差')
    results_per_lambda = []
    baseline_rho = None
    baseline_recon = None

    for lam in LAMBDA_GRID:
        # Weighted distance matrix: ||x - c||² - λ · ||U^T c||²
        weighted = sq_dist - lam * task_score_per_code.unsqueeze(0)  # (N, K)
        # argmin over K
        q1_indices = weighted.argmin(dim=1)  # (N,)
        q1_recon = C_1[q1_indices]  # (N, d)

        # ρ^{task}(q_1'(λ))
        rho_task = compute_rho_task_vector(q1_recon, U)

        # 重建误差 E[||x - q_1||²]
        recon_err = ((emb - q1_recon) ** 2).sum(dim=1).mean().item()

        # ρ^{task}(x - q_1) — 残差比
        residual = emb - q1_recon
        rho_residual = compute_rho_task_vector(residual, U)

        # 码字利用 (用了多少个码字)
        n_unique = q1_indices.unique().shape[0]

        if lam == 0:
            baseline_rho = rho_task
            baseline_recon = recon_err
            rho_growth_abs = 0.0
            recon_growth_rel = 0.0
        else:
            rho_growth_abs = rho_task - baseline_rho
            recon_growth_rel = (recon_err - baseline_recon) / baseline_recon * 100

        print(f'\n  λ = {lam:.2f}:')
        print(f'    ρ^{{task}}(q_1\') = {rho_task:.4f} (Δ_abs = {rho_growth_abs:+.4f})')
        print(f'    重建误差       = {recon_err:.4f} (Δ_rel = {recon_growth_rel:+.1f}%)')
        print(f'    ρ^{{task}}(x - q_1\') = {rho_residual:.4f}')
        print(f'    使用码字数: {n_unique}/{K}')

        results_per_lambda.append({
            'lambda': float(lam),
            'rho_task_q1': float(rho_task),
            'rho_task_residual': float(rho_residual),
            'reconstruction_error': float(recon_err),
            'rho_growth_abs': float(rho_growth_abs),
            'recon_growth_rel_pct': float(recon_growth_rel),
            'n_unique_codes_used': int(n_unique),
        })

    # 7. 判定
    print('\n[Step 7] 判定 (按 kill 线)')
    rho_at_max_lambda = results_per_lambda[-1]['rho_task_q1']
    recon_at_max_lambda = results_per_lambda[-1]['recon_growth_rel_pct']

    # 看是否有任何 λ 同时满足: ρ >= 0.20 + recon growth <= 50%
    any_pass_20 = next((r for r in results_per_lambda if r['rho_task_q1'] >= 0.20 and r['recon_growth_rel_pct'] <= 50), None)
    any_pass_strict = next((r for r in results_per_lambda if r['rho_task_q1'] >= 0.20 and r['recon_growth_rel_pct'] <= 20), None)
    any_pull_back = next((r for r in results_per_lambda[1:] if r['rho_task_q1'] > baseline_rho * 1.5), None)

    if any_pass_strict is not None:
        verdict = (
            f'PASS_STRICT: 在 λ={any_pass_strict["lambda"]:.2f} 时, '
            f'ρ^{{task}}(q_1\') = {any_pass_strict["rho_task_q1"]:.3f} ≥ 0.20 '
            f'+ 重建误差仅 +{any_pass_strict["recon_growth_rel_pct"]:.1f}% (≤ 20%). '
            f'加权最近邻是可行方案. → 现象 2: 真重训 L1 层即可验证端到端收益. '
            f'VALUE: 证实 ρ^task ↔ 重建 是可调的, 不是硬权衡.'
        )
    elif any_pass_20 is not None:
        verdict = (
            f'PASS_RELAXED: 在 λ={any_pass_20["lambda"]:.2f} 时, '
            f'ρ^{{task}}(q_1\') = {any_pass_20["rho_task_q1"]:.3f} ≥ 0.20, '
            f'但重建误差 +{any_pass_20["recon_growth_rel_pct"]:.1f}% (>20% 但 <50%). '
            f'中间地带: 需要权衡. 建议 λ={any_pass_strict["lambda"]:.2f if any_pass_strict else "N/A"} 作为轻量起点.'
        )
    else:
        if any_pull_back is not None:
            verdict = (
                f'PARTIAL_PULLBACK: λ={any_pull_back["lambda"]:.2f} 时 ρ^{{task}}(q_1\') '
                f'从 {baseline_rho:.3f} 提升到 {any_pull_back["rho_task_q1"]:.3f} '
                f'(+{(any_pull_back["rho_task_q1"]/baseline_rho - 1)*100:.0f}%, >1.5x), '
                f'但仍 < 0.20. 重建误差 +{any_pull_back["recon_growth_rel_pct"]:.1f}%. '
                f'加权最近邻有部分效果, 但 0.20 目标超过其能力边界.'
            )
        else:
            verdict = (
                f'KILL_POST_HOC: 任何 λ 下 ρ^{{task}}(q_1\') 都未能从 {baseline_rho:.3f} '
                f'拉到 0.20, 同时重建误差涨幅可能过大. 信息丢失 vs 重建质量是硬权衡, '
                f'post-hoc 分配规则改不动这个权衡. → 现象 2 (真重训) 是必要手段, '
                f'但要带"任务相关方差占比"作为代价函数才能改.'
            )

    print(f'\n[Verdict] {verdict}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea2_task_regularizer_lambda_scan.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'post_hoc_weighted_nearest_neighbor',
            'description': (
                'Re-assign items to codewords under new rule '
                'q_1\'(λ) = argmin_c [||x-c||² - λ · ||U^T c||²]. '
                'Same codebook, only assignment rule changes.'
            ),
            'params': {
                'lambda_grid': LAMBDA_GRID,
                'm_task': M_TASK,
                'n_items': N,
                'n_codewords': K,
                'emb_dim': d,
                'seed': SEED,
            },
            'task_subspace': {
                'n_classes': int(n_classes),
                'top10_eigvals': [float(v) for v in eigvals_top],
                'total_var': float(total_var),
                'explained_ratio': float(explained),
                'U_shape': list(U.shape),
            },
            'task_score_per_code_stats': {
                'min': float(task_score_per_code.min()),
                'max': float(task_score_per_code.max()),
                'mean': float(task_score_per_code.mean()),
                'std': float(task_score_per_code.std()),
            },
            'baseline': {
                'rho_task_q1_at_lambda_0': float(baseline_rho),
                'recon_err_at_lambda_0': float(baseline_recon),
            },
            'results_per_lambda': results_per_lambda,
            'verdict': verdict,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task22_directional_neural_collapse.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    return verdict, results_per_lambda


if __name__ == '__main__':
    main()
