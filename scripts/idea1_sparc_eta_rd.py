#!/usr/bin/env python3
"""Idea 1 SPARC 现象 1+2: AQ vs RQ 率失真效率 η_RD + 能量占比对齐

测什么:
- 白化 Stage 1 embedding → 算特征值 λ_i (作为高斯源近似)
- Shannon RD 界 D*(R) (reverse water-filling, 二分 θ)
- 实际 RQ 失真 D_actual^{RQ} (从 rqidx.q_lst 直接算)
- η_RD^{RQ} = D*(R_total) / D_actual^{RQ}
- AQ: 因 codebook 向量不可访问, 用 collision/Recall 作为定性 proxy
- 现象 2: per-layer 能量占比 e_l = E[||q_l||²] / Σ E[||q_l||²]

判定:
- η_RD^{RQ} 在合理范围 (> 0.30) → RD 边界有意义
- η_RD^{RQ} 远 < 0.30 → 量化离 RD 界很远, 任何"结构优势"都需重新审视
- 跨算法 e_l 与 SPARC 最优 e_l^SPARC 的 Pearson r → 现象 2

复用:
- Stage 1 embedding (11924, 2048)
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from scipy.stats import pearsonr

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_sparc'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

RQIDX_PATHS = {
    'A_RQ_VAE':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_MMQ':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_GSRQ':    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF_K256_64_16': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}

# AQ proxy 数据 (from result/task21/task19_eval_AQ.json)
AQ_PROXY = {
    'collision_rate': 0.0976,
    'recall_at_5': 0.0790,
    'recall_at_10': 0.1145,
    'ndcg_at_5': 0.0542,
    'ndcg_at_10': 0.0657,
}


def whiten_pca(X, n_components=None):
    """PCA 白化
    返回:
    - X_white: 白化后 (N, n_components)
    - eigvals_per_sample: 协方差矩阵特征值 (per-sample), 注意: torch.linalg.svd 返回 S 是 sqrt(N×λ)
    """
    N = X.shape[0]
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    d = X.shape[1]
    if n_components is None:
        n_components = d
    V = Vh[:n_components].T  # (d, n_components)
    X_pca = X_centered @ V  # (N, n_components)
    X_white = X_pca / (S[:n_components].clamp(min=1e-8))
    # 真正的 cov 特征值 λ_i = S_i² / N
    eigvals_per_sample = (S[:n_components] ** 2) / N
    return X_white, eigvals_per_sample.numpy()


def shannon_rd_bound(eigvals, R_target, eps=1e-6):
    """reverse water-filling 求 D*(R)
    D*(R) = Σ min(θ, λ_i)
    R = Σ (1/2) log2(λ_i / min(θ, λ_i))

    输入: eigvals (d,) 升序, R_target 比特/样本
    返回: D_star (方差)
    """
    eigvals = np.sort(eigvals)[::-1]  # 降序
    d = len(eigvals)

    # 二分 θ
    theta_lo, theta_hi = 1e-12, eigvals.max() * 2

    def rate_at_theta(theta):
        if theta <= 0:
            return 0.0
        terms = []
        for lam in eigvals:
            if lam <= theta:
                terms.append(0.0)
            else:
                terms.append(0.5 * np.log2(lam / theta))
        return sum(terms)

    for _ in range(100):
        theta_mid = (theta_lo + theta_hi) / 2
        r = rate_at_theta(theta_mid)
        if r > R_target:
            theta_lo = theta_mid
        else:
            theta_hi = theta_mid
        if abs(theta_hi - theta_lo) < eps:
            break

    theta = (theta_lo + theta_hi) / 2
    D_star = sum(min(theta, lam) for lam in eigvals)
    return float(D_star), float(theta)


def main():
    print('=' * 70)
    print('Idea 1 SPARC 现象 1+2: Shannon RD 界 + RQ 失真 + 能量占比')
    print('=' * 70)

    # 1. 加载 + 白化
    print('\n[Step 1] 加载 Stage 1 embedding + PCA 白化')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N, d = emb.shape
    print(f'  Embedding: {tuple(emb.shape)}')

    emb_white, eigvals = whiten_pca(emb)
    print(f'  Whitened: {tuple(emb_white.shape)}, top-5 eigvals (per-sample, after N-norm) = {[round(float(e), 4) for e in eigvals[:5]]}')
    total_var = float(eigvals.sum())
    print(f'  Total variance (Σ λ_i, per-sample) = {total_var:.4f}')

    # 2. Shannon RD 界 (对 24 bits / 32 bits 算)
    # 注意: 24 bits / 2048 dims = 0.012 bits/dim, 极低码率
    # 跨 d 维有效码率 = 24 bits / 11924 items = 24 bits per sample
    # 这里 R_target = bits per sample (Gaussian source)
    R_RQ_total = 24.0  # 3 × 8 = 24 bits per sample
    R_total_AQ_like = 32.0  # 4 × 8 = 32 bits per sample
    print('\n[Step 2] Shannon RD 界 (reverse water-filling, R = bits per sample)')
    D_RQ, theta_RQ = shannon_rd_bound(eigvals, R_target=R_RQ_total)
    print(f'  R = {R_RQ_total} bits/sample → D*(R) = {D_RQ:.4f} (θ = {theta_RQ:.6f})')
    D_AQ, theta_AQ = shannon_rd_bound(eigvals, R_target=R_total_AQ_like)
    print(f'  R = {R_total_AQ_like} bits/sample → D*(R) = {D_AQ:.4f} (θ = {theta_AQ:.6f})')

    # 3. 各 RQ 算法的实际失真
    print('\n[Step 3] 各 RQ 算法实际失真 D_actual')
    results = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            print(f'[WARN] {algo_name}: rqidx 不存在')
            continue

        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        r_lst = data['r_lst']
        q_lst = data['q_lst']

        # D_actual = E[||x - Σ q_l||²] → 这里 x 是 emb_white (白化后)
        x_hat = sum(q_lst)  # (N, d_whiten)
        # 注: rqidx 中的 q_lst 是非白化空间的, 直接误差用原始空间
        x_hat_raw = x_hat  # 保留 (rqidx 已经在原始空间)
        x_target = emb - emb.mean(dim=0, keepdim=True)  # 中心化 (相对 mean)
        # 误差
        residual = x_target - sum([q for q in q_lst])
        D_actual = (residual ** 2).sum(dim=1).mean().item()

        # per-layer 能量
        energies = [(q ** 2).sum(dim=1).mean().item() for q in q_lst]
        total_energy = sum(energies)
        e_per_layer = [e / total_energy for e in energies]

        # η_RD
        if algo_name == 'A_RQ_VAE':
            # A 用了 3 层 + 1 dedup ≈ 24 bits 主残差
            eta_RD = D_RQ / D_actual if D_actual > 0 else None
        else:
            eta_RD = D_RQ / D_actual if D_actual > 0 else None

        print(f'\n[{algo_name}]')
        print(f'  D_actual = {D_actual:.4f}')
        print(f'  η_RD(RQ 24 bits) = D*/D_actual = {eta_RD:.4f}' if eta_RD else f'  η_RD = None')
        print(f'  per-layer e_l = {[round(e, 3) for e in e_per_layer]}')

        results.append({
            'algorithm': algo_name,
            'D_actual': D_actual,
            'eta_RD_RQ_24bits': eta_RD,
            'energies_per_layer': energies,
            'e_per_layer_normalized': e_per_layer,
        })

    # 4. 现象 1 判定 (RQ)
    print('\n[Step 4] 现象 1 判定 (RQ 侧)')
    for r in results:
        eta = r['eta_RD_RQ_24bits']
        if eta is None:
            v = 'INVALID'
        elif eta < 0.30:
            v = 'BELOW_FLOOR (远离 RD 界, 结构优势不来自 RD 最优)'
        elif eta < 0.5:
            v = 'NEAR_BOUND (η 在 0.3-0.5, 中等 RD 效率)'
        else:
            v = 'STRONG (η > 0.5, 接近 RD 界)'
        print(f'  {r["algorithm"]}: η = {eta:.4f} → {v}')

    # 5. AQ proxy 报道
    print('\n[Step 5] AQ proxy 数据 (无 codebook 可访问, 用 collision+Recall 估)')
    print(f'  AQ: collision={AQ_PROXY["collision_rate"]:.4f}, '
          f'Recall@10={AQ_PROXY["recall_at_10"]:.4f}, '
          f'NDCG@10={AQ_PROXY["ndcg_at_10"]:.4f}')

    # 6. 现象 2: 能量占比 vs SPARC 理论最优
    print('\n[Step 6] 现象 2: 能量占比 SPARC 对齐')
    # SPARC 理论最优: 各层 rate = log2(M_l), 贡献能量按 water-fill 划分
    # 简化版: 用 reverse water-filling 在 R_l 上分配, 给出每层理论能量占比
    # 在白化空间, 每层 R_l 独立 → 各层能量分配均匀
    # 实际 e_l 应近似均匀 (或者按 reverse water-fill 递减)
    # 这里用单 RQ 自己的层 rate 作为参照, 算自相关

    rq_results_phen2 = []
    for r, path in zip(results, RQIDX_PATHS.values()):
        if not os.path.exists(path):
            continue
        data = torch.load(path, weights_only=False, map_location='cpu')
        codebooks = data['codebooks']
        K_per_layer = [c.shape[0] if c is not None else 1 for c in codebooks]
        bits_per_layer = [np.log2(K) for K in K_per_layer]

        e_actual = r['e_per_layer_normalized']

        # SPARC 最优: 每个 bit 对应的能量贡献相等 (白化空间均匀)
        # → e_l^SPARC ∝ R_l (按 rate 占比)
        total_bits = sum(bits_per_layer)
        e_sparc = [b / total_bits for b in bits_per_layer]

        # Pearson r
        if len(e_actual) >= 2 and all(x > 0 for x in e_sparc):
            rho, pval = pearsonr(e_actual, e_sparc)
        else:
            rho, pval = None, None

        print(f'\n[{r["algorithm"]}]')
        print(f'  K_per_layer = {K_per_layer}, bits_per_layer = {bits_per_layer}')
        print(f'  e_actual = {[round(e, 3) for e in e_actual]}')
        print(f'  e_sparc   = {[round(e, 3) for e in e_sparc]}')
        print(f'  Pearson r = {rho:.3f}, p = {pval:.3f}' if rho is not None else '  Pearson r = None')

        if rho is None:
            v = 'INSUFFICIENT_DATA'
        elif rho > 0.7:
            v = f'PASS (r = {rho:.3f})'
        elif rho < 0.3:
            v = f'KILL (r = {rho:.3f})'
        else:
            v = f'WEAK (r = {rho:.3f})'

        rq_results_phen2.append({
            'algorithm': r['algorithm'],
            'e_actual': e_actual,
            'e_sparc_theoretical': e_sparc,
            'pearson_r': rho,
            'pearson_p': pval,
            'verdict': v,
        })

    # 7. 保存
    out_json = os.path.join(OUT_DIR, 'idea1_phen1_eta_RD.json')
    with open(out_json, 'w') as f:
        json.dump({
            'shannon_bound': {
                'R_total_RQ_24bits': R_RQ_total,
                'D_star_RQ': D_RQ,
                'theta_RQ': theta_RQ,
                'R_total_AQ_32bits': R_total_AQ_like,
                'D_star_AQ': D_AQ,
                'theta_AQ': theta_AQ,
                'total_variance': float(total_var),
                'top5_eigvals': [float(e) for e in eigvals[:5]],
            },
            'per_algorithm_actual': results,
            'phen1_verdicts': [{'algorithm': r['algorithm'], 'eta': r['eta_RD_RQ_24bits']} for r in results],
            'aq_proxy': AQ_PROXY,
            'phen2_energy_alignment': rq_results_phen2,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task8_sparc_rate_distortion.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # 8. η_RD bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [r['algorithm'] for r in results]
        etas = [r['eta_RD_RQ_24bits'] for r in results]
        ax.bar(names, etas, color=['steelblue', 'seagreen', 'salmon', 'orchid'][:len(names)],
               edgecolor='k')
        ax.axhline(0.30, color='red', ls='--', alpha=0.5, label='kill line 0.30')
        ax.axhline(0.50, color='green', ls='--', alpha=0.5, label='strong line 0.50')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('η_RD = D*(R=24) / D_actual')
        ax.set_title('Idea 1 SPARC: RQ 率失真效率 (RQ vs Shannon 界)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15)
        ax.legend()
        out_png = os.path.join(OUT_DIR, 'eta_RD_comparison.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')

        # 现象 2: 能量占比对比
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        n_algo = len(rq_results_phen2)
        layer_max = max(len(r['e_actual']) for r in rq_results_phen2)
        x = np.arange(layer_max)
        bar_w = 0.35
        for i, r in enumerate(rq_results_phen2):
            offset = (i - n_algo/2) * bar_w
            ax2.bar(x + offset, r['e_actual'], width=bar_w*0.9,
                    label=f"{r['algorithm']} actual", alpha=0.7)
            ax2.plot(x + offset, r['e_sparc_theoretical'], 'o-',
                     label=f"{r['algorithm']} SPARC", color='red', alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'l={l}' for l in range(layer_max)])
        ax2.set_ylabel('e_l (energy fraction)')
        ax2.set_title('Idea 1 SPARC 现象 2: per-layer 能量占比 (actual vs SPARC)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=7)
        out_png2 = os.path.join(OUT_DIR, 'energy_ratio_comparison.png')
        plt.tight_layout()
        plt.savefig(out_png2, dpi=120)
        print(f'[SAVED] {out_png2}')
    except Exception as e:
        print(f'[WARN] 画图失败: {e}')

    return results, rq_results_phen2


if __name__ == '__main__':
    main()
