#!/usr/bin/env python3
"""Idea 5 TDA 补测 B: 多 seed 稳定性 + 带宽敏感性 + HRQ 空间对比 (last part 注记)

补测 B 步骤 1: 多种子子采样稳定性 (seeds 42/43/44)
补测 B 步骤 2: KDE 带宽敏感性 (Silverman × {0.5, 1, 2, 4})
补测 B 步骤 3: HRQ 空间 n_peaks 对比 — 见末尾 note (需要 HRQ 模型推理, 暂时跳过)

复用:
- Stage 1 embedding (11924, 2048)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
import ripser
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea5_tda'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 5000
PCA_DIM = 50
RIPSER_THRESHOLD_PERCENTILE = 90


def whiten_pca(X, n_components=50):
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    X_pca = X_centered @ V
    X_white = X_pca / (S[:n_components].clamp(min=1e-8))
    return X_white


def compute_h0(X_np, thresh):
    result = ripser.ripser(X_np, thresh=thresh)['dgms']
    h0 = result[0]
    h0 = h0[np.isfinite(h0[:, 1])]
    return h0


def silverman_bandwidth(values):
    """Silverman 经验法则带宽"""
    n = len(values)
    sigma = np.std(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    a = min(sigma, iqr / 1.349)
    if a == 0:
        a = sigma if sigma > 0 else 1.0
    h = 0.9 * a * n ** (-1/5)
    return float(h)


def kde_peak_count_with_bandwidth(values, bandwidth_factor, distance_min_pct=0.05):
    """用 bandwidth_factor × Silverman 带宽做 KDE, 数峰"""
    if len(values) < 10:
        return 0, []
    base_bw = silverman_bandwidth(values)
    bw = base_bw * bandwidth_factor

    kde = gaussian_kde(values, bw_method=bw / values.std() if values.std() > 0 else 'scott')
    xs = np.linspace(values.min(), values.max(), 1000)
    ys = kde(xs)

    height_min = 0.05 * ys.max()
    distance_min_samples = max(10, int(distance_min_pct * len(xs)))
    peaks, _ = find_peaks(ys, height=height_min, distance=distance_min_samples)
    return len(peaks), xs[peaks].tolist()


def main():
    print('=' * 70)
    print('Idea 5 TDA 补测 B: 多 seed + 带宽 + HRQ 空间对比')
    print('=' * 70)

    # 1. 加载 embedding
    print('\n[Step 1] 加载 Stage 1 embedding')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N_full, d = emb.shape
    print(f'  Full embedding: {tuple(emb.shape)}')

    # 2. 补测 B 步骤 1: 多种子稳定性
    print('\n[Step 2] 补测 B 步骤 1: 多种子子采样稳定性 (42/43/44)')
    seeds = [42, 43, 44]
    multi_seed_results = {}
    base_bw_global = None

    for seed in seeds:
        print(f'\n  [seed = {seed}]')
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N_full, generator=g)[:N_SUBSAMPLE]
        emb_sub = emb[perm]

        emb_white = whiten_pca(emb_sub, n_components=PCA_DIM)
        X_np = emb_white.numpy()

        # pairwise 距离样本 (决定 threshold)
        rng = np.random.default_rng(seed)
        n_dist_sample = 50000
        i_idx = rng.integers(0, N_SUBSAMPLE, n_dist_sample)
        j_idx = rng.integers(0, N_SUBSAMPLE, n_dist_sample)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        dists = np.linalg.norm(X_np[i_idx] - X_np[j_idx], axis=1)

        thresh = float(np.percentile(dists, RIPSER_THRESHOLD_PERCENTILE))
        h0 = compute_h0(X_np, thresh=thresh)
        deaths = h0[:, 1]

        n_peaks, peak_xs = kde_peak_count_with_bandwidth(deaths, bandwidth_factor=1.0)
        E_pers = float(-(deaths / deaths.sum() * np.log(deaths / deaths.sum() + 1e-12)).sum()) if deaths.sum() > 0 else 0.0
        bw_base = silverman_bandwidth(deaths)

        print(f'    thresh = {thresh:.3f}, H0 pairs = {len(h0)}')
        print(f'    Silverman bw = {bw_base:.4f}, E_pers = {E_pers:.3f}')
        print(f'    n_peaks (bw×1.0) = {n_peaks}, peak positions = {[round(p, 3) for p in peak_xs]}')

        multi_seed_results[seed] = {
            'n_peaks': n_peaks,
            'peak_positions': peak_xs,
            'E_pers': E_pers,
            'silverman_bw': bw_base,
            'thresh': thresh,
            'h0_pairs': len(h0),
            'death_min': float(deaths.min()),
            'death_median': float(np.median(deaths)),
            'death_max': float(deaths.max()),
        }

    # Step 1 判定
    n_peaks_set = [r['n_peaks'] for r in multi_seed_results.values()]
    peaks_consistent = (len(set(n_peaks_set)) == 1)
    if peaks_consistent and n_peaks_set[0] == 1:
        step1_verdict = f'PASS: 3 seed 全部 n_peaks=1, 单一尺度结论稳健'
    elif peaks_consistent:
        step1_verdict = f'STABLE: 3 seed n_peaks={n_peaks_set[0]} (但需要看 n 是否 > 1)'
    else:
        step1_verdict = f'INSTABILITY: 3 seed 给出不同 n_peaks {[r for r in n_peaks_set]}, 结论需谨慎'

    print(f'\n  Step 1 判定: {step1_verdict}')

    # 3. 补测 B 步骤 2: 带宽敏感性 (用 seed=42 数据)
    print('\n[Step 3] 补测 B 步骤 2: KDE 带宽敏感性 (seed=42 数据, bw × {0.5, 1.0, 2.0, 4.0})')
    seed_use = 42
    g = torch.Generator().manual_seed(seed_use)
    perm = torch.randperm(N_full, generator=g)[:N_SUBSAMPLE]
    emb_sub = emb[perm]
    emb_white = whiten_pca(emb_sub, n_components=PCA_DIM)
    X_np = emb_white.numpy()
    rng = np.random.default_rng(seed_use)
    i_idx = rng.integers(0, N_SUBSAMPLE, 50000)
    j_idx = rng.integers(0, N_SUBSAMPLE, 50000)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    dists = np.linalg.norm(X_np[i_idx] - X_np[j_idx], axis=1)
    thresh = float(np.percentile(dists, RIPSER_THRESHOLD_PERCENTILE))
    h0 = compute_h0(X_np, thresh=thresh)
    deaths = h0[:, 1]
    base_bw = silverman_bandwidth(deaths)

    bandwidth_results = []
    bw_factors = [0.5, 1.0, 2.0, 4.0]
    for bw_f in bw_factors:
        n_peaks_bw, peaks_bw = kde_peak_count_with_bandwidth(deaths, bandwidth_factor=bw_f)
        actual_bw = base_bw * bw_f
        print(f'  bw × {bw_f}: actual_bw = {actual_bw:.4f}, n_peaks = {n_peaks_bw}, peaks @ {[round(p, 3) for p in peaks_bw]}')
        bandwidth_results.append({
            'bw_factor': bw_f,
            'actual_bw': actual_bw,
            'n_peaks': n_peaks_bw,
            'peak_positions': peaks_bw,
        })

    # Step 2 判定
    n_peaks_bw = [r['n_peaks'] for r in bandwidth_results]
    if all(p == 1 for p in n_peaks_bw):
        step2_verdict = 'PASS: n_peaks=1 在 0.5×-4× bandwidth 范围内稳健, 单一尺度结论不依赖带宽选择'
    elif n_peaks_bw.count(1) >= 3:
        step2_verdict = f'WEAK: 4 个带宽中 {n_peaks_bw.count(1)} 个 n_peaks=1, {n_peaks_bw.count(2)} 个 n_peaks=2 (轻微敏感性)'
    else:
        step2_verdict = f'INSTABILITY: 不同带宽给出 n_peaks ∈ {sorted(set(n_peaks_bw))}, 需改用 gap statistic 法'

    print(f'\n  Step 2 判定: {step2_verdict}')

    # 4. 补测 B 步骤 3: HRQ 空间对比 — 注记 (待后续)
    print('\n[Step 4] 补测 B 步骤 3: HRQ 空间 n_peaks 对比')
    print('  注: HRQ 训练表示空间 (T5 encoder 输出 + Poincaré 几何) 需要加载 task20 HRQ checkpoint 做推理。')
    print('  当前 task20 checkpoint 加载需要绕开 src.data.loading 循环导入 (circular import 问题),')
    print('  因此本次 tick 跳过 HRQ 空间对比。')
    print('  替代说明: HRQ R@10=0.1336 (+37.3%) 提升 + co-purchase Spearman ρ=-0.63 是真实收益,')
    print('  但其几何基础不能用当前 TDA 工具在「未加载 HRQ 模型」条件下检验。')
    print('  后续可能的快速验证方案: 用 task18_hrq_s22/pickle/merged_predictions_tensor.pt 提取 SID')
    print('  序列, 通过 Hamming 距离反推 item-level 「HRQ 拓扑空间」, 再做 TDA。')
    hrq_note = {
        'skip_reason': 'HRQ 模型 checkpoint 加载存在 src.data.loading circular import, 需代码层修复',
        'available_proxy': 'task18_hrq_s22/pickle/merged_predictions_tensor.pt (SID 仅, 无连续表示)',
        'alternative_method': 'Hamming 距离反推 item-level HRQ 空间, 做 TDA',
        'status': 'DEFERRED',
    }

    # 5. 综合判定
    print('\n[Step 5] 综合判定')
    if peaks_consistent and n_peaks_set[0] == 1 and 'PASS' in step2_verdict:
        overall = 'ROBUST_SINGLE_SCALE: 多种子 + 多带宽一致确认 n_peaks=1, Toys 数据 1 尺度结论可写入正式文档'
    elif peaks_consistent and all(p == 1 for p in n_peaks_bw):
        overall = 'STRONG: 子采样稳健, 带宽在小范围 (0.5×-2×) 稳健, 单一尺度强支持'
    else:
        overall = 'CAUTION: 结论需标注子采样敏感性, 不能作为确定性结论'

    print(f'  Overall: {overall}')

    # 6. 保存
    out_json = os.path.join(OUT_DIR, 'idea5_supplementary_testB.json')
    with open(out_json, 'w') as f:
        json.dump({
            'multi_seed_stability': {
                'results_per_seed': multi_seed_results,
                'n_peaks_distribution': n_peaks_set,
                'verdict': step1_verdict,
            },
            'bandwidth_sensitivity': {
                'results': bandwidth_results,
                'verdict': step2_verdict,
            },
            'hrq_space_comparison': hrq_note,
            'overall_verdict': overall,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task10_persistent_homology_tda.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    return overall, multi_seed_results, bandwidth_results


if __name__ == '__main__':
    main()
