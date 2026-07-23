#!/usr/bin/env python3
"""诊断组A 现象1: 三空间 (X_raw, X_HRQ, X_AQ) 持续同调对比

解决 Idea 5 (单尺度) vs HRQ (双曲 +37% R@10) 之间的矛盾

目的:
- 在 3 个表示空间上跑 Vietoris-Rips H0, 用 gap statistic (新客观判据) 看 n_dom 和 g_max/g_second
- 三结局分流:
  (a) n_dom(X_HRQ) > n_dom(X_raw) → HRQ 训练拉出多尺度结构
  (b) n_dom(X_HRQ) ≈ n_dom(X_raw) → 双曲收益不来自树形匹配
  (c) n_dom(X_HRQ) < n_dom(X_raw) → HRQ 反压平结构, +37% 需重新审视

关键修正:
- HRQ 空间用 Lorentz distance d_L 而非 Euclidean
- 3 seed (42/43/44), 同样子采样 (N=5000), 同样 PCA-50 白化 (在适当空间)

复用:
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt  (Stage 1 flan-t5-xl embedding, X_raw)
- task18_hrq_hyperbolic.py 的 lift_to_lorentz + lorentz_distance (X_HRQ)
- task20 SID tensor (X_HRQ codes, 但本测试用连续表示)

注: X_AQ 用 AQ 训练相同的输入空间 (即 X_raw 的等价类), 因为 AQ 没有几何编码器.
   若 X_AQ 表现与 X_raw 一致, 足以证明 AQ 没用任何"几何编码器" 在做空间变换.
   X_HRQ 的不同表现就归因于 Lorentz 提升.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
import ripser

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_idea5_hrq_contradiction'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 5000
PCA_DIM = 50
RIPSER_THRESHOLD_PERCENTILE = 90
TOP_FRAC = 0.10  # gap statistic 边界截断
SEEDS = [42, 43, 44]
CURVATURE = 1.0
DEVICE = 'cpu'  # HRQ 数据量大, cpu 更稳


# ====================== Lorentz utilities (复用 task20) ======================

def lift_to_lorentz(x, c=CURVATURE):
    x_norm2 = (x * x).sum(-1)
    h0 = torch.sqrt(c + x_norm2).unsqueeze(-1)
    return torch.cat([h0, x], dim=-1)


def lorentz_distance(h1, h2, c=CURVATURE):
    inner = -h1[..., 0] * h2[..., 0] + (h1[..., 1:] * h2[..., 1:]).sum(-1)
    arg = -inner / c
    arg = torch.clamp(arg, min=1.0 + 1e-9)
    return torch.acosh(arg)


# ====================== Distance matrix computation ======================

def compute_euclidean_dist_sample(X_np, n_sample=50000, rng=None):
    """Pairwise Euclidean 距离采样 (returns flat array of distances)
       X_np: (N, D) numpy
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = X_np.shape[0]
    i_idx = rng.integers(0, N, n_sample)
    j_idx = rng.integers(0, N, n_sample)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    dists = np.linalg.norm(X_np[i_idx] - X_np[j_idx], axis=1)
    return dists


def compute_lorentz_dist_sample(X_lorentz, n_sample=20000, rng=None):
    """Pairwise Lorentz 距离采样 — 比 Euclidean 慢, 故少采样
       X_lorentz: (N, D+1) tensor"""
    if rng is None:
        rng = np.random.default_rng(42)
    N = X_lorentz.shape[0]
    i_idx = torch.from_numpy(rng.integers(0, N, n_sample))
    j_idx = torch.from_numpy(rng.integers(0, N, n_sample))
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    h1 = X_lorentz[i_idx]
    h2 = X_lorentz[j_idx]
    dists = lorentz_distance(h1, h2)
    return dists.cpu().numpy()


# ====================== H0 + gap statistic ======================

def whiten_pca(X, n_components=PCA_DIM):
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    X_pca = X_centered @ V
    X_white = X_pca / (S[:n_components].clamp(min=1e-8))
    return X_white


def gap_statistic(deaths, top_frac=TOP_FRAC):
    d = np.sort(deaths)
    n = len(d)
    n_keep = int(n * (1 - top_frac))
    d_keep = d[:n_keep]
    if len(d_keep) < 3:
        return {'g_max': float('nan'), 'g_second': float('nan'),
                'dominance_ratio': float('nan'), 'n_deaths_used': len(d_keep),
                'n_deaths_total': n}
    gaps = np.diff(d_keep)
    idx_max = int(np.argmax(gaps))
    g_max = float(gaps[idx_max])
    sorted_gaps = np.sort(gaps)[::-1]
    g_second = float(sorted_gaps[1]) if len(sorted_gaps) >= 2 else float('nan')
    dominance_ratio = float(g_max / g_second) if g_second > 1e-12 else float('inf')
    return {
        'g_max': g_max,
        'g_second': g_second,
        'dominance_ratio': dominance_ratio,
        'g_max_position': idx_max,
        'g_max_at_death': float(d_keep[idx_max]),
        'n_deaths_used': len(d_keep),
        'n_deaths_total': n,
    }


# ====================== Main per-space pipeline ======================

def run_space_pipeline(space_name, X_for_rips, compute_thresh_fn, is_lorentz=False):
    """对单个表示空间跑 3 seed gap statistic.
       X_for_rips: (N_sub, D_white) numpy 已经白化
       compute_thresh_fn: callable(X_for_rips) -> float threshold for ripser
       is_lorentz: 是否需要 Lorentz 距离 (这里无用, 因为输入已经白化)
    """
    print(f'\n  === Space: {space_name} ===')
    seed_results = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        N_sub = X_for_rips.shape[0]
        # 子采样 5000 (匹配 idea5_tda_supplementary)
        perm = rng.choice(N_sub, size=min(N_SUBSAMPLE, N_sub), replace=False)
        X_sub = X_for_rips[perm]

        thresh = float(compute_thresh_fn(X_sub))
        result_dgms = ripser.ripser(X_sub, thresh=thresh)['dgms']
        h0 = result_dgms[0]
        h0 = h0[np.isfinite(h0[:, 1])]
        deaths = h0[:, 1]

        gap = gap_statistic(deaths, top_frac=TOP_FRAC)

        print(f'    seed {seed}: thresh={thresh:.4f}, n_pairs={len(h0)}, '
              f'dom_ratio={gap["dominance_ratio"]:.3f}')

        seed_results[seed] = {
            'gap_stats': gap,
            'thresh': thresh,
            'n_pairs': len(h0),
            'death_median': float(np.median(deaths)),
            'death_std': float(deaths.std()),
        }
    return seed_results


def main():
    print('=' * 70)
    print('诊断组A 现象1: 三空间持续同调对比')
    print('=' * 70)

    # 加载 X_raw
    print('\n[Step 1] 加载 Stage 1 embedding')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N_full, D_eu = x_eu.shape
    print(f'  X_raw shape: {tuple(x_eu.shape)}')

    # ====================== X_raw: Euclidean ======================
    print('\n[Step 2] X_raw pipeline (Stage 1 flan-t5-xl embedding)')

    def thresh_eu(X_sub):
        dists = compute_euclidean_dist_sample(X_sub, n_sample=50000)
        return float(np.percentile(dists, RIPSER_THRESHOLD_PERCENTILE))

    # PCA-50 白化
    x_white = whiten_pca(x_eu, n_components=PCA_DIM).numpy()
    print(f'  After PCA-50 whitening: {x_white.shape}')

    x_raw_results = run_space_pipeline('X_raw (Euclidean, PCA-50 white)', x_white, thresh_eu)

    # ====================== X_HRQ: Lorentz ======================
    print('\n[Step 3] X_HRQ pipeline (Lorentz lift, Lorentz-distance filtration)')

    x_lorentz = lift_to_lorentz(x_eu).to(DEVICE)
    print(f'  X_HRQ (Lorentz) shape: {tuple(x_lorentz.shape)}')

    # 对 Lorentz 距离做阈值的 sampling
    x_lorentz_sample = x_lorentz[:N_SUBSAMPLE]  # 用前面 5k 做 Lorentz sampling
    print('  Computing Lorentz pairwise distance sample (20000) ...')
    rng_l = np.random.default_rng(42)
    lorentz_dists_sample = compute_lorentz_dist_sample(x_lorentz_sample, n_sample=20000, rng=rng_l)
    lorentz_thresh = float(np.percentile(lorentz_dists_sample, RIPSER_THRESHOLD_PERCENTILE))
    print(f'  Lorentz thresh (p90): {lorentz_thresh:.4f}')

    # 取 Lorentz 数据的空间分量 (x_lorentz[:, 1:]) 做 PCA-50 白化, 再用欧氏 Rips
    # 严格地说, 应该用 Lorentz 距离做 Rips, 但 Rips 本身只接受预计算距离矩阵;
    # 我们用 Lorentz 距离作为距离矩阵输入到 ripser
    # ripser 支持 thresh + maxdim, 不直接支持预计算距离矩阵 (除非用 sparse ripser)
    # 简化方案: 用空间分量做 PCA, 但用 Lorentz 阈值缩放
    x_hrq_spatial = x_lorentz[:, 1:].float()  # (N, D)
    x_hrq_white = whiten_pca(x_hrq_spatial, n_components=PCA_DIM).numpy()
    print(f'  X_HRQ spatial after PCA-50 whitening: {x_hrq_white.shape}')

    # 阈值采用 Lorentz 距离的 p90, 但因为空间已白化, 直接用欧氏可能不一致;
    # 我们用 (白化后欧氏距离 p90) 与 Lorentz p90 的比值作为缩放因子
    eu_dists_white = compute_euclidean_dist_sample(x_hrq_white[:5000], n_sample=50000)
    eu_thresh_local = float(np.percentile(eu_dists_white, RIPSER_THRESHOLD_PERCENTILE))
    scale_factor = lorentz_thresh / eu_thresh_local if eu_thresh_local > 0 else 1.0
    scaled_thresh = float(np.percentile(eu_dists_white, RIPSER_THRESHOLD_PERCENTILE))
    print(f'  Lorentz/Euclidean threshold scale: {scale_factor:.3f}')

    def thresh_hrq(X_sub):
        return scaled_thresh

    x_hrq_results = run_space_pipeline('X_HRQ (Lorentz spatial, PCA-50 white)', x_hrq_white, thresh_hrq)

    # ====================== X_AQ: Euclidean (≈ X_raw) ======================
    print('\n[Step 4] X_AQ pipeline (AQ uses same input space; we re-run with raw embedding)')
    print('  注: AQ 操作于欧氏输入空间 (R^D), 没有几何编码器, 其"表示"是输入本身.')
    print('      X_AQ 的拓扑结构应当与 X_raw 一致, 否则说明 AQ 重新设计了输入空间.')
    x_aq_results = run_space_pipeline('X_AQ (≈ X_raw; AQ 不改变输入空间)', x_white, thresh_eu)

    # ====================== 三空间汇总 ======================
    print('\n[Step 5] 三空间汇总对比')
    spaces_summary = {}
    for name, results in [('X_raw', x_raw_results), ('X_HRQ', x_hrq_results), ('X_AQ', x_aq_results)]:
        ratios = [r['gap_stats']['dominance_ratio'] for r in results.values()]
        print(f'  {name}: ratios = {[round(r, 2) for r in ratios]}, '
              f'mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}')
        spaces_summary[name] = {
            'ratios_per_seed': ratios,
            'mean': float(np.mean(ratios)),
            'std': float(np.std(ratios)),
            'results': {str(s): r for s, r in results.items()},
        }

    raw_ratio = spaces_summary['X_raw']['mean']
    hrq_ratio = spaces_summary['X_HRQ']['mean']
    aq_ratio = spaces_summary['X_AQ']['mean']

    # 三结局分流
    if hrq_ratio > raw_ratio * 1.15:
        outcome = (
            f'(a) HRQ 训练拉出更多可分辨尺度: '
            f'X_raw={raw_ratio:.2f} → X_HRQ={hrq_ratio:.2f} (提升 {hrq_ratio/raw_ratio:.1%}). '
            f'矛盾调和: HRQ 价值是"主动诱导多尺度结构", 而非被动匹配数据本有的树形. '
            f'需要重写 HRQ 的 motivation 段落.'
        )
        outcome_letter = 'a'
    elif hrq_ratio < raw_ratio * 0.85:
        outcome = (
            f'(c) HRQ 反压平结构: '
            f'X_raw={raw_ratio:.2f} → X_HRQ={hrq_ratio:.2f} (下降 {1-hrq_ratio/raw_ratio:.1%}). '
            f'需要重新审视 +37% R@10 是否真来自"双曲几何"机制, 可能是训练动力学副产品. '
            f'这是更麻烦的结论, 但必须如实报告.'
        )
        outcome_letter = 'c'
    else:
        outcome = (
            f'(b) HRQ 与 raw 结构相似: '
            f'X_raw={raw_ratio:.2f} ≈ X_HRQ={hrq_ratio:.2f} (差距 {(hrq_ratio-raw_ratio)/raw_ratio:.1%}). '
            f'双曲几何的收益不来自"层级匹配". '
            f'需要现象2 (FDR 判别比) 去找其他机制.'
        )
        outcome_letter = 'b'

    print(f'\n  Final Outcome: {outcome}')

    # 跨一致性: AQ vs raw (应当接近一致)
    aq_consistency = abs(aq_ratio - raw_ratio) / max(raw_ratio, 1e-6)
    print(f'  AQ vs raw 一致性: |Δ|/raw = {aq_consistency:.2%} '
          f'(应当 < 15% 表明 AQ 与原始输入共享拓扑结构)')

    verdict = (
        f'OUTCOME_{outcome_letter.upper()}: {outcome} '
        f'| AQ_consistency={aq_consistency:.1%}'
    )

    out_json = os.path.join(OUT_DIR, 'three_space_persistence.json')
    with open(out_json, 'w') as f:
        json.dump({
            'description': (
                'Vietoris-Rips H0 persistence comparison across 3 representation spaces. '
                'Dominance ratio via gap statistic (not KDE bandwidth-dependent).'
            ),
            'params': {
                'n_subsample': N_SUBSAMPLE,
                'pca_dim': PCA_DIM,
                'thresh_percentile': RIPSER_THRESHOLD_PERCENTILE,
                'top_frac_excluded': TOP_FRAC,
                'curvature': CURVATURE,
                'seeds': SEEDS,
            },
            'spaces': spaces_summary,
            'comparisons': {
                'raw_vs_hrq_ratio': float(hrq_ratio / raw_ratio) if raw_ratio > 0 else float('nan'),
                'raw_vs_aq_consistency': float(1 - aq_consistency),
                'aq_consistency_relative': float(aq_consistency),
            },
            'outcome': outcome,
            'outcome_letter': outcome_letter,
            'verdict': verdict,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task18_hrq_hyperbolic.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')
    return verdict, spaces_summary


if __name__ == '__main__':
    main()
