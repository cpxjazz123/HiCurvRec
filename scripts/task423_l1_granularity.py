"""task423: L1 粒度诊断 — "桶变大、粒度变粗" 叙事的量化验证

对每个 variant (E_E_E_E / H_E_E_E / H_H_E_E / H_H_H_H) 在 L1 层计算:
  1. items per L1 code 分布 (mean / std / percentiles)
  2. 头部分位点累积覆盖度: top-K 个最常用码字覆盖多少 fraction of items
  3. Gini 系数 (item-level L1 granularity disparity)
  4. 桶大小变异系数 CV = std/mean

注: items per code = code count. 平均 = total_items / unique_codes_used.
   假设: 若 H_E_E_E 的"平均每个码字覆盖商品数 >> baseline",
         支持"粒度变粗"叙事。
"""

import os
import json
import numpy as np
import torch

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/task_artifacts/results/exp388v5/task423_l1_granularity'
os.makedirs(OUT_DIR, exist_ok=True)

SID_PATHS = {
    'E_E_E_E':   f'{GRID}/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'H_E_E_E':   f'{GRID}/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_E_E':   f'{GRID}/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_H_H':   f'{GRID}/logs/inference/runs/task388v4_hhhh_s22/pickle/merged_predictions_tensor.pt',
}
N_ITEMS = 11924
CODEBOOK_SIZE = 256  # max code value + 1


def gini(counts):
    """Gini from positive counts; range [0,1], 0 = perfect equality."""
    c = np.sort(counts[counts > 0].astype(np.float64))
    if c.size == 0:
        return 0.0
    n = c.size
    cum = np.cumsum(c)
    lorenz = cum / cum[-1]
    # 简化 Gini: G = (2*sum(i*x_i) - (n+1)*sum(x_i)) / (n*sum(x_i)),  i = 1..n
    i = np.arange(1, n + 1)
    g = (2.0 * (i * c).sum() - (n + 1) * c.sum()) / (n * c.sum())
    return float(g)


def per_level_granularity(sid_l, codebook_size):
    counts = np.bincount(sid_l, minlength=codebook_size)
    used = counts[counts > 0]
    n_used = used.size
    n_items = counts.sum()
    if n_used == 0:
        return None
    mean_per_code = float(n_items / n_used)
    std_per_code = float(used.std())
    cv = std_per_code / mean_per_code if mean_per_code > 0 else 0.0
    g = gini(counts)

    # Percentiles of items-per-code distribution (computed over active codes)
    pcts = np.percentile(used, [10, 25, 50, 75, 90, 95, 99])

    # Top-K coverage: cumulative fraction of items covered by top-K most-used codes
    sorted_desc = np.sort(counts)[::-1]
    cum = np.cumsum(sorted_desc) / n_items
    cov_top1 = float(cum[0]) if len(cum) > 0 else 0.0
    cov_top5 = float(cum[4]) if len(cum) > 4 else float(cum[-1])
    cov_top10 = float(cum[9]) if len(cum) > 9 else float(cum[-1])
    cov_top20 = float(cum[19]) if len(cum) > 19 else float(cum[-1])

    # 50% coverage: how many top codes do you need to cover 50% of items?
    half_cover_n = int(np.searchsorted(cum, 0.5) + 1)
    pct_50_codes_cover = 0.5
    pct_50_top_K_frac = half_cover_n / n_used

    # Effective K = exp(H) over marginal distribution
    p = used / used.sum()
    H = -float((p * np.log(p)).sum())
    eff_K = float(np.exp(H))

    return {
        'n_items': int(n_items),
        'n_codes_used': int(n_used),
        'mean_items_per_code': mean_per_code,
        'std_items_per_code': std_per_code,
        'cv_items_per_code': float(cv),
        'gini': g,
        'eff_K': eff_K,
        'pct_items_per_code': {'p10': float(pcts[0]), 'p25': float(pcts[1]),
                               'p50': float(pcts[2]), 'p75': float(pcts[3]),
                               'p90': float(pcts[4]), 'p95': float(pcts[5]),
                               'p99': float(pcts[6])},
        'coverage_top1': cov_top1,
        'coverage_top5': cov_top5,
        'coverage_top10': cov_top10,
        'coverage_top20': cov_top20,
        'codes_needed_for_50pct_items': half_cover_n,
        'frac_codes_for_50pct_items': float(pct_50_top_K_frac),
    }


def main():
    summary = {}
    print(f"{'='*80}")
    print(f"{'Variant':<8} {'used':>6} {'mean':>8} {'std':>8} {'CV':>6} {'Gini':>6} {'effK':>7} "
          f"{'cov@1':>7} {'cov@5':>7} {'cov@10':>7} {'cov@20':>7} {'50%C%':>6}")
    print(f"{'='*80}")

    for variant, path in SID_PATHS.items():
        sid = torch.load(path, map_location='cpu', weights_only=False).numpy().astype(np.int64)
        L1 = sid[1]
        gs = per_level_granularity(L1, CODEBOOK_SIZE)
        summary[variant] = {'L1': gs, 'L0': per_level_granularity(sid[0], CODEBOOK_SIZE),
                            'L2': per_level_granularity(sid[2], CODEBOOK_SIZE),
                            'L3': per_level_granularity(sid[3], CODEBOOK_SIZE)}

        line = (f"{variant:<8} {gs['n_codes_used']:>6d} "
                f"{gs['mean_items_per_code']:>8.2f} {gs['std_items_per_code']:>8.2f} "
                f"{gs['cv_items_per_code']:>6.3f} {gs['gini']:>6.3f} "
                f"{gs['eff_K']:>7.1f} "
                f"{gs['coverage_top1']:>7.3f} {gs['coverage_top5']:>7.3f} "
                f"{gs['coverage_top10']:>7.3f} {gs['coverage_top20']:>7.3f} "
                f"{gs['frac_codes_for_50pct_items']:>6.2f}")
        print(line)

    # Save
    out_json = f'{OUT_DIR}/summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved: {out_json}")

    # Markdown report
    md_path = f'{OUT_DIR}/report.md'
    with open(md_path, 'w') as f:
        f.write("# task423 — L1 粒度诊断报告\n\n")
        f.write("**目的**: 检验 \"H 版本 L1 粒度变粗\" 叙事\n\n")
        f.write("## 1. 关键发现\n\n")
        # Headline statistic
        baseline = summary['E_E_E_E']['L1']
        hee = summary['H_E_E_E']['L1']
        print()

        f.write("| Variant | n_codes_used | mean_items/code | std | CV | Gini | eff_K | cov@1 | cov@5 | cov@10 | cov@20 |\n")
        f.write("|---------|--------------|-----------------|-----|-----|------|-------|-------|-------|--------|--------|\n")
        for variant in ['E_E_E_E', 'H_E_E_E', 'H_H_E_E', 'H_H_H_H']:
            gs = summary[variant]['L1']
            f.write(f"| {variant} | {gs['n_codes_used']} | {gs['mean_items_per_code']:.2f} | "
                    f"{gs['std_items_per_code']:.2f} | {gs['cv_items_per_code']:.3f} | "
                    f"{gs['gini']:.3f} | {gs['eff_K']:.1f} | "
                    f"{gs['coverage_top1']:.3f} | {gs['coverage_top5']:.3f} | "
                    f"{gs['coverage_top10']:.3f} | {gs['coverage_top20']:.3f} |\n")
        f.write("\n## 2. Items per code 分布 (active codes only)\n\n")
        f.write("| Variant | p10 | p25 | p50 (median) | p75 | p90 | p95 | p99 |\n")
        f.write("|---------|-----|-----|--------------|-----|-----|-----|-----|\n")
        for variant in ['E_E_E_E', 'H_E_E_E', 'H_H_E_E', 'H_H_H_H']:
            pcts = summary[variant]['L1']['pct_items_per_code']
            f.write(f"| {variant} | {pcts['p10']:.1f} | {pcts['p25']:.1f} | "
                    f"{pcts['p50']:.1f} | {pcts['p75']:.1f} | "
                    f"{pcts['p90']:.1f} | {pcts['p95']:.1f} | {pcts['p99']:.1f} |\n")
        f.write("\n## 3. 50% 覆盖率\n\n")
        f.write("| Variant | codes needed for 50% item coverage | fraction of active codes |\n")
        f.write("|---------|-------------------------------------|--------------------------|\n")
        for variant in ['E_E_E_E', 'H_E_E_E', 'H_H_E_E', 'H_H_H_H']:
            gs = summary[variant]['L1']
            f.write(f"| {variant} | {gs['codes_needed_for_50pct_items']} | "
                    f"{gs['frac_codes_for_50pct_items']:.3f} |\n")
    print(f"✓ Saved: {md_path}")


if __name__ == '__main__':
    main()
