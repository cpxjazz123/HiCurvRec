"""task424: L1 超级码字 vs 商品热门度 + Zipf 定律检验

Step 1 (low cost): 验证 baseline 超级码字（cov@20=61.2% items）是否主要由
  热门商品构成。具体做法: 把 L1 code 按 item count 降序排列, 对 top-K codes
  (K=1, 5, 10, 20, 50) 计算其平均 item-popularity, 对比全体 item 的 mean。
  若是热门商品 → 平均 popularity > 总体 mean, 反之亦然。

Step 2 (low cost): L1 token frequency 是否符合 Zipf 定律?
  - log(rank) vs log(frequency) 的最小二乘斜率 α
  - 拟合 R²
  - KL 散度 vs 完美 Zipf (α=1)
  - 头部 10 / 50 / 100 项累积覆盖率

Step 3 (high cost, 暂不做): 压扁消融 — 不在本脚本
"""

import os
import json
import numpy as np
import torch
from scipy.stats import linregress

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/task_artifacts/results/exp388v5/task424_super_pop_zipf'
os.makedirs(OUT_DIR, exist_ok=True)

SID_PATHS = {
    'E_E_E_E':   f'{GRID}/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'H_E_E_E':   f'{GRID}/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_E_E':   f'{GRID}/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt',
    'H_H_H_H':   f'{GRID}/logs/inference/runs/task388v4_hhhh_s22/pickle/merged_predictions_tensor.pt',
}
POP_PATH = f'{GRID}/result/task418_aqe_head_tail/item_popularity.npy'
N_ITEMS = 11924
CODEBOOK_SIZE = 256
R10_REF = {'E_E_E_E': 0.09731, 'H_E_E_E': 0.04332, 'H_H_E_E': 0.00680, 'H_H_H_H': 0.00036}


def super_code_popularity_test(sid_l1, popularity, codebook_size, k_list=(1, 5, 10, 20, 50)):
    """对每个 variant, 按 L1 code usage 降序选 top-K, 计算这些 code 的 item 平均 popularity。"""
    counts = np.bincount(sid_l1, minlength=codebook_size)
    items_per_code = [[] for _ in range(codebook_size)]
    for item_idx, code in enumerate(sid_l1):
        items_per_code[code].append(popularity[item_idx])
    avg_pop_per_code = np.array([np.mean(ipc) if len(ipc) > 0 else 0.0 for ipc in items_per_code])
    # Sort by count desc
    sorted_idx = np.argsort(counts)[::-1]
    total_pop_mean = float(popularity.mean())

    out = {'total_pop_mean': total_pop_mean,
           'total_pop_median': float(np.median(popularity))}
    for k in k_list:
        topk_idx = sorted_idx[:k]
        topk_items = np.concatenate([items_per_code[i] for i in topk_idx if items_per_code[i]])
        topk_pop_mean = float(topk_items.mean())
        topk_pop_median = float(np.median(topk_items))
        # 与全体 mean 的差异
        z_score = (topk_pop_mean - total_pop_mean) / (popularity.std() / np.sqrt(len(topk_items)) + 1e-9)
        out[f'top{k}'] = {
            'k': k,
            'n_items': int(len(topk_items)),
            'mean_pop': topk_pop_mean,
            'median_pop': topk_pop_median,
            'global_mean_pop': total_pop_mean,
            'lift': topk_pop_mean - total_pop_mean,
            'z_score': float(z_score),
        }
    return out


def zipf_analysis(sid_l1, codebook_size):
    """Zipf 分析: 把 L1 code usage 按降序排, log-log 拟合。"""
    counts = np.bincount(sid_l1, minlength=codebook_size).astype(np.float64)
    freq_desc = np.sort(counts)[::-1]
    freq_desc = freq_desc[freq_desc > 0]
    if freq_desc.size < 2:
        return None
    rank = np.arange(1, len(freq_desc) + 1, dtype=np.float64)
    log_rank = np.log10(rank)
    log_freq = np.log10(freq_desc)
    res = linregress(log_rank, log_freq)
    slope = float(res.slope)
    intercept = float(res.intercept)
    r2 = float(res.rvalue ** 2)

    # 拟合残差 (log scale)
    expected = slope * log_rank + intercept
    rmse_log = float(np.sqrt(((log_freq - expected) ** 2).mean()))

    # Head 10 / 50 / 100 coverage
    n = freq_desc.sum()
    cum = np.cumsum(freq_desc) / n
    cov10 = float(cum[9]) if len(cum) > 9 else float(cum[-1])
    cov50 = float(cum[49]) if len(cum) > 49 else float(cum[-1])
    cov100 = float(cum[99]) if len(cum) > 99 else float(cum[-1])

    # 标准化 α: Zipf 标准 α=1, 偏离量 = |α+1|
    alpha_dev = abs(slope + 1)

    return {
        'n_active_codes': int(freq_desc.size),
        'slope_alpha': slope,
        'intercept': intercept,
        'r2': r2,
        'rmse_log10': rmse_log,
        'alpha_minus_one_dev': alpha_dev,
        'head10_cov': cov10,
        'head50_cov': cov50,
        'head100_cov': cov100,
        'top_freq_code_items': int(freq_desc[0]),
        'min_freq_code_items': int(freq_desc[-1]),
    }


def main():
    popularity = np.load(POP_PATH)
    assert popularity.shape == (N_ITEMS,)
    pop_global_mean = popularity.mean()
    print(f"Global popularity: mean={pop_global_mean:.2f}, median={np.median(popularity):.1f}, std={popularity.std():.2f}\n")

    results = {}
    print("="*100)
    print(f"{'Variant':<8}  {'Code':<6} {'N':>5} {'mean_pop':>9} {'lift':>8}  {'z':>6}  {'α(slope)':>9} {'R²':>7} {'α_dev':>7}  {'cov@10':>7} {'cov@50':>7}  {'R@10':>7}")
    print("="*100)

    for variant, path in SID_PATHS.items():
        sid = torch.load(path, map_location='cpu', weights_only=False).numpy().astype(np.int64)
        L1 = sid[1]

        sp = super_code_popularity_test(L1, popularity, CODEBOOK_SIZE)
        za = zipf_analysis(L1, CODEBOOK_SIZE)

        results[variant] = {
            'super_pop': sp,
            'zipf': za,
            'R@10_ref': R10_REF.get(variant, None),
        }

        # Print: top1, top5, α etc.
        for tag, label in [('top1', 'top1'), ('top5', 'top5'), ('top10', 'top10'),
                           ('top20', 'top20'), ('top50', 'top50')]:
            r = sp[tag]
            z = za
            if label == 'top5':
                print(f"{variant:<8}  {label:<6} {r['n_items']:>5d} {r['mean_pop']:>9.2f} "
                      f"{r['lift']:>8.2f}  {r['z_score']:>6.2f}  "
                      f"{z['slope_alpha']:>9.3f} {z['r2']:>7.3f} {z['alpha_minus_one_dev']:>7.3f}  "
                      f"{z['head10_cov']:>7.3f} {z['head50_cov']:>7.3f}  "
                      f"{R10_REF.get(variant, 0):>7.4f}")
            elif label == 'top1':
                print(f"{variant:<8}  {label:<6} {r['n_items']:>5d} {r['mean_pop']:>9.2f} "
                      f"{r['lift']:>8.2f}  {r['z_score']:>6.2f}  "
                      f"{' ':>9} {' ':>7} {' ':>7}  {' ':>7} {' ':>7}  "
                      f"{R10_REF.get(variant, 0):>7.4f}")
        print()

    # Save
    out_json = f'{OUT_DIR}/summary.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {out_json}")

    # Markdown report
    md = []
    md.append("# task424 — L1 超级码字 vs 热门度 + Zipf 检验\n\n")
    md.append(f"**日期**: 2026-07-15\n")
    md.append(f"**商品总数**: {N_ITEMS}\n")
    md.append(f"**全局 item popularity**: mean={pop_global_mean:.2f}, median={np.median(popularity):.1f}\n\n")
    md.append("## 1. Step 1 — 超级码字 vs item popularity\n\n")
    md.append("| Variant | Top-K | N items | mean pop | mean pop (global) | lift | z-score |\n")
    md.append("|---------|-------|---------|----------|-------------------|------|---------|\n")
    for v in SID_PATHS:
        for k_tag in ['top1', 'top5', 'top10', 'top20', 'top50']:
            r = results[v]['super_pop'][k_tag]
            md.append(f"| {v} | {k_tag} | {r['n_items']} | {r['mean_pop']:.2f} | "
                      f"{r['global_mean_pop']:.2f} | {r['lift']:+.2f} | {r['z_score']:.2f} |\n")
    md.append("\n## 2. Step 2 — Zipf 拟合\n\n")
    md.append("| Variant | n_active | slope α | R² | α - |α+1| | cov@10 | cov@50 | top1 items | min items | R@10 |\n")
    md.append("|---------|----------|---------|----|------------|--------|---------|-------------|------------|---------|\n")
    for v in SID_PATHS:
        z = results[v]['zipf']
        md.append(f"| {v} | {z['n_active_codes']} | {z['slope_alpha']:+.3f} | {z['r2']:.3f} | "
                  f"{z['alpha_minus_one_dev']:.3f} | {z['head10_cov']:.3f} | {z['head50_cov']:.3f} | "
                  f"{z['top_freq_code_items']} | {z['min_freq_code_items']} | {results[v]['R@10_ref']:.4f} |\n")

    md.append("\n## 3. R@10 vs Zipf α summary\n\n")
    md.append("| Variant | slope α | R@10 | 与 α=−1 接近度 |\n")
    md.append("|---------|---------|------|----------------|\n")
    for v in SID_PATHS:
        z = results[v]['zipf']
        md.append(f"| {v} | {z['slope_alpha']:+.3f} | {results[v]['R@10_ref']:.4f} | "
                  f"{'VERY CLOSE' if z['alpha_minus_one_dev'] < 0.1 else ('CLOSE' if z['alpha_minus_one_dev'] < 0.2 else 'FAR')} |\n")

    md_path = f'{OUT_DIR}/report.md'
    with open(md_path, 'w') as f:
        f.writelines(md)
    print(f"✓ Saved: {md_path}")


if __name__ == '__main__':
    main()
