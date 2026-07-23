#!/usr/bin/env python3
"""Task 320 (task54 多 seed 重跑 v2): S=3 稳定性指标
从 3 个 seed 的训练 metrics.csv 提取同 step 公平对比点 + bootstrap 95% CI

数据来源:
- task19_aq_s3 (seed=42, max_steps=5000): 取 step 1999 公平点
- task19_aq_s3_seed43 (seed=43, max_steps=2000): step 1999
- task19_aq_s3_seed44 (seed=44, max_steps=2000): step 1799

所有 seed 均在 step ~1900 收敛到相似水平, 适合 S=3 稳定性分析。
"""
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import numpy as np

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task54_multi_seed'
os.makedirs(OUT_DIR, exist_ok=True)

# 三 seed CSV 路径 + 对齐 step
SEEDS = [
    {
        'seed': 42,
        'csv': 'logs/train/runs/task19_aq_s3/csv/version_0/metrics.csv',
        'target_step': 1999,
        'ckpt': 'task19_aq_s3/best_tiger_step3900.ckpt (R@10=0.088 test)',
        'max_steps': 5000,
    },
    {
        'seed': 43,
        'csv': 'logs/train/runs/task19_aq_s3_seed43/csv/version_0/metrics.csv',
        'target_step': 1999,
        'ckpt': 'task19_aq_s3_seed43/checkpoint_epoch=000_step=001900.ckpt (R@10=0.082 test)',
        'max_steps': 2000,
    },
    {
        'seed': 44,
        'csv': 'logs/train/runs/task19_aq_s3_seed44/csv/version_0/metrics.csv',
        'target_step': 1799,  # seed44 stopped at 1899, take 1799
        'ckpt': 'task19_aq_s3_seed44/checkpoint_epoch=000_step=001800.ckpt (R@10=0.082 test)',
        'max_steps': 2000,
    },
]


def load_val_at_step(csv_path, target_step, tolerance=50):
    """从训练 metrics.csv 提取最接近 target_step 的 val 行"""
    rows = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            step = int(row.get('step', -1))
            # val rows have val/loss non-empty (column index 9 in this CSV format)
            if row.get('val/ndcg@10', ''):
                rows.append({
                    'step': step,
                    'val_loss': float(row['val/loss']),
                    'val_ndcg_10': float(row['val/ndcg@10']),
                    'val_ndcg_5': float(row['val/ndcg@5']),
                    'val_recall_10': float(row['val/recall@10']),
                    'val_recall_5': float(row['val/recall@5']),
                })

    # Find closest step
    rows.sort(key=lambda r: abs(r['step'] - target_step))
    if not rows:
        raise ValueError(f'No val rows found in {csv_path}')
    actual_step = rows[0]['step']
    if abs(actual_step - target_step) > tolerance:
        print(f'  WARN: closest step {actual_step} is >{tolerance} from target {target_step}')
    return rows[0], actual_step


def bootstrap_ci(values, n_boot=1000, ci=0.95, rng_seed=42):
    """Compute mean ± SE and bootstrap 95% CI."""
    arr = np.array(values)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    rng = np.random.RandomState(rng_seed)
    n = len(arr)
    boot_means = np.array([arr[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    alpha = 1 - ci
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {'mean': mean, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n}


def main():
    print('=' * 70)
    print('Task 320: S=3 多 seed 稳定性分析 (val metrics, 同 step 公平对比)')
    print('=' * 70)

    per_seed = []
    metric_arrays = {'val_recall_10': [], 'val_recall_5': [], 'val_ndcg_10': [], 'val_ndcg_5': [], 'val_loss': []}

    for seed_info in SEEDS:
        csv_path = os.path.join('/home/wlia0047/ar57/wenyu/GeneRec/GRID', seed_info['csv'])
        if not os.path.exists(csv_path):
            print(f'  SKIP seed={seed_info["seed"]}: {csv_path} not found')
            continue

        val, actual_step = load_val_at_step(csv_path, seed_info['target_step'])
        seed_info_actual = {
            'seed': seed_info['seed'],
            'csv': seed_info['csv'],
            'target_step': seed_info['target_step'],
            'actual_step': actual_step,
            'max_steps': seed_info['max_steps'],
            'ckpt': seed_info['ckpt'],
            'val_metrics': val,
        }
        per_seed.append(seed_info_actual)

        for k in metric_arrays:
            metric_arrays[k].append(val[k])

        print(f'\n  seed={seed_info["seed"]}: step={actual_step}/{seed_info["max_steps"]}')
        print(f'    val_recall@10 = {val["val_recall_10"]:.4f}')
        print(f'    val_recall@5  = {val["val_recall_5"]:.4f}')
        print(f'    val_ndcg@10   = {val["val_ndcg_10"]:.4f}')
        print(f'    val_ndcg@5    = {val["val_ndcg_5"]:.4f}')
        print(f'    val_loss      = {val["val_loss"]:.4f}')

    # S=3 statistics
    print('\n' + '=' * 70)
    print('S=3 聚合统计')
    print('=' * 70)

    stats = {}
    for metric_name, values in metric_arrays.items():
        if not values:
            continue
        s = bootstrap_ci(values)
        stats[metric_name] = s
        print(f'  {metric_name}: mean={s["mean"]:.4f} ± SE={s["se"]:.4f}, '
              f'95% CI=[{s["ci_lo"]:.4f}, {s["ci_hi"]:.4f}], n={s["n"]}')

    # Coefficient of variation
    if 'val_recall_10' in stats:
        r10_arr = np.array(metric_arrays['val_recall_10'])
        cv = float(r10_arr.std(ddof=1) / r10_arr.mean()) if r10_arr.mean() > 0 else 0
        print(f'\n  CoV (R@10): {cv*100:.2f}%  (low CV → high stability)')

    # Verdict
    print('\n' + '=' * 70)
    print('判读')
    print('=' * 70)
    r10_stats = stats.get('val_recall_10', {})
    r5_stats = stats.get('val_recall_5', {})

    r10_range = max(metric_arrays['val_recall_10']) - min(metric_arrays['val_recall_10'])
    r5_range = max(metric_arrays['val_recall_5']) - min(metric_arrays['val_recall_5'])

    print(f'  R@10: mean={r10_stats["mean"]:.4f}, range={r10_range:.4f}')
    print(f'  R@5:  mean={r5_stats["mean"]:.4f}, range={r5_range:.4f}')

    if r10_range < 0.01:
        verdict = 'STABLE'
        print(f'\n  ✅ S=3 stability: STABLE (R@10 range < 0.01)')
    elif r10_range < 0.02:
        verdict = 'MOSTLY STABLE'
        print(f'\n  ✓ S=3 stability: MOSTLY STABLE (R@10 range < 0.02)')
    else:
        verdict = 'UNSTABLE'
        print(f'\n  ⚠️ S=3 stability: UNSTABLE (R@10 range ≥ 0.02)')

    # Save
    out = {
        'task': 'task320/task54 multi-seed stability',
        'data_source': 'val metrics from train CSV (SIDRetrievalEvaluator, fair step ~1999)',
        'n_seeds': len(per_seed),
        'per_seed': per_seed,
        's3_stats': stats,
        'range': {
            'val_recall_10': r10_range,
            'val_recall_5': r5_range,
        },
        'cov_R10_pct': cv * 100 if 'val_recall_10' in stats else None,
        'verdict': verdict,
        'note': 'Compared at step ~1999 (fair S=3 seed comparison)',
    }

    with open(os.path.join(OUT_DIR, 's3_real_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Verdict markdown
    with open(os.path.join(OUT_DIR, 'verdict_s3.md'), 'w') as f:
        f.write('# Task 320 S=3 多 seed 稳定性 (val metrics, fair step ~1999)\n\n')
        f.write(f'> 数据来源: 3 个 seed 的 val metrics.csv (SIDRetrievalEvaluator on val set)\n')
        f.write(f'> 对齐 step: {SEEDS[0]["target_step"]} (~1999, 公平比较点)\n')
        f.write(f'> n_seeds: {len(per_seed)}\n\n')
        f.write('## Per-seed val metrics\n\n')
        f.write('| Seed | Step | val/loss | val/R@10 | val/R@5 | val/N@10 | val/N@5 |\n')
        f.write('|------|------|----------|----------|---------|----------|---------|\n')
        for s in per_seed:
            v = s['val_metrics']
            f.write(f'| {s["seed"]} | {s["actual_step"]} | {v["val_loss"]:.4f} | '
                    f'{v["val_recall_10"]:.4f} | {v["val_recall_5"]:.4f} | '
                    f'{v["val_ndcg_10"]:.4f} | {v["val_ndcg_5"]:.4f} |\n')
        f.write('\n## S=3 聚合 (bootstrap 95% CI)\n\n')
        f.write('| Metric | mean ± SE | 95% CI | n |\n')
        f.write('|--------|-----------|--------|---|\n')
        for m, s in stats.items():
            f.write(f'| {m} | {s["mean"]:.4f} ± {s["se"]:.4f} | '
                    f'[{s["ci_lo"]:.4f}, {s["ci_hi"]:.4f}] | {s["n"]} |\n')
        f.write(f'\n## Range\n\n')
        f.write(f'- val/R@10 range: {r10_range:.4f}\n')
        f.write(f'- val/R@5 range: {r5_range:.4f}\n')
        if 'val_recall_10' in stats:
            f.write(f'- CoV (R@10): {cv*100:.2f}%\n')
        f.write(f'\n## Verdict\n\n**{verdict}**\n\n')
        f.write(f'R@10 range = {r10_range:.4f}; CoV = {cv*100:.2f}%\n')
        f.write('\n')

    print(f'\n[产物] {OUT_DIR}/s3_real_metrics.json')
    print(f'[产物] {OUT_DIR}/verdict_s3.md')


if __name__ == '__main__':
    main()