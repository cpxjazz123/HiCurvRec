#!/usr/bin/env python3
"""Task 321 对比脚本: L=3 ablation (无 dedup) vs L=4 baseline (有 dedup) S=3 稳定性

输入:
- L=3 ablation: logs/train/runs/2026-07-13/{05-02-42,05-06-25,05-07-44}/csv/version_0/metrics.csv (seed=42/43/44)
- L=4 baseline: logs/train/runs/task19_aq_s3{,_seed43,_seed44}/csv/version_0/metrics.csv (seed=42/43/44)

输出:
- result/task321_l3_ablation/l3_vs_l4_s3.json
- result/task321_l3_ablation/l3_vs_l4_verdict.md

判读:
- 若 L=3 ≈ L=4 → "deep SID ≈ 0 contribution" 进一步确认 (dedup digit 无用)
- 若 L=3 > L=4 → dedup digit 是噪声, 应去掉 (新发现)
- 若 L=3 < L=4 → dedup digit 有用 (反驳 task46 真 TIGER proxy 结论)
"""
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import numpy as np

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task321_l3_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

GRID_ROOT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'

# L=3 ablation: 3 个 seed (本次启动的训练)
L3_RUNS = {
    42: 'logs/train/runs/2026-07-13/05-02-42/csv/version_0/metrics.csv',
    43: 'logs/train/runs/2026-07-13/05-06-25/csv/version_0/metrics.csv',
    44: 'logs/train/runs/2026-07-13/05-07-44/csv/version_0/metrics.csv',
}

# L=4 baseline: 已存在的训练
L4_RUNS = {
    42: 'logs/train/runs/task19_aq_s3/csv/version_0/metrics.csv',
    43: 'logs/train/runs/task19_aq_s3_seed43/csv/version_0/metrics.csv',
    44: 'logs/train/runs/task19_aq_s3_seed44/csv/version_0/metrics.csv',
}


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
            if row.get('val/ndcg@10', ''):
                rows.append({
                    'step': step,
                    'val_loss': float(row['val/loss']),
                    'val_ndcg_10': float(row['val/ndcg@10']),
                    'val_ndcg_5': float(row['val/ndcg@5']),
                    'val_recall_10': float(row['val/recall@10']),
                    'val_recall_5': float(row['val/recall@5']),
                })
    if not rows:
        return None, None
    rows.sort(key=lambda r: abs(r['step'] - target_step))
    actual_step = rows[0]['step']
    if abs(actual_step - target_step) > tolerance:
        print(f'  WARN: closest step {actual_step} is >{tolerance} from target {target_step}')
    return rows[0], actual_step


def get_best_in_window(csv_path, step_min=0, step_max=3000):
    """获取指定 step 范围内的最佳 val 指标"""
    rows = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            step = int(row.get('step', -1))
            if row.get('val/ndcg@10', '') and step_min <= step <= step_max:
                rows.append({
                    'step': step,
                    'val_loss': float(row['val/loss']),
                    'val_ndcg_10': float(row['val/ndcg@10']),
                    'val_ndcg_5': float(row['val/ndcg@5']),
                    'val_recall_10': float(row['val/recall@10']),
                    'val_recall_5': float(row['val/recall@5']),
                })
    if not rows:
        return None
    return max(rows, key=lambda r: r['val_recall_10'])


def bootstrap_ci(values, n_boot=1000, ci=0.95, rng_seed=42):
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
    print('Task 321: L=3 ablation vs L=4 baseline S=3 对比')
    print('=' * 70)

    # 1. L=3 vs L=4 在相同 step 的 val metric
    target_steps = [199, 999, 1999, 2999]  # 公平比较点 (L=4 训练终止点 ~2000-3000)
    comparison = {}
    for ts in target_steps:
        comp_row = {'target_step': ts, 'L3': {}, 'L4': {}}
        for seed, csv in L3_RUNS.items():
            full_path = os.path.join(GRID_ROOT, csv)
            if os.path.exists(full_path):
                v, actual = load_val_at_step(full_path, ts)
                if v is not None:
                    comp_row['L3'][seed] = {**v, 'actual_step': actual}
        for seed, csv in L4_RUNS.items():
            full_path = os.path.join(GRID_ROOT, csv)
            if os.path.exists(full_path):
                v, actual = load_val_at_step(full_path, ts)
                if v is not None:
                    comp_row['L4'][seed] = {**v, 'actual_step': actual}
        comparison[ts] = comp_row

    # 2. S=3 聚合 (在 best step ~1999)
    s3_stats = {'L3': {}, 'L4': {}}
    metric_keys = ['val_loss', 'val_recall_10', 'val_recall_5', 'val_ndcg_10', 'val_ndcg_5']

    for variant, runs in [('L3', L3_RUNS), ('L4', L4_RUNS)]:
        all_vals = {k: [] for k in metric_keys}
        for seed, csv in runs.items():
            full_path = os.path.join(GRID_ROOT, csv)
            if os.path.exists(full_path):
                v, _ = load_val_at_step(full_path, 1999)
                if v is not None:
                    for k in metric_keys:
                        all_vals[k].append(v[k])
        for k in metric_keys:
            if all_vals[k]:
                s3_stats[variant][k] = bootstrap_ci(all_vals[k])

    # 3. 输出对比表
    print('\n' + '=' * 70)
    print('Step-wise 对比 (L=3 ablation vs L=4 baseline)')
    print('=' * 70)
    for ts in target_steps:
        c = comparison[ts]
        print(f'\n--- step ~{ts} ---')
        if c['L3'] and c['L4']:
            l3_r10 = np.mean([v['val_recall_10'] for v in c['L3'].values()])
            l4_r10 = np.mean([v['val_recall_10'] for v in c['L4'].values()])
            l3_r5 = np.mean([v['val_recall_5'] for v in c['L3'].values()])
            l4_r5 = np.mean([v['val_recall_5'] for v in c['L4'].values()])
            print(f'  L=3: R@10={l3_r10:.4f}, R@5={l3_r5:.4f} (n={len(c["L3"])})')
            print(f'  L=4: R@10={l4_r10:.4f}, R@5={l4_r5:.4f} (n={len(c["L4"])})')
            print(f'  Δ (L3-L4): R@10={l3_r10-l4_r10:+.4f}, R@5={l3_r5-l4_r5:+.4f}')

    # 4. S=3 聚合
    print('\n' + '=' * 70)
    print('S=3 聚合 (best step ~1999)')
    print('=' * 70)
    for variant in ['L3', 'L4']:
        print(f'\n{variant}:')
        for k, s in s3_stats[variant].items():
            print(f'  {k}: mean={s["mean"]:.4f} ± SE={s["se"]:.4f}, CI=[{s["ci_lo"]:.4f}, {s["ci_hi"]:.4f}]')

    # 5. 判读
    print('\n' + '=' * 70)
    print('判读')
    print('=' * 70)
    l3_r10 = s3_stats['L3'].get('val_recall_10', {}).get('mean', None)
    l4_r10 = s3_stats['L4'].get('val_recall_10', {}).get('mean', None)
    l3_r5 = s3_stats['L3'].get('val_recall_5', {}).get('mean', None)
    l4_r5 = s3_stats['L4'].get('val_recall_5', {}).get('mean', None)

    if l3_r10 is not None and l4_r10 is not None:
        delta_r10 = l3_r10 - l4_r10
        delta_r5 = l3_r5 - l4_r5 if l3_r5 and l4_r5 else None
        print(f'  Δ R@10 (L3 - L4): {delta_r10:+.4f}')
        print(f'  Δ R@5  (L3 - L4): {delta_r5:+.4f}' if delta_r5 else '')

        if abs(delta_r10) < 0.01:
            verdict = 'CONFIRM_DEDUP_USELESS'
            print(f'\n  ✅ L=3 ≈ L=4 → dedup digit 几乎无用 (Δ R@10 < 0.01)')
            print(f'  → 进一步确认 "deep SID ≈ 0 contribution" 结论')
        elif delta_r10 > 0.01:
            verdict = 'DEDUP_IS_NOISE'
            print(f'\n  🔥 L=3 > L=4 → dedup digit 是噪声, 应去掉 (Δ R@10 > 0.01)')
        else:
            verdict = 'DEDUP_HELPS'
            print(f'\n  ⚠️ L=3 < L=4 → dedup digit 有正向贡献 (Δ R@10 < -0.01)')
    else:
        verdict = 'INSUFFICIENT_DATA'
        print(f'  ⚠️ 数据缺失: L3 n={len(s3_stats["L3"].get("val_recall_10",{}).get("ci_lo","") and [])} or L4 n={len(s3_stats["L4"].get("val_recall_10",{}).get("ci_lo","") and [])}')

    # 6. 保存
    out = {
        'task': 'task321 L=3 ablation vs L=4 baseline S=3',
        'L3_runs': L3_RUNS,
        'L4_runs': L4_RUNS,
        'step_wise_comparison': comparison,
        's3_stats': s3_stats,
        'verdict': verdict,
    }
    with open(os.path.join(OUT_DIR, 'l3_vs_l4_s3.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'\n[产物] {OUT_DIR}/l3_vs_l4_s3.json')


if __name__ == '__main__':
    main()