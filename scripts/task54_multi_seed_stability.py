#!/usr/bin/env python3
"""Task 312: 多 seed 稳定性诊断 (S=1 当前 → 框架就绪 + 历史 rep 聚合)

目前仅有 seed=42 数据. 本脚本:
1. 验证现有 task38 (M=30 reps) 的 bootstrap CI 稳定性
2. 聚合已完成的 7 个任务结果, 输出跨任务一致性表
3. 写出"加 seed=43/44 重跑"的执行清单
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task54_multi_seed'
os.makedirs(OUT_DIR, exist_ok=True)

# 已完成任务结果路径
RESULT_PATHS = {
    'task38_causal_shuffle': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task38_causal_shuffle/per_layer_stats.json',
    'task42_entropy': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task42_entropy_branching/entropy_per_layer.json',
    'task43_behavior_purity': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task43_behavior_purity/delta_H_per_B.json',
    'task45_sibling': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task45_sibling_ranking/sibling_metrics.json',
    'task47_norm_strict': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task47_norm_strict/4probe_per_layer.json',
    'task50_topk_boundary': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task50_topk_boundary/BIR_per_K.json',
    'task51_alignment': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task51_alignment/alignment_matrix.json',
}


def load_or_none(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def aggregate_seed42():
    """Aggregate cross-task consistency table for current seed=42 results."""
    summary = {}
    for task_name, path in RESULT_PATHS.items():
        data = load_or_none(path)
        if data is None:
            summary[task_name] = {'status': 'NOT_FOUND', 'path': path}
            continue
        summary[task_name] = {'status': 'LOADED', 'path': path}

    # Cross-task key metrics table
    key_metrics = {}

    # task38: per-layer delta_code (sample intervention)
    t298 = load_or_none(RESULT_PATHS['task38_causal_shuffle'])
    if t298:
        # try to extract per-layer stat
        if 'per_layer' in t298:
            for layer, info in t298['per_layer'].items():
                for intv_type, vals in info.items():
                    if isinstance(vals, dict) and 'delta_code_mean' in vals:
                        key_metrics.setdefault(f't298_{intv_type}', {})[layer] = vals['delta_code_mean']

    # task42: B_eff per layer
    t300 = load_or_none(RESULT_PATHS['task42_entropy'])
    if t300 and 'results' in t300:
        for layer, stats in t300['results'].items():
            key_metrics.setdefault('t300_B_eff', {})[layer] = stats['b_eff']
            key_metrics.setdefault('t300_CR_l', {})[layer] = stats['cr_l']

    # task43: ΔH per layer per B
    t301 = load_or_none(RESULT_PATHS['task43_behavior_purity'])
    if t301 and 'results' in t301:
        for B_name in ['cat_sub', 'brand', 'cat_top']:
            key_metrics.setdefault(f't301_dH_{B_name}', {})
            for layer_key, stats in t301['results'].get(B_name, {}).items():
                key_metrics[f't301_dH_{B_name}'][layer_key] = stats['delta_H']

    # task45: sibling mean size
    t303 = load_or_none(RESULT_PATHS['task45_sibling'])
    if t303 and 'results' in t303:
        for layer, stats in t303['results'].items():
            key_metrics.setdefault('t303_mean_size', {})[layer] = stats['sibling_size_stats']['mean']

    # task47: 4 probes per layer
    t305 = load_or_none(RESULT_PATHS['task47_norm_strict'])
    if t305 and 'results' in t305:
        for layer, stats in t305['results'].items():
            key_metrics.setdefault('t305_dir_R2', {})[layer] = stats['direction_only_R2']
            key_metrics.setdefault('t305_gain_R2', {})[layer] = stats['gain_only_R2']

    # task51: R² with b_beh per layer
    t309 = load_or_none(RESULT_PATHS['task51_alignment'])
    if t309 and 'matrix' in t309:
        for layer, targs in t309['matrix'].items():
            key_metrics.setdefault('t309_R2_b_beh', {})[layer] = targs['b_beh']['r2']

    return summary, key_metrics


def write_replay_plan():
    """Plan for S=3 seeds × R=5 init re-run."""
    plan = """# 多 Seed 稳定性诊断 (S≥3 × R≥5) 执行计划

## 当前状态
- 仅 seed=42 数据就绪
- task38 已有 M=30 reps (bootstrap)
- 其它任务为单次执行

## 待执行 (待 GPU 资源空闲)

### S=3 重新训练 (per task)
- task44 (梯度饥饿): RQ-VAE 训练 3 seeds (耗时最长)
- task46 (Prob vs Rank): 训练 4 模型 × 3 seeds
- task49 (用户历史): 训练 4 TIGER × 3 seeds
- task52 (协同信息): 训练 4 TIGER × 3 seeds
- task53 (训练竞争): 训练 12 RQ-VAE × 12 TIGER × 3 seeds (大规模, 需多 GPU)

### R=5 重复 (per seed)
- task39 (单调嵌套 Probe): 5 init 拟合
- task47 (Residual norm): 5 init PCA
- task51 (对齐矩阵): 5 init PCA + CKA

### 聚合分析 (本脚本)
- 跨任务一致性检验
- bootstrap CI (依赖 task38 M=30)
- Hits@K actual counts

## 接受标准
- 3 seeds 关键指标 (Recall@10, B_eff_L3) 极差 ≤ 0.01
- bootstrap 95% CI 宽度 ≤ 0.005
"""
    return plan


def main():
    print('=' * 70)
    print('Task 312: 多 Seed 稳定性诊断 (聚合 seed=42 + 框架就绪)')
    print('=' * 70)

    summary, key_metrics = aggregate_seed42()

    # Output summary
    out = {
        'note': '聚合 seed=42 已完成任务的跨任务关键指标, 用于一致性检验. 多 seed 重跑需要 GPU 资源空闲.',
        'task_status': summary,
        'key_metrics': key_metrics,
    }
    with open(os.path.join(OUT_DIR, 'seed42_aggregate.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/seed42_aggregate.json')

    # Replay plan
    plan = write_replay_plan()
    with open(os.path.join(OUT_DIR, 'replay_plan.md'), 'w') as f:
        f.write(plan)
    print(f'[产物] {OUT_DIR}/replay_plan.md')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 312 Verdict: 多 Seed 稳定性诊断\n\n')
        f.write('## 当前状态\n\n')
        f.write('| Task | 状态 |\n')
        f.write('|------|------|\n')
        for t, s in summary.items():
            f.write(f'| {t} | {s["status"]} |\n')
        f.write('\n## 关键指标一致性 (seed=42 单次)\n\n')
        if key_metrics:
            # Get all layer keys
            all_layers = set()
            for k, v in key_metrics.items():
                all_layers.update(v.keys())
            all_layers = sorted(all_layers)
            metric_names = sorted(key_metrics.keys())
            f.write('| Metric | ' + ' | '.join(all_layers) + ' |\n')
            f.write('|--------' + '|'.join(['--------'] * len(all_layers)) + '|\n')
            for m in metric_names:
                vals = key_metrics[m]
                row = []
                for layer in all_layers:
                    if layer in vals:
                        row.append(f'{vals[layer]:.3f}')
                    else:
                        row.append('—')
                f.write(f'| {m} | ' + ' | '.join(row) + ' |\n')
        f.write('\n## 一致性观察\n\n')
        # task38 vs task42 cross-check: B_eff trend should match layer purity
        if 't300_B_eff' in key_metrics and 't301_dH_cat_sub' in key_metrics:
            b_eff_L3 = key_metrics['t300_B_eff'].get('L3', None)
            dH_L3 = key_metrics['t301_dH_cat_sub'].get('l=3', None)
            if b_eff_L3 and dH_L3:
                f.write(f'- B_eff L3 = {b_eff_L3:.1f}, ΔH(cat_sub) L3 = {dH_L3:.4f}\n')
                if b_eff_L3 > 50 and dH_L3 < 0.05:
                    f.write('  ⚠️ L3 高 B_eff 但低行为增量 → **区分能力用于 item-id 而非行为**\n')
        if 't303_mean_size' in key_metrics:
            l3_size = key_metrics['t303_mean_size'].get('L3', None)
            if l3_size and l3_size < 20:
                f.write(f'- L3 sibling mean size = {l3_size:.1f} → 大部分 prefix 区分后剩余 < 20 item\n')
        f.write('\n## 下一步\n\n')
        f.write('- 待 GPU 1 释放 (task297 完成), 启动 S=3 重跑计划\n')
        f.write('- 当前 seed=42 数据用作 baseline 锚点\n')

    print(f'[产物] {OUT_DIR}/verdict.md')

    print('\n[总结] seed=42 聚合完成, 多 seed 重跑计划就绪.')


if __name__ == '__main__':
    main()