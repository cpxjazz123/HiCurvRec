#!/usr/bin/env python3
"""P1 诊断: teacher forcing vs 自回归生成脱节

数据来源:
- val/loss, val/recall@10: task19_aq_s3/csv/version_0/metrics.csv (NextK 目标位置)
- end-to-end Recall@10: result/task21/task19_eval_AQ_v*.json (真正下一项)

关键发现:
- val_R@10_NextK (step 1999→3899):  0.0836 → 0.0921  (+10.2%) ↑ 持续上涨
- e2e R@10    (ckpt v2000→v3900):     0.1361 → 0.1297  (-4.7%) ↓ 顶峰后衰退
- val/loss    (step 1999→3899):       8.531 → 9.222    (+8.1%)  ↑ 反弹

→ 模型在"预测 -K 位置"上越来越准, 但在"预测真正下一项"上反而变差
→ val_R@10_NextK 不能代表 e2e 能力, 必须用 beam search predict-next 替代

注: 数据来自现有产物 (无新训练), 全部基于 task21 已有的 dense eval JSON + 训练 CSV。
"""
import json
import os

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/p1_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

# From training CSV: val/loss and val/recall@10 at the steps right before ckpt saves
VAL_METRICS = {
    1599: {'val_loss': 8.558, 'val_recall_at_10_nextK': 0.0788, 'val_recall_at_5_nextK': 0.0556},
    1799: {'val_loss': 8.553, 'val_recall_at_10_nextK': 0.0820, 'val_recall_at_5_nextK': 0.0555},
    1999: {'val_loss': 8.531, 'val_recall_at_10_nextK': 0.0836, 'val_recall_at_5_nextK': 0.0576},
    2199: {'val_loss': 8.539, 'val_recall_at_10_nextK': 0.0850, 'val_recall_at_5_nextK': 0.0596},
    3899: {'val_loss': 9.222, 'val_recall_at_10_nextK': 0.0921, 'val_recall_at_5_nextK': 0.0639},
}

# From result/task21 JSONs: end-to-end Recall@K on TEST set
E2E_RESULTS = {
    1600: {'Recall@5': 0.0838, 'Recall@10': 0.1188, 'NDCG@5': 0.0573, 'NDCG@10': 0.0687},
    1800: {'Recall@5': 0.0858, 'Recall@10': 0.1275, 'NDCG@5': 0.0590, 'NDCG@10': 0.0725},
    2000: {'Recall@5': 0.0945, 'Recall@10': 0.1361, 'NDCG@5': 0.0648, 'NDCG@10': 0.0783},
    3900: {'Recall@5': 0.0871, 'Recall@10': 0.1297, 'NDCG@5': 0.0593, 'NDCG@10': 0.0730},
}

# Seed43/44 for cross-seed sanity
E2E_SEEDS = {
    1900: {'seed': 43, 'Recall@5': 0.0880, 'Recall@10': 0.1293, 'NDCG@5': 0.0605, 'NDCG@10': 0.0738},
    1800: {'seed': 44, 'Recall@5': 0.0893, 'Recall@10': 0.1281, 'NDCG@5': 0.0611, 'NDCG@10': 0.0735},
}


def main():
    print('=' * 70)
    print('P1 诊断: teacher forcing val_R@10_NextK vs end-to-end R@10')
    print('=' * 70)

    # Build joint table
    rows = []
    # Step 2000 is in E2E_RESULTS but val was at step 1999
    # Step 3900 is in E2E_RESULTS but val was at step 3899
    for ckpt_step in sorted(set(list(E2E_RESULTS.keys()))):
        val_key = ckpt_step - 1 if ckpt_step % 100 == 0 else ckpt_step
        val = VAL_METRICS.get(val_key, {})
        e2e = E2E_RESULTS.get(ckpt_step, {})
        rows.append({
            'step': ckpt_step,
            'val_step': val_key,
            'val_loss': val.get('val_loss'),
            'val_R@10_NextK': val.get('val_recall_at_10_nextK'),
            'e2e_R@10': e2e.get('Recall@10'),
            'e2e_R@5': e2e.get('Recall@5'),
            'e2e_NDCG@10': e2e.get('NDCG@10'),
        })

    # Print table
    print(f"\n{'step':>6} {'val_loss':>10} {'val_R@10_NextK':>16} {'e2e_R@5':>10} {'e2e_R@10':>10} {'e2e_N@10':>10}  divergence?")
    print('-' * 80)
    for r in rows:
        div_flag = ''
        if r['val_R@10_NextK'] is not None and r['e2e_R@10'] is not None:
            div_flag = '(see trend)'
        vl = f"{r['val_loss']:>10.3f}" if r['val_loss'] is not None else f"{'-':>10}"
        vr = f"{r['val_R@10_NextK']:>16.4f}" if r['val_R@10_NextK'] is not None else f"{'-':>16}"
        print(f"{r['step']:>6} {vl} {vr} "
              f"{r['e2e_R@5']:>10.4f} {r['e2e_R@10']:>10.4f} "
              f"{r['e2e_NDCG@10']:>10.4f}  {div_flag}")

    # === Trend analysis ===
    print('\n' + '=' * 70)
    print('TREND ANALYSIS — 关键诊断')
    print('=' * 70)
    # Compare first vs last checkpoint we have e2e for
    if 2000 in E2E_RESULTS and 3900 in E2E_RESULTS:
        e2e_first = E2E_RESULTS[2000]['Recall@10']
        e2e_last = E2E_RESULTS[3900]['Recall@10']
        e2e_delta = e2e_last - e2e_first
        val_first = VAL_METRICS[1999]['val_recall_at_10_nextK']
        val_last = VAL_METRICS[3899]['val_recall_at_10_nextK']
        val_delta = val_last - val_first
        loss_first = VAL_METRICS[1999]['val_loss']
        loss_last = VAL_METRICS[3899]['val_loss']
        loss_delta = loss_last - loss_first

        print(f"\n  [step 1999/2000 → step 3899/3900] — over 1900 training steps:")
        print(f"    val_R@10_NextK : {val_first:.4f} → {val_last:.4f}  Δ = {val_delta:+.4f} ({(val_delta/val_first)*100:+.1f}%)")
        print(f"    val/loss       : {loss_first:.3f} → {loss_last:.3f}  Δ = {loss_delta:+.3f} ({(loss_delta/loss_first)*100:+.1f}%)")
        print(f"    e2e R@10 (TEST): {e2e_first:.4f} → {e2e_last:.4f}  Δ = {e2e_delta:+.4f} ({(e2e_delta/e2e_first)*100:+.1f}%)")

        # Best e2e step
        best_step = max(E2E_RESULTS, key=lambda s: E2E_RESULTS[s]['Recall@10'])
        print(f"\n  Best e2e R@10 step = {best_step} (R@10 = {E2E_RESULTS[best_step]['Recall@10']:.4f})")
        print(f"  Final (step 3900) e2e R@10 = {E2E_RESULTS[3900]['Recall@10']:.4f}")
        print(f"  Drop from best to step 3900: {E2E_RESULTS[best_step]['Recall@10'] - E2E_RESULTS[3900]['Recall@10']:+.4f}")

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — P1: 假设被证实 (有反向分叉)')
    print('=' * 70)
    if 2000 in E2E_RESULTS and 3900 in E2E_RESULTS and 1999 in VAL_METRICS and 3899 in VAL_METRICS:
        val_delta = VAL_METRICS[3899]['val_recall_at_10_nextK'] - VAL_METRICS[1999]['val_recall_at_10_nextK']
        e2e_delta = E2E_RESULTS[3900]['Recall@10'] - E2E_RESULTS[2000]['Recall@10']
        if val_delta > 0 and e2e_delta < 0:
            print('  ✓ 假设证实: val_R@10_NextK 持续上涨 (+{:.4f}), '
                  'e2e R@10 在 step 2000 后下降 ({:+.4f})'.format(val_delta, e2e_delta))
            print('  → 训练时监控的指标与最终交付能力已脱节')
            print('  → 必须用真实 beam search 预测下一项的 val_R@10_Generative 替代')

    # Save
    out = {
        'val_metrics_per_step': VAL_METRICS,
        'e2e_metrics_per_step': E2E_RESULTS,
        'cross_seed_e2e': E2E_SEEDS,
        'interpretation': {
            'phenomenon': 'val_R@10_NextK (predict position -K) and e2e R@10 (predict NEXT item) diverge',
            'val_R@10_NextK_trend': '+10.2% over 1900 steps',
            'e2e_R@10_trend': 'peaks at step 2000 then declines -4.7% by step 3900',
            'val_loss_trend': '8.53 → 9.22, +8.1% (also detects overfitting)',
            'best_e2e_step': 2000,
            'best_e2e_R@10': 0.1361,
            'final_step_e2e_R@10': 0.1297,
            'drop_from_best': -0.0064,
        },
    }
    out_path = os.path.join(OUT_DIR, 'p1_diagnostic.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()