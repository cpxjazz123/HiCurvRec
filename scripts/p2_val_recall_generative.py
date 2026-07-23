#!/usr/bin/env python3
"""P2 (final): val_R@10_Generative — 用现有数据直接对比

目的:
- 现有 task21 metrics.csv 已经在 val (evaluation split) 上记录:
  - val/loss (teacher forcing CE)
  - val/recall@10 (autoregressive generate() with next_k=4 = "predict last item")
- 现有 e2e eval JSON 在 test (testing split) 上记录同样的指标
- P2 把这两条曲线对齐, 验证: val_R@10_Generative (autoregressive) 是不是 e2e 指标的好代理

不重新跑任何推理, 全部基于已生成产物。

关键改动 vs P1:
- P1 看 val_R@10_NextK (predict position -K = predict last item) 涨 vs e2e 降 → 出现脱节
- P2 提的问题: val_R@10_Generative (predict truly-next item, sliding-window style) 是不是更可靠?
"""
import json
import os
import csv

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/p2_val_generative'
os.makedirs(OUT_DIR, exist_ok=True)


def read_val_metrics_csv():
    """Read val/recall@10 and val/loss from training metrics.csv."""
    path = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task19_aq_s3/csv/version_0/metrics.csv'
    out = []
    if not os.path.exists(path):
        print(f'NOT FOUND: {path}')
        return out
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {}
            # Val metrics are logged at epoch end, with global_step as 'step'
            try:
                step = int(row.get('step', '0'))
            except Exception:
                continue
            r['step'] = step
            # val/recall@10 (or recall@5), val/loss
            for k in row:
                if k.startswith('val/') and row[k] not in ('', None):
                    try:
                        r[k] = float(row[k])
                    except Exception:
                        pass
            if 'val/loss' in r or 'val/recall@10' in r:
                out.append(r)
    # Keep only val rows
    out = [r for r in out if any(k.startswith('val/') for k in r)]
    return out


def read_e2e_evals():
    """Read e2e test eval JSONs to get Recall@K on test set."""
    eval_dir = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task8'
    out = {}
    for fname in os.listdir(eval_dir):
        if not fname.startswith('task19_eval_AQ_v'):
            continue
        if fname.endswith('.json'):
            ckpt_step = fname.replace('task19_eval_AQ_v', '').replace('.json', '')
            try:
                ckpt_step = int(ckpt_step)
            except Exception:
                continue
            with open(os.path.join(eval_dir, fname)) as f:
                ev = json.load(f)
            # Handle different eval output formats
            metrics = ev.get('metrics', ev)
            r10 = metrics.get('val/recall@10', metrics.get('Recall@10', metrics.get('test_R@10')))
            r5 = metrics.get('val/recall@5', metrics.get('Recall@5', metrics.get('test_R@5')))
            n10 = metrics.get('val/ndcg@10', metrics.get('NDCG@10', metrics.get('test_NDCG@10')))
            out[ckpt_step] = {
                'Recall@5': r5,
                'Recall@10': r10,
                'NDCG@10': n10,
            }
    return out


def main():
    print('=' * 70)
    print('P2 (final): val_R@10_Generative ↔ e2e R@10 一致性分析')
    print('=' * 70)

    val_rows = read_val_metrics_csv()
    e2e = read_e2e_evals()

    # Build step-by-step val table
    print('\nValidation curves (from training CSV):')
    print(f"{'step':>6} {'val/loss':>10} {'val_R@5':>10} {'val_R@10':>10} {'val_NDCG@10':>12}")
    val_by_step = {}
    for r in val_rows:
        s = r['step']
        val_by_step[s] = r
        line = (f"{s:>6} "
                f"{r.get('val/loss', float('nan')):>10.4f} "
                f"{r.get('val/recall@5', float('nan')):>10.4f} "
                f"{r.get('val/recall@10', float('nan')):>10.4f} "
                f"{r.get('val/ndcg@10', float('nan')):>12.4f}")
        print(line)
    print(f'\n  Total val steps logged: {len(val_rows)}')

    # Compute val_R@10_Generative (last item prediction via autoregressive eval)
    # Task23 reports this AS val/recall@10. Aligned with e2e test R@10 from JSONs.
    print('\nE2E test R@10 from JSON:')
    for ckpt_step in sorted(e2e.keys()):
        m = e2e[ckpt_step]
        print(f"  ckpt v{ckpt_step}: R@5={m.get('Recall@5'):.4f}  R@10={m.get('Recall@10'):.4f}  NDCG@10={m.get('NDCG@10'):.4f}")

    # Align: val at step 1999 → e2e at ckpt 2000 (both are around the same training step)
    pairs = []
    for ckpt_step in sorted(e2e.keys()):
        val_step = ckpt_step - 1
        if val_step in val_by_step:
            v = val_by_step[val_step]
            e = e2e[ckpt_step]
            pairs.append({
                'train_step': val_step,
                'ckpt_step': ckpt_step,
                'val_R@10_Generative': v.get('val/recall@10'),
                'val_NDCG@10': v.get('val/ndcg@10'),
                'e2e_test_R@10': e.get('Recall@10'),
                'e2e_test_R@5': e.get('Recall@5'),
                'e2e_test_NDCG@10': e.get('NDCG@10'),
                'val_loss': v.get('val/loss'),
            })

    print('\n' + '=' * 70)
    print('Aligned val_R@10_Generative ↔ e2e R@10  (one row per ckpt)')
    print('=' * 70)
    print(f"{'ckpt':>6} {'val_R@10':>10} {'e2e_R@10':>10} {'val_loss':>10} {'divergence':>10}")
    for p in pairs:
        vr = p['val_R@10_Generative']
        er = p['e2e_test_R@10']
        diff = (vr - er) if (vr is not None and er is not None) else float('nan')
        div = '+OK' if abs(diff) < 0.005 else ('WARN_GAP' if diff != diff else '?')
        print(f"{p['ckpt_step']:>6} "
              f"{(vr if vr is not None else float('nan')):>10.4f} "
              f"{(er if er is not None else float('nan')):>10.4f} "
              f"{(p['val_loss'] if p['val_loss'] is not None else float('nan')):>10.4f} "
              f"{diff:>+10.4f} {div}")

    # Trend analysis: rank correlation between val and e2e as training proceeds
    print('\n' + '=' * 70)
    print('KEY DIAGNOSIS — is val_R@10_Generative a reliable proxy for e2e R@10?')
    print('=' * 70)
    # Identify best e2e step
    if pairs:
        best_e2e = max(pairs, key=lambda p: p.get('e2e_test_R@10') or 0)
        best_val = max(pairs, key=lambda p: p.get('val_R@10_Generative') or 0)
        print(f"\n  Best e2e R@10 ckpt = v{best_e2e['ckpt_step']} "
              f"(R@10 = {best_e2e['e2e_test_R@10']:.4f})")
        print(f"  Best val_R@10_Generative ckpt = v{best_val['ckpt_step']} "
              f"(R@10 = {best_val['val_R@10_Generative']:.4f})")
        if best_e2e['ckpt_step'] == best_val['ckpt_step']:
            print('  ✓ val_R@10_Generative 选择正确的 ckpt (与 e2e 最优一致)')
        else:
            print(f'  ✗ val_R@10_Generative 选了不同的 ckpt (val 选 v{best_val["ckpt_step"]}, 实际 best 是 v{best_e2e["ckpt_step"]})')

    # Check trend between consecutive steps
    if len(pairs) >= 2:
        # Find any divergence points
        divergences = []
        for i in range(1, len(pairs)):
            prev, cur = pairs[i-1], pairs[i]
            v_diff = (cur['val_R@10_Generative'] or 0) - (prev['val_R@10_Generative'] or 0)
            e_diff = (cur['e2e_test_R@10'] or 0) - (prev['e2e_test_R@10'] or 0)
            if (v_diff > 0) and (e_diff < 0):
                divergences.append((cur['ckpt_step'], v_diff, e_diff))
        if divergences:
            print(f'\n  发现 {len(divergences)} 个 train→eval 方向反转点:')
            for step, v, e in divergences:
                print(f'    step {step}: val_R@10_Generative {v:+.4f}, e2e_R@10 {e:+.4f}')
        else:
            print(f'\n  ✓ val_R@10_Generative 与 e2e_R@10 趋势一致 (无方向反转)')

    # Save
    out = {
        'val_curve': val_rows,
        'e2e_results': e2e,
        'aligned_pairs': pairs,
        'best_e2e_ckpt': best_e2e['ckpt_step'] if pairs else None,
        'best_val_ckpt': best_val['ckpt_step'] if pairs else None,
        'verdict': 'val_R@10_Generative reliable proxy' if (pairs and best_e2e['ckpt_step'] == best_val['ckpt_step']) else 'val_R@10_Generative diverges from e2e',
    }
    out_path = os.path.join(OUT_DIR, 'p2_val_R10_generative.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
