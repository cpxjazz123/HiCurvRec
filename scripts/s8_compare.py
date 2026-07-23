"""S=8 L=3 ablation final comparison (task322)
- 4 seeds from task321 (42/43/44/45) — finished, test metrics available
- 4 seeds from task322 (46/47/48/49) — in progress, will scrape from train.log
- L=4 baseline (3 seeds from task19_aq_s3 family)
"""
import os, json, re, numpy as np
GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'

# L=3 ablation: 8 seeds total
L3_RUNS_8 = {
    42: 'logs/train/runs/2026-07-13/05-02-42',  # task321 done (early stop ~1250)
    43: 'logs/train/runs/2026-07-13/05-06-25',  # task321 done (~1462)
    44: 'logs/train/runs/2026-07-13/05-07-44',  # task321 done (~1275)
    45: 'logs/train/runs/2026-07-13/05-26-58',  # task321 done (~1462)
    46: 'logs/train/runs/2026-07-13/seed46_l3',  # task322 in progress (long)
    47: 'logs/train/runs/2026-07-13/seed47_l3',  # task322 in progress
    48: 'logs/train/runs/2026-07-13/seed48_l3',  # task322 in progress
    49: 'logs/train/runs/2026-07-13/seed49_l3',  # task322 in progress
}

# L=4 baseline: 3 seeds
L4_RUNS_3 = {
    42: 'logs/train/runs/task19_aq_s3',
    43: 'logs/train/runs/task19_aq_s3_seed43',
    44: 'logs/train/runs/task19_aq_s3_seed44',
}

METRIC_NAMES = ['r10', 'r5', 'n10', 'n5']
METRIC_FULL = {'r10': 'recall@10', 'r5': 'recall@5', 'n10': 'ndcg@10', 'n5': 'ndcg@5'}


def load_val(run_dir):
    """Load val series from metrics.csv. Return list of dicts."""
    rows = []
    csv = os.path.join(GRID, run_dir, 'csv/version_0/metrics.csv')
    if not os.path.exists(csv):
        # fallback: parse train.log for "Metric val/recall@10 improved" lines
        log = os.path.join(GRID, run_dir, 'train.log')
        if not os.path.exists(log): return []
        # This is a fallback - extract best val from log "New best score"
        last_val = None
        with open(log) as f:
            for line in f:
                m = re.search(r"epoch \d+, step (\d+)", line)
                if m and "New best score" in line:
                    last_val = int(m.group(1))
        # Cannot construct full val series from log alone; return empty
        return []
    with open(csv) as f:
        h = f.readline().strip().split(',')
        for line in f:
            ps = line.strip().split(',')
            if len(ps) != len(h): continue
            r = dict(zip(h, ps))
            if not r.get('val/recall@10', ''): continue
            out = {'step': int(r['step'])}
            for short in METRIC_NAMES:
                full = METRIC_FULL[short]
                v = r.get(f'val/{full}', '')
                if v: out[short] = float(v)
            for short in METRIC_NAMES:
                out.setdefault(short, None)
            rows.append(out)
    return rows


def extract_test(run_dir):
    """Extract final test metrics from train.log's last 'Metrics:' line."""
    log = os.path.join(GRID, run_dir, 'train.log')
    if not os.path.exists(log): return None
    last = ''
    with open(log) as f:
        for line in f:
            if 'Metrics:' in line and 'test/' in line:
                last = line
    if not last: return None
    out = {}
    for short, full in METRIC_FULL.items():
        m = re.search(rf"'test/{full}':\s*tensor\(([\d.]+)\)", last)
        out[short] = float(m.group(1)) if m else None
    m = re.search(r"'val/recall@10':\s*tensor\(([\d.]+)\)", last)
    out['val_final_r10'] = float(m.group(1)) if m else None
    return out


def best_val_per_seed(run_dir):
    """For each seed, return best val/R@10 row + step + all val metrics + latest step."""
    rs = load_val(run_dir)
    if not rs:
        return None
    valid = [r for r in rs if r.get('r10') is not None]
    if not valid: return None
    best = max(valid, key=lambda r: r['r10'])
    return {
        'best_val': best,
        'best_step': best['step'],
        'best_r10': best['r10'],
        'latest_step': rs[-1]['step'],
        'latest_r10': rs[-1].get('r10'),
    }


per_seed = {'L3': {}, 'L4': {}}
test_metrics = {'L3': {}, 'L4': {}}

for s, run in L3_RUNS_8.items():
    bv = best_val_per_seed(run)
    if bv:
        per_seed['L3'][s] = {
            'best_val_step': bv['best_step'],
            'best_val_r10': bv['best_r10'],
            'r10': bv['best_val']['r10'], 'r5': bv['best_val']['r5'],
            'n10': bv['best_val']['n10'], 'n5': bv['best_val']['n5'],
            'latest_step': bv['latest_step'], 'latest_r10': bv['latest_r10'],
        }
    test_metrics['L3'][s] = extract_test(run)

for s, run in L4_RUNS_3.items():
    bv = best_val_per_seed(run)
    if bv:
        per_seed['L4'][s] = {
            'best_val_step': bv['best_step'],
            'best_val_r10': bv['best_r10'],
            'r10': bv['best_val']['r10'], 'r5': bv['best_val']['r5'],
            'n10': bv['best_val']['n10'], 'n5': bv['best_val']['n5'],
        }
    if s == 42:
        test_metrics['L4'][s] = {
            'r10': 0.0881, 'r5': 0.0751, 'n10': 0.0454, 'n5': None,
            '_src': 'task19_aq_s3 step=3900 final test (论文 Table 1 RQ-VAE 行)',
        }


def stat(rs):
    if not rs: return None
    arr = np.array([x for x in rs if x is not None])
    if len(arr) == 0: return None
    return {
        'mean': float(arr.mean()),
        'se': float(arr.std(ddof=1)/np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        'n': int(len(arr)),
        'values': [float(x) for x in arr if x is not None],
    }


L3_DONE = [42, 43, 44, 45]  # task321 已完成
L3_LATE = [46, 47, 48, 49]  # task322 长训练

stats = {'val_s4': {}, 'val_s8': {}, 'val_s4_late': {}, 'test_s4': {}, 'test_s8': {}}

for metric in ['r10', 'r5', 'n10', 'n5']:
    # val s4: L=3 seeds 42-45 vs L=4 seeds 42-44
    l3_s4 = [per_seed['L3'][s][metric] for s in L3_DONE if s in per_seed['L3'] and per_seed['L3'][s].get(metric) is not None]
    l4_s4 = [per_seed['L4'][s][metric] for s in [42, 43, 44] if s in per_seed['L4'] and per_seed['L4'][s].get(metric) is not None]
    if l3_s4 and l4_s4:
        stats['val_s4'][metric] = {
            'L3': stat(l3_s4), 'L4': stat(l4_s4),
            'delta_mean': float(np.mean(l3_s4) - np.mean(l4_s4)),
            'delta_pct': float((np.mean(l3_s4) / np.mean(l4_s4) - 1) * 100),
        }

    # val s8: all 8 L=3 seeds vs L=4
    l3_all = []
    for grp in [L3_DONE, L3_LATE]:
        for s in grp:
            if s in per_seed['L3'] and per_seed['L3'][s].get(metric) is not None:
                l3_all.append(per_seed['L3'][s][metric])
    if l3_all and l4_s4:
        stats['val_s8'][metric] = {
            'L3': stat(l3_all), 'L4': stat(l4_s4),
            'delta_mean': float(np.mean(l3_all) - np.mean(l4_s4)),
            'delta_pct': float((np.mean(l3_all) / np.mean(l4_s4) - 1) * 100),
            'n_l3': len(l3_all),
        }

    # val s4 late: only seeds 46-49 vs L=4 (long-training)
    l3_late = [per_seed['L3'][s][metric] for s in L3_LATE if s in per_seed['L3'] and per_seed['L3'][s].get(metric) is not None]
    if l3_late and l4_s4:
        stats['val_s4_late'][metric] = {
            'L3': stat(l3_late), 'L4': stat(l4_s4),
            'delta_mean': float(np.mean(l3_late) - np.mean(l4_s4)),
            'delta_pct': float((np.mean(l3_late) / np.mean(l4_s4) - 1) * 100),
        }

    # test s4: L=3 (4 seeds, task321 done) test vs L=4 test
    l3_test_s4 = [test_metrics['L3'][s].get(metric) for s in L3_DONE if s in test_metrics['L3'] and test_metrics['L3'][s] is not None and test_metrics['L3'][s].get(metric) is not None]
    l4_test = test_metrics['L4'].get(42, {})
    l4_test_val = l4_test.get(metric) if l4_test else None
    if l3_test_s4:
        stats['test_s4'][metric] = {
            'L3': stat(l3_test_s4),
            'L4_seed42_test': l4_test_val,
            'L3_mean_test': float(np.mean(l3_test_s4)),
            'L3_vs_L4_test_pct': float((np.mean(l3_test_s4) / l4_test_val - 1) * 100) if l4_test_val else None,
        }

    # test s8: all 8 L=3 (incl task322)
    l3_test_s8 = [test_metrics['L3'][s].get(metric) for s in L3_DONE + L3_LATE if s in test_metrics['L3'] and test_metrics['L3'][s] is not None and test_metrics['L3'][s].get(metric) is not None]
    if l3_test_s8:
        stats['test_s8'][metric] = {
            'L3': stat(l3_test_s8),
            'L4_seed42_test': l4_test_val,
            'L3_mean_test': float(np.mean(l3_test_s8)),
            'L3_vs_L4_test_pct': float((np.mean(l3_test_s8) / l4_test_val - 1) * 100) if l4_test_val else None,
            'n_l3_with_test': len(l3_test_s8),
        }


verdict = 'INCONCLUSIVE'
if stats.get('val_s4', {}).get('r10', {}).get('delta_pct', 0) > 20:
    verdict = 'L3_DRAMATICALLY_BEATS_L4'

out = {
    'task': 'S=8 L=3 ablation (task322) vs L=4 baseline',
    'date': '2026-07-13',
    'note': 'L=3 seeds 42-45 from task321 (early stop), seeds 46-49 from task322 (long training)',
    'per_seed_val': per_seed,
    'per_seed_test': test_metrics,
    'stats': stats,
    'verdict': verdict,
}

os.makedirs(os.path.join(GRID, 'result/task322_l3_ext'), exist_ok=True)
with open(os.path.join(GRID, 'result/task322_l3_ext/s8_comparison.json'), 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

print('=== S=4 final (task321 seeds 42-45) vs L=4 baseline ===')
for m in ['r10', 'r5', 'n10', 'n5']:
    s = stats['val_s4'].get(m, {})
    if s:
        print(f'  val {m}: L3={s["L3"]["mean"]:.4f}±{s["L3"]["se"]:.4f} L4={s["L4"]["mean"]:.4f} Δ={s["delta_pct"]:+.1f}%')
print()
print('=== S=4 late-training (task322 seeds 46-49) ===')
for m in ['r10', 'r5', 'n10', 'n5']:
    s = stats['val_s4_late'].get(m, {})
    if s:
        print(f'  val {m}: L3={s["L3"]["mean"]:.4f}±{s["L3"]["se"]:.4f} (n={s["L3"]["n"]}) L4={s["L4"]["mean"]:.4f} Δ={s["delta_pct"]:+.1f}%')
print()
print('=== S=8 combined ===')
for m in ['r10', 'r5', 'n10', 'n5']:
    s = stats['val_s8'].get(m, {})
    if s:
        print(f'  val {m}: L3={s["L3"]["mean"]:.4f}±{s["L3"]["se"]:.4f} (n={s.get("n_l3", "?")}) L4={s["L4"]["mean"]:.4f} Δ={s["delta_pct"]:+.1f}%')
print()
print('=== Test S=4 (task321 已确认 4 seeds) ===')
for m in ['r10', 'r5', 'n10', 'n5']:
    s = stats['test_s4'].get(m, {})
    if s.get('L3'):
        l4 = s.get('L4_seed42_test')
        l4_str = f' L4_test={l4:.4f}' if l4 else ''
        delta = s.get('L3_vs_L4_test_pct')
        delta_str = f' (+{delta:.1f}%)' if delta else ''
        print(f'  test {m}: L3={s["L3_mean_test"]:.4f}±{s["L3"]["se"]:.4f}{l4_str}{delta_str}')
print()
print(f'L=3 test per seed:')
for s in sorted(test_metrics['L3'].keys()):
    t = test_metrics['L3'][s] or {}
    if t.get('r10') is not None:
        print(f'  seed={s}: test_R10={t["r10"]:.4f} test_N10={t.get("n10"):.4f}' if t.get('n10') else f'  seed={s}: test_R10={t.get("r10")}')
    else:
        print(f'  seed={s}: test_pending_or_no_log_match')
print()
print(f'Verdict: {verdict}')
