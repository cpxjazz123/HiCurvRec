#!/usr/bin/env python3
"""P4 aggregate: 收集 3-seed 训练结果, 给出接受标准判定"""
import argparse
import json
import os


def get_best_step_from_csv(seed_dir):
    """从 training CSV 中读取 best step (按 val/recall@10 排序).
    自动找到任意 version_*/metrics.csv."""
    csv_dir = os.path.join(seed_dir, 'csv')
    csv_path = None
    if os.path.isdir(csv_dir):
        versions = sorted(os.listdir(csv_dir))
        for v in reversed(versions):  # latest version first
            cand = os.path.join(csv_dir, v, 'metrics.csv')
            if os.path.exists(cand):
                csv_path = cand
                break
    if csv_path is None or not os.path.exists(csv_path):
        return None
    best_row = None
    with open(csv_path) as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    r10_idx = header.index('val/recall@10') if 'val/recall@10' in header else None
    step_idx = header.index('step') if 'step' in header else None
    if r10_idx is None or step_idx is None:
        return None
    for line in lines[1:]:
        cols = line.strip().split(',')
        if len(cols) <= max(r10_idx, step_idx):
            continue
        try:
            r10 = float(cols[r10_idx])
            step = int(cols[step_idx])
        except Exception:
            continue
        if best_row is None or r10 > best_row[1]:
            best_row = (step, r10)
    return best_row  # (step, val/recall@10)


def get_test_metric(seed_dir):
    """从 test_eval 或 trainer 自动 test 中读取测试 R@10.
    优先级:
      1. test_eval/metrics.json (launcher 跑的 inference 输出)
      2. csv/version_*/metrics.csv 中的 test/recall@10 (trainer.test() 自动跑)
      3. train.log 中的 test_R@10 (manual parse)
    """
    # 1. explicit test_eval dir
    candidates = [
        os.path.join(seed_dir, 'test_eval', 'metrics.json'),
        os.path.join(seed_dir, 'test_eval', 'pickle', 'eval_metrics.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get('Recall@10', data.get('test_R@10', None))
            except Exception:
                continue

    # 2. metrics.csv 中的 test/recall@10
    csv_dir = os.path.join(seed_dir, 'csv')
    if os.path.isdir(csv_dir):
        versions = sorted(os.listdir(csv_dir))
        for v in reversed(versions):
            csv_path = os.path.join(csv_dir, v, 'metrics.csv')
            if not os.path.exists(csv_path):
                continue
            try:
                with open(csv_path) as f:
                    lines = f.readlines()
                if not lines:
                    continue
                header = lines[0].strip().split(',')
                r10_idx = header.index('test/recall@10') if 'test/recall@10' in header else None
                if r10_idx is None:
                    continue
                for line in reversed(lines[1:]):
                    cols = line.strip().split(',')
                    if len(cols) > r10_idx and cols[r10_idx]:
                        return float(cols[r10_idx])
            except Exception:
                continue

    # 3. parse train.log for "test/recall@10 ... 0.xxx"
    train_log = os.path.join(seed_dir, 'train.log')
    if os.path.exists(train_log):
        try:
            import re
            with open(train_log) as f:
                content = f.read()
            # Look for table row pattern
            m = re.search(r'test/recall@10[│|]\s*([\d.]+)', content)
            if m:
                return float(m.group(1))
        except Exception:
            pass

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed_dirs', nargs='+', required=True)
    ap.add_argument('--out', default='p4_summary.json')
    args = ap.parse_args()

    print('=' * 70)
    print('P4: 3-seed 复现验证')
    print('=' * 70)

    seed_results = []
    for sd in args.seed_dirs:
        seed = os.path.basename(sd.rstrip('/')).replace('seed_', '')
        best = get_best_step_from_csv(sd)
        test_r10 = get_test_metric(sd)
        seed_results.append({
            'seed': seed,
            'seed_dir': sd,
            'best_step_val_R10': best,
            'test_R10': test_r10,
        })
        print(f'  seed {seed}: best ckpt step={best}, test_R@10={test_r10}')

    # Accept criteria
    accept_criteria = {}
    steps = [r['best_step_val_R10'][0] for r in seed_results if r['best_step_val_R10']]
    if len(steps) >= 3:
        step_range = max(steps) - min(steps)
        accept_criteria['best_step_range'] = step_range
        accept_criteria['best_step_pass'] = (step_range <= 1500)
        print(f'\n  Best step range across 3 seeds: {step_range} (阈值 ≤ 1500): '
              f'{"✓ PASS" if step_range <= 1500 else "✗ FAIL"}')
    else:
        accept_criteria['best_step_range'] = None
        accept_criteria['best_step_pass'] = None

    r10_vals = [r['test_R10'] for r in seed_results if r['test_R10'] is not None]
    if len(r10_vals) >= 3:
        import statistics
        sd = statistics.stdev(r10_vals)
        accept_criteria['test_R10_std'] = sd
        accept_criteria['test_R10_pass'] = (sd <= 0.005)
        print(f'  Test R@10 std across 3 seeds: {sd:.5f} (阈值 ≤ 0.005): '
              f'{"✓ PASS" if sd <= 0.005 else "✗ FAIL"}')
    else:
        accept_criteria['test_R10_std'] = None
        accept_criteria['test_R10_pass'] = None

    summary = {
        'seed_results': seed_results,
        'accept_criteria': accept_criteria,
        'overall_pass': (
            accept_criteria.get('best_step_pass', False)
            and accept_criteria.get('test_R10_pass', False)
        ),
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
