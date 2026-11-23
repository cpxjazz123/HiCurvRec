"""Recompute legacy timing windows; never infer missing historical provenance."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue252/HISTORICAL_TIMING.json'


def record(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(times):
    values = sorted(times)
    return {'epochs': len(times), 'timer_sum_s': sum(times),
            'timer_mean_s': statistics.mean(times), 'timer_median_s': statistics.median(times),
            'timer_p95_s': values[math.ceil(.95 * len(values)) - 1]}


def audit():
    wide = ROOT / 'results/stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq/logs/train_migrated.log'
    lines = wide.read_text().splitlines()
    parsed = {}
    for line in lines:
        match = re.search(r'epoch=(\d+).*\btime=([\d.]+)s', line)
        if match:
            epoch = int(match[1])
            if epoch in parsed:
                raise ValueError('ambiguous repeated historical epoch')
            parsed[epoch] = float(match[2])
    wide_result = {
        **record(wide), 'epoch_first': 1001, 'epoch_last': 1030,
        **summarize([parsed[e] for e in range(1001, 1031)]),
        'completed_epoch': max(parsed), 'window_wall_s': None,
        'actual_start_records': lines[:3], 'seed': None, 'executed_source_sha256': None,
        'hardware': None, 'warning': 'Migrated text has no execution-source/GPU identity or window timestamps; timer sum is not inclusive wall.'}
    role = ROOT / 'results/stage2_RQ-VAE/TIGER_graphlet_role_Q/logs/training_metrics.jsonl'
    records = [json.loads(line) for line in role.read_text().splitlines() if line.strip()]
    train = {r['epoch']: r for r in records if r.get('event') == 'train'}
    start = next(r for r in records if r.get('event') == 'train_start')
    end = next(r for r in records if r.get('event') == 'train_end')
    source = ROOT / 'stage2_RQ-VAE/_archive/TIGER_graphlet_role_rq/train_rqvae.py'
    role_result = {
        **record(role), 'epoch_first': 970, 'epoch_last': 999,
        **summarize([train[e]['epoch_time_s'] for e in range(970, 1000)]),
        'window_wall_s': train[999]['cumulative_time_s'] - train[969]['cumulative_time_s'],
        'window_timestamp_wall_s': train[999]['timestamp'] - train[969]['timestamp'],
        'actual_start_record': start, 'actual_end_record': end,
        'current_archived_source': record(source), 'executed_source_sha256': None,
        'hardware': None,
        'timer_boundary': 'Current archived source starts per-epoch timer before loader; stops before post-loop metric all_reduce and before periodic full-catalogue evaluation. Cumulative records include intervening gaps.',
        'warning': 'Current archived source hash is not an execution-time binding. GPU/driver/clock absent. These windows do not certify the current mechanism.'}
    return {'schema': 'genrec.issue252.historical_timing/1',
            'audit_source': record(Path(__file__)), 'wide_l1': wide_result, 'role_Q': role_result}


def main():
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: {k: v for k, v in value.items() if k.startswith('timer_') or k in ('epochs', 'window_wall_s')}
                      for key, value in result.items() if key in ('wide_l1', 'role_Q')}, indent=2))


if __name__ == '__main__':
    main()
