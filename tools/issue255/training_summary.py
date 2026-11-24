"""Summarise the formal Stage2 run: the timing window, the gate mixture and the calibration."""
from __future__ import annotations

import json
from pathlib import Path

from tools.issue253.performance import summarize_window

from tools.issue255.identity import ROOT, RUN as RUN_DIRECTORY
RUN = RUN_DIRECTORY / 'result'
LEVELS = 3
EXPERTS = 3


def main() -> int:
    stream = RUN / 'logs/training_metrics.jsonl'
    rows = [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
    train = [row for row in rows if row.get('event') == 'train']
    boundaries = [row for row in rows if row.get('event') == 'epoch_boundary']
    window = [row for row in boundaries if 31 <= row['epoch'] <= 60]
    performance = summarize_window(window)
    last = train[-1]
    gate = last['gate_mean']
    mixture = {f'level{level + 1}': {'negative': gate[level * EXPERTS],
                                     'zero': gate[level * EXPERTS + 1],
                                     'positive': gate[level * EXPERTS + 2]}
               for level in range(LEVELS)}
    calibration = next((row for row in rows if row.get('event') == 'curvature_calibration'), {})
    scales = next((row for row in rows if row.get('event') == 'level_scale_calibration'), {})
    summary = {
        'epochs_recorded': len(train),
        'last_epoch': last['epoch'],
        'window_epochs': [window[0]['epoch'], window[-1]['epoch']] if window else None,
        'window_mean_seconds': performance['mean_seconds'],
        'window_median_seconds': performance['median_seconds'],
        'window_p95_seconds': performance['p95_seconds'],
        'window_total_seconds': performance['window_seconds'],
        'interepoch_seconds': performance['interepoch_seconds'],
        'final_loss': last['loss'], 'final_recon': last['recon'],
        'final_curvature_estimate': last['curvature_estimate'],
        'final_gradient_norm_before_clip': last['gradient_norm_before_clip'],
        'final_gate_entropy': last['gate_entropy'],
        'final_attention_entropy': last['auxiliary_attention_entropy'],
        'final_level_codebook_gradient_norm': last.get('level_codebook_gradient_norm'),
        'last_epoch_gate_mixture': mixture,
        'curvature_calibration': {key: calibration.get(key) for key in
                                  ('original_gradient', 'reconstruction_gradient', 'curvature_gradient',
                                   'original_curvature_cosine', 'reconstruction_curvature_cosine', 'lambda',
                                   'tangent_scale', 'target_scale')},
        'level_scale_calibration': scales,
        'collision_events': [row.get('collision') for row in train if row.get('collision') is not None][-3:],
    }
    path = ROOT / 'results/research/issue255/TRAINING_SUMMARY.json'
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
