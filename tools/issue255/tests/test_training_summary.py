"""Regression checks for reported timing and measured gate populations."""
import json

import pytest

from tools.issue255 import training_summary


def run_summary(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    (run / 'logs').mkdir(parents=True)
    (tmp_path / 'results/research/issue255').mkdir(parents=True)
    rows = []
    for index, epoch in enumerate(range(31, 61)):
        start = index * 0.6
        rows.append({'event': 'epoch_boundary', 'epoch': epoch,
                     'start': start, 'end': start + 0.5, 'epoch_time_s': 0.5})
    rows.append({'event': 'train', 'epoch': 3000,
                 'gate_mean': [0.8, 0.1, 0.1, 0.2, 0.3, 0.5, 0.1, 0.7, 0.2],
                 'loss': 1.0, 'recon': 0.5, 'curvature_estimate': 2.0,
                 'gradient_norm_before_clip': 0.1, 'gate_entropy': 0.2,
                 'auxiliary_attention_entropy': 0.3})
    (run / 'logs/training_metrics.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))
    monkeypatch.setattr(training_summary, 'ROOT', tmp_path)
    monkeypatch.setattr(training_summary, 'RUN', run)
    assert training_summary.main() == 0
    return json.loads((tmp_path / 'results/research/issue255/TRAINING_SUMMARY.json').read_text())


def test_inclusive_window_keeps_between_epoch_work(tmp_path, monkeypatch):
    result = run_summary(tmp_path, monkeypatch)
    assert result['window_total_seconds'] == pytest.approx(17.9)
    assert result['window_mean_seconds'] == pytest.approx(17.9 / 30)
    assert result['window_median_seconds'] == pytest.approx(0.6)
    assert result['window_p95_seconds'] == pytest.approx(0.6)
    assert result['interepoch_seconds'] == pytest.approx(2.9)


def test_training_gate_mean_is_not_catalogue_measurement(tmp_path, monkeypatch):
    result = run_summary(tmp_path, monkeypatch)
    assert 'catalogue_mean_gate_mixture' not in result
    assert result['last_epoch_gate_mixture']['level1'] == {
        'negative': 0.8, 'zero': 0.1, 'positive': 0.1}
    assert result['last_epoch_gate_mixture']['level2'] == {
        'negative': 0.2, 'zero': 0.3, 'positive': 0.5}
