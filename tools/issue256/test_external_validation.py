"""A rank-shard denominator must not silently replace the frozen split population."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.issue256 import external_validation as evaluation


def prepared(monkeypatch, tmp_path, population):
    valid = tmp_path / 'valid.parquet'
    pq.write_table(pa.table({'user': list(range(5)), 'history': [[0]] * 5, 'target': [1] * 5}), valid)
    output = tmp_path / 'evaluation'
    output.mkdir()
    identity = {'valid': evaluation.record(valid)}
    monkeypatch.setattr(evaluation, 'VALID', valid)
    monkeypatch.setattr(evaluation, 'OUTPUT', output)
    monkeypatch.setattr(evaluation, 'inputs', lambda: ({}, identity))
    for rank in range(4):
        numbers = list(range(rank, population, 4))
        predictions = output / ('rank%d.jsonl' % rank)
        predictions.write_text(''.join(json.dumps({'global_row': row, 'target_rank': None}) + '\n'
                                       for row in numbers))
        (output / ('rank%d.json' % rank)).write_text(json.dumps({
            'rank': rank, 'population': population, 'identity': identity,
            'predictions': evaluation.record(predictions), 'n_eval': len(numbers),
            'recall_sums': {'5': 0., '10': 0.}, 'ndcg_sums': {'5': 0., '10': 0.}}))
    return output


def test_reducer_rejects_partial_frozen_validation(monkeypatch, tmp_path):
    output = prepared(monkeypatch, tmp_path, 4)
    with pytest.raises(ValueError, match='validation population differs from the complete frozen split'):
        evaluation.reduce_rows()
    assert not (output / 'VALIDATION.json').exists()


def test_reducer_preserves_complete_all_miss_population(monkeypatch, tmp_path):
    output = prepared(monkeypatch, tmp_path, 5)
    assert evaluation.reduce_rows() == 0
    result = json.loads((output / 'VALIDATION.json').read_text())
    assert result['metrics'] == {'n_eval': 5, 'validation_recall@5': 0., 'validation_recall@10': 0.,
                                 'validation_ndcg@5': 0., 'validation_ndcg@10': 0.}
