"""Independently reduce persisted formal predictions; no model/test replay."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.issue256.identity import ROOT, ARM, RUN_ID, EVIDENCE
from tools.issue256.external_validation import inputs, record
from tools.issue256.extract_catalogues import context
from tools.issue253.reconcile_results import reconcile_per_user

REFERENCE = ROOT / ('results/stage3_T5Train/TIGER_three_expert_attention_rq/logs/'
                    'Amazon_2023_Instruments/Nov-24-2026_21-16-28/test_final.json')


def close(actual, expected, name):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1.e-10, abs_tol=1.e-10):
        raise ValueError('independent final metric mismatch: ' + name)


def main():
    import pyarrow.parquet as pq

    final, identity = inputs()
    run, paths, payload, stage2_identity = context()
    stage3 = Path(final['stage3_run_id']).resolve()
    stage3.relative_to((ROOT / 'results/stage3_T5Train' / ARM).resolve())
    if stage3 / 'test_final.json' != Path(identity['test_final']['path']):
        raise ValueError('Stage3 result crosses run directories')
    for final_key, stage2_key in (('sid', 'sid'), ('stage2_checkpoint', 'checkpoint'),
                                ('archive_binding', 'binding'), ('stage2_export_manifest', 'export_manifest')):
        descriptor = record(paths[stage2_key])
        path_key = 'sid_path' if final_key == 'sid' else final_key
        if Path(final[path_key]).resolve() != Path(descriptor['path']) or final[final_key + '_sha256'] != descriptor['sha256']:
            raise ValueError('Stage2/Stage3 input pairing mismatch: ' + final_key)
    if final['stage2_run_id'] != RUN_ID or Path(final['stage2_source_tree']).resolve() != Path(run['source']):
        raise ValueError('Stage3 source/workspace pairing differs')
    config = final['stage3_input_config']
    reference = json.loads(REFERENCE.read_text())['stage3_input_config']
    ignored = {'code_path', 'log_path', 'save_path', 'dataset_path', 'vocab_size', 'eos_token_id'}
    if {k: v for k, v in config.items() if k not in ignored} != {k: v for k, v in reference.items() if k not in ignored}:
        raise ValueError('non-path/non-derived Stage3 protocol differs from frozen comparison')
    if Path(config['dataset_path']).resolve() != (ROOT / 'results/stage0_build_parquet').resolve():
        raise ValueError('Stage3 split directory changed')
    raw_sid = json.loads(Path(final['sid_path']).read_text())
    largest = max(max(row) for row in raw_sid.values())
    expected_eos = largest + config['n_user_tokens'] + 2
    if config['eos_token_id'] != expected_eos or config['vocab_size'] != expected_eos + 1:
        raise ValueError('actual SID-derived EOS/vocabulary is incorrect')
    metrics_path = stage3 / 'training_metrics.jsonl'
    stream = [json.loads(line) for line in metrics_path.open() if line.strip()]
    train = [r for r in stream if r['event'] == 'train']
    starts = [r for r in stream if r['event'] == 'train_start']
    if ([r['epoch'] for r in train] != list(range(1, 151)) or len(starts) != 1
            or stream[-1]['event'] != 'train_end' or not all(math.isfinite(r['loss']) for r in train)):
        raise ValueError('actual Stage3 lacks complete finite150-epoch training and train_end')
    start = starts[0]
    if (start['global_train_batch'] != 16384 or start['launch_mode'] != 'ddp4'
            or start['batch_size'] != 4096 or start['infer_size'] != 1024
            or start['code_path_sha256'] != final['sid_sha256']):
        raise ValueError('actual Stage3 layout/input differs from frozen four-card recipe')
    test_path = ROOT / 'results/stage0_build_parquet/test.parquet'
    population = pq.ParquetFile(test_path).metadata.num_rows
    if population != 57439 or final['n_eval'] != population:
        raise ValueError('complete test population differs')
    shards = [json.loads((stage3 / '_test_shards' / ('rank%d.json' % r)).read_text()) for r in range(4)]
    reduced = reconcile_per_user(shards, expected_users=population)
    predicted_ranks = [None] * population
    seen = set()
    prediction_files = []
    for rank in range(4):
        path = stage3 / 'predictions' / ('rank%d.jsonl' % rank)
        expected_rows = list(range(rank, population, 4))
        row_numbers = []
        local = []
        for line in path.open():
            row = json.loads(line)
            number, target_rank = row['global_row'], row['target_rank']
            if number in seen or type(number) is not int or not 0 <= number < population:
                raise ValueError('missing/duplicate/out-of-range formal prediction identity')
            selected = [int(value['item']) for value in row['topk']]
            actual_rank = selected.index(row['target_item']) if row['target_item'] in selected else None
            if target_rank != actual_rank or len(selected) != 10 or len(set(selected)) != 10:
                raise ValueError('persisted rank differs from actual final item-level top10')
            seen.add(number)
            predicted_ranks[number] = target_rank
            row_numbers.append(number)
            local.append(target_rank)
        if row_numbers != expected_rows or local != [r['target_rank'] for r in shards[rank]['per_user']]:
            raise ValueError('prediction/per-user shard identity or ranks disagree')
        prediction_files.append(record(path))
    if seen != set(range(population)):
        raise ValueError('partial formal prediction population')
    from_predictions = {'n_eval': population}
    for k in (5, 10):
        hits = [rank for rank in predicted_ranks if rank is not None and rank < k]
        from_predictions['test_recall@%d' % k] = len(hits) / population
        from_predictions['test_ndcg@%d' % k] = math.fsum(1 / math.log2(rank + 2) for rank in hits) / population
    for key, value in reduced.items():
        if key != 'n_eval':
            close(value, final[key], key)
            close(from_predictions[key], value, 'predictions/' + key)
    if context()[-1] != stage2_identity or inputs()[1] != identity:
        raise ValueError('paired final source/artifacts changed during reduction')
    report = {'schema': 'genrec.issue256.stage3_reconciliation/1', 'status': 'PASS',
        'run_id': RUN_ID, 'identity': identity, 'stage2_identity': stage2_identity,
        'actual_epochs': len(train), 'source_protocol_reference': record(REFERENCE),
        'all_frozen_non_path_non_derived_config_equal': True,
        'derived_vocabulary': {'max_raw_sid': largest, 'eos': expected_eos, 'vocab': expected_eos + 1},
        'metrics': reduced, 'prediction_reduction': from_predictions,
        'strict_target_met': reduced['test_recall@10'] > .065,
        'artifacts': {'metrics': record(metrics_path), 'native_log': record(stage3 / 'HG_Rec.log'),
            'stage3_checkpoint': record(final['best_checkpoint']),
            'test_split': record(test_path),
            'shards': [record(stage3 / '_test_shards' / ('rank%d.json' % r)) for r in range(4)],
            'predictions': prediction_files},
        'evaluator': record(__file__), 'reducer': record(ROOT / 'tools/issue253/reconcile_results.py'),
        'test_access': 'Independent read-only reduction of persisted final test rows; no model replay or error-item selection.'}
    output = EVIDENCE / 'STAGE3_RECONCILIATION.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'metrics': reduced,
                      'strict_target_met': report['strict_target_met'], 'report': record(output)}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
