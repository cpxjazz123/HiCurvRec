"""CPU-only rank, provenance and durable composite-pairing regressions."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.issue254.paired import sample_identity
from tools.issue252 import archive_audit
from tools.issue255 import paired_comparison as paired
from tools.issue255 import reconcile_results as reconcile


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def descriptor(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def metrics(ranks, k):
    selected = [r for r in ranks if r is not None and r < k]
    return len(selected) / len(ranks), sum(1 / math.log2(r + 2) for r in selected) / len(ranks)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    root = tmp_path
    evidence = root / 'evidence'
    evidence.mkdir()
    trainer = root / 'stage3_T5Train/train_HG-Rec.py'
    trainer.parent.mkdir()
    trainer.write_text('FROZEN_TRAINER = True\n')
    test_path = root / 'results/stage0_build_parquet/test.parquet'
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(b'isolated test split identity')
    rows = []
    for i in range(6):
        row = {'global_row': i, 'user': 100 + i, 'history': [14],
               'seen_history': [14], 'target_item': 15}
        row['sample_sha256'] = sample_identity(row)
        rows.append(row)
    protocol = {'num_epochs': 150, 'seed': 42, 'beam_size': 20,
                'eval_interval': 5, 'eval_start_epoch': 75,
                'early_eval_epochs': [25, 50, 75, 100, 125],
                'no_eval': True, 'skip_test': False, 'topk_list': [5, 10],
                'batch_size': 4096, 'infer_size': 1024, 'dropout_rate': 0.1,
                'dataset_name': 'Amazon_2023_Instruments', 'n_user_tokens': 1,
                'vocab_size': 19, 'eos_token_id': 18}

    def stage2(run_id, arm):
        run = root / 'runs' / run_id / arm
        source, result = run / 'src', run / 'result'
        (source / 'model').mkdir(parents=True)
        result.mkdir()
        (source / 'train_rqvae.py').write_text('FROZEN = True\n')
        (source / 'model/__init__.py').write_text('')
        sid = dump(result / 'item_sids.json',
                   {str(i): [0, 0, i, 0 if arm == 'old_arm' else 20] for i in range(16)})
        checkpoint = result / 'rqvae_best.pth'
        checkpoint.write_bytes(run_id.encode())
        export = dump(result / 'export_manifest.json', {
            'schema': 'genrec.stage2.export_manifest/1', 'tokens_match': True,
            'same_l3': 1.0, 'differing_items': 0, 'tree': str(source),
            'sid': sid['path'], 'sid_sha256': sid['sha256'],
            'checkpoint': str(checkpoint), 'checkpoint_sha256': descriptor(checkpoint)['sha256']})
        candidates = archive_audit.source_hashes(source)
        binding = dump(result / 'archive_binding.json', {
            'schema': 'genrec.stage2.archive_binding/1', 'repository': str(root),
            'run_id': run_id, 'source_tree': str(source), 'candidate_sources': candidates,
            'infrastructure_sources': {}, 'repository_configs': {},
            'checkpoint_provenance': {'candidate_sources': candidates, 'infrastructure_sources': {}},
            'sid': sid, 'checkpoint': descriptor(checkpoint), 'export_manifest': export,
            'audit': descriptor(Path(archive_audit.__file__).resolve())})
        artifacts = {'sid': sid, 'checkpoint': descriptor(checkpoint), 'export': export,
                     'binding': binding, 'workspace': dump(run / 'workspace.json', {'run_id': run_id}),
                     'metrics': dump(result / 'logs/training_metrics.jsonl', {'event': 'train_end'}),
                     'training_summary': dump(evidence / (run_id + '_summary.json'), {'run_id': run_id}),
                     'layered_metrics': dump(result / 'layers.json', {'run_id': run_id})}
        return {'run_id': run_id, 'arm': arm, 'source_tree': str(source),
                'epoch': 3000, 'artifacts': artifacts}

    old2 = stage2('frozen253', 'old_arm')
    new2 = stage2('issue255_formal_r1', 'new_arm')
    dump(evidence / 'FINAL_STAGE2.json', new2)

    def stage3(stage2_record, name, ranks):
        arm = stage2_record['arm']
        run = root / 'results/stage3_T5Train' / arm / 'logs/Amazon_2023_Instruments' / name
        run.mkdir(parents=True)
        config = dict(protocol, dataset_path=str(test_path.parent),
                      code_path=stage2_record['artifacts']['sid']['path'],
                      log_path=str(run.parent.parent),
                      save_path=str(root / 'results/stage3_T5Train' / arm / 'ckpt'))
        if arm != 'old_arm':
            config.update(vocab_size=24, eos_token_id=23)
        ckpt = Path(config['save_path']) / config['dataset_name'] / name / 'HG_Rec_best.pth'
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b'completed Stage3 weights')
        final = {'stage2_run_id': stage2_record['run_id'],
                 'stage2_source_tree': stage2_record['source_tree'],
                 'stage3_run_id': str(run), 'stage3_input_config': config,
                 'stage3_completed_epochs': 150, 'stage3_world_size': 4,
                 'stage3_trainer_sha256': descriptor(trainer)['sha256'],
                 'n_eval': len(rows), 'best_checkpoint': str(ckpt)}
        for key, field in [('sid', 'sid_path'), ('checkpoint', 'stage2_checkpoint'),
                           ('export', 'stage2_export_manifest'), ('binding', 'archive_binding')]:
            entry = stage2_record['artifacts'][key]
            final[field] = entry['path']
            final['sid_sha256' if key == 'sid' else field + '_sha256'] = entry['sha256']
        for k in (5, 10):
            final['test_recall@%d' % k], final['test_ndcg@%d' % k] = metrics(ranks, k)
        dump(run / 'test_final.json', final)
        stream = [{'event': 'train_start', 'num_epochs': 150, 'no_eval': True,
                   'batch_size': 4096, 'infer_size': 1024, 'global_train_batch': 16384,
                   'launch_mode': 'ddp4', 'world_size': 4, 'rank': 0,
                   'log_path': str(run), 'code_path': final['sid_path'],
                   'code_path_sha256': final['sid_sha256']}]
        stream.extend({'event': 'train', 'epoch': i} for i in range(1, 151))
        stream.extend([{'event': 'test', 'epoch': 150, 'n_eval': len(rows),
                        'test_shard_world_size': 4,
                        **{'test_%s_at_%d' % (metric, k): final['test_%s@%d' % (metric, k)]
                           for metric in ('recall', 'ndcg') for k in (5, 10)}},
                       {'event': 'train_end'}])
        (run / 'training_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in stream))
        sid_rows = json.loads(Path(stage2_record['artifacts']['sid']['path']).read_text())
        canonical = {int(item): [value + 1 for value in code] for item, code in sid_rows.items()}
        for rank in range(4):
            predictions, users, selected = [], [], []
            for index in range(rank, len(rows), 4):
                target_rank = ranks[index]
                candidates = list(range(10))
                if target_rank is not None:
                    candidates.insert(target_rank, 15)
                    candidates.pop()
                predictions.append({'global_row': index, 'target_item': 15,
                                    'target_sid': canonical[15], 'target_rank': target_rank,
                                    'topk': [{'item': item, 'sid_canonical': canonical[item],
                                              'sid_generated': canonical[item]} for item in candidates]})
                users.append({'local_row_index': len(users), 'target_rank': target_rank})
                selected.append(target_rank)
            path = run / 'predictions' / ('rank%d.jsonl' % rank)
            path.parent.mkdir(exist_ok=True)
            path.write_text(''.join(json.dumps(p) + '\n' for p in predictions))
            dump(run / '_test_shards' / ('rank%d.json' % rank), {
                'per_user': users, 'n_eval': len(users),
                'recall_sums': {str(k): metrics(selected, k)[0] * len(selected) for k in (5, 10)},
                'ndcg_sums': {str(k): metrics(selected, k)[1] * len(selected) for k in (5, 10)}})
        return run

    old = stage3(old2, 'frozen', [0, 4, 5, 9, None, None])
    new = stage3(new2, 'corrected', [None, 5, 4, None, 0, 9])
    for module in (paired, reconcile):
        for name, value in {'ROOT': root, 'EVIDENCE': evidence, 'RUN_ID': new2['run_id'],
                            'ARM': new2['arm'], 'RUN': Path(new2['source_tree']).parent,
                            'RESULT': Path(new2['source_tree']).parent / 'result'}.items():
            monkeypatch.setattr(module, name, value, raising=False)
    monkeypatch.setattr(paired, 'ISSUE253', old)
    monkeypatch.setattr(paired, 'STAGE3_LOGS', new.parent.parent)
    monkeypatch.setattr(paired, 'STAGE3_TRAINER', trainer)
    monkeypatch.setattr(paired, 'BASELINE_TRAINER_SHA256', descriptor(trainer)['sha256'], raising=False)
    monkeypatch.setattr(paired, 'POPULATION', len(rows), raising=False)
    monkeypatch.setattr(paired, 'TEST_PARQUET', test_path, raising=False)
    monkeypatch.setattr(paired, 'read_test_rows', lambda: copy.deepcopy(rows))
    return {'root': root, 'evidence': evidence, 'rows': rows, 'old': old, 'new': new,
            'new2': new2, 'make_stage3': stage3, 'protocol': json.loads((old / 'test_final.json').read_text())['stage3_input_config']}


def test_zero_based_nullable_boundaries_and_net_counts():
    pairs = list(zip([0, 4, 5, 9, 10, None], [None, 0, 4, 10, 9, 0]))
    result = reconcile.reduce_ranks(pairs)
    for k, old_count, new_count in [(5, 2, 3), (10, 4, 4)]:
        cell = result[str(k)]
        assert cell['old_recall'] == old_count / 6
        assert cell['new_recall'] == new_count / 6
        old_gain = 1 + 1 / math.log2(6)
        new_gain = 2 + 1 / math.log2(6)
        if k == 10:
            old_gain += 1 / math.log2(7) + 1 / math.log2(11)
            new_gain += 1 / math.log2(11)
        assert cell['old_ndcg'] == pytest.approx(old_gain / 6)
        assert cell['new_ndcg'] == pytest.approx(new_gain / 6)
        assert cell['net_hit_change'] == new_count - old_count
    assert result['5']['old_only'] == 1
    assert result['5']['new_only'] == 2
    assert result['10']['old_only'] == result['10']['new_only'] == 2
    assert result['5']['net_ndcg_change'] == pytest.approx(result['5']['new_ndcg'] - result['5']['old_ndcg'])


@pytest.mark.parametrize('field,value', [('sid_sha256', 'wrong'), ('stage2_run_id', 'issue255_formal'),
    ('stage2_source_tree', '/wrong-source'), ('stage2_checkpoint_sha256', 'wrong'),
    ('archive_binding_sha256', 'wrong'), ('stage3_trainer_sha256', 'wrong'),
    ('stage3_world_size', 1), ('n_eval', 5), ('stage3_completed_epochs', 149)])
def test_rejects_mismatched_completed_generation(pipeline, field, value):
    path = pipeline['new'] / 'test_final.json'
    payload = json.loads(path.read_text())
    payload[field] = value
    dump(path, payload)
    with pytest.raises(ValueError):
        paired.resolve_stage3(paired.stage2_identity(), pipeline['protocol'], len(pipeline['rows']))


@pytest.mark.parametrize('field,value', [('seed', 43), ('beam_size', 10), ('num_epochs', 149),
    ('no_eval', False), ('skip_test', True), ('dropout_rate', 0.2), ('batch_size', 1024)])
def test_rejects_protocol_changes(pipeline, field, value):
    path = pipeline['new'] / 'test_final.json'
    payload = json.loads(path.read_text())
    payload['stage3_input_config'][field] = value
    dump(path, payload)
    with pytest.raises(ValueError):
        paired.resolve_stage3(paired.stage2_identity(), pipeline['protocol'], len(pipeline['rows']))


def test_rejects_partial_epoch_stream(pipeline):
    path = pipeline['new'] / 'training_metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows if row.get('epoch') != 83))
    with pytest.raises(ValueError):
        paired.resolve_stage3(paired.stage2_identity(), pipeline['protocol'], len(pipeline['rows']))


def test_rejects_ambiguous_completed_generation(pipeline):
    pipeline['make_stage3'](pipeline['new2'], 'also_completed', [0, 4, 5, 9, None, None])
    with pytest.raises(ValueError, match='ambiguous'):
        paired.resolve_stage3(paired.stage2_identity(), pipeline['protocol'], len(pipeline['rows']))


def test_final_stage2_checks_every_artifact(pipeline):
    path = Path(pipeline['new2']['artifacts']['training_summary']['path'])
    path.write_text('changed summary')
    with pytest.raises(ValueError):
        paired.stage2_identity()


def test_no_arg_pipeline_preserves_ledger_and_receipt(pipeline):
    assert paired.main() == 0
    evidence = pipeline['evidence']
    receipt = json.loads((evidence / 'STAGE3_RESULT.json').read_text())
    assert receipt['run_id'] == 'issue255_formal_r1'
    assert receipt['test_final'] == descriptor(pipeline['new'] / 'test_final.json')
    assert receipt['stage2']['artifacts']['binding'] == pipeline['new2']['artifacts']['binding']
    aggregate = json.loads((evidence / 'PAIRED_STAGE3.json').read_text())
    ledger_path = evidence / 'evaluation/paired_users.jsonl'
    assert aggregate['records'] == descriptor(ledger_path)
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(ledger) == len(pipeline['rows'])
    for actual, sample, old, new in zip(ledger, pipeline['rows'], [0, 4, 5, 9, None, None], [None, 5, 4, None, 0, 9]):
        assert (actual['global_row'], actual['user'], actual['sample_sha256'], actual['target_item']) == (
            sample['global_row'], sample['user'], sample['sample_sha256'], sample['target_item'])
        assert (actual['rank_issue253'], actual['rank_issue255']) == (old, new)
        assert actual['paired_inputs_sha256'] == aggregate['paired_inputs_sha256']
        for k in (5, 10):
            assert actual['hit%d_issue255' % k] == (new is not None and new < k)
            assert actual['ndcg%d_issue255' % k] == pytest.approx(0 if new is None or new >= k else 1 / math.log2(new + 2))
    result = reconcile.stage3(0.0)
    assert result['status'] == 'COMPLETE'
    assert result['population'] == 6
    assert result['cutoffs']['5']['old_only'] == 2
    assert result['cutoffs']['5']['new_only'] == 2
    assert result['cutoffs']['10']['net_hit_change'] == 0
    with pytest.raises(ValueError, match='collision'):
        reconcile.stage3(0.4)
    ledger_path.write_text(ledger_path.read_text() + '{}\n')
    with pytest.raises(ValueError):
        reconcile.stage3(0.0)


def test_composite_identity_mismatch_and_partial_predictions_rejected(pipeline):
    old = paired.load_arm(pipeline['old'], pipeline['rows'])
    new = paired.load_arm(pipeline['new'], pipeline['rows'])
    new['rows'][0]['user'] += 1000
    with pytest.raises(ValueError, match='identity'):
        paired.build_ledger(old, new, pipeline['rows'])
    new = paired.load_arm(pipeline['new'], pipeline['rows'])
    new['rows'].pop()
    with pytest.raises(ValueError, match='population'):
        paired.build_ledger(old, new, pipeline['rows'])


def test_incomplete_stage3_produces_no_scientific_outputs(pipeline):
    (pipeline['new'] / 'test_final.json').unlink()
    with pytest.raises(ValueError):
        paired.main()
    assert not (pipeline['evidence'] / 'STAGE3_RESULT.json').exists()
    assert not (pipeline['evidence'] / 'PAIRED_STAGE3.json').exists()
    assert not (pipeline['evidence'] / 'evaluation/paired_users.jsonl').exists()


def test_sid_derived_vocabulary_is_not_a_training_protocol_change(pipeline):
    old_config = pipeline['protocol']
    new_config = json.loads((pipeline['new'] / 'test_final.json').read_text())['stage3_input_config']
    assert old_config['vocab_size'] == 19 and new_config['vocab_size'] == 24
    assert old_config['eos_token_id'] == 18 and new_config['eos_token_id'] == 23
    paired.resolve_stage3(paired.stage2_identity(), old_config, len(pipeline['rows']))


@pytest.mark.parametrize('field', ['vocab_size', 'eos_token_id'])
def test_wrong_sid_derived_vocabulary_is_rejected(pipeline, field):
    path = pipeline['new'] / 'test_final.json'
    final = json.loads(path.read_text())
    final['stage3_input_config'][field] += 1
    dump(path, final)
    with pytest.raises(ValueError, match='SID-derived'):
        paired.resolve_stage3(paired.stage2_identity(), pipeline['protocol'], len(pipeline['rows']))
