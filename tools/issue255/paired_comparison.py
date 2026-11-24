"""Read-only pairing of the exact corrected Issue255 and frozen Issue253 runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tools.issue252.archive_audit import validate_binding
from tools.issue254 import contract
from tools.issue254.paired import build_ledger as frozen_ledger
from tools.issue254.paired import load_arm, read_test_rows
from tools.issue255.identity import ARM, EVIDENCE, RESULT, ROOT, RUN, RUN_ID
from tools.issue255 import historical

ISSUE253 = ROOT / ('results/stage3_T5Train/TIGER_three_expert_attention_rq/logs/'
                   'Amazon_2023_Instruments/Nov-24-2026_21-16-28')
BASELINE_TRAINER_SHA256 = '54ca6bfbcd55a42e718660dd0004370a41a5e1c2c5038a86bbfc84f32b963690'
STAGE3_LOGS = ROOT / 'results/stage3_T5Train' / ARM / 'logs'
STAGE3_TRAINER = ROOT / 'stage3_T5Train/train_HG-Rec.py'
TEST_PARQUET = ROOT / 'results/stage0_build_parquet/test.parquet'
POPULATION = 57439
CORE_PROTOCOL = {
    'num_epochs': 150, 'seed': 42, 'beam_size': 20, 'eval_interval': 5,
    'eval_start_epoch': 75, 'early_eval_epochs': [25, 50, 75, 100, 125],
    'no_eval': True, 'skip_test': False, 'topk_list': [5, 10],
    'batch_size': 4096, 'infer_size': 1024,
}
PATH_CONFIG = {'code_path', 'log_path', 'save_path', 'dataset_path'}
SID_CONFIG = {'vocab_size', 'eos_token_id'}
GROUP_NAMES = {
    contract.BOTH: 'both_hit', contract.ONLY_252: 'old_only',
    contract.ONLY_253: 'new_only', contract.NEITHER: 'both_miss',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def descriptor(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def check_descriptor(record, expected=None):
    path = Path(record['path']).resolve()
    if expected is not None and path != Path(expected).resolve():
        raise ValueError('artifact path mismatch: ' + str(path))
    if not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError('artifact changed or missing: ' + str(path))
    return path


def stage2_identity():
    """Recheck every final audit artifact, including the bound training summary."""
    path = EVIDENCE / 'FINAL_STAGE2.json'
    final = read_json(path)
    if (final['run_id'] != RUN_ID or final['arm'] != ARM or final['epoch'] != 3000
            or Path(final['source_tree']).resolve() != (RUN / 'src').resolve()):
        raise ValueError('FINAL_STAGE2 is not the approved corrected generation')
    artifacts = final['artifacts']
    required = {'sid', 'checkpoint', 'export', 'binding', 'workspace', 'metrics',
                'layered_metrics', 'training_summary'}
    if not required <= artifacts.keys():
        raise ValueError('FINAL_STAGE2 lacks required bound artifacts')
    expected = {'sid': RESULT / 'item_sids.json', 'checkpoint': RESULT / 'rqvae_best.pth',
                'export': RESULT / 'export_manifest.json', 'binding': RESULT / 'archive_binding.json',
                'workspace': RUN / 'workspace.json', 'metrics': RESULT / 'logs/training_metrics.jsonl'}
    for name, record in artifacts.items():
        check_descriptor(record, expected.get(name))
    binding, _ = validate_binding(artifacts['binding']['path'], artifacts['sid']['path'],
                                  artifacts['checkpoint']['path'], RUN / 'src')
    if binding['run_id'] != RUN_ID:
        raise ValueError('archive binding run identity mismatch')
    return {'run_id': RUN_ID, 'arm': ARM, 'source_tree': str((RUN / 'src').resolve()),
            'final_stage2': descriptor(path), 'artifacts': artifacts,
            'candidate_sources': binding['candidate_sources'],
            'infrastructure_sources': binding['infrastructure_sources'],
            'repository_configs': binding['repository_configs']}


def baseline_identity(final):
    fields = {'sid': 'sid_path', 'checkpoint': 'stage2_checkpoint',
              'export': 'stage2_export_manifest', 'binding': 'archive_binding'}
    artifacts = {}
    for name, field in fields.items():
        digest_key = 'sid_sha256' if name == 'sid' else field + '_sha256'
        artifacts[name] = {'path': final[field], 'sha256': final[digest_key]}
        check_descriptor(artifacts[name])
    binding, _ = historical.validate_binding(artifacts['binding']['path'], artifacts['sid']['path'],
                                             artifacts['checkpoint']['path'], final['stage2_source_tree'])
    if binding['run_id'] != final['stage2_run_id']:
        raise ValueError('baseline archive run identity mismatch')
    return {'run_id': binding['run_id'], 'source_tree': binding['source_tree'],
            'artifacts': artifacts, 'candidate_sources': binding['candidate_sources'],
            'infrastructure_sources': binding['infrastructure_sources'],
            'repository_configs': binding['repository_configs']}


def validate_completed(run_dir, stage2, protocol, population, trainer_sha256):
    """Bind actual final inputs, all frozen settings and the complete training stream."""
    run_dir = Path(run_dir).resolve()
    final_path = run_dir / 'test_final.json'
    final = read_json(final_path)
    if (final['stage2_run_id'] != stage2['run_id']
            or Path(final['stage2_source_tree']).resolve() != Path(stage2['source_tree']).resolve()
            or Path(final['stage3_run_id']).resolve() != run_dir):
        raise ValueError('Stage3 run/source identity mismatch')
    for name, field in (('sid', 'sid_path'), ('checkpoint', 'stage2_checkpoint'),
                        ('export', 'stage2_export_manifest'), ('binding', 'archive_binding')):
        digest_key = 'sid_sha256' if name == 'sid' else field + '_sha256'
        if (Path(final[field]).resolve() != Path(stage2['artifacts'][name]['path']).resolve()
                or final[digest_key] != stage2['artifacts'][name]['sha256']):
            raise ValueError('Stage3 input identity mismatch: ' + field)
        check_descriptor(stage2['artifacts'][name])
    if (final['n_eval'] != population or final['stage3_world_size'] != 4
            or final['stage3_completed_epochs'] != 150
            or final['stage3_trainer_sha256'] != trainer_sha256):
        raise ValueError('Stage3 population, completion or trainer mismatch')
    config = final['stage3_input_config']
    for key, value in CORE_PROTOCOL.items():
        if config.get(key) != value or protocol.get(key) != value:
            raise ValueError('frozen Stage3 protocol mismatch: ' + key)
    excluded = PATH_CONFIG | SID_CONFIG
    if ({k: v for k, v in config.items() if k not in excluded}
            != {k: v for k, v in protocol.items() if k not in excluded}):
        raise ValueError('Stage3 configuration differs from frozen Issue253 protocol')
    # These are data-derived values in the frozen trainer, not tunable settings.
    # Recompute the exact RecBole namespace from this run's bound SID.
    sid_rows = read_json(stage2['artifacts']['sid']['path'])
    max_sid = max(max(row) for row in sid_rows.values())
    expected_eos = max_sid + config['n_user_tokens'] + 2
    if config.get('eos_token_id') != expected_eos or config.get('vocab_size') != expected_eos + 1:
        raise ValueError('SID-derived Stage3 vocabulary/EOS protocol mismatch')
    if (Path(config['code_path']).resolve() != Path(final['sid_path']).resolve()
            or Path(config['dataset_path']).resolve() != TEST_PARQUET.parent.resolve()
            or (Path(config['log_path']) / config['dataset_name'] / run_dir.name).resolve() != run_dir):
        raise ValueError('Stage3 configured input/output path mismatch')
    checkpoint = (Path(config['save_path']) / config['dataset_name'] / run_dir.name / 'HG_Rec_best.pth')
    if Path(final['best_checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Stage3 checkpoint generation mismatch')
    metrics_path = run_dir / 'training_metrics.jsonl'
    stream = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    starts = [row for row in stream if row.get('event') == 'train_start']
    tests = [row for row in stream if row.get('event') == 'test']
    ends = [row for row in stream if row.get('event') == 'train_end']
    if ([row['epoch'] for row in stream if row.get('event') == 'train'] != list(range(1, 151))
            or len(starts) != 1 or len(tests) != 1 or len(ends) != 1
            or stream[0] != starts[0] or stream[-1] != ends[0]):
        raise ValueError('incomplete or duplicated Stage3 training population')
    start, test = starts[0], tests[0]
    expected_start = {'num_epochs': 150, 'no_eval': True, 'batch_size': config['batch_size'],
                      'infer_size': config['infer_size'], 'global_train_batch': config['batch_size'] * 4,
                      'world_size': 4, 'rank': 0, 'launch_mode': 'ddp4',
                      'code_path_sha256': final['sid_sha256']}
    if (any(start.get(k) != v for k, v in expected_start.items())
            or Path(start['code_path']).resolve() != Path(final['sid_path']).resolve()
            or Path(start['log_path']).resolve() != run_dir
            or test['epoch'] != 150 or test['n_eval'] != population
            or test['test_shard_world_size'] != 4):
        raise ValueError('Stage3 metrics provenance/population mismatch')
    for metric in ('recall', 'ndcg'):
        for cutoff in contract.CUTOFFS:
            if not math.isclose(test['test_%s_at_%d' % (metric, cutoff)],
                                final['test_%s@%d' % (metric, cutoff)], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Stage3 final/stream metric mismatch')
    files = {}
    for folder, suffix in (('predictions', '.jsonl'), ('_test_shards', '.json')):
        paths = sorted((run_dir / folder).glob('rank*' + suffix))
        if {p.name for p in paths} != {'rank%d%s' % (r, suffix) for r in range(4)}:
            raise ValueError('Stage3 prediction shard population mismatch')
        files[folder] = {p.name: descriptor(p) for p in paths}
    return {'run_dir': str(run_dir), 'test_final': descriptor(final_path),
            'metrics': descriptor(metrics_path), 'checkpoint': descriptor(checkpoint),
            'trainer_sha256': trainer_sha256, **files}


def resolve_stage3(stage2, protocol, population):
    """There must be one exact completed generation, never an mtime-selected run."""
    matching = []
    sid = stage2['artifacts']['sid']
    for path in sorted(STAGE3_LOGS.glob('*/*/test_final.json')):
        final = read_json(path)
        if (final.get('stage2_run_id') == stage2['run_id']
                or final.get('sid_sha256') == sid['sha256']
                or Path(final.get('sid_path', '')).resolve() == Path(sid['path']).resolve()):
            matching.append(path.parent)
    if len(matching) > 1:
        raise ValueError('ambiguous completed Stage3 generation for corrected SID/run')
    if not matching:
        raise ValueError('no completed Stage3 generation for corrected SID/run; scientific results remain NA')
    return validate_completed(matching[0], stage2, protocol, population, sha256(STAGE3_TRAINER))


def build_ledger(old, new, test_rows):
    # The frozen consumer checks user + full-history/seen-history hash + target,
    # as well as complete global-row coverage. Rename its arms, not its semantics.
    ledger, _ = frozen_ledger(old, new, test_rows)
    result = []
    for record in ledger:
        row = {key.replace('issue253', 'issue255').replace('issue252', 'issue253'): value
               for key, value in record.items()}
        for cutoff in contract.CUTOFFS:
            row['group%d' % cutoff] = GROUP_NAMES[row['group%d' % cutoff]]
        result.append(row)
    return result


def summarise(ledger, old, new):
    pairs = [(row['rank_issue253'], row['rank_issue255']) for row in ledger]
    result = {}
    for cutoff in contract.CUTOFFS:
        counts = contract.group_counts(pairs, cutoff)
        cell = {'groups': {GROUP_NAMES[k]: v for k, v in counts.items()},
                'net_change': contract.net_change(counts, len(pairs)),
                'net_hit_change': counts[contract.ONLY_253] - counts[contract.ONLY_252]}
        for index, name, arm in ((0, 'issue253', old), (1, 'issue255', new)):
            for metric, reducer in (('recall', contract.recall), ('ndcg', contract.ndcg)):
                value = reducer(pairs, cutoff, index)
                cell[metric + '_' + name] = value
                if not math.isclose(value, arm['final']['test_%s@%d' % (metric, cutoff)],
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError('independently paired headline disagrees with final result')
        cell['net_ndcg_change'] = cell['ndcg_issue255'] - cell['ndcg_issue253']
        result[str(cutoff)] = cell
    return result


def main():
    stage2 = stage2_identity()
    raw = read_json(stage2['artifacts']['sid']['path'])
    collision = 1 - len({tuple(row[:3]) for row in raw.values()}) / len(raw)
    if not collision < .4:
        raise ValueError('three-code collision >= 0.4; Stage3 pairing is forbidden')
    test_rows = read_test_rows()
    if len(test_rows) != POPULATION:
        raise ValueError('frozen test population differs')
    baseline_final = read_json(ISSUE253 / 'test_final.json')
    baseline = baseline_identity(baseline_final)
    protocol = baseline_final['stage3_input_config']
    old_receipt = validate_completed(ISSUE253, baseline, protocol, POPULATION, BASELINE_TRAINER_SHA256)
    new_receipt = resolve_stage3(stage2, protocol, POPULATION)
    old = load_arm(ISSUE253, test_rows)
    new = load_arm(Path(new_receipt['run_dir']), test_rows)
    ledger = build_ledger(old, new, test_rows)
    cutoffs = summarise(ledger, old, new)
    inputs = {'test_split': descriptor(TEST_PARQUET), 'stage2': stage2,
              'issue253_stage2': baseline, 'issue253': old_receipt, 'issue255': new_receipt}
    inputs_sha = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    evaluation = EVIDENCE / 'evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    records_path = evaluation / 'paired_users.jsonl'
    with records_path.open('w', encoding='utf-8') as handle:
        for record in ledger:
            record.update(paired_inputs_sha256=inputs_sha,
                          test_split_sha256=inputs['test_split']['sha256'],
                          issue253_test_final_sha256=old_receipt['test_final']['sha256'],
                          issue255_test_final_sha256=new_receipt['test_final']['sha256'],
                          stage2_run_id=RUN_ID, stage3_run_id=new_receipt['run_dir'])
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    summary = {'schema': 'genrec.issue255.paired/2', 'run_id': RUN_ID, 'population': len(ledger),
               'issue253_run': str(ISSUE253), 'new_run': new_receipt['run_dir'],
               'cutoffs': cutoffs, 'records': descriptor(records_path), 'paired_inputs': inputs,
               'paired_inputs_sha256': inputs_sha}
    summary_path = EVIDENCE / 'PAIRED_STAGE3.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    receipt = {'schema': 'genrec.issue255.stage3_result/1', 'run_id': RUN_ID, 'arm': ARM,
               'stage2': stage2, 'run_dir': new_receipt['run_dir'],
               'test_final': new_receipt['test_final'], 'stage3': new_receipt,
               'issue253': old_receipt, 'paired': descriptor(summary_path)}
    (EVIDENCE / 'STAGE3_RESULT.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
