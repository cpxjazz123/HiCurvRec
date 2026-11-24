"""Freeze the evidence for Issue254: identities, hashes and join preconditions.

Read-only: it reads the two finished generations, the frozen assets and the
test split, and writes a single JSON record of what it actually saw.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT/'results/research/issue254'

ARMS = {
    'issue252': {
        'arm': 'TIGER_gnn_attention_sectional_rq',
        'stage2_run': 'issue252_formal_r2',
        'stage3_run': ('results/stage3_T5Train/TIGER_gnn_attention_sectional_rq/logs/'
                       'Amazon_2023_Instruments/Nov-24-2026_15-34-06'),
        'r2_number': '100',
    },
    'issue253': {
        'arm': 'TIGER_three_expert_attention_rq',
        'stage2_run': 'stage2_787483d6ef9842d1a7c4593a61ae925c',
        'stage3_run': ('results/stage3_T5Train/TIGER_three_expert_attention_rq/logs/'
                       'Amazon_2023_Instruments/Nov-24-2026_21-16-28'),
        'r2_number': '101',
    },
}

TEST_PARQUET = ROOT/'results/stage0_build_parquet/test.parquet'
ASSETS = ROOT/'assets'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    if not path.exists():
        return {'path': str(path), 'present': False}
    if path.is_dir():
        return {'path': str(path), 'present': True, 'kind': 'directory'}
    return {'path': str(path), 'present': True, 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def arm_evidence(name, spec):
    stage2 = ROOT/'results/stage2_RQ-VAE/_runs'/spec['stage2_run']/spec['arm']/'result'
    stage3 = ROOT/spec['stage3_run']
    final = json.loads((stage3/'test_final.json').read_text())
    binding = json.loads((stage2/'archive_binding.json').read_text())
    export = json.loads((stage2/'export_manifest.json').read_text())
    ledger = json.loads((ROOT/'results/research/issue139/r2_ledger.json').read_text())[spec['r2_number']]
    out = {'arm': spec['arm'], 'stage2_run_id': spec['stage2_run'],
           'stage2_result': record(stage2), 'stage2_source': record(stage2.parent/'src'),
           'stage3_run_dir': str(stage3.relative_to(ROOT)), 'r2_number': spec['r2_number'],
           'stage3_test_final': record(stage3/'test_final.json'),
           'stage3_metrics': record(stage3/'training_metrics.jsonl'),
           'stage3_log': record(stage3/'HG_Rec.log'),
           'stage3_checkpoint': record(final['best_checkpoint']),
           'stage2_metrics': record(stage2/'logs/training_metrics.jsonl'),
           'stage3_shards': [record(stage3/'_test_shards'/('rank%d.json' % r)) for r in range(4)],
           'stage3_predictions': [record(stage3/'predictions'/('rank%d.jsonl' % r)) for r in range(4)]}
    for field, filename, final_field, remote in [
            ('stage2_checkpoint', 'rqvae_best.pth', 'stage2_checkpoint_sha256', 'rqvae_best.pth'),
            ('stage2_sid', 'item_sids.json', 'sid_sha256', 'item_sids.json'),
            ('stage2_export_manifest', 'export_manifest.json', 'stage2_export_manifest_sha256',
             spec['arm']+'/export_manifest.json'),
            ('stage2_archive_binding', 'archive_binding.json', 'archive_binding_sha256',
             spec['arm']+'/archive_binding.json')]:
        out[field] = record(stage2/filename)
        actual = out[field]['sha256']
        if actual != final[final_field] or actual != ledger['files'][remote]:
            raise ValueError(name+' Stage3/R2 generation mismatch: '+field)
    if not export['tokens_match'] or export['differing_items'] != 0:
        raise ValueError('checkpoint SID not proven identical')
    for field in ['sid', 'checkpoint', 'export_manifest']:
        if binding[field]['sha256'] != sha256(binding[field]['path']):
            raise ValueError('archive binding mismatch: '+field)
    inventory = {}
    for path, expected in binding['candidate_sources'].items():
        actual = sha256(Path(binding['source_tree'])/path)
        if actual != expected:
            raise ValueError('immutable source changed: '+path)
        inventory[path] = actual
    trainer = ROOT/'stage3_T5Train/train_HG-Rec.py'
    if sha256(trainer) != final['stage3_trainer_sha256']:
        raise ValueError('Stage3 trainer changed')
    out['source_inventory'] = {'files': len(inventory),
                               'sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
                               'mapping': 'stage2_archive_binding.candidate_sources',
                               'all_match': True}
    out['r2'] = {'key': ledger['key'], 'files': len(ledger['files']),
                 'generation_hashes_match': True, 'receipt_reused': True}
    out['stage3_config'] = final['stage3_input_config']
    out['stage3_trainer'] = record(trainer)
    out['stage3_dataset_codec'] = record(ROOT/'stage3_T5Train/data/dataset.py')
    out['stage3_dataloader'] = record(ROOT/'stage3_T5Train/data/dataloader.py')
    out['stage2_trainer_copy_is_not_authority'] = record(stage2/'item_sids.trainer.json')
    return out


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
    by_path = {entry['path']: entry for entry in manifest['assets']}
    wanted = ['assets/index.npz', 'assets/node_moments.npz', 'assets/item_summary.npz',
              'assets/train/visible_forward.npz', 'assets/train/visible_reverse.npz',
              'assets/train/visible_undirected.npz', 'assets/shards/shard_000.npz',
              'assets/v8_runtime_layout/targets.npy',
              'assets/v8_runtime_layout/target_offsets.npy',
              'assets/v8_runtime_layout/two_hop_nodes.npy',
              'assets/v8_runtime_layout/two_hop_offsets.npy',
              'assets/v8_runtime_layout/visible_forward_bits.npy',
              'results/research/issue247/assets/topology_features.npz',
              'results/research/infra246/assets/motifs/motif_counts.npz',
              'results/research/infra246/assets/motifs/motif_instances_visible.npz',
              'results/research/infra246/assets/motifs/motif_instances_4node_visible.npz',
              'results/stage0_build_parquet/test.parquet',
              'stage1_GeneEmbedding/output/sentence_t5.npy']
    assets = []
    for path in wanted:
        entry = by_path.get(path, {})
        assets.append({'path': path, 'use': entry.get('use'), 'registered': path in by_path,
                       'manifest_sha256': entry.get('sha256'), 'actual_sha256': sha256(ROOT/path)
                       if (ROOT/path).is_file() else None,
                       'bytes': (ROOT/path).stat().st_size if (ROOT/path).is_file() else None})
        if path in by_path and assets[-1]['manifest_sha256'] != assets[-1]['actual_sha256']:
            raise ValueError('registered asset hash mismatch: '+path)
    record_ = {
        'schema': 'genrec.issue254.evidence/1',
        'issue': 254,
        'purpose': 'evaluation',
        'test_parquet': record(TEST_PARQUET),
        'asset_manifest': record(ASSETS/'manifest.json'),
        'assets': assets,
        'generations': {name: arm_evidence(name, spec) for name, spec in ARMS.items()},
        'r2_ledger': record(ROOT/'results/research/issue139/r2_ledger.json'),
        'issue253_final_evaluation': record(ROOT/'results/research/issue253/FINAL_EVALUATION.json'),
        'issue253_reconciliation': record(ROOT/'results/research/issue253/RECONCILIATION.json'),
        'issue252_report': record(ROOT/'results/research/issue252/REPORT.md'),
        'issue253_report': record(ROOT/'results/research/issue253/REPORT.md'),
    }
    generations = record_['generations']
    a, b = [generations[name]['stage3_config'] for name in ('issue252', 'issue253')]
    differences = {key: [a.get(key), b.get(key)] for key in a.keys() | b.keys()
                   if a.get(key) != b.get(key)}
    allowed = {'code_path', 'log_path', 'save_path', 'vocab_size', 'eos_token_id'}
    if set(differences) - allowed:
        raise ValueError('non-catalogue Stage3 configuration changed')
    record_['protocol_comparison'] = {
        'differences': differences, 'other_fields_identical': True,
        'search_beam': 20, 'persisted_candidates': 10, 'population': 57439,
        'note': 'vocabulary/EOS derived from fourth collision suffix; not an identical-vocabulary counterfactual'}
    record_['verdicts'] = {name: record(ROOT/'results/research'/name/'VERDICT.md')
                           for name in ('issue252', 'issue253')}
    record_['r2_receipts'] = {
        'issue252': record(ROOT/'results/research/issue252/RECONCILIATION.json'),
        'issue253': record(ROOT/'results/research/issue253/R2_RECEIPT.json')}
    prior = json.loads((ROOT/'results/research/issue253/RECONCILIATION.json').read_text())
    if prior['evidence']['test_input']['sha256'] != record_['test_parquet']['sha256']:
        raise ValueError('original test split hash changed')
    record_['test_split_original_sha_verified'] = True
    record_['history_identity_limitation'] = (
        'Original prediction dumps have no explicit user/history. Identity is reconstructed from the '
        'hash-verified split and trainer sharding, with every target and canonical SID independently checked. '
        'No per-sample historical input tensor digest was saved.')
    path = EVIDENCE/'EVIDENCE.json'
    path.write_text(json.dumps(record_, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(path), 'sha256': sha256(path),
                      'test_parquet': record_['test_parquet'],
                      'generations': {name: {'stage2_sid': gen['stage2_sid'].get('sha256'),
                                             'stage2_checkpoint': gen['stage2_checkpoint'].get('sha256'),
                                             'stage3_test_final': gen['stage3_test_final'].get('sha256')}
                                       for name, gen in record_['generations'].items()}},
                     indent=2))


if __name__ == '__main__':
    main()
