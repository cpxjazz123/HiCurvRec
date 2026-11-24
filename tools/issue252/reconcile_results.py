"""Independent reduction of the exact completed Issue252 generation.

No training, SID export, latest-run discovery or metric-selected input. Recall
is recomputed from per-user zero-based ranks, not the trainer's accumulators;
curvature is reaggregated from the independently measured per-centre records.
"""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from tools.issue252.archive_audit import sha256, validate_binding

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
ARM = 'TIGER_gnn_attention_sectional_rq'
RUN = ROOT / 'results/stage2_RQ-VAE/_runs/issue252_formal_r2' / ARM
RESULT = RUN / 'result'
STAGE3 = ROOT / 'results/stage3_T5Train' / ARM / 'logs/Amazon_2023_Instruments/Nov-24-2026_15-34-06'
EVIDENCE = ROOT / 'results/research/issue252'
OUTPUT = EVIDENCE / 'RECONCILIATION.json'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def _close(actual, expected, label):
    _require(math.isfinite(actual) and math.isfinite(expected)
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
             'independent metric disagrees: ' + label)


def remeasure_test(shards):
    """Four disjoint persisted shards; global user weighting, rank zero is first."""
    _require(len(shards) == 4 and sorted(s['rank'] for s in shards) == [0, 1, 2, 3],
             'missing or duplicate test rank')
    ranks = []
    for shard in shards:
        users = shard['per_user']
        _require(type(shard['n_eval']) is int and shard['n_eval'] == len(users) and users,
                 'per-user population differs from shard denominator')
        _require([row['local_row_index'] for row in users] == list(range(len(users))),
                 'missing or duplicate local evaluation row')
        for row in users:
            rank = row['target_rank']
            _require(rank is None or type(rank) is int and 0 <= rank < 20,
                     'invalid zero-based target rank')
            ranks.append(rank)
    result = {'n_eval': len(ranks)}
    for k in (5, 10):
        hits = [rank for rank in ranks if rank is not None and rank < k]
        result['test_recall@%d' % k] = len(hits) / len(ranks)
        result['test_ndcg@%d' % k] = math.fsum(1 / math.log2(rank + 2) for rank in hits) / len(ranks)
    return result


def bound_archive(ledger, key, sid_sha256, checkpoint_sha256):
    """Unkeyed historical entries cannot supply this exact SID/checkpoint pair."""
    matches = [(number, entry) for number, entry in ledger.items() if entry.get('key') == key]
    _require(len(matches) == 1, 'missing or ambiguous exact R2 archive')
    number, archive = matches[0]
    _require(archive['files']['item_sids.json'] == sid_sha256
             and archive['files']['rqvae_best.pth'] == checkpoint_sha256,
             'R2 archive input generation differs')
    return number, archive


def _window(rows):
    _require(rows and all(math.isfinite(r['start']) and math.isfinite(r['end'])
                         and r['end'] > r['start'] for r in rows), 'invalid epoch boundary')
    for left, right in zip(rows, rows[1:]):
        _require(right['epoch'] == left['epoch'] + 1 and right['start'] >= left['end'],
                 'nonconsecutive or overlapping epochs')
    values = sorted(row['end'] - row['start'] for row in rows)
    position = .95 * (len(values) - 1)
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    wall = rows[-1]['end'] - rows[0]['start']
    return {'epochs': len(rows), 'first_epoch': rows[0]['epoch'], 'last_epoch': rows[-1]['epoch'],
            'wall_seconds': wall, 'mean_seconds': wall / len(rows),
            'median_seconds': statistics.median(values),
            'p95_seconds': values[lower] + (position - lower) * (values[upper] - values[lower])}


def _curvature(report):
    rows = list(report['per_item'].values())
    count = sum(row['count'] for row in rows)
    _require(count == report['count'] == 28391948, 'full frozen curvature coverage differs')
    macro = math.fsum(row['mean_abs_error'] for row in rows) / len(rows)
    micro = math.fsum(row['mean_abs_error'] * row['count'] for row in rows) / count
    _close(macro, report['macro_mae'], 'curvature macro MAE')
    _close(micro, report['micro_mae'], 'curvature micro MAE')
    recalls = report['sign']['recall']
    balanced = math.fsum(recalls[name]['recall'] for name in report['sign']['present_signs']) / len(report['sign']['present_signs'])
    _close(balanced, report['sign']['balanced_accuracy'], 'curvature balanced accuracy')
    return {'count': count, 'centres': len(rows), 'macro_mae': macro, 'micro_mae': micro,
            'balanced_accuracy': balanced, 'sign_recalls': recalls}


def main():
    sid, checkpoint = RESULT / 'item_sids.json', RESULT / 'rqvae_best.pth'
    binding, _ = validate_binding(RESULT / 'archive_binding.json', sid, checkpoint, RUN / 'src')
    workspace = _json(RUN / 'workspace.json')
    original = EVIDENCE / 'admission_history' / (workspace['training_admission']['sha256'] + '.json')
    _require(sha256(original) == workspace['training_admission']['sha256'],
             'original pre-training receipt was not preserved')
    public = _json(EVIDENCE / 'PUBLIC_VERIFICATION.json')
    _require(public.get('passed') is True and {row['name']: row['status'] for row in public['checks']}
             == {'entry': 'PASS', 'behavior_graph': 'PASS'}, 'current public admission is incomplete')

    payload = _json(sid)
    _require(set(payload) == {str(i) for i in range(24587)}, 'SID catalogue is incomplete')
    triples = [tuple(payload[str(i)][:3]) for i in range(24587)]
    _require(all(len(row) == 3 and all(type(code) is int and 0 <= code < 256 for code in row)
                 for row in triples), 'invalid actual exported code')
    prefixes2, prefixes3 = Counter(row[:2] for row in triples), Counter(triples)
    entropy = lambda counts: -math.fsum(n / 24587 * math.log2(n / 24587) for n in counts.values())
    catalogue = {'items': 24587, 'distinct_l2_prefixes': len(prefixes2),
                 'distinct_l3_triples': len(prefixes3), 'l3_collision': 1 - len(prefixes3) / 24587,
                 'l3_refinement_gain_bits': entropy(prefixes3) - entropy(prefixes2),
                 'codebook_occupancy': [len({row[level] for row in triples}) for level in range(3)]}
    layer_path = ROOT / 'results/stage2_RQ-VAE' / ARM / 'stage2_metrics' / (sha256(sid) + '.json')
    layer = _json(layer_path)
    _require(layer['sid'] == _record(sid) and layer['run_id'] == binding['run_id'],
             'layer record belongs to another SID generation')
    for key, value in catalogue.items():
        if isinstance(value, list):
            _require(value == layer['catalogue'][key], 'occupancy disagrees')
        else:
            _close(value, layer['catalogue'][key], key)

    metrics_path = RESULT / 'logs/training_metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    trains = [r for r in metrics if r['event'] == 'train']
    boundaries = [r for r in metrics if r['event'] == 'epoch_boundary']
    _require([r['epoch'] for r in trains] == list(range(1, 3001))
             and [r['epoch'] for r in boundaries] == list(range(1, 3001)), 'Stage2 is not 3000 complete epochs')
    _require(all(r.get('run_id') == binding['run_id'] for r in metrics), 'Stage2 metric run identity differs')
    _require(metrics[-1]['event'] == 'train_end' and metrics[-1]['checkpoint_selection'] == 'final_epoch',
             'Stage2 final completion/selection missing')
    _require(all(math.isfinite(r[key]) for r in trains for key in ('loss', 'recon', 'curvature_estimate',
                                                                  'gradient_norm_before_clip')), 'nonfinite formal loss/gradient')
    _require(all(len(r['structural_coverage']) == 4 and all(c['epoch_updates'] == 6
                 for c in r['structural_coverage']) for r in boundaries), 'structural updates were skipped')
    start = next(r for r in metrics if r['event'] == 'train_start')
    _require(start['epochs'] == 3000 and start['total_batch_size'] == 4096
             and start['batch_size_per_rank'] == 1024, 'Stage2 frozen recipe differs')
    timing = {'formal_1_3000': _window(boundaries), 'formal_31_60': _window(boundaries[30:60]),
              'formal_31_3000': _window(boundaries[30:]),
              'initialization_seconds': next(r['initialization_seconds'] for r in metrics
                                             if r['event'] == 'initialization_complete'),
              'trainer_total_seconds': metrics[-1]['total_wall_time_s'],
              'last_boundary_to_end_seconds': metrics[-1]['timestamp'] - boundaries[-1]['timestamp']}

    final_path = STAGE3 / 'test_final.json'
    final = _json(final_path)
    for name, path in (('sid', sid), ('stage2_checkpoint', checkpoint),
                       ('archive_binding', RESULT / 'archive_binding.json')):
        path_key = 'sid_path' if name == 'sid' else name
        _require(Path(final[path_key]).resolve() == path.resolve()
                 and final[name + '_sha256'] == sha256(path), 'Stage3 input identity differs: ' + name)
    _require(final['stage2_run_id'] == binding['run_id'] and final['stage3_run_id'] == str(STAGE3)
             and final['stage3_completed_epochs'] == 150 and final['stage3_world_size'] == 4,
             'Stage3 formal generation or epoch budget differs')
    config = final['stage3_input_config']
    _require(all(config[k] == value for k, value in {'num_epochs': 150, 'seed': 42, 'beam_size': 20,
                 'no_eval': True, 'skip_test': False}.items()), 'Stage3 frozen protocol differs')
    stage3_metrics_path = STAGE3 / 'training_metrics.jsonl'
    stream = [json.loads(line) for line in stage3_metrics_path.read_text().splitlines() if line.strip()]
    _require([r['epoch'] for r in stream if r['event'] == 'train'] == list(range(1, 151))
             and stream[-1]['event'] == 'train_end', 'Stage3 epochs/completion are incomplete')
    shard_paths = [STAGE3 / '_test_shards' / ('rank%d.json' % rank) for rank in range(4)]
    recall = remeasure_test([_json(path) for path in shard_paths])
    _require(recall['n_eval'] == final['n_eval'], 'Stage3 evaluated population differs')
    for key, value in recall.items():
        _close(value, final[key], key)
    _require(final['stage3_trainer_sha256'] == sha256(ROOT / 'stage3_T5Train/train_HG-Rec.py'),
             'Stage3 trainer source changed')

    full_path = EVIDENCE / 'FINAL_EVALUATION.json'
    full = _json(full_path)
    _require(full['provenance']['run_id'] == binding['run_id']
             and full['provenance']['artifacts']['sid'] == _record(sid), 'full evaluation belongs to another run')
    geometry = {name: _curvature(full[name]['curvature']) for name in ('hard_q', 'continuous_z')}
    ledger_path = ROOT / 'results/research/issue139/r2_ledger.json'
    key = ARM + '/' + sha256(sid)[:12]
    number, archive = bound_archive(_json(ledger_path), key, sha256(sid), sha256(checkpoint))
    result = {'schema': 'genrec.issue252.reconciliation/1', 'run_id': binding['run_id'],
              'catalogue': catalogue, 'timing': timing, 'recall_from_per_user_ranks': recall,
              'curvature_from_independent_per_centre_records': geometry,
              'historical_tiger': full['historical_tiger'],
              'scientific_threshold': {'test_R@10_greater_than': .065, 'met': recall['test_recall@10'] > .065},
              'last_training_gate_mean': trains[-1]['gate_mean'],
              'legacy_conditional_g3': layer['validation']['conditional_g3'],
              'public_admission_timing': 'post-training reconciliation; not backdated to formal dispatch',
              'r2': {'number': number, 'target': 'r2:genrec/' + number + '/', 'key': key},
              'evidence': {'original_training_admission': _record(original), 'public_verification': _record(EVIDENCE / 'PUBLIC_VERIFICATION.json'),
                           'stage2_metrics': _record(metrics_path), 'stage3_metrics': _record(stage3_metrics_path),
                           'test_final': _record(final_path), 'full_evaluation': _record(full_path),
                           'layer_record': _record(layer_path), 'source': _record(__file__),
                           'test_shards': [_record(path) for path in shard_paths]}}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(OUTPUT), 'sha256': sha256(OUTPUT),
                      'recall': recall, 'r2': result['r2']}, indent=2))


if __name__ == '__main__':
    main()
