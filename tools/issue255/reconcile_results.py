"""Recompute headlines from immutable SID, per-centre rows and final test ranks.

This reducer does not import the mechanism metric reducers. It rechecks the
full cell populations, not a second geometric forward; numerical geometry is
covered separately by the independent known-answer/reference tests.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.issue255.identity import ARM, EVIDENCE, RESULT, ROOT, RUN, RUN_ID


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1.e-10, abs_tol=1.e-10):
        raise ValueError('%s disagrees: %r vs %r' % (name, actual, expected))


def curvature(cell):
    path = Path(cell['per_item']['path'])
    if sha256(path) != cell['per_item']['sha256']:
        raise ValueError('per-centre curvature data changed')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len({row['item'] for row in rows}) != len(rows):
        raise ValueError('duplicate centre in full-reference cell')
    count = sum(row['count'] for row in rows)
    if count != 28391948 or len(rows) != 24326:
        raise ValueError('incomplete curvature population')
    result = {'count': count, 'centres': len(rows),
              'macro_mae': statistics.fmean(row['mean_abs_error'] for row in rows),
              'micro_mae': math.fsum(row['count'] * row['mean_abs_error'] for row in rows) / count,
              'recall': {}}
    for name in ('negative', 'zero', 'positive'):
        population = sum(row['count_' + name] for row in rows)
        if population:
            recall = math.fsum(row['count_' + name] * row['recall_' + name]
                               for row in rows if row['count_' + name]) / population
            result['recall'][name] = recall
            close(recall, cell['sign']['recall'][name]['recall'], 'sign recall ' + name)
    result['balanced_sign_accuracy'] = statistics.fmean(result['recall'].values())
    close(result['macro_mae'], cell['macro_mae'], 'macro MAE')
    close(result['micro_mae'], cell['micro_mae'], 'micro MAE')
    close(result['balanced_sign_accuracy'], cell['sign']['balanced_accuracy'], 'balanced sign')
    return result


def reduce_ranks(pairs):
    """Independent zero-based, nullable rank arithmetic (no consumer reducers)."""
    if not pairs:
        raise ValueError('empty paired population')
    result = {}
    for cutoff in (5, 10):
        hits = [0, 0]
        gains = [[], []]
        groups = Counter()
        for left, right in pairs:
            outcomes = []
            for index, rank in enumerate((left, right)):
                if rank is not None and (not isinstance(rank, int) or rank < 0):
                    raise ValueError('invalid zero-based rank')
                hit = rank is not None and rank < cutoff
                outcomes.append(hit)
                hits[index] += int(hit)
                gains[index].append(1 / math.log2(rank + 2) if hit else 0.0)
            groups[tuple(outcomes)] += 1
        ndcg = [math.fsum(column) / len(pairs) for column in gains]
        result[str(cutoff)] = {
            'old_recall': hits[0] / len(pairs), 'new_recall': hits[1] / len(pairs),
            'old_ndcg': ndcg[0], 'new_ndcg': ndcg[1],
            'both_hit': groups[(True, True)], 'old_only': groups[(True, False)],
            'new_only': groups[(False, True)], 'both_miss': groups[(False, False)],
            'net_hit_change': hits[1] - hits[0],
            'net_change': (hits[1] - hits[0]) / len(pairs),
            'net_ndcg_change': ndcg[1] - ndcg[0]}
    return result


def stage3(collision):
    receipt_path = EVIDENCE / 'STAGE3_RESULT.json'
    if not collision < .4:
        if receipt_path.exists():
            raise ValueError('Stage3 receipt cannot waive three-code collision >= 0.4')
        return {'status': 'NOT_RUN_COLLISION_GATE', 'recall': 'NA',
                'reason': 'three-code collision >= 0.4; no gate waiver or invented recall'}
    if not receipt_path.is_file():
        raise ValueError('eligible SID still requires its frozen Stage3 final result')
    from tools.issue255 import paired_comparison as paired
    stage2 = paired.stage2_identity()
    receipt = read_json(receipt_path)
    baseline_final = read_json(paired.ISSUE253 / 'test_final.json')
    baseline = paired.baseline_identity(baseline_final)
    protocol = baseline_final['stage3_input_config']
    test = paired.read_test_rows()
    if len(test) != paired.POPULATION:
        raise ValueError('test population differs')
    old_receipt = paired.validate_completed(paired.ISSUE253, baseline, protocol, len(test),
                                            paired.BASELINE_TRAINER_SHA256)
    new_receipt = paired.resolve_stage3(stage2, protocol, len(test))
    if (receipt['run_id'] != RUN_ID or receipt['arm'] != ARM or receipt['stage2'] != stage2
            or receipt['stage3'] != new_receipt or receipt['issue253'] != old_receipt
            or receipt['test_final'] != new_receipt['test_final']
            or receipt['run_dir'] != new_receipt['run_dir']):
        raise ValueError('Stage3 result receipt identity mismatch')
    aggregate_path = paired.check_descriptor(receipt['paired'], EVIDENCE / 'PAIRED_STAGE3.json')
    aggregate = read_json(aggregate_path)
    inputs = {'test_split': paired.descriptor(paired.TEST_PARQUET), 'stage2': stage2,
              'issue253_stage2': baseline, 'issue253': old_receipt, 'issue255': new_receipt}
    inputs_sha = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (aggregate['run_id'] != RUN_ID or aggregate['population'] != len(test)
            or aggregate['paired_inputs'] != inputs or aggregate['paired_inputs_sha256'] != inputs_sha
            or aggregate['new_run'] != new_receipt['run_dir']
            or Path(aggregate['issue253_run']).resolve() != paired.ISSUE253.resolve()):
        raise ValueError('paired aggregate input identity mismatch')
    records_path = paired.check_descriptor(aggregate['records'], EVIDENCE / 'evaluation/paired_users.jsonl')
    old = paired.load_arm(paired.ISSUE253, test)
    new = paired.load_arm(Path(new_receipt['run_dir']), test)
    expected = paired.build_ledger(old, new, test)
    seen = 0
    with records_path.open(encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            if index >= len(expected):
                raise ValueError('extra paired ledger population')
            row = expected[index]
            row.update(paired_inputs_sha256=inputs_sha,
                       test_split_sha256=inputs['test_split']['sha256'],
                       issue253_test_final_sha256=old_receipt['test_final']['sha256'],
                       issue255_test_final_sha256=new_receipt['test_final']['sha256'],
                       stage2_run_id=RUN_ID, stage3_run_id=new_receipt['run_dir'])
            if json.loads(line) != row:
                raise ValueError('persisted paired composite identity/rank/metric differs')
            seen += 1
    if seen != len(test):
        raise ValueError('missing paired ledger population')
    cutoffs = reduce_ranks([(old['ranks'][row['global_row']], new['ranks'][row['global_row']])
                           for row in test])
    for cutoff, cell in cutoffs.items():
        summary = aggregate['cutoffs'][cutoff]
        for name in ('both_hit', 'old_only', 'new_only', 'both_miss'):
            if cell[name] != summary['groups'][name]:
                raise ValueError('paired group count disagrees')
        for name in ('net_hit_change', 'net_change', 'net_ndcg_change'):
            close(cell[name], summary[name], 'paired ' + name)
        for name, arm in (('old', 'issue253'), ('new', 'issue255')):
            final = old['final'] if name == 'old' else new['final']
            for metric in ('recall', 'ndcg'):
                measured = cell[name + '_' + metric]
                close(measured, final['test_%s@%s' % (metric, cutoff)], 'Stage3 ' + metric)
                close(measured, summary[metric + '_' + arm], 'paired ' + metric)
    return {'status': 'COMPLETE', 'population': len(test), 'cutoffs': cutoffs,
            'records': aggregate['records'], 'test_final': new_receipt['test_final'],
            'receipt': {'path': str(receipt_path), 'sha256': sha256(receipt_path)}}


def main():
    import numpy as np
    from scipy.sparse import load_npz
    from tools.issue252.run_workspace import load_run
    from tools.issue255.paired_comparison import stage2_identity
    load_run(ROOT, RUN_ID, ARM)
    final = stage2_identity()
    raw = read_json(RESULT / 'item_sids.json')
    if set(raw) != {str(i) for i in range(24587)}:
        raise ValueError('SID catalogue is incomplete')
    rows = [raw[str(i)] for i in range(24587)]
    triples = {tuple(row[:3]) for row in rows}
    collision = 1 - len(triples) / len(rows)
    counts = [Counter(tuple(row[:level]) for row in rows) for level in (1, 2, 3)]
    entropy = [-math.fsum(n / len(rows) * math.log2(n / len(rows)) for n in count.values())
               for count in counts]
    sid_result = {'items': len(rows), 'distinct_prefixes': [len(count) for count in counts],
                  'three_code_collision': collision, 'l3_refinement_gain_bits': entropy[2] - entropy[1],
                  'codebook_used': [len({row[level] for row in rows}) for level in range(3)]}
    graph = load_npz(ROOT / 'assets/train/visible_forward.npz').tocsr()
    degrees = np.diff(graph.indptr)
    source = np.repeat(np.arange(len(rows)), degrees)
    l1 = np.asarray([row[0] for row in rows])
    hit = l1[source] == l1[graph.indices]
    shared = np.bincount(source, weights=hit.astype(np.int64), minlength=len(rows))
    sharing = {'micro': int(hit.sum()) / len(hit),
               'out_centre_macro': statistics.fmean((shared[degrees > 0] / degrees[degrees > 0]).tolist())}
    evaluation_path = EVIDENCE / 'evaluation/FINAL_EVALUATION.json'
    evaluation = read_json(evaluation_path)
    generation = evaluation['generations']['issue255']
    close(collision, generation['sid_structure']['three_code_collision'], 'SID collision')
    close(sharing['micro'], generation['visible_directed_l1_sharing']['per_edge_rate'], 'edge sharing')
    close(sharing['out_centre_macro'], generation['visible_directed_l1_sharing']['out_direction']['macro_rate'],
          'centre sharing')
    cells = {name: {level: curvature(data['curvature']) for level, data in arm['layers'].items()}
             for name, arm in evaluation['generations'].items()}
    stream = [json.loads(line) for line in (RESULT / 'logs/training_metrics.jsonl').read_text().splitlines() if line]
    boundaries = [row for row in stream if row.get('event') == 'epoch_boundary' and 31 <= row['epoch'] <= 60]
    timing = (boundaries[-1]['end'] - boundaries[0]['start']) / len(boundaries)
    close(timing, read_json(final['artifacts']['training_summary']['path'])['window_mean_seconds'],
          'inclusive formal timing')
    result = {'schema': 'genrec.issue255.reconciliation/1', 'run_id': RUN_ID,
              'catalogue': sid_result, 'visible_edge_sharing': sharing, 'curvature': cells,
              'formal_inclusive_window_seconds': timing, 'stage3': stage3(collision),
              'source': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
              'evaluation': {'path': str(evaluation_path), 'sha256': sha256(evaluation_path)},
              'scope': 'independent SID/edge/rank arithmetic and per-centre full-population reductions'}
    (EVIDENCE / 'RECONCILIATION.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
