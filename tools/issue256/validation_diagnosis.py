"""Paired user/item diagnosis on existing complete validation predictions only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tools.issue256.identity import ROOT, EVIDENCE
from tools.issue252.final_evaluation import load_sid
from tools.issue252.archive_audit import sha256


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def checked(descriptor):
    path = Path(descriptor['path'])
    if record(path) != descriptor:
        raise ValueError('validation diagnosis input changed: ' + str(path))
    return path


def best_prefix(sids, targets, selected):
    matches = sids[selected, :3] == sids[targets, None, :3]
    return np.logical_and.accumulate(matches, axis=-1).sum(axis=-1).max(axis=-1)


def diagnose(evidence):
    import pandas as pd

    evidence = Path(evidence)
    output = evidence / 'recommendation_validation'
    receipt_path = output / 'VALIDATION.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'COMPLETE' or len(receipt['shards']) != 4:
        raise ValueError('complete paired validation must precede diagnosis')
    identity = receipt['identity']
    valid = checked(identity['valid'])
    sid = checked(identity['sid'])
    checked(identity['checkpoint'])
    frame = pd.read_parquet(valid)
    count = len(frame)
    if count <= 0 or receipt['metrics']['n_eval'] != count:
        raise ValueError('validation diagnosis must retain the complete frozen population')
    sids = load_sid(sid, 24587, [256] * 3)
    targets = frame['target'].to_numpy(dtype=np.int64)
    selected = np.empty((count, 10), dtype=np.int64)
    ranks = np.full(count, 10, dtype=np.int16)
    visited = np.zeros(count, dtype=bool)
    for rank, descriptor in enumerate(receipt['shards']):
        shard_path = checked(descriptor)
        shard = json.loads(shard_path.read_text())
        if shard['identity'] != identity or shard['rank'] != rank or shard['population'] != count:
            raise ValueError('mixed validation generations')
        numbers = []
        for line in checked(shard['predictions']).open():
            row = json.loads(line)
            number = row['global_row']
            if type(number) is not int or not 0 <= number < count or visited[number]:
                raise ValueError('invalid/duplicate validation row identity')
            items = [int(item['item']) for item in row['topk']]
            if (len(items) != 10 or len(set(items)) != 10 or min(items) < 0 or max(items) >= len(sids)
                    or row['target_item'] != int(targets[number])):
                raise ValueError('validation target/top10 identity differs from its Parquet row')
            expected_rank = items.index(int(targets[number])) if int(targets[number]) in items else None
            if row['target_rank'] != expected_rank:
                raise ValueError('validation target rank does not describe the selected items')
            selected[number] = items
            ranks[number] = 10 if expected_rank is None else expected_rank
            visited[number] = True
            numbers.append(number)
        if numbers != list(range(rank, count, 4)):
            raise ValueError('partial/misordered validation shard')
    if not visited.all():
        raise ValueError('partial validation diagnosis is forbidden')
    prefix = best_prefix(sids, targets, selected)
    seen_history = frame['seen_history'] if 'seen_history' in frame else frame['history']
    seen_target = np.fromiter((int(target) in history for target, history in zip(targets, seen_history)),
                              dtype=bool, count=count)
    discount = 1 / np.log2(ranks.astype(np.float64) + 2)

    def metrics(mask):
        number = int(mask.sum())
        if not number:
            return {'status': 'NA', 'n_eval': 0, 'reason': 'empty validation stratum'}
        result = {'status': 'OK', 'n_eval': number}
        for k in (5, 10):
            hit = mask & (ranks < k)
            result['recall@%d' % k] = int(hit.sum()) / number
            result['ndcg@%d' % k] = float(discount[hit].sum()) / number
        return result

    full = metrics(np.ones(count, dtype=bool))
    for key, actual in receipt['metrics'].items():
        if key != 'n_eval' and not math.isclose(full[key.removeprefix('validation_')], actual,
                                               rel_tol=1.e-10, abs_tol=1.e-10):
            raise ValueError('paired validation diagnosis changed a frozen metric')
    misses = ranks >= 10
    miss_count = int(misses.sum())
    miss_prefix = {'L%d' % level: {'count': int((misses & (prefix == level)).sum()),
        'fraction_of_all': int((misses & (prefix == level)).sum()) / count,
        'fraction_of_misses': int((misses & (prefix == level)).sum()) / miss_count if miss_count else None}
                   for level in range(4)}
    if sum(row['count'] for row in miss_prefix.values()) != miss_count:
        raise ValueError('miss-prefix groups do not partition validation failures')
    mechanism_path = evidence / 'evaluation/FINAL_EVALUATION.json'
    mechanism = json.loads(mechanism_path.read_text())
    if mechanism['run_id'] != identity['run_id']:
        raise ValueError('training geometry covariates belong to another candidate')
    with np.load(checked(mechanism['generation']['per_item_sid']), allow_pickle=False) as blob:
        if not np.array_equal(blob['sid'], sids):
            raise ValueError('training-side item identities differ from validation SID')
        out_degree = blob['out_degree']
        reconstruction = blob['reconstruction_l2']
    curvature = np.full((len(sids), 3), np.nan, dtype=np.float64)
    for level in range(3):
        cell = mechanism['generation']['layers']['L%d' % (level + 1)]['curvature']
        for line in checked(cell['per_item']).open():
            row = json.loads(line)
            curvature[row['item'], level] = row['mean_abs_error']

    def covariates(mask):
        items = targets[mask]
        if not len(items):
            return {'status': 'NA', 'reason': 'empty validation stratum'}
        result = {'validation_rows': len(items), 'mean_training_out_degree': float(out_degree[items].mean()),
                  'mean_training_catalogue_reconstruction_l2': float(reconstruction[items].mean()),
                  'training_curvature': {}}
        for level in range(3):
            values = curvature[items, level]
            finite = np.isfinite(values)
            result['training_curvature']['L%d' % (level + 1)] = {
                'covered_validation_rows': int(finite.sum()),
                'mean_item_training_curvature_mae': float(values[finite].mean()) if finite.any() else None}
        return result

    paired_path = output / 'PAIRED_ROWS.npz'
    np.savez(paired_path, global_row=np.arange(count), user=frame['user'].to_numpy(dtype=np.int64),
        target=targets, target_rank=ranks, best_prefix=prefix, target_in_seen_history=seen_target,
        training_out_degree=out_degree[targets], training_reconstruction_l2=reconstruction[targets],
        training_curvature_mae=curvature[targets])
    per_item_path = output / 'ITEM_ERRORS.npz'
    np.savez(per_item_path, item=np.arange(len(sids)),
        validation_rows=np.bincount(targets, minlength=len(sids)),
        hits10=np.bincount(targets[ranks < 10], minlength=len(sids)))
    report = {'schema': 'genrec.issue256.validation_diagnosis/1', 'status': 'COMPLETE',
        'identity': identity, 'source_validation_receipt': record(receipt_path),
        'mechanism_report': record(mechanism_path), 'metrics': full,
        'miss_prefix_partition': miss_prefix,
        'target_seen_history': {'seen': metrics(seen_target), 'not_seen': metrics(~seen_target)},
        'training_covariates': {'hit10': covariates(ranks < 10), 'miss10': covariates(misses)},
        'paired_rows': record(paired_path), 'item_errors': record(per_item_path),
        'evaluator': record(__file__), 'test_access': False, 'used_as_training_data': False,
        'limitations': 'Descriptive validation pairing only. Training-graph geometry covariates are in-sample, not held-out curvature or causal error explanations. Missing lawful triples remain missing.'}
    report_path = output / 'VALIDATION_DIAGNOSIS.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'metrics': full, 'miss_prefix_partition': miss_prefix,
                      'target_seen_history': report['target_seen_history'], 'report': record(report_path)}, indent=2), flush=True)
    return 0


def main():
    return diagnose(EVIDENCE)


if __name__ == '__main__':
    raise SystemExit(main())
