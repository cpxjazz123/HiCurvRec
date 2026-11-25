"""Recompute R04 headline diagnostics from raw SID/catalogue/per-centre records."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tools.issue256.r04.identity import ROOT, EVIDENCE, RUN_ID
from tools.issue256.r04.extract_catalogues import context, record


def close(a, b, label):
    if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1.e-10, abs_tol=1.e-10):
        raise ValueError('independent Stage2 metric differs: ' + label)


def curvature(cell):
    published = cell['curvature']
    path = Path(published['per_item']['path'])
    if record(path) != published['per_item']:
        raise ValueError('curvature per-centre artifact changed')
    rows = [json.loads(line) for line in path.open()]
    if len(rows) != 24326 or len({r['item'] for r in rows}) != 24326:
        raise ValueError('partial/duplicate curvature centre population')
    count = sum(r['count'] for r in rows)
    if count != 28391948 or published['count'] != count:
        raise ValueError('partial curvature occurrence population')
    macro = math.fsum(r['mean_abs_error'] for r in rows) / len(rows)
    micro = math.fsum(r['count'] * r['mean_abs_error'] for r in rows) / count
    close(macro, published['macro_mae'], 'curvature macro')
    close(micro, published['micro_mae'], 'curvature micro')
    recalls = {}
    for name in ('negative', 'zero', 'positive'):
        n = sum(r['count_' + name] for r in rows)
        if n == 0:
            if published['sign']['recall'][name]['status'] != 'NA':
                raise ValueError('absent curvature sign must be NA')
            recalls[name] = None
            continue
        value = math.fsum(r['count_' + name] * r['recall_' + name]
                          for r in rows if r['count_' + name]) / n
        close(value, published['sign']['recall'][name]['recall'], name + ' recall')
        recalls[name] = value
    balanced = math.fsum(r for r in recalls.values() if r is not None) / sum(r is not None for r in recalls.values())
    close(balanced, published['sign']['balanced_accuracy'], 'balanced sign accuracy')
    return {'count': count, 'centres': len(rows), 'macro_mae': macro, 'micro_mae': micro,
            'balanced_accuracy': balanced, 'sign_recall': recalls}


def main():
    _, paths, _, identity = context()
    report_path = EVIDENCE / 'evaluation/FINAL_EVALUATION.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'COMPLETE' or report['context'] != identity:
        raise ValueError('full mechanism report uses another generation')
    generation = report['generation']
    raw = json.loads(Path(paths['sid']).read_text())
    if set(raw) != {str(i) for i in range(24587)}:
        raise ValueError('partial SID population')
    rows = [raw[str(i)] for i in range(24587)]
    distinct = [len({tuple(row[:level]) for row in rows}) for level in (1, 2, 3)]
    collision = 1 - distinct[2] / len(rows)
    close(collision, generation['sid_structure']['three_code_collision'], 'three-code collision')
    if distinct[1] != generation['sid_structure']['distinct_l2_prefixes']:
        raise ValueError('two-level distinct prefix count differs')
    catalogue_path = Path(generation['catalogue']['path'])
    if record(catalogue_path) != generation['catalogue']:
        raise ValueError('final CUDA catalogue changed')
    with np.load(catalogue_path, allow_pickle=False) as blob:
        reconstruction = blob['reconstruction']
    target = np.load(ROOT / 'stage1_GeneEmbedding/output/sentence_t5.npy', mmap_mode='r')
    squared, l2 = [], []
    for start in range(0, len(target), 1024):
        difference = reconstruction[start:start + 1024].astype(np.float64) - target[start:start + 1024]
        squared.append(float(np.square(difference).sum()))
        l2.append(float(np.linalg.norm(difference, axis=1).sum()))
    mse, mean_l2 = math.fsum(squared) / target.size, math.fsum(l2) / len(target)
    close(mse, generation['reconstruction']['mean_squared_error'], 'reconstruction MSE')
    close(mean_l2, generation['reconstruction']['mean_l2_error'], 'reconstruction L2')
    banks = generation['strict_heldout']['banks']
    rates = {}
    for name, bank in banks.items():
        path = Path(bank['asset']['path'])
        if record(path) != bank['asset'] or not name.startswith('strict_struct_val_'):
            raise ValueError('validation population changed or test bank entered analysis')
        with np.load(path, allow_pickle=False) as blob:
            pairs = blob['pairs']
        hits = sum(rows[int(a)][0] == rows[int(b)][0] for a, b in pairs)
        rates[name] = hits / len(pairs)
        close(rates[name], bank['prefix']['L1']['rate'], name)
    control = rates['strict_struct_val_degree_matched']
    delta = rates['strict_struct_val_direct'] - control
    layers = {name: curvature(cell) for name, cell in generation['layers'].items()}
    if set(layers) != {'L1', 'L2', 'L3'}:
        raise ValueError('missing hard-prefix curvature level')
    if context()[-1] != identity:
        raise ValueError('final production source/input generation changed')
    result = {'schema': 'genrec.issue256.stage2_reconciliation/1', 'status': 'PASS',
        'run_id': RUN_ID, 'identity': identity, 'report': record(report_path),
        'evaluator': record(__file__), 'distinct_prefixes': distinct, 'l3_collision': collision,
        'reconstruction': {'mean_squared_error': mse, 'mean_l2_error': mean_l2},
        'validation_l1_rates': rates, 'validation_direct_minus_control': delta,
        'layers': layers, 'preregistered_predictions': {
            'collision_below_0_4': collision < .4,
            'reconstruction_mse_no_worse_than_parent255': mse <= .00018233678135234408},
        'test_access': False,
        'limitations': 'Training-graph geometry is in-sample. Periodic budget/validation associations do not establish a causal explanation of Recall. Final global-gradient and recommendation forecasts are reconciled separately from their complete raw evidence.'}
    output = EVIDENCE / 'STAGE2_RECONCILIATION.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'collision': collision, 'distinct': distinct,
                      'mse': mse, 'validation_direct_excess': delta,
                      'predictions': result['preregistered_predictions'], 'report': record(output)}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
