"""Extract parent train/validation evidence; never consume test error item lists."""
from __future__ import annotations

import json
from pathlib import Path

from tools.issue252.archive_audit import sha256

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
OUTPUT = ROOT / 'results/research/issue256'
PARENT = ROOT / 'results/research/issue255'


def load(path):
    return json.loads(Path(path).read_text())


def validation_only(value):
    """Keep existing complete validation banks, excluding all structure-test banks."""
    if isinstance(value, dict):
        return {key: validation_only(item) for key, item in value.items()
                if 'test' not in key.lower()}
    if isinstance(value, list):
        return [validation_only(item) for item in value
                if not isinstance(item, dict)
                or not any('test' in str(item.get(key, '')).lower()
                           for key in ('split', 'name', 'bank', 'path'))]
    return value


def main():
    path = PARENT / 'evaluation/FINAL_EVALUATION.json'
    full = load(path)
    generations = {}
    for name, data in full['generations'].items():
        generations[name] = {key: validation_only(data[key]) for key in
            ('layers', 'gates', 'reconstruction', 'sid_structure',
             'strict_heldout', 'visible_directed_l1_sharing', 'prefix_identity')}
    pressure = load(OUTPUT / 'PARENT_GRADIENT_PRESSURE.json')
    metrics_path = ROOT / ('results/stage2_RQ-VAE/_runs/issue255_formal_r1/'
                           'TIGER_prefix_curvature_gate_rq/result/logs/training_metrics.jsonl')
    stream = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    epochs = [row for row in stream if row.get('event') == 'train']
    # Preserve schema and exact records for the shared reducer instead of
    # guessing a time field from another generation's log convention.
    report = {
        'schema': 'genrec.issue256.parent_diagnostics/1',
        'source': {'path': str(path), 'sha256': sha256(path)},
        'generations': generations,
        'curvature_coverage': full['coverage'],
        'units': full['units'], 'metric': full['comparison'],
        'formal_parent_metrics': {'path': str(metrics_path), 'sha256': sha256(metrics_path),
            'events': sorted({row.get('event', '') for row in stream}),
            'initialization': [row for row in stream if row.get('event') in
                               ('curvature_calibration', 'level_scale_calibration', 'initialization_complete')],
            'first_train_record': epochs[0] if epochs else None,
            'last_train_record': epochs[-1] if epochs else None},
        'actual_pretraining_module_pressure': pressure,
        'recommendation_validation_metrics': None,
        'recommendation_validation_missing_reason': 'No checkpoint-paired external validation inference was saved in the parent evidence.',
        'parent_recommendation_test': {'issue252': {'r5': 0.03278260415397204,
            'r10': 0.0492870697609638, 'ndcg5': 0.021616071827379253,
            'ndcg10': 0.02692583040624563}, 'issue253': {'r5': 0.0314246417939031,
            'r10': 0.046431866850049616, 'ndcg5': 0.02080348571822262,
            'ndcg10': 0.02563402906712319}, 'issue255': None},
        'test_use': 'Previously published historical aggregate endpoints only; no test-error rows, derived features or structure-test values selected for this diagnostic.',
        'qualification': 'Gradient pressure comes from the bound actual one-epoch diagnostic, not the formal final weights. It does not prove the training-collapse cause.',
        'command': 'python3 -m tools.issue256.parent_diagnostics', 'exit_code': 0,
    }
    path = OUTPUT / 'rounds/r01/PARENT_DIAGNOSTICS.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('saved training/validation parent evidence', path)
    print('actual sampled maximum weighted-curvature/original norm:', pressure['max_weighted_ratio'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
