"""Issue #287: freeze the candidate (formula, thresholds, identity, applicability domain).

Usage: python3.10 results/research/issue287/freeze_candidate287.py <run_dir> <round_tag>
The file records everything a later, separately authorized Stage3 validation would need: the
mechanism's frozen configuration, the frozen sample set, the frozen margin, the checkpoint and SID
identities and the measured applicability domain (P/N/C).  No label is read here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

P = Path('results/research/issue287')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    run = Path(sys.argv[1]); tag = sys.argv[2]
    result = run / 'result'
    manifest = json.load(open(result / 'export_manifest.json'))
    metrics = [json.loads(line) for line in open(result / 'logs' / 'training_metrics.jsonl')]
    calibration = next(m for m in metrics if m.get('event') == 'pair_calibration')
    probe = next(m for m in metrics if m.get('event') == 'l1_probe')
    train_end = next(m for m in metrics if m.get('event') == 'train_end')
    train_tree = run / 'src'
    tree_files = sorted(p for p in train_tree.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    out = {
        'schema': 'genrec.issue287.frozen_candidate/1',
        'round': tag,
        'arm': manifest.get('tree', '').split('/')[-2] if manifest.get('tree') else None,
        'run_id': run.name,
        'source_tree': str(train_tree),
        'source_tree_files': {str(p.relative_to(train_tree)): sha(p) for p in tree_files},
        'checkpoint': manifest['checkpoint'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
        'sid_path': manifest['sid'], 'sid_sha256': manifest['sid_sha256'],
        'tokens_sha256': manifest['tokens_sha256'], 'export_manifest': manifest,
        'mechanism': {
            'supervised_level': 'L1 (the selected L1 codeword; the forward value is the hard selection)',
            'frozen_pair_index': 'results/research/issue274/pairs_index.npz',
            'frozen_margin': calibration['margin'],
            'gradient_budget': calibration['budget'],
            'lambda_calibrated': calibration['lambda'],
            'groups_per_step': calibration['groups_per_step'],
            'ramp_steps': calibration['ramp_steps'],
            'seed': 42, 'global_batch': 4096, 'epochs': 3000, 'checkpoint_selection': 'final_epoch',
            'mechanism_boundary': 'the curvature enters only the offline selection; the loss is a '
                                  'distance ranking only',
        },
        'measured_path': {'pair_calibration': calibration, 'l1_probe': probe},
        'training': {'total_wall_time_s': train_end.get('total_wall_time_s'),
                     'final_collision_train_forward': train_end.get('final_collision'),
                     'checkpoint_selection': train_end.get('checkpoint_selection')},
        'applicability_domain': 'the measured P/N/C of this exact export; a Stage3 validation must '
                                 'stay inside it',
        'not_a_recall_claim': 'no Stage3 run was started; no Recall is claimed or predicted here',
    }
    (P / f'FROZEN_CANDIDATE_{tag.upper()}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'round': tag, 'sid': out['sid_sha256'], 'lambda': calibration['lambda'],
                      'margin': calibration['margin'],
                      'wall_time_s': out['training']['total_wall_time_s']}, indent=1))


if __name__ == '__main__':
    main()
