"""Audit the one exact R01 final checkpoint; never rewrite SID or train a model."""
from __future__ import annotations

import json
from pathlib import Path
import time

from tools.issue256.r02.identity import ROOT, ARM, RUN_ID, RUN, SOURCE, EVIDENCE, PYTHON
from tools.research.stage2 import launcher


def audit():
    import numpy as np
    import torch
    from tools.issue252.archive_audit import create_binding, sha256
    from tools.issue252.final_evaluation import load_sid, validate_final_checkpoint
    from tools.issue252.layer_metrics import record_layers
    from tools.issue252.run_workspace import load_run
    from tools.issue253.performance import summarize_window
    from tools.research.stage2 import export_check, sid_quality

    started = time.monotonic()
    run = load_run(ROOT, RUN_ID, ARM)
    tree, result = Path(run['source']), Path(run['result'])
    checkpoint, sid = result / 'rqvae_best.pth', result / 'item_sids.json'
    payload = validate_final_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
    if payload['run_id'] != RUN_ID or Path(payload['source_tree']).resolve() != tree.resolve():
        raise ValueError('final checkpoint crosses run/source generations')
    metrics = result / 'logs/training_metrics.jsonl'
    stream = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    train = [row for row in stream if row.get('event') == 'train']
    if [row['epoch'] for row in train] != list(range(1, 3001)):
        raise ValueError('the formal run must contain exactly all3000 epochs')
    if len([row for row in stream if row.get('event') == 'train_end']) != 1:
        raise ValueError('the formal run requires one actual train_end')
    calibrations = [row for row in stream if row.get('event') == 'curvature_calibration']
    if len(calibrations) != 1:
        raise ValueError('the initial coefficient must have been calibrated exactly once')
    calibration = calibrations[0]
    if (calibration['calibration_world_size'] != 4
            or calibration['calibration_batch_items'] != 1024
            or len(calibration['module_budgets']) != 17
            or max(g['weighted_ratio'] for g in calibration['module_budgets'].values()) > 1 + 2e-6
            or float(payload['state_dict']['curvature_weight']) != calibration['lambda']):
        raise ValueError('stored coefficient differs from the actual frozen global module budget')
    trainer = export_check._load_trainer(tree)
    trainer.set_seed(trainer.SEED)
    exported_at = time.monotonic()
    manifest = export_check.check(tree, checkpoint, sid, rewrite=False)
    (EVIDENCE / 'REEXPORT_ATTEMPT.json').write_text(json.dumps({
        'run_id': RUN_ID, 'manifest': manifest, 'sid_rewrite_allowed': False,
        'float32_matmul_precision': torch.get_float32_matmul_precision()}, indent=2) + '\n')
    if manifest['differing_items'] != 0 or manifest['rewritten_from_checkpoint']:
        raise ValueError('the actual final checkpoint does not exactly reproduce every SID')
    rows = load_sid(sid, 24587, (256, 256, 256))
    if not np.array_equal(trainer._extend_collisions(rows[:, :3], [256] * 3), rows):
        raise ValueError('canonical fourth collision suffix differs from the actual export')
    export_path = result / 'export_manifest.json'
    export_check.write_manifest(manifest, export_path)
    export_seconds = time.monotonic() - exported_at
    binding_path = result / 'archive_binding.json'
    create_binding(tree, checkpoint, sid, export_path, output=binding_path)
    layered = record_layers(sid, repo=ROOT, arm=ARM)
    quality = sid_quality.evaluate(sid, checkpoint_path=checkpoint)
    from tools.issue256.r02.final_gradients import probe
    post_training = probe(tree, payload)
    window = [row for row in stream if row.get('event') == 'epoch_boundary' and 31 <= row['epoch'] <= 60]
    timing = summarize_window(window)
    last = train[-1]
    summary = {'run_id': RUN_ID, 'source_tree': str(tree), 'epochs_recorded': len(train),
               'last_epoch': last['epoch'], 'inclusive_window': timing,
               'curvature_calibration': calibration,
               'level_scale_calibration': [row for row in stream if row.get('event') == 'level_scale_calibration'],
               'last_training_record': last,
               'initialization': [row for row in stream if row.get('event') == 'initialization_complete']}
    summary_path = EVIDENCE / 'TRAINING_SUMMARY.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    paths = {'checkpoint': checkpoint, 'sid': sid, 'export': export_path,
             'binding': binding_path, 'layered_metrics': Path(layered['path']),
             'workspace': RUN / 'workspace.json', 'training_summary': summary_path, 'metrics': metrics,
             'post_training_modules': Path(post_training['path'])}
    report = {'schema': 'genrec.issue256.final_stage2/1', 'run_id': RUN_ID,
              'arm': ARM, 'source_tree': str(tree), 'epoch': payload['epoch'],
              'artifacts': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()},
              'same_checkpoint_all_items_exact': True, 'collision_suffix_exact': True,
              'quality': quality, 'reexport_wall_seconds': export_seconds,
              'audit_wall_seconds': time.monotonic() - started,
              'evidence_timing': 'post-formal audit; never a substitute for pre-dispatch proof',
              'evaluator': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)}}
    (EVIDENCE / 'FINAL_STAGE2.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0


def main():
    if launcher.is_child_rank():
        import os
        return audit() if int(os.environ['RANK']) == 0 else 0
    return launcher.launch(SOURCE / 'final_audit_entry.py', EVIDENCE / 'final_audit',
                           python=PYTHON, log_name='final_audit.log')


if __name__ == '__main__':
    raise SystemExit(main())
