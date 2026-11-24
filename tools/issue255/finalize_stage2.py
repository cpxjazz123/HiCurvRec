"""Audit the one final Stage2 generation; never train or rewrite its SID."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.stage2 import launcher

from tools.issue255.identity import ARM, RUN_ID, RUN, EVIDENCE as OUTPUT
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


def audit():
    import numpy as np
    import torch
    from tools.issue252.archive_audit import create_binding, sha256
    from tools.issue252.final_evaluation import load_sid, validate_final_checkpoint
    from tools.issue252.layer_metrics import record_layers
    from tools.issue252.run_workspace import load_run
    from tools.research.stage2 import export_check, sid_quality
    from tools.issue255 import training_summary

    started = time.monotonic()
    run = load_run(ROOT, RUN_ID, ARM)
    tree, result = Path(run['source']), Path(run['result'])
    checkpoint, sid = result / 'rqvae_best.pth', result / 'item_sids.json'
    payload = validate_final_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
    if payload['run_id'] != RUN_ID:
        raise ValueError('final checkpoint belongs to another run')
    records = [json.loads(line) for line in (result / 'logs/training_metrics.jsonl').read_text().splitlines() if line]
    if [r['epoch'] for r in records if r.get('event') == 'train'] != list(range(1, 3001)):
        raise ValueError('the formal training stream does not contain all 3000 epochs')
    if not any(r.get('event') == 'train_end' for r in records):
        raise ValueError('the formal run has no train_end')
    # set_seed also restores the frozen deterministic/TF32 numerical policy.
    # A cold checker otherwise uses a different GEMM precision from training.
    trainer = export_check._load_trainer(tree)
    trainer.set_seed(trainer.SEED)
    export_started = time.monotonic()
    manifest = export_check.check(tree, checkpoint, sid, rewrite=False)
    (OUTPUT / 'REEXPORT_ATTEMPT.json').write_text(
        json.dumps({'run_id': RUN_ID, 'manifest': manifest,
                    'float32_matmul_precision': torch.get_float32_matmul_precision(),
                    'sid_rewrite_allowed': False}, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    if manifest['differing_items'] != 0 or manifest['rewritten_from_checkpoint']:
        raise ValueError('final checkpoint re-export is not itemwise exact')
    rows = load_sid(sid, 24587, (256, 256, 256))
    trainer = export_check._load_trainer(tree)
    if not np.array_equal(trainer._extend_collisions(rows[:, :3], [256, 256, 256]), rows):
        raise ValueError('fourth collision suffix does not match the canonical export')
    export_path = result / 'export_manifest.json'
    export_check.write_manifest(manifest, export_path)
    export_seconds = time.monotonic() - export_started
    binding_path = result / 'archive_binding.json'
    create_binding(tree, checkpoint, sid, export_path, output=binding_path)
    layered = record_layers(sid, repo=ROOT, arm=ARM)
    quality = sid_quality.evaluate(sid, checkpoint_path=checkpoint)
    if training_summary.main() != 0:
        raise ValueError('training summary failed')
    artifacts = {}
    for name, path in {'checkpoint': checkpoint, 'sid': sid, 'export': export_path,
                       'binding': binding_path, 'layered_metrics': Path(layered['path']),
                       'workspace': RUN / 'workspace.json',
                       'training_summary': OUTPUT / 'TRAINING_SUMMARY.json',
                       'metrics': result / 'logs/training_metrics.jsonl'}.items():
        artifacts[name] = {'path': str(path), 'sha256': sha256(path)}
    report = {'schema': 'genrec.issue255.final_stage2/1', 'run_id': RUN_ID,
              'arm': ARM, 'source_tree': str(tree), 'epoch': payload['epoch'],
              'artifacts': artifacts, 'same_checkpoint_all_items_exact': True,
              'collision_suffix_exact': True, 'quality': quality,
              'reexport_wall_seconds': export_seconds,
              'audit_wall_seconds': time.monotonic() - started,
              'evidence_timing': 'post-formal audit; not pre-dispatch admission',
              'evaluator': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)}}
    (OUTPUT / 'FINAL_STAGE2.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0


def main():
    if launcher.is_child_rank():
        import os
        return audit() if int(os.environ['RANK']) == 0 else 0
    return launcher.launch(ROOT / 'stage2_RQ-VAE' / ARM / 'final_audit_entry.py',
                           OUTPUT / 'final_audit', python=PYTHON, log_name='final_audit.log')


if __name__ == '__main__':
    raise SystemExit(main())
