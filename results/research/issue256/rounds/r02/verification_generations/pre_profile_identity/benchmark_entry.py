"""Labelled 60-complete-epoch diagnostic; frozen 3000-epoch recipe is untouched."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import train_rqvae as training
from tools.issue253.performance import summarize_window

ROOT = training.REPO_ROOT
KIND = 'benchmark'
DIAGNOSTIC_EPOCHS = 60


def source_binding(kind):
    """This tree's own source binding.

    The shared #252 admission policy only knows the retired arm identities, so
    the benchmark records its own hash inventory instead of pretending to be
    another candidate.
    """
    tree = Path(__file__).resolve().parent
    files = {}
    for path in sorted(tree.rglob('*')):
        if (not path.is_file() or path.suffix not in ('.py', '.json')
                or '__pycache__' in path.parts or 'logs' in path.parts):
            continue
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT / 'tools/issue256/r02').rglob('*.py')):
        if '__pycache__' not in path.parts:
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    from tools.issue252.archive_audit import imported_repo_sources
    files.update(imported_repo_sources(tree, ROOT))
    for relative in ('assets/manifest.json', 'tools/issue252/archive_audit.py',
                     'tools/issue256/r02/launch_diagnostics.py',
                     'tools/issue256/r02/layer_validation_entry.py',
                     'tools/research/stage2/launcher.py',
                     'tools/issue253/performance.py',
                     'results/research/issue256/rounds/r02/CONTRACT.md',
                     'results/research/issue256/rounds/r02/MUTATIONS.md',
                     'results/research/issue256/rounds/r02/reference.py',
                     'results/research/issue256/ACCEPTANCE.json',
                     'results/research/issue256/EVALUATION.md'):
        path = ROOT / relative
        if path.is_file():
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'kind': kind, 'tree': str(tree.relative_to(ROOT)), 'files': files}


def configure(kind):
    if int(os.environ.get('WORLD_SIZE', '0')) != 4 or int(os.environ.get('MASTER_PORT', '0')) != 50202:
        raise RuntimeError('diagnostics require the standard four-rank/50202 launcher')
    if (training.EPOCHS, training.BATCH_SIZE_PER_RANK, training.SEED) != (3000, 1024, 42):
        raise RuntimeError('frozen training recipe changed')
    controller = os.getppid()
    start_tick = (Path('/proc') / str(controller) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    run_id = 'issue256_r02_' + kind + '_' + str(controller) + '_' + start_tick
    output = ROOT / 'results/research/issue256/rounds/r02' / kind / run_id
    output.mkdir(parents=True, exist_ok=True)
    training.RUN_ID = run_id
    training.RESULT_DIR = output
    training.LOG_DIR = output / 'logs'
    training.METRICS_PATH = training.LOG_DIR / 'training_metrics.jsonl'
    training.CHECKPOINT = output / 'forbidden_formal_checkpoint.pth'
    training.OUTPUT_SID = output / 'forbidden_formal_sids.npy'
    training.OUTPUT_JSON = output / 'forbidden_formal_item_sids.json'
    return run_id, output


def main():
    run_id, output = configure(KIND)
    rank = int(os.environ['RANK'])
    binding = source_binding(KIND)
    report = {'run_id': run_id, 'rank': rank, 'worker_pid': os.getpid(),
              'controller_pid': os.getppid(), 'formal_training': False,
              'diagnostic_epochs': DIAGNOSTIC_EPOCHS, 'source_hashes': binding,
              'recipe': {'epochs': training.EPOCHS, 'per_rank_batch': 1024,
                         'global_batch': 4096, 'world_size': 4, 'seed': 42,
                         'config_sha256': training.CONFIG.sha256}, 'status': 'running'}
    started = time.monotonic()
    try:
        training.main(diagnostic_epochs=DIAGNOSTIC_EPOCHS)
        if rank == 0:
            records = [json.loads(line) for line in training.METRICS_PATH.read_text().splitlines()]
            from tools.issue256.r02.calibration_receipts import validate
            calibrations = [row for row in records if row['event'] == 'curvature_calibration']
            validate(calibrations, epochs=60, interval=50, steps_per_epoch=6)
            report['actual_calibration_records'] = calibrations
            boundaries = [row for row in records if row['event'] == 'epoch_boundary']
            if [row['epoch'] for row in boundaries] != list(range(1, 61)):
                raise RuntimeError('benchmark did not complete all 60 inclusive epoch boundaries')
            for row in boundaries:
                coverage = row.get('structural_coverage', [])
                if len(coverage) != 4 or any(receipt['epoch_updates'] <= 0
                    or receipt['epoch_target_occurrences'] <= 0
                    or receipt['epoch_reference_draws'] <= 0 for receipt in coverage):
                    raise RuntimeError('missing actual four-rank structural epoch coverage')
            window = [row for row in boundaries if 31 <= row['epoch'] <= 60]
            report['epoch_boundaries'] = boundaries
            report['performance'] = summarize_window(window)
            report['initialization_seconds'] = boundaries[0]['start'] - started
            report['periodic_evaluation_epochs'] = [row['epoch'] for row in records
                if row['event'] == 'train' and row.get('raw_unique') is not None]
            if 50 not in report['periodic_evaluation_epochs']:
                raise RuntimeError('inclusive timing window did not include actual epoch50 evaluation')
            if not report['performance']['passed']:
                raise RuntimeError('inclusive epochs31..60 mean exceeds the authorized 0.8-second complete-mechanism ceiling')
        if any(p.exists() for p in (training.CHECKPOINT, training.OUTPUT_SID, training.OUTPUT_JSON)):
            raise RuntimeError('diagnostic incorrectly emitted formal final artifacts')
        if binding != source_binding(KIND):
            raise RuntimeError('source changed during benchmark')
        report['status'] = 'pass'
    except Exception:
        report['status'] = 'fail'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        (output / ('rank_%d.json' % rank)).write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
