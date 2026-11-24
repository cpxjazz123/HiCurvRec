"""Collect the per-module gradient/update evidence from one actual validation run.

Reads the four rank reports of one validation diagnostic and writes
MODULE_CHECKS.json: per-module forward finiteness, actual total-loss gradient
norms, pre/post-clip norms, real optimizer updates, and the original/structural
VJP norms and cosines. Nothing here is inferred from a synthetic loss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
EVIDENCE = ROOT/'results/research/issue253'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect(run_directory, output):
    directory = Path(run_directory)
    modules = {}
    local_losses = {}
    local_first = {}
    for index in range(4):
        path = directory/('rank_%d.json' % index)
        if not path.is_file():
            raise SystemExit('missing rank report: ' + str(path))
        report = _read(path)
        if str(report.get('status')).lower() != 'pass':
            raise SystemExit('rank %d did not pass the validation diagnostic' % index)
        for name, row in report['normal_backward'].items():
            entry = modules.setdefault(name, {'ranks': {}})
            entry['ranks'][str(index)] = {
                'total_loss_gradient': row,
                'post_clip_gradient': report['normal_backward_post_clip'][name],
                'optimizer_update': report['actual_optimizer_update'][name],
                'forward': report['normal_forward'].get(name),
                'original_vjp': report['raw_real_batch_VJP']['original'].get(name),
                'structural_vjp': report['raw_real_batch_VJP']['structural'].get(name),
                'weighted_structural_vjp': report['weighted_structural_VJP'].get(name),
                'loss_gradient_direction': report['loss_gradient_direction'].get(name)}
        if index == 0:
            reference = {
                'run_id': report['run_id'], 'rank': index,
                'first_step_losses': report['first_step_losses'],
                'completed_local_losses': report['completed_local_losses'],
                'actual_structural_hard_q_decode_exact':
                    report['actual_structural_hard_q_decode_exact'],
                'actual_gate_weights': report.get('actual_gate_weights'),
                'completed_parameter_hashes': report['completed_parameter_hashes'],
                'shared_equivalence': report.get('shared_equivalence'),
                'export': report.get('export'),
                'switches': report.get('switches')}
            local_losses[0] = report['completed_local_losses']
            local_first[0] = report['first_step_losses']
        else:
            # Only the DDP-synchronised parameters are cross-rank identical;
            # per-rank losses belong to that rank's own batch.
            if report.get('completed_parameter_hashes') != reference.get('completed_parameter_hashes'):
                raise SystemExit('rank %d disagrees on the completed parameter hashes' % index)
            local_losses[index] = report['completed_local_losses']
            local_first[index] = report['first_step_losses']
    result = {
        'schema': 'genrec.issue253.module_checks/1',
        'scope': 'Actual one-epoch four-rank validation diagnostic: real per-rank '
                 'batch, real original reconstruction loss and real sampled A.1 '
                 'curvature loss. No synthetic loss is used.',
        'source_run_directory': str(directory),
        'reference_rank': reference,
        'per_rank_local_losses': {str(key): value for key, value in local_losses.items()},
        'per_rank_first_step_losses': {str(key): value for key, value in local_first.items()},
        'modules': modules,
        'gradient_reached_every_module': all(
            row['ranks']['0']['total_loss_gradient']['norm'] > 0 for row in modules.values()),
        'optimizer_updated_every_module': all(
            row['ranks']['0']['optimizer_update']['norm'] > 0 for row in modules.values()),
        'not_claimed': ['unbiased full-reference gradient', 'single-total-loss proof',
                        'cross-rank reuse of learned outputs']}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n',
                            encoding='utf-8')
    print(json.dumps({'output': str(output), 'modules': len(modules)}, indent=2))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: collect_module_checks.py <validation-run-dir> <output.json>')
    raise SystemExit(collect(sys.argv[1], sys.argv[2]))
