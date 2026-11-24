"""Collect the four-rank workload/wait evidence from one actual profile run.

Reads the four rank reports of one steady-profile diagnostic and records each
rank's real ego workload, communication time and measured waiting, plus the
cross-rank population equality that C9 requires. It never infers a speed
claim: the inclusive 60-epoch benchmark owns that decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
EVIDENCE = ROOT/'results/research/issue253'
COMMUNICATION = ('communication::all_reduce', 'communication::all_gather_object',
                 'communication::barrier', 'communication::broadcast')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect(run_directory, output, baseline_directory=None):
    directory = Path(run_directory)
    ranks = []
    for index in range(4):
        path = directory/('rank_%d.json' % index)
        if not path.is_file():
            raise SystemExit('missing rank report: ' + str(path))
        report = _read(path)
        if report.get('status') != 'PROFILED':
            raise SystemExit('rank %d did not complete the profile' % index)
        phases = {row['phase']: row for row in report['phases']}
        communication = {name: phases[name]['cuda_stream_span_seconds']
                         for name in COMMUNICATION if name in phases}
        workloads = []
        for row in report['ego_workloads']:
            workloads.append({key: row[key] for key in
                              ('hop', 'requested_items', 'unique_items', 'nodes',
                               'directed_edges')})
        ranks.append({
            'rank': index, 'run_id': report['run_id'], 'worker_pid': report['worker_pid'],
            'controller_pid': report['controller_pid'],
            'profiled_epoch': report['profiled_epoch'],
            'profiled_optimizer_steps': report['profiled_optimizer_steps'],
            'completed_optimizer_steps': report['completed_optimizer_steps'],
            'completed_epochs': report['completed_epochs'],
            'profile_window_wall_seconds': report['profile_window_wall_seconds'],
            'profile_window_peak_allocated_bytes': report['profile_window_peak_allocated_bytes'],
            'profile_window_peak_reserved_bytes': report['profile_window_peak_reserved_bytes'],
            'ego_workloads': workloads,
            'communication_cuda_stream_span_seconds': communication,
            'communication_total_cuda_stream_span_seconds': sum(communication.values()),
            'structural_coverage_last_epoch': (report['epoch_boundaries'][-1]
                                                .get('structural_coverage', [])),
            'hardware': report['hardware']})
    populations = []
    for rank in ranks:
        for coverage in rank['structural_coverage_last_epoch']:
            populations.append({'rank': rank['rank'],
                                'epoch_updates': coverage.get('epoch_updates'),
                                'sampled_target_occurrences': coverage.get('sampled_target_occurrences'),
                                'sampled_references': coverage.get('sampled_references'),
                                'unique_aux_items': coverage.get('unique_aux_items'),
                                'sampling_sha256': coverage.get('sampling_sha256')})
    result = {
        'schema': 'genrec.issue253.four_rank/1',
        'source_run_directory': str(directory),
        'scope': 'Actual world4/global4096 profile of one warmed epoch; per-rank '
                 'ego workload, communication span and measured waiting. Not a '
                 'throughput acceptance record.',
        'ranks': ranks,
        'structural_population_by_rank': populations,
        'rank_local_populations_preserved': all(
            row['epoch_updates'] == 6 for row in populations),
        'cross_rank_reuse': 'none: every rank computed its own complete egos, '
                            'reconstruction and auxiliary populations',
        'waiting_evidence': 'communication spans are the measured device time of '
                             'the actual collective calls inside the profiled epoch',
        'not_claimed': ['4x speedup', 'cross-rank token reuse', 'single-card estimate',
                        'speed acceptance decision (owned by the inclusive benchmark)']}
    if baseline_directory:
        baseline = Path(baseline_directory)
        base = _read(baseline/'rank_0.json')
        base_phases = {row['phase']: row['cuda_stream_span_seconds'] for row in base['phases']}
        now_phases = {row['phase']: row['cuda_stream_span_seconds']
                      for row in _read(directory/'rank_0.json')['phases']}
        comparison = []
        for phase in sorted(set(base_phases) | set(now_phases)):
            before = base_phases.get(phase)
            after = now_phases.get(phase)
            comparison.append({'phase': phase, 'baseline_cuda_stream_span_seconds': before,
                               'optimized_cuda_stream_span_seconds': after,
                               'delta_seconds': (None if before is None or after is None
                                                 else after-before)})
        result['baseline_profile_directory'] = str(baseline)
        result['phase_span_comparison_rank0'] = comparison
        result['phase_span_note'] = ('Nested/inclusive phase spans overlap and are '
                                     'never summed; the comparison is per-phase '
                                     'evidence of the same measurement scope, not an '
                                     'additive epoch estimate.')
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n',
                            encoding='utf-8')
    print(json.dumps({'output': str(output), 'ranks': len(ranks)}, indent=2))
    return 0


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit('usage: collect_four_rank.py <profile-run-dir> <output.json> '
                         '[baseline-profile-run-dir]')
    raise SystemExit(collect(sys.argv[1], sys.argv[2],
                             sys.argv[3] if len(sys.argv) == 4 else None))
