"""Paired analysis of the mixed-curvature and per-role geometry contrasts.

Every number is read from the persisted Stage3 final records; nothing is trained,
replayed or selected here. The fixed test has been read by earlier rounds, so the
result is an exploratory comparison with that bias stated, not an unbiased
generalisation estimate.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

ROOT = Path('/fs04/ar57/wenyu/GeneRec')
STAGE3 = ROOT / 'results/stage3_T5Train'
OUT = ROOT / 'results/research/issue256/analysis'


def finals():
    for path in sorted(STAGE3.glob('*/logs/*/*/test_final.json')):
        arm = path.relative_to(STAGE3).parts[0]
        payload = json.loads(path.read_text())
        if payload.get('test_recall@10') is None:
            continue
        yield arm, path.parent.name, payload


def main():
    rows = {}
    for arm, run, payload in finals():
        rows.setdefault(arm, []).append((run, payload))
    measured = {}
    for arm, entries in rows.items():
        values = sorted(payload['test_recall@10'] for _, payload in entries)
        measured[arm] = {'values': values, 'median': statistics.median(values),
                         'runs': [run for run, _ in entries]}
    repeats = {arm: {'values': [payload['test_recall@10'] for _, payload in entries],
                     'spread': max(payload['test_recall@10'] for _, payload in entries)
                     - min(payload['test_recall@10'] for _, payload in entries),
                     'same_sid': len({payload.get('sid_sha256') for _, payload in entries}) == 1}
                for arm, entries in rows.items() if len(entries) > 1}
    spreads = sorted(entry['spread'] for entry in repeats.values())
    contrasts = []

    def contrast(name, left, right, rationale):
        if left not in measured or right not in measured:
            return
        a, b = measured[left]['median'], measured[right]['median']
        contrasts.append({'name': name, 'left': left, 'right': right,
                          'left_values': measured[left]['values'],
                          'right_values': measured[right]['values'],
                          'delta': a - b, 'rationale': rationale,
                          'exceeds_max_repeat_spread': (a - b) > (spreads[-1] if spreads else 0.0),
                          'exceeds_median_repeat_spread': (a - b) > (statistics.median(spreads) if spreads else 0.0)})

    contrast('mixed curvature vs pure Euclidean (same capacity)',
             'TIGER_mixed_rq_F0', 'TIGER_mixed_rq_E',
             'F0 assigns codes in the E16 x H16 mixed space; E is its pure E16 x E16 twin with the same capacity, seed and protocol.')
    contrast('per-role geometry vs shared global geometry',
             'TIGER_role_geom_rq', 'TIGER_role_geom_global_rq',
             'Same mechanism with one global geometric bias instead of a per-role bias.')
    contrast('per-role geometry vs shuffled role assignment',
             'TIGER_role_geom_rq', 'TIGER_role_geom_shuffled_rq',
             'Same mechanism with the role-to-geometry correspondence permuted, i.e. the geometry no longer follows the graphlet role.')
    contrast('local per-prefix curvature weight vs global weight',
             'TIGER_mixed_rq_L', 'TIGER_mixed_rq_G',
             'L sets the curvature weight per prefix; G sets one global weight.')
    contrast('periodic budget refresh vs single refresh (this lineage)',
             'TIGER_periodic_budget_prefix_rq', 'TIGER_module_budget_prefix_rq',
             'R02 refreshes the maximal global module budget every 50 native epochs; R01 refreshes it once.')
    contrast('periodic budget refresh vs best measured arm',
             'TIGER_periodic_budget_prefix_rq', 'TIGER_struct_mixed_rq',
             'Against the highest measured final Recall in this worktree.')

    report = {'schema': 'genrec.issue256.geometry_contrasts/1',
              'bias_disclosure': 'Exploratory paired comparison of already-read final test values; no selection, training or model replay.',
              'test_population': 57439,
              'measured_arms': {arm: entry['values'] for arm, entry in sorted(measured.items())},
              'repeat_runs': repeats,
              'noise_floor': {'max_repeat_spread': spreads[-1] if spreads else None,
                              'median_repeat_spread': statistics.median(spreads) if spreads else None,
                              'min_repeat_spread': spreads[0] if spreads else None,
                              'arms_with_repeats': len(repeats)},
              'contrasts': contrasts}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'geometry_contrasts.json').write_text(json.dumps(report, indent=2) + '\n')
    print('repeat noise: max %.6f median %.6f min %.6f over %d repeated arms'
          % (report['noise_floor']['max_repeat_spread'], report['noise_floor']['median_repeat_spread'],
             report['noise_floor']['min_repeat_spread'], report['noise_floor']['arms_with_repeats']))
    for row in contrasts:
        print('%-58s %+0.6f  (left %s, right %s) >max=%s >median=%s'
              % (row['name'], row['delta'], row['left'], row['right'],
                 row['exceeds_max_repeat_spread'], row['exceeds_median_repeat_spread']))
    print('written', OUT / 'geometry_contrasts.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
