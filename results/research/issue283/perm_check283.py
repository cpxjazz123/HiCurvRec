"""Issue #283: verify the analytic uniform permutation null with empirical permutations.

For 3 SIDs x 10 permutations x 3 levels, per-class observed pair/ALL rates are compared with the
analytic expectation E (CONTRACT §4) within Monte-Carlo error.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue283')
from results.research.issue283 import joint283 as J

SIDS = 3
PERMS = 10
NC = len(J.CLASS_NAMES)


def main() -> None:
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    paths = []
    seen = set()
    for g in table['groups']:
        for r in g['rows']:
            sha = r['sid_sha256']
            if sha in seen:
                continue
            seen.add(sha)
            rec = json.loads(open(r['features']['_source']).read())
            paths.append((sha, rec['sid']['path']))
    import stats283 as S
    S.setup()
    A, M, B, CLS = S.G['A'], S.G['M'], S.G['B'], S.G['CLS']
    support = S.G['support']
    rng = np.random.default_rng(42)
    out = {'schema': 'genrec.issue283.perm_check/1', 'perms': PERMS, 'sids': []}
    for sha, path in paths[:SIDS]:
        codes = J.codes_matrix(path)
        rec = {'sid_sha256': sha, 'levels': {}}
        for k in range(1, 4):
            col = codes[:, k - 1]
            n_g = np.bincount(col, minlength=1024)
            uni = J.null_rates(n_g, J.N)
            emp = np.zeros((PERMS, NC, 4))  # am, mb, ab, all per class per permutation
            for p in range(PERMS):
                perm = col[rng.permutation(J.N)]
                ca, cm, cb = perm[A], perm[M], perm[B]
                pat = J.pattern_of(ca == cm, cm == cb, ca == cb)
                cnt = np.bincount(CLS.astype(np.int64) * 5 + pat,
                                  minlength=NC * 5).reshape(-1, 5).astype(float)
                rate = cnt / np.maximum(support[:, None], 1)
                emp[p, :, 0] = rate[:, 0] + rate[:, 1]
                emp[p, :, 1] = rate[:, 0] + rate[:, 2]
                emp[p, :, 2] = rate[:, 0] + rate[:, 3]
                emp[p, :, 3] = rate[:, 0]
            obs = {}
            for ci, name in enumerate(J.CLASS_NAMES):
                if not support[ci]:
                    continue
                means = emp[:, ci, :].mean(axis=0)
                ses = emp[:, ci, :].std(axis=0, ddof=1) / np.sqrt(PERMS)
                se_all = float(ses[3])
                z = None if se_all < 1e-9 else round(float((means[3] - uni['triple']) / se_all), 3)
                obs[name] = {'analytic_pair': round(uni['pair'], 6),
                             'emp_pair': round(float(means[:3].mean()), 6),
                             'pair_mc_se': round(float(ses[:3].mean()), 6),
                             'analytic_all': round(uni['triple'], 6),
                             'emp_all': round(float(means[3]), 6),
                             'all_mc_se': round(se_all, 8),
                             'all_abs_diff': round(float(means[3] - uni['triple']), 8),
                             'all_mc_z': z}
            rec['levels'][str(k)] = obs
        out['sids'].append(rec)
        print('done', sha[:12], flush=True)
    zs = [abs(v['all_mc_z']) for rec in out['sids'] for lvl in rec['levels'].values()
          for v in lvl.values() if v['all_mc_z'] is not None]
    degenerate = sum(1 for rec in out['sids'] for lvl in rec['levels'].values()
                     for v in lvl.values() if v['all_mc_z'] is None)
    out['summary'] = {'n_tests': len(zs), 'n_degenerate_se_zero': degenerate,
                      'max_abs_z': max(zs) if zs else None,
                      'median_abs_z': float(np.median(zs)) if zs else None,
                      'within_3sd': int(sum(1 for z in zs if z <= 3)),
                      'worst': sorted([{'class': k, 'level': lv, 'z': v['all_mc_z'],
                                        'abs_diff': v['all_abs_diff']}
                                       for rec in out['sids'] for lv, lvl in rec['levels'].items()
                                       for k, v in lvl.items() if v['all_mc_z'] is not None],
                                      key=lambda x: -abs(x['z']))[:5]}
    (J.P / 'PERM_CHECK.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['summary'], indent=1))


if __name__ == '__main__':
    main()
