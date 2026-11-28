"""Issue #285: audit of #284/#277/#283 artifacts, labels, fold-local strata and same-source checks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue285')
import common285 as W

P = Path('results/research/issue285')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    out = {'schema': 'genrec.issue285.audit/1'}
    files = ['REPORT.md', 'VERDICT.md', 'MODEL_284.json', 'CHAIN_ANALYSIS.json', 'CHAIN_TABLE.json',
             'MATCHED_284.json', 'PROSPECTIVE_284.json', 'AUDIT_284.json']
    out['p284_files'] = {f: {'sha256': sha(f'results/research/issue284/{f}'), 'exists': True} for f in files}
    out['p284_commit'] = 'b106d01e61a73d903aad7e58f77e63a42afaa745'
    out['p277_predictor'] = {'sha256': sha('results/research/issue277/PREDICTOR.json')}
    out['p283_join_summary'] = {'sha256': sha('results/research/issue283/JOIN_SUMMARY.json')}
    d = W.load_table()
    y = d['y']; fold = d['fold']
    out['labels'] = {'n': len(y), 'n_strong': int(d['strong'].sum()), 'n_weak': int((~d['strong']).sum()),
                     'n_high': int((y > 0.06).sum()), 'n_high_strong': int(((y > 0.06) & d['strong']).sum()),
                     'n_units': len(set(d['unit'])),
                     'fold_sizes': {str(f): int((fold == f).sum()) for f in range(5)}}
    # ER = P/N identity
    ratio = d['P'] / d['N']
    out['identity_ER'] = {'max_abs_ER_minus_P_over_N': float(np.max(np.abs(d['ER'] - ratio))),
                          'median_ER': float(np.median(d['ER']))}
    # correlations and same-source check
    feats = {'N': d['N'], 'C': d['C'], 'P': d['P'], 'ALL': d['ALL'], 'y': y}
    corr = {}
    names = list(feats)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr[f'{a}-{b}'] = round(float(stats.spearmanr(feats[a], feats[b]).correlation), 4)
    out['spearman'] = corr
    M = np.corrcoef(np.vstack([feats[a] for a in ('N', 'C', 'P')]))
    ev = np.linalg.eigvalsh(M)
    out['same_source'] = {'eigvals_NCP': [round(float(v), 4) for v in ev],
                          'pc1_share': round(float(ev[-1] / ev.sum()), 4)}
    # fold-local strata
    g = W.gates(d)
    cells = W.cell_labels(g['level_N'], g['level_C'])
    out['gates'] = {'edges_N': {k: [round(x, 6) for x in v] for k, v in g['edges_N'].items()},
                    'edges_C': {k: [round(x, 6) for x in v] for k, v in g['edges_C'].items()},
                    'counts': {c: int((cells == c).sum()) for c in ('LL', 'LN', 'LC', 'HH', 'HN', 'HC', 'mid')}}
    # overlap of the two low sets + Jaccard
    ln, lc = g['level_N'] == -1, g['level_C'] == -1
    out['overlap'] = {'lowN': int(ln.sum()), 'lowC': int(lc.sum()), 'intersection': int((ln & lc).sum()),
                      'jaccard': round(float((ln & lc).sum() / max((ln | lc).sum(), 1)), 4),
                      'lowN_given_lowC': round(float((ln & lc).sum() / max(lc.sum(), 1)), 4),
                      'lowC_given_lowN': round(float((ln & lc).sum() / max(ln.sum(), 1)), 4)}
    # cell composition
    comp = {}
    for c in ('LL', 'LN', 'LC', 'HH', 'other'):
        m = cells == c
        if m.sum():
            comp[c] = {'n': int(m.sum()), 'strong': int(d['strong'][m].sum()),
                       'high': int((y[m] > 0.06).sum()),
                       'recall_mean': round(float(y[m].mean()), 5),
                       'recall_min': round(float(y[m].min()), 5), 'recall_max': round(float(y[m].max()), 5)}
    out['cells'] = comp
    # global-tercile cross-check (the #284 bands)
    qn, qc, gln, glc = W.gates_global(d)
    out['global_crosscheck'] = {
        'edges_N': [round(float(x), 6) for x in qn], 'edges_C': [round(float(x), 6) for x in qc],
        'lowN_n': int((gln == -1).sum()), 'lowC_n': int((glc == -1).sum()),
        'rho_lowN': round(float(stats.spearmanr(d['ALL'][gln == -1], y[gln == -1]).correlation), 4),
        'rho_lowC': round(float(stats.spearmanr(d['ALL'][glc == -1], y[glc == -1]).correlation), 4)}
    # the 2 time-postordered labels as a non-blind sensitivity (no fold; global bins only)
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    all_rows = {}
    for gg in table['groups']:
        for r in gg['rows']:
            all_rows[r['sid_sha256']] = r
    sens = []
    for sid_pref, recall in (('9f6bf673', 0.0479465171747419), ('61f898498c19', 0.04846881039015303)):
        row = next((r for s, r in all_rows.items() if s.startswith(sid_pref)), None)
        if row and all(isinstance((row.get('features') or {}).get(k), (int, float))
                       for k in ('l1_neg_hit', 'l3_collision', 'l1_pos_hit')):
            f = row['features']
            sens.append({'sid_prefix': sid_pref, 'recall': recall, 'N': f['l1_neg_hit'],
                         'C': f['l3_collision'], 'P': f['l1_pos_hit'],
                         'global_lowN': bool(f['l1_neg_hit'] <= qn[0]),
                         'global_lowC': bool(f['l3_collision'] <= qc[0])})
    out['time_postordered_sensitivity'] = sens
    (P / 'AUDIT_285.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('labels', 'overlap', 'same_source', 'cells', 'global_crosscheck',
                                          'time_postordered_sensitivity')}, indent=1)[:3500])


if __name__ == '__main__':
    main()
