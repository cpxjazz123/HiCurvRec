"""Issue #284: primary/secondary matched-SID contrasts for chain ALL and its conditional residual."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
import common284 as C

P = Path('results/research/issue284')
TOL = {'l1_pos_hit': ('abs', 0.01), 'l1_hard_er': ('rel', 0.20), 'l2_pos_hit': ('rel', 0.20),
       'l3_collision': ('abs', 0.01), 'l3_refine_bits': ('rel', 0.10),
       'l1_distinct': ('rel', 0.10), 'l1_entropy': ('rel', 0.05)}
MARG = 0.005


def matches(rows, ch, marginal):
    out = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if marginal:
            for k in ('p_am', 'p_mb', 'p_ab'):
                if abs(ch[i][k] - ch[j][k]) > MARG:
                    break
            else:
                pass
            if any(abs(ch[i][k] - ch[j][k]) > MARG for k in ('p_am', 'p_mb', 'p_ab')):
                continue
        ok = True
        for key, (mode, tol) in TOL.items():
            vi = rows[i]['x8'].get(key, ch[i].get(key))
            vj = rows[j]['x8'].get(key, ch[j].get(key))
            if vi is None or vj is None:
                ok = False; break
            if mode == 'abs':
                if abs(vi - vj) > tol:
                    ok = False; break
            else:
                if abs(vi - vj) / max(abs(vi), abs(vj), 1e-12) > tol:
                    ok = False; break
        if ok:
            out.append((i, j))
    return out


def cluster_spearman(pairs, x, y, unit, reps=2000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit))
    idx_of = {u: np.where(np.array(unit) == u)[0] for u in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        keep = set(int(k) for k in pick)
        sel = [(a, b) for a, b in pairs if unit[a] in keep and unit[b] in keep]
        if len(sel) < 10:
            continue
        d = np.array([x[a] - x[b] for a, b in sel]); dy = np.array([y[a] - y[b] for a, b in sel])
        if np.std(d) == 0 or np.std(dy) == 0:
            continue
        vals.append(float(stats.spearmanr(d, dy).correlation))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def contrast(rows, ch, pairs, x, y, unit):
    entry = {'n_pairs': len(pairs)}
    if len(pairs) < 3:
        return entry
    d = np.array([x[a] - x[b] for a, b in pairs]); dy = np.array([y[a] - y[b] for a, b in pairs])
    entry['spearman'] = round(float(stats.spearmanr(d, dy).correlation), 4) if np.std(d) > 0 else None
    entry['kendall'] = round(float(stats.kendalltau(d, dy).correlation), 4) if np.std(d) > 0 else None
    entry['ci95'] = cluster_spearman(pairs, x, y, unit)
    entry['sign_agreement'] = round(float(np.mean(np.sign(d) == np.sign(dy))), 4)
    order = np.argsort(-np.abs(d))
    entry['counterexamples'] = [{'i': int(pairs[o][0]), 'j': int(pairs[o][1]),
                                 'd_feature': round(float(d[o]), 6), 'd_recall': round(float(dy[o]), 6)}
                                for o in order[:5] if np.sign(d[o]) != np.sign(dy[o])][:3]
    # leave-one-out over SIDs that appear in the pairs
    singles = sorted({a for p in pairs for a in p})
    loo = []
    for s in singles:
        sub = [(a, b) for a, b in pairs if a != s and b != s]
        if len(sub) < 10:
            continue
        dd = np.array([x[a] - x[b] for a, b in sub]); yy = np.array([y[a] - y[b] for a, b in sub])
        if np.std(dd) > 0:
            loo.append(float(stats.spearmanr(dd, yy).correlation))
    entry['loo'] = {'n_dropped': len(singles), 'min': round(float(min(loo)), 4) if loo else None,
                    'max': round(float(max(loo)), 4) if loo else None}
    # strong-only and recall halves
    strong = np.array([rows[a]['strong'] and rows[b]['strong'] for a, b in pairs])
    if strong.sum() >= 10:
        dd = np.array([x[a] - x[b] for (a, b), s in zip(pairs, strong) if s])
        yy = np.array([y[a] - y[b] for (a, b), s in zip(pairs, strong) if s])
        entry['strong_only'] = {'n_pairs': int(strong.sum()),
                                'spearman': round(float(stats.spearmanr(dd, yy).correlation), 4)
                                if np.std(dd) > 0 else None}
    med = float(np.median(y))
    hi = np.array([min(y[a], y[b]) > med for a, b in pairs])
    for name, mask in (('high_high', hi), ('low_low', np.array([max(y[a], y[b]) <= med for a, b in pairs]))):
        if mask.sum() >= 10:
            dd = np.array([x[a] - x[b] for (a, b), s in zip(pairs, mask) if s])
            yy = np.array([y[a] - y[b] for (a, b), s in zip(pairs, mask) if s])
            entry[name] = {'n_pairs': int(mask.sum()),
                           'spearman': round(float(stats.spearmanr(dd, yy).correlation), 4)
                           if np.std(dd) > 0 else None}
    entry['coverage_sids'] = int(len(singles))
    return entry


def main() -> None:
    rows = C.table_rows()
    ch = json.load(open(P / 'CHAIN_TABLE.json'))['entries']
    cf = [ch[r['sha']] for r in rows]
    unit = C.clusters(rows)
    y = np.array([r['y'] for r in rows])
    feats = {name: np.array([c.get(name, np.nan) for c in cf], float)
             for name in ('all1_chain', 'all_norm', 'all_resid', 'cond_refine', 'p3_split_given_l1_all')}
    out = {'schema': 'genrec.issue284.matched/1', 'n_rows': len(rows), 'n_units': len(set(unit)),
           'primary': {}, 'secondary': {}, 'collinearity_all_norm': {}}
    for k in C.FEATS6 + ['l2_pos_hit', 'l2_neg_hit']:
        v = np.array([r['x8'][k] for r in rows], float)
        out['collinearity_all_norm'][f'all_norm_vs_{k}'] = round(
            float(stats.spearmanr(feats['all_norm'], v).correlation), 4)
    out['collinearity_all_norm']['all_norm_vs_l1_entropy'] = round(
        float(stats.spearmanr(feats['all_norm'], np.array([c['l1_entropy'] for c in cf])).correlation), 4)
    for tag, marginal in (('primary', True), ('secondary', False)):
        pairs = matches(rows, cf, marginal)
        for name in ('all1_chain', 'all_norm', 'cond_refine'):
            out[tag][name] = contrast(rows, cf, pairs, feats[name], y, unit)
        out[tag]['n_pairs_total'] = len(pairs)
    (P / 'MATCHED_284.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:3500])


if __name__ == '__main__':
    main()
