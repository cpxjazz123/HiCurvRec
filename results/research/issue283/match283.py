"""Issue #283: matched-SID contrasts, residual association and counterexamples (CONTRACT §6)."""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from results.research.issue283 import joint283 as J

FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
          'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS6 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)
XTOL = {'l1_pos_hit': ('abs', 0.01), 'l1_hard_er': ('rel', 0.20), 'l2_pos_hit': ('rel', 0.20),
        'l3_collision': ('abs', 0.01), 'l3_refine_bits': ('rel', 0.10)}
MARGINAL_TOL = 0.005  # near-equal pairwise rates (the issue's "marginals matched" condition)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rev = tot = 0
    for i in range(len(y)):
        for k in range(i + 1, len(y)):
            if (y[i] - y[k]) * (p[i] - p[k]) < 0:
                rev += 1
            if y[i] != y[k]:
                tot += 1
    return {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2))),
            'spearman': float(stats.spearmanr(y, p).correlation) if np.std(p) > 0 else None,
            'kendall': float(stats.kendalltau(y, p).correlation) if np.std(p) > 0 else None,
            'reversals': rev / tot if tot else None}


def load():
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    join = json.load(open('results/research/issue283/JOIN_TABLE.json'))
    rows = []
    seen = set()
    for g in table['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS8)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        sha = r['sid_sha256']
        if sha in seen:
            continue
        seen.add(sha)
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [s.get('source') for s in r.get('label_sources', [])]
        rows.append({'sha': sha, 'y': float(np.mean(labels)),
                     'x': {k: float(r['features'][k]) for k in FEATS8},
                     'strong': ('L1_ledger' in ident) or any(str(s).startswith('inventory_unit') for s in srcs),
                     'arms': [a.get('arm') for a in r.get('arms', [])],
                     'group': g['group']})
    return rows, folds, join


def feats_of(entry, support):
    """Per-SID derived joint features (rates + lifts + family aggregates)."""
    classes = entry['classes']
    f = {}
    fam_counts = {}
    for name, d in classes.items():
        ci = J.CLASS_ID[name]
        fam = J.FAMILY[ci]
        for k in (1, 2, 3):
            lvl = d[str(k)]
            cnt = np.asarray(lvl['counts'], dtype=float)
            S = cnt.sum()
            f[f'all{k}_{name}'] = float(cnt[0] / S)
            f[f'all{k}_{name}_lift'] = float(lvl['lift']['all_u'])
            f[f'all{k}_{name}_liftd'] = float(lvl['lift']['all_d'])
            f[f'amb{k}_{name}'] = float((cnt[1] + cnt[2]) / S)
            f[f'ab{k}_{name}'] = float(cnt[3] / S)
            f[f'am{k}_{name}'] = float(cnt[1] / S)
            f[f'mb{k}_{name}'] = float(cnt[2] / S)
            key = (fam, k)
            fc = fam_counts.setdefault(key, np.zeros(5))
            fc += cnt
    # family aggregates
    for (fam, k), cnt in fam_counts.items():
        S = cnt.sum()
        f[f'all{k}_fam_{fam}'] = float(cnt[0] / S)
        f[f'amb{k}_fam_{fam}'] = float((cnt[1] + cnt[2]) / S)
        f[f'ab{k}_fam_{fam}'] = float(cnt[3] / S)
        f[f'am{k}_fam_{fam}'] = float(cnt[1] / S)
        f[f'mb{k}_fam_{fam}'] = float(cnt[2] / S)
    # pre-registered descriptive contrasts
    f['all1_chain_lift'] = f.get('all1_chain_lift', np.nan)
    f['all3_chain_lift'] = f.get('all3_chain_lift', np.nan)
    f['am_mb_chain'] = f['am1_chain'] - f['mb1_chain']
    f['amb_ab_merge'] = f['amb1_merge'] - f['ab1_merge']
    f['amb_ab_fork'] = f['amb1_fork'] - f['ab1_fork']
    f['am_mb_ab_chain'] = (f['am1_chain'] + f['mb1_chain']) / 2 - f['ab1_chain']
    f['all_ab_triangle'] = f['all1_fam_triangle'] - f['ab1_fam_triangle']
    f['all_ab_recip'] = f['all1_fam_reciprocal'] - f['ab1_fam_reciprocal']
    return f


def matched_pair_feature(rows, feature):
    """Pairs passing the frozen tolerances; returns list of (i,j,d_feature,d_y,r_i,r_j)."""
    out = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        xi, xj = rows[i]['x'], rows[j]['x']
        ok = True
        for key, (mode, tol) in XTOL.items():
            if mode == 'abs':
                if abs(xi[key] - xj[key]) > tol:
                    ok = False; break
            else:
                denom = max(abs(xi[key]), abs(xj[key]), 1e-12)
                if abs(xi[key] - xj[key]) / denom > tol:
                    ok = False; break
        if not ok:
            continue
        out.append((i, j, feature[i] - feature[j], rows[i]['y'] - rows[j]['y']))
    return out


def marginal_matched_pairs(rows, feat, entry_feats):
    """Near-equal pairwise marginals for the chain structure (CONTRACT §6 primary contrast)."""
    out = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = entry_feats[i], entry_feats[j]
        if abs(a['am1_chain'] - b['am1_chain']) > MARGINAL_TOL: continue
        if abs(a['mb1_chain'] - b['mb1_chain']) > MARGINAL_TOL: continue
        if abs(a['ab1_chain'] - b['ab1_chain']) > MARGINAL_TOL: continue
        out.append((i, j, a['all1_chain'] - b['all1_chain'], rows[i]['y'] - rows[j]['y']))
    return out


def cluster_bootstrap_spearman(pairs, units, reps=2000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(units))
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        keep = set(pick.tolist())
        sub = [(d, dy) for (i, j, d, dy) in pairs if units[i] in keep and units[j] in keep]
        if len(sub) < 10:
            continue
        d = np.array([p[0] for p in sub]); dy = np.array([p[1] for p in sub])
        if np.std(d) == 0 or np.std(dy) == 0:
            continue
        out.append(float(stats.spearmanr(d, dy).correlation))
    return ([float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None)


def inner_alpha(X, y, groups):
    best, best_err = 1.0, np.inf
    for alpha in ALPHAS:
        errs = []
        for g in sorted(set(groups.tolist())):
            mask = groups != g
            if mask.sum() < 5:
                continue
            sc = StandardScaler().fit(X[mask])
            rg = Ridge(alpha=alpha).fit(sc.transform(X[mask]), y[mask])
            errs.append(np.mean(np.abs(rg.predict(sc.transform(X[~mask])) - y[~mask])))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best


def oof(X, y, fold):
    """Nested OOF identical to model283/#277: inner alpha only on training rows (no outer leak)."""
    pred = np.zeros(len(y))
    for g in sorted(set(fold.tolist())):
        tr = fold != g
        a = inner_alpha(X[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
        pred[~tr] = rg.predict(sc.transform(X[~tr]))
    return pred


def main() -> None:
    rows, folds, join = load()
    y = np.array([r['y'] for r in rows])
    fold = np.array([folds['assignment'][r['group']] for r in rows])
    X = np.array([[r['x'][k] for k in FEATS6] for r in rows])
    A = oof(X, y, fold)
    resid = y - A
    base = metrics(y, A)
    entry_feats = [feats_of(join['entries'][r['sha']], None) for r in rows]
    # rebuild near-duplicate label clusters from the label-source signatures (127 units)
    clusters = {}
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    rowmap = {}
    for g in table['groups']:
        for r0 in g['rows']:
            rowmap[r0['sid_sha256']] = r0
    for i, r in enumerate(rows):
        r0 = rowmap[r['sha']]
        sigk = tuple(sorted((str(s.get('source')), round(float(s.get('recall', 0)), 6))
                            for s in r0.get('label_sources', [])))
        clusters.setdefault(sigk, []).append(i)
    unit_of_row = {}
    for u, (sigk, idxs) in enumerate(clusters.items()):
        for i in idxs:
            unit_of_row[i] = u
    units = [unit_of_row[i] for i in range(len(rows))]
    feats = [{k: v for k, v in f.items()} for f in entry_feats]
    contrast_names = ['all1_chain_lift', 'all3_chain_lift', 'am_mb_chain', 'amb_ab_merge',
                      'amb_ab_fork', 'am_mb_ab_chain', 'all_ab_triangle', 'all_ab_recip',
                      'all1_merge_lift']
    results = {'schema': 'genrec.issue283.matched/1', 'baseline_A': base,
               'n_rows': len(rows), 'n_units': len(set(units)), 'contrasts': {},
               'marginal_matched_chain': {}, 'residuals': {}, 'r2_diagnostic': {},
               'recall_groups': {}}
    med = float(np.median(y))
    for name in contrast_names:
        vals = np.array([f.get(name, np.nan) for f in feats])
        m = np.isfinite(vals)
        if m.sum() < 10:
            continue
        lo = vals[m & (y <= med)]; hi = vals[m & (y > med)]
        results['recall_groups'][name] = {
            'n': int(m.sum()), 'median_label': med,
            'low_mean': round(float(lo.mean()), 6), 'high_mean': round(float(hi.mean()), 6),
            'low_median': round(float(np.median(lo)), 6), 'high_median': round(float(np.median(hi)), 6),
            'spearman_all': round(float(stats.spearmanr(vals[m], y[m]).correlation), 4)
            if np.std(vals[m]) > 0 else None}
    for name in contrast_names:
        vals = np.array([f.get(name, np.nan) for f in feats])
        if np.all(np.isnan(vals)):
            continue
        pairs = matched_pair_feature(rows, vals)
        entry = {'n_pairs': len(pairs)}
        if len(pairs) >= 3:
            d = np.array([p[2] for p in pairs]); dy = np.array([p[3] for p in pairs])
            entry['spearman'] = float(stats.spearmanr(d, dy).correlation) if np.std(d) > 0 and np.std(dy) > 0 else None
            entry['kendall'] = float(stats.kendalltau(d, dy).correlation) if np.std(d) > 0 and np.std(dy) > 0 else None
            entry['spearman_ci95'] = cluster_bootstrap_spearman(pairs, units)
            same = int(np.sum(np.sign(d) == np.sign(dy)))
            entry['sign_agreement'] = same / len(pairs)
            order = np.argsort(-np.abs(d))
            entry['counterexamples'] = [
                {'i': int(pairs[o][0]), 'j': int(pairs[o][1]), 'd_feature': round(float(pairs[o][2]), 6),
                 'd_recall': round(float(pairs[o][3]), 6)} for o in order[:5]
                if np.sign(pairs[o][2]) != np.sign(pairs[o][3])][:3]
        results['contrasts'][name] = entry
    # marginal-matched chain: ALL vs single-pair sharing at near-equal marginals
    mm = marginal_matched_pairs(rows, y, feats)
    entry = {'n_pairs': len(mm)}
    if len(mm) >= 3:
        d = np.array([p[2] for p in mm]); dy = np.array([p[3] for p in mm])
        entry['spearman'] = float(stats.spearmanr(d, dy).correlation) if np.std(d) > 0 and np.std(dy) > 0 else None
        entry['spearman_ci95'] = cluster_bootstrap_spearman(mm, units)
        entry['sign_agreement'] = int(np.sum(np.sign(d) == np.sign(dy))) / len(mm)
        order = np.argsort(-np.abs(d))
        entry['counterexamples'] = [
            {'i': int(mm[o][0]), 'j': int(mm[o][1]), 'd_all': round(float(mm[o][2]), 6),
             'd_recall': round(float(mm[o][3]), 6)} for o in order[:5] if np.sign(mm[o][2]) != np.sign(mm[o][3])][:3]
    results['marginal_matched_chain'] = entry
    # residual associations with BH FDR over the tested contrasts
    rhos = {}
    for name in contrast_names:
        vals = np.array([f.get(name, np.nan) for f in feats])
        m = np.isfinite(vals)
        if m.sum() > 10 and np.std(vals[m]) > 0:
            rhos[name] = float(stats.spearmanr(vals[m], resid[m]).correlation)
    pvals = {k: float(2 * stats.t.sf(abs(v) * np.sqrt(len(rows) - 2) / np.sqrt(max(1 - v * v, 1e-12)),
                                      df=len(rows) - 2)) for k, v in rhos.items()}
    order = sorted(pvals, key=lambda k: pvals[k])
    bh = {}
    m = len(order)
    for rank, key in enumerate(order, 1):
        bh[key] = min(1.0, pvals[key] * m / rank)
    # enforce monotone BH
    running = 1.0
    for key in reversed(order):
        running = min(running, bh[key]); bh[key] = running
    results['residuals'] = {'n': int(np.isfinite(resid).sum()),
                            'spearman': {k: round(v, 4) for k, v in rhos.items()},
                            'pvalues': {k: round(v, 5) for k, v in pvals.items()},
                            'bh_qvalues': {k: round(v, 5) for k, v in bh.items()}}
    r2 = json.load(open('results/research/issue276/AUDIT_274.json'))['rounds']['R2']
    results['r2_diagnostic'] = {'status': 'NOT_AVAILABLE',
                                'reason': '#274 R2 sid_sha256 is not in the #276 table and has no label/features',
                                **r2}
    (J.P / 'MATCHED.json').write_text(json.dumps(results, indent=1) + '\n')
    print(json.dumps({'baseline': base, 'contrasts': results['contrasts'],
                      'marginal_matched': results['marginal_matched_chain'],
                      'residuals': results['residuals']}, indent=1)[:4000])


if __name__ == '__main__':
    main()
