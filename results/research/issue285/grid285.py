"""Issue #285: joint-condition grid, interaction tests, monotonicity, counterexamples, robustness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue285')
import common285 as W

P = Path('results/research/issue285')
MIN_SUPPORT = 10


def spearman(x, y):
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(stats.spearmanr(x, y).correlation)


def kendall(x, y):
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(stats.kendalltau(x, y).correlation)


def reversals(x, y):
    n = len(x); inv = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            tot += 1
            if (x[i] - x[j]) * (y[i] - y[j]) < 0:
                inv += 1
    return {'inversions': inv, 'comparable': tot, 'rate': inv / tot if tot else None}


def slope(x, y):
    if len(x) < 4 or np.std(x) == 0:
        return None
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
    return float(np.polyfit(xs, ys, 1)[0])


def bootstrap_cell(cell_idx, x, y, unit, reps=2000, seed=42):
    """Cluster bootstrap of the within-cell Spearman."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit))
    idx_of = {u: [i for i in cell_idx if unit[i] == u] for u in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = []
        for u in pick:
            sel.extend(idx_of[int(u)])
        if len(sel) < 5:
            continue
        v = spearman(x[sel], y[sel])
        if v is not None:
            vals.append(v)
    return ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 10 else None)


def bootstrap_diff(idx_a, idx_b, x, y, unit, reps=2000, seed=42):
    """Cluster bootstrap of rho(A) - rho(B) with the same resampled clusters."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit))
    map_a = {u: [i for i in idx_a if unit[i] == u] for u in uniq}
    map_b = {u: [i for i in idx_b if unit[i] == u] for u in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel_a, sel_b = [], []
        for u in pick:
            sel_a.extend(map_a[int(u)]); sel_b.extend(map_b[int(u)])
        if len(sel_a) < 5 or len(sel_b) < 5:
            continue
        ra = spearman(x[sel_a], y[sel_a]); rb = spearman(x[sel_b], y[sel_b])
        if ra is not None and rb is not None:
            vals.append(ra - rb)
    return ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 10 else None)


def partial_spearman(cell_idx, x, y, controls, unit, reps=1000, seed=42):
    """Partial Spearman of x and y after rank-residualising out the controls within the cell."""
    xi = np.array(cell_idx)
    rx = stats.rankdata(x[xi]); ry = stats.rankdata(y[xi])
    R = np.column_stack([stats.rankdata(c[xi]) for c in controls])
    A = np.column_stack([np.ones(len(xi)), R])
    beta_y, *_ = np.linalg.lstsq(A, ry, rcond=None)
    beta_x, *_ = np.linalg.lstsq(A, rx, rcond=None)
    ey = ry - A @ beta_y; ex = rx - A @ beta_x
    v = spearman(ex, ey)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit))
    mapc = {u: [i for i in xi if unit[i] == u] for u in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = []
        for u in pick:
            sel.extend(mapc[int(u)])
        if len(sel) < 8:
            continue
        s = np.array(sel)
        rx2 = stats.rankdata(x[s]); ry2 = stats.rankdata(y[s])
        R2 = np.column_stack([stats.rankdata(c[s]) for c in controls])
        A2 = np.column_stack([np.ones(len(s)), R2])
        ay, *_ = np.linalg.lstsq(A2, ry2, rcond=None)
        ax, *_ = np.linalg.lstsq(A2, rx2, rcond=None)
        vv = spearman(rx2 - A2 @ ax, ry2 - A2 @ ay)
        if vv is not None:
            vals.append(vv)
    return {'rho': v, 'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            if len(vals) > 10 else None}


def main() -> None:
    d = W.load_table()
    y = d['y']; unit = d['unit']; fold = d['fold']
    g = W.gates(d)
    cells = W.cell_labels(g['level_N'], g['level_C'])
    feats = {'P': d['P'], 'ER': d['ER'], 'ALL': d['ALL'], 'ALL_fork': d['ALL_fork'],
             'ALL_merge': d['ALL_merge']}
    out = {'schema': 'genrec.issue285.grid/1', 'min_support': MIN_SUPPORT,
           'cells': {}, 'interaction': {}, 'monotonicity': {}, 'counterexamples': {},
           'discrimination': {}, 'partial': {}, 'robustness': {}}
    for c in ('LL', 'LN', 'LC', 'HH', 'other'):
        idx = np.where(cells == c)[0]
        if len(idx) == 0:
            continue
        entry = {'n': int(len(idx)), 'clusters': len(set(unit[i] for i in idx)),
                 'strong': int(d['strong'][idx].sum()), 'high': int((y[idx] > 0.06).sum()),
                 'high_strong': int(((y[idx] > 0.06) & d['strong'][idx]).sum()),
                 'support_ok': bool(len(idx) >= MIN_SUPPORT),
                 'recall': {'mean': round(float(y[idx].mean()), 5), 'min': round(float(y[idx].min()), 5),
                            'max': round(float(y[idx].max()), 5),
                            'range': round(float(y[idx].max() - y[idx].min()), 5)}}
        for name, v in feats.items():
            m = idx[np.isfinite(v[idx])]
            entry[name] = {'spearman': round(spearman(v[m], y[m]), 4) if spearman(v[m], y[m]) is not None else None,
                           'kendall': round(kendall(v[m], y[m]), 4) if kendall(v[m], y[m]) is not None else None,
                           'slope': round(slope(v[m], y[m]), 4) if slope(v[m], y[m]) is not None else None,
                           'ci95': bootstrap_cell(list(m), v, y, unit),
                           'reversals': reversals(v[m], y[m])}
        out['cells'][c] = entry
    # interaction LL vs HH (both >= min support) for each feature
    ll = np.where(cells == 'LL')[0]; hh = np.where(cells == 'HH')[0]
    for name, v in feats.items():
        r_ll = spearman(v[ll], y[ll]); r_hh = spearman(v[hh], y[hh])
        dci = bootstrap_diff(list(ll), list(hh), v, y, unit)
        out['interaction'][name] = {
            'rho_LL': round(r_ll, 4), 'rho_HH': round(r_hh, 4),
            'delta': round(r_ll - r_hh, 4),
            'delta_ci95': dci,
            'supported': bool(dci and dci[0] > 0 and len(ll) >= MIN_SUPPORT and len(hh) >= MIN_SUPPORT)}
    out['interaction']['note'] = ("supported requires the delta-rho cluster-bootstrap CI to exclude 0 and "
                                  "both cells >= min_support; one-significant-one-not is not an interaction")
    # monotonicity / reversal counts per cell for P and ALL
    for c in ('LL', 'HH'):
        idx = np.where(cells == c)[0]
        out['monotonicity'][c] = {name: reversals(feats[name][idx], y[idx]) for name in ('P', 'ER', 'ALL')}
    # counterexamples in LL (largest |ΔP| / |ΔALL| discordant pairs, equality excluded)
    for name in ('P', 'ALL'):
        v = feats[name][ll]; yy = y[ll]
        pairs = []
        for i in range(len(ll)):
            for j in range(len(ll)):
                if i >= j or yy[i] == yy[j]:
                    continue
                if (v[i] - v[j]) * (yy[i] - yy[j]) < 0:
                    pairs.append((abs(v[i] - v[j]), int(ll[i]), int(ll[j]), float(v[i] - v[j]), float(yy[i] - yy[j])))
        pairs.sort(reverse=True)
        out['counterexamples'][name] = [{'i': p[1], 'j': p[2], 'd_feature': round(p[3], 6),
                                         'd_recall': round(p[4], 6)} for p in pairs[:5]]
    # lowN/lowC alone for high-recall discrimination (AUC, cluster bootstrap)
    rng = np.random.default_rng(42)
    uniq = sorted(set(unit))
    for name, v in (('N', d['N']), ('C', d['C'])):
        pos = y > 0.06
        if pos.sum() < 2:
            continue
        auc = float(stats.mannwhitneyu(v[pos], v[~pos], alternative='two-sided').statistic /
                    (pos.sum() * (~pos).sum()))
        idx_of = {u: [i for i in range(len(y)) if unit[i] == u] for u in uniq}
        boots = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = []
            for u in pick:
                sel.extend(idx_of[int(u)])
            s = np.array(sel); p = y[s] > 0.06
            if p.sum() < 2 or (~p).sum() < 2:
                continue
            boots.append(float(stats.mannwhitneyu(v[s][p], v[s][~p], alternative='two-sided').statistic /
                               (p.sum() * (~p).sum())))
        out['discrimination'][name] = {'auc_high_recall': round(auc, 4),
                                       'ci95': [round(float(np.percentile(boots, 2.5)), 4),
                                                round(float(np.percentile(boots, 97.5)), 4)],
                                       'note': 'AUC of the raw axis for recalling >0.06 (not a predictor fit)'}
    # partial association within LL after N,C (and the six metrics)
    for name in ('P', 'ALL'):
        out['partial'].setdefault('LL', {})[name] = partial_spearman(list(ll), feats[name], y,
                                                                     [d['N'], d['C']], unit)
    # robustness: strong-only, drop high tail, leave-one-out, time-postordered sensitivity
    strong_ll = ll[d['strong'][ll]]
    out['robustness']['LL_strong_only'] = {
        'n': int(len(strong_ll)), 'high': int((y[strong_ll] > 0.06).sum()),
        'P_spearman': round(spearman(feats['P'][strong_ll], y[strong_ll]), 4),
        'ALL_spearman': round(spearman(feats['ALL'][strong_ll], y[strong_ll]), 4)}
    no_high = ll[y[ll] <= 0.06]
    out['robustness']['LL_without_high_tail'] = {
        'n': int(len(no_high)),
        'P_spearman': round(spearman(feats['P'][no_high], y[no_high]), 4),
        'ALL_spearman': round(spearman(feats['ALL'][no_high], y[no_high]), 4)}
    loo = {}
    for name in ('P', 'ALL'):
        vals = []
        for drop in ll:
            keep = [i for i in ll if i != drop]
            v = spearman(feats[name][keep], y[keep])
            if v is not None:
                vals.append(v)
        loo[name] = {'min': round(float(min(vals)), 4), 'max': round(float(max(vals)), 4)}
    out['robustness']['LL_leave_one_out'] = loo
    # range restriction: truncate LL to the common recall interval shared with HH
    lo = max(float(y[ll].min()), float(y[hh].min())); hi = min(float(y[ll].max()), float(y[hh].max()))
    ll_tr = ll[(y[ll] >= lo) & (y[ll] <= hi)]
    out['range_restriction'] = {
        'common_interval': [round(lo, 5), round(hi, 5)], 'LL_n': int(len(ll)), 'LL_truncated_n': int(len(ll_tr)),
        'LL_truncated_high': int((y[ll_tr] > 0.06).sum()),
        'LL_truncated_P_spearman': round(spearman(feats['P'][ll_tr], y[ll_tr]), 4),
        'LL_truncated_ALL_spearman': round(spearman(feats['ALL'][ll_tr], y[ll_tr]), 4),
        'LL_recall_sd': round(float(y[ll].std(ddof=1)), 5), 'HH_recall_sd': round(float(y[hh].std(ddof=1)), 5),
        'LL_full_P_spearman': round(spearman(feats['P'][ll], y[ll]), 4),
        'note': 'truncation to the recall interval shared by LL and HH; the remainder is the non-range component'}
    # per-SID table for the deliverable
    rows = []
    for i, r in enumerate(d['rows']):
        rows.append({'sid_sha256': r['sha'], 'y': float(y[i]), 'fold': int(fold[i]),
                     'strong': bool(d['strong'][i]), 'cell': str(cells[i]),
                     'N': round(float(d['N'][i]), 6), 'C': round(float(d['C'][i]), 6),
                     'P': round(float(d['P'][i]), 6), 'ER': round(float(d['ER'][i]), 6),
                     'ALL': round(float(d['ALL'][i]), 6)})
    (P / 'PER_SID_285.json').write_text(json.dumps(rows, indent=0) + '\n')
    (P / 'GRID_285.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('cells', 'interaction', 'monotonicity', 'discrimination',
                                          'partial', 'robustness')}, indent=1)[:4500])


if __name__ == '__main__':
    main()
