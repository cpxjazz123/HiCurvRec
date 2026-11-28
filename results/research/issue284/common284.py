"""Issue #284 shared pipeline: rows, SID codes, chain features, baseline OOF, clusters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
from results.research.issue283 import joint283 as J

P = Path('results/research/issue284')
P283 = Path('results/research/issue283')
FEATS6 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
          'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)
NC = len(J.CLASS_NAMES)
_ROWS = {}


def load_row_tables():
    if 'A' in _ROWS:
        return _ROWS
    A, M, B, CLS = [], [], [], []
    for name in sorted((P283 / 'rows').glob('shard_*.npz')):
        z = np.load(name)
        A.append(z['a']); M.append(z['m']); B.append(z['b']); CLS.append(z['cls'])
    _ROWS.update(A=np.concatenate(A), M=np.concatenate(M), B=np.concatenate(B),
                 CLS=np.concatenate(CLS).astype(np.int8))
    return _ROWS


def table_rows():
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    rows, seen = [], set()
    for g in table['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float))
                                             for k in FEATS8)]
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
                     'x6': {k: float(r['features'][k]) for k in FEATS6},
                     'x8': {k: float(r['features'][k]) for k in FEATS8},
                     'strong': ('L1_ledger' in ident) or any(str(s).startswith('inventory_unit') for s in srcs),
                     'group': g['group'],
                     'sid_path': None})
    table_rows.folds = folds
    return rows


def sid_path_of(sha):
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    for g in table['groups']:
        for r in g['rows']:
            if r['sid_sha256'] == sha:
                src = (r.get('features') or {}).get('_source')
                if not src:
                    return None
                rec = json.loads(Path(src).read_text())
                return rec['sid']['path']
    return None


def chain_counts(codes, classes=('chain',)):
    """Per-class 5-pattern counts at k=1,2,3 + transitions for the given role classes (row table)."""
    R = load_row_tables()
    A, M, B, CLS = R['A'], R['M'], R['B'], R['CLS']
    out = {}
    for cname in classes:
        ci = J.CLASS_ID[cname]
        mask = CLS == ci
        if not mask.any():
            out[cname] = None
            continue
        a, m, b = A[mask], M[mask], B[mask]
        trans = {}
        counts = {}
        cum = [np.ones(len(a), dtype=bool) for _ in range(3)]
        prev = None
        for k in range(1, 4):
            ca, cm, cb = codes[a, k - 1], codes[m, k - 1], codes[b, k - 1]
            e = (ca == cm, cm == cb, ca == cb)
            for i in range(3):
                cum[i] = cum[i] & e[i]
            pat = J.pattern_of(cum[0], cum[1], cum[2])
            counts[k] = np.bincount(pat, minlength=5).tolist()
            if prev is not None:
                tr = np.bincount(prev * 5 + pat, minlength=25)
                trans[k] = tr.reshape(5, 5).tolist()
            prev = pat
        out[cname] = {'counts': counts, 'trans': trans}
    return out


def chain_features(codes):
    """Frozen per-SID chain features (CONTRACT §1) from one code matrix."""
    cc = chain_counts(codes, ('chain', 'chain_both', 'fork', 'merge', 'revchain'))
    c = cc['chain']
    c1 = np.asarray(c['counts'][1], float); S = c1.sum()
    p_am = (c1[0] + c1[1]) / S; p_mb = (c1[0] + c1[2]) / S; p_ab = (c1[0] + c1[3]) / S
    all1 = c1[0] / S
    lo, hi = all_bounds(p_am, p_mb, p_ab)
    f = {'all1_chain': all1, 'all2_chain': c['counts'][2][0] / sum(c['counts'][2]),
         'all3_chain': c['counts'][3][0] / sum(c['counts'][3]),
         'p_am': p_am, 'p_mb': p_mb, 'p_ab': p_ab,
         'all_lo': lo, 'all_hi': hi,
         'all_norm': (all1 - lo) / (hi - lo) if hi > lo else 0.0,
         'all_resid': all1 - lo,
         'n_chain': int(S)}
    tr2 = np.asarray(c['trans'][2], float); tr3 = np.asarray(c['trans'][3], float)
    cond = conditional_from_transitions(tr2, tr3)
    f.update(cond)
    for other in ('chain_both', 'fork', 'merge', 'revchain'):
        o = cc[other]
        cnt = np.asarray(o['counts'][1], float)
        f[f'all1_{other}'] = cnt[0] / cnt.sum() if cnt.sum() else None
    # L1 occupancy / entropy
    l1 = np.bincount(codes[:, 0], minlength=int(codes[:, 0].max()) + 1).astype(float)
    nz = l1[l1 > 0]; N = l1.sum()
    f['l1_distinct'] = int(len(nz))
    f['l1_entropy'] = float(-(nz / N * np.log2(nz / N)).sum())
    return f


def all_bounds(p_am, p_mb, p_ab):
    """Feasible interval of ALL given the three pairwise marginals (five-partition non-negativity)."""
    return (max(0.0, (p_am + p_mb + p_ab - 1.0) / 2.0), min(p_am, p_mb, p_ab))


def conditional_from_transitions(tr2, tr3):
    """Exact conditional refinement metrics from the k1->k2 and k2->k3 transition counts."""
    tr2 = np.asarray(tr2, float); tr3 = np.asarray(tr3, float)
    denom = tr2[0].sum()
    if not denom:
        return {'l2_all_given_l1_all': None, 'cond_refine': None,
                'p3_split_given_l1_all': None, 'l3_all_given_l1_all': None}
    p_l2_all = tr2[0, 0] / denom
    num_split = 0.0; num_all3 = 0.0
    for p2 in range(5):
        n_p2 = tr3[p2].sum()
        if n_p2:
            num_split += tr2[0, p2] * (tr3[p2, 4] / n_p2)
            num_all3 += tr2[0, p2] * (tr3[p2, 0] / n_p2)
    return {'l2_all_given_l1_all': float(p_l2_all), 'cond_refine': float(1.0 - p_l2_all),
            'p3_split_given_l1_all': float(num_split / denom),
            'l3_all_given_l1_all': float(num_all3 / denom)}


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
            'reversals': rev / tot if tot else None,
            'bias_low': float(np.mean((y - p)[y <= np.median(y)])),
            'bias_high': float(np.mean((y - p)[y > np.median(y)]))}


def confusion(y, p, cut=0.06):
    y, p = np.array(y, float), np.array(p, float)
    return {'positives': int((y > cut).sum()), 'tp': int(((p > cut) & (y > cut)).sum()),
            'fp': int(((p > cut) & (y <= cut)).sum()), 'fn': int(((p <= cut) & (y > cut)).sum()),
            'tn': int(((p <= cut) & (y <= cut)).sum())}


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


def oof(X, y, groups):
    pred = np.zeros(len(y))
    for g in sorted(set(groups.tolist())):
        tr = groups != g
        a = inner_alpha(X[tr], y[tr], groups[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
        pred[~tr] = rg.predict(sc.transform(X[~tr]))
    return pred


def clusters(rows):
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    rowmap = {r0['sid_sha256']: r0 for g in table['groups'] for r0 in g['rows']}
    groups = {}
    for i, r in enumerate(rows):
        sig = tuple(sorted((str(s.get('source')), round(float(s.get('recall', 0)), 6))
                           for s in rowmap[r['sha']].get('label_sources', [])))
        groups.setdefault(sig, []).append(i)
    unit = {}
    for u, idx in enumerate(groups.values()):
        for i in idx:
            unit[i] = u
    return [unit[i] for i in range(len(rows))]


def cluster_bootstrap(pred_a, pred_b, y, unit, reps=2000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit))
    idx_of = {u: np.where(np.array(unit) == u)[0] for u in uniq}
    diffs = []
    base = np.abs(y - pred_a); alt = np.abs(y - pred_b)
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[int(k)] for k in pick])
        diffs.append(float(base[idx].mean() - alt[idx].mean()))
    return {'mae_diff': float(base.mean() - alt.mean()),
            'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
