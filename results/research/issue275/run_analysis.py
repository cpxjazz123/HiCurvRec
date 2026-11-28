"""Issue #275 — pre-registered analysis: controls, LOFO/time splits, PySR, bootstrap CIs.

Reads JOINT_TABLE.json, writes ANALYSIS_RESULTS.json. Deterministic; seeded 42.
"""
import json, math, warnings
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
P = Path('results/research/issue275')
FEATS_SID = ['l1_pos_hit','l1_neg_hit','l1_hard_er','l2_distinct','l3_collision','l3_refine_bits']
FEATS_VEC = FEATS_SID + ['quant_error','effective_rank']
SEED = 42

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    m = {'mae': float(np.mean(np.abs(y-p))), 'rmse': float(np.sqrt(np.mean((y-p)**2)))}
    if len(y) > 2 and np.std(p) > 0:
        m['spearman'] = float(stats.spearmanr(y, p).correlation)
        m['kendall'] = float(stats.kendalltau(y, p).correlation)
    else:
        m['spearman'] = None; m['kendall'] = None
    rev = 0; tot = 0
    for i in range(len(y)):
        for k in range(i+1, len(y)):
            if (y[i]-y[k])*(p[i]-p[k]) < 0: rev += 1
            if y[i] != y[k]: tot += 1
    m['reversals'] = rev/tot if tot else None
    return m



def main():
    j = json.load(open(P/'JOINT_TABLE.json'))
    rows = [r for r in j['rows'] if r['label_recall@10'] is not None and r['l1_pos_hit'] is not None]
    rows.sort(key=lambda r: r['sid_sha256'])
    def vec(r, feats): return [float(r[f]) for f in feats]
    fam_counts = Counter(r['family'] for r in rows)
    outer_fams = sorted([f for f,c in fam_counts.items() if c >= 3])

    def fit_predict(train, test, feats):
        Xtr = np.array([vec(r, feats) for r in train]); ytr = np.array([r['label_recall@10'] for r in train])
        Xte = np.array([vec(r, feats) for r in test]); yte = np.array([r['label_recall@10'] for r in test])
        out = {}
        out['mean'] = np.full(len(yte), ytr.mean()); out['median'] = np.full(len(yte), np.median(ytr))
        # best single feature by train Spearman
        best, best_rho = None, -2
        for i,f in enumerate(feats):
            if np.std(Xtr[:,i]) == 0: continue
            rho = stats.spearmanr(Xtr[:,i], ytr).correlation
            if rho is not None and rho > best_rho: best, best_rho = i, rho
        if best is not None:
            # fit a 1-D linear map on the best feature
            A = np.vstack([Xtr[:,best], np.ones(len(ytr))]).T
            coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
            out['best_single'] = Xte[:,best]*coef[0] + coef[1]; out['best_feature'] = feats[best]
        else:
            out['best_single'] = out['mean']; out['best_feature'] = None
        # ridge with inner grouped CV
        sc = StandardScaler().fit(Xtr)
        best_alpha, best_err = 1.0, np.inf
        fam = np.array([r['family'] for r in train])
        for alpha in (0.01, 0.1, 1.0, 10.0):
            errs = []
            for f in sorted(set(fam)):
                m = fam != f
                if m.sum() < 5: continue
                rg = Ridge(alpha=alpha).fit(sc.transform(Xtr[m]), ytr[m])
                errs.append(np.mean(np.abs(rg.predict(sc.transform(Xtr[~m])) - ytr[~m])))
            if errs and np.mean(errs) < best_err: best_alpha, best_err = alpha, np.mean(errs)
        rg = Ridge(alpha=best_alpha).fit(sc.transform(Xtr), ytr)
        out['ridge'] = rg.predict(sc.transform(Xte)); out['ridge_alpha'] = best_alpha
        # prior formula: l1_pos_hit alone, linear map
        i = feats.index('l1_pos_hit')
        A = np.vstack([Xtr[:,i], np.ones(len(ytr))]).T
        coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
        out['prior_l1pos'] = Xte[:,i]*coef[0] + coef[1]
        # PySR
        from pysr import PySRRegressor
        import os
        (P/'pysr').mkdir(exist_ok=True)
        model = PySRRegressor(niterations=100, populations=8, binary_operators=['+','-','*','/'],
                              unary_operators=['square'], maxsize=10, maxdepth=4,
                              timeout_in_seconds=60, random_state=SEED, deterministic=True, procs=0,
                              multithreading=False, model_selection='best', progress=False, verbosity=0)
        old_cwd = os.getcwd()
        os.chdir(P/'pysr')
        try:
            model.fit(Xtr, ytr, variable_names=feats)
        finally:
            os.chdir(old_cwd)
        out['pysr'] = model.predict(Xte)
        expr = model.sympy()
        out['pysr_expr'] = str(expr)
        out['pysr_score_train'] = float(model.score(Xtr, ytr))
        # independent recomputation of the SymPy expression on the held-out rows
        import sympy as sp
        syms = [sp.Symbol(n) for n in feats]
        fn = sp.lambdify(syms, expr, modules=['numpy'])
        val = np.asarray(fn(*[Xte[:, i] for i in range(len(feats))]), dtype=float)
        out['pysr_expr_values'] = val
        out['pysr_expr_max_abs_diff'] = (float(np.max(np.abs(val - out['pysr'])))
                                         if np.all(np.isfinite(val)) else None)
        try:
            eq = model.equations_
            out['pareto'] = eq[['complexity', 'loss', 'equation']].to_dict('records')
        except Exception:
            out['pareto'] = []
        return out, yte

    def lofo(feats, label):
        preds = {}; exprs = {}
        for fam in outer_fams:
            tr = [r for r in rows if r['family'] != fam]; te = [r for r in rows if r['family'] == fam]
            if len(tr) < 8 or len(te) < 2: continue
            out, yte = fit_predict(tr, te, feats)
            for k in ('mean','median','best_single','ridge','prior_l1pos','pysr'):
                preds.setdefault(k, []); preds[k].extend(list(out[k]))
            exprs[fam] = {'expr': out['pysr_expr'],
                          'expr_max_abs_diff': out['pysr_expr_max_abs_diff'],
                          'pareto': [{'complexity': float(e['complexity']), 'loss': float(e['loss']),
                                      'equation': str(e['equation'])} for e in out['pareto']]}
        y = []
        for fam in outer_fams:
            te = [r for r in rows if r['family'] == fam]
            if len(te) >= 2: y.extend([r['label_recall@10'] for r in te])
        res = {'label': label, 'n': len(y), 'folds': exprs,
               'metrics': {k: metrics(y, v) for k, v in preds.items()}}
        # paired bootstrap over families: pysr vs strongest control (min MAE)
        fams = [f for f in outer_fams if len([r for r in rows if r['family']==f]) >= 2]
        err = {k: [] for k in preds}
        off = 0
        for fam in fams:
            te = [r for r in rows if r['family'] == fam]
            n = len(te)
            for k in preds: err[k].append(np.abs(np.array(preds[k][off:off+n]) - np.array([r['label_recall@10'] for r in te])))
            off += n
        best_ctrl = min((k for k in preds if k != 'pysr'), key=lambda k: np.mean(np.concatenate(err[k])))
        rng = np.random.default_rng(SEED); diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(err['pysr']), len(err['pysr']))
            a = np.mean([np.mean(err['pysr'][i]) for i in idx])
            b = np.mean([np.mean(err[best_ctrl][i]) for i in idx])
            diffs.append(a-b)
        res['strongest_control'] = best_ctrl
        res['mae_diff_pysr_minus_control'] = float(np.mean(np.concatenate(err['pysr'])) - np.mean(np.concatenate(err[best_ctrl])))
        res['mae_diff_ci95'] = [float(np.percentile(diffs,2.5)), float(np.percentile(diffs,97.5))]
        res['predictions'] = {k: [float(x) for x in v] for k, v in preds.items()}
        res['y'] = [float(x) for x in y]
        return res

    results = {'schema':'genrec.issue275.analysis/1', 'seed':SEED,
               'n_rows': len(rows), 'family_counts': dict(fam_counts), 'outer_families': outer_fams}
    results['Q_SID'] = lofo(FEATS_SID, 'Q_SID')
    # Q_VEC ablation
    vec_rows = [r for r in rows if r['quant_error'] is not None]
    results['Q_VEC_audit'] = {'n': len(vec_rows), 'families': dict(Counter(r['family'] for r in vec_rows))}
    if len(vec_rows) >= 20:
        saved = rows; rows = vec_rows
        fam_counts = Counter(r['family'] for r in rows)
        outer_fams = sorted([f for f,c in fam_counts.items() if c >= 3])
        results['Q_VEC'] = lofo(FEATS_VEC, 'Q_VEC')
        rows = saved
    # threshold evaluation
    y = np.array(results['Q_SID']['y']); p = np.array(results['Q_SID']['predictions']['pysr'])
    results['threshold_0.06'] = {'positives': int((y >= 0.06).sum()),
                                 'evaluable': bool((y >= 0.06).sum() > 0 and (y < 0.06).sum() > 0),
                                 'tp': int(((p >= 0.06) & (y >= 0.06)).sum()),
                                 'fp': int(((p >= 0.06) & (y < 0.06)).sum())}
    # time split (report-only)
    by_date = {}
    for r in rows:
        d = r.get('date') or 'unknown'; by_date.setdefault(d, []).append(r)
    results['time_split'] = {d: len(v) for d, v in sorted(by_date.items())}
    if len(sorted(by_date)) >= 2:
        ds = sorted(by_date); early, late = ds[0], ds[-1]
        results['time_split_note'] = ('underpowered' if min(len(by_date[early]), len(by_date[late])) < 8 else 'usable')
    (P/'ANALYSIS_RESULTS.json').write_text(json.dumps(results, indent=1))
    print('done', json.dumps({k: results['Q_SID']['metrics'][k] for k in ('mean','pysr')}, indent=0)[:400])
    print('best control', results['Q_SID']['strongest_control'], 'diff', results['Q_SID']['mae_diff_pysr_minus_control'], 'ci', results['Q_SID']['mae_diff_ci95'])


if __name__ == '__main__':
    main()
