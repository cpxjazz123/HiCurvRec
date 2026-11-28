"""Issue #280 — SID-only directional decoder for unseen directed edges (CONTRACT.md §2–§5)."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import average_precision_score

P = Path('results/research/issue280')
N = 24587
SEED = 42
ALPHAS = (0.1, 1.0, 10.0)
EPSILONS = (1e-3, 1e-2)


def load_graph():
    tr = pq.read_table('results/stage0_build_parquet/train.parquet',
                       columns=['history', 'target']).to_pylist()
    ea, eb = [], []
    for r in tr:
        h = r['history']
        if len(h) >= 2:
            a = np.asarray(h, dtype=np.int64)
            ea.append(a[:-1]); eb.append(a[1:])
    last = np.array([r['history'][-1] if r['history'] else -1 for r in tr], dtype=np.int64)
    ok = last >= 0
    ea.append(last[ok]); eb.append(np.array([r['target'] for r in tr], dtype=np.int64)[ok])
    ea = np.concatenate(ea); eb = np.concatenate(eb)
    uk, cnt = np.unique(ea * N + eb, return_counts=True)
    return (uk // N).astype(np.int64), (uk % N).astype(np.int64), cnt.astype(np.int64)


def build_tables(codes, ea, eb, ew):
    """Per-level numerator/denominator lookup arrays from distinct edges (weighted)."""
    c1 = codes[:, 0]; c2 = codes[:, 1]; c3 = codes[:, 2]
    s1, s2, s3 = c1[ea], (c1[ea] * 256 + c2[ea]), ((c1[ea] * 256 + c2[ea]) * 256 + c3[ea])
    b1, b2, b3 = c1[eb], c2[eb], c3[eb]
    t1, t2 = b1, b1 * 256 + b2
    num = [(s1, 0, b1), (s2, t1, b2), (s3, t2, b3)]
    den = [(0, 0, b1), (0, t1, b2), (0, t2, b3)]
    levels = []
    for (sn, tn, cl), (sd, td, cld) in zip(num, den):
        kn = (sn * 256 + tn) * 256 + cl
        kd = (sd * 256 + td) * 256 + cld
        ukn = np.unique(kn).astype(np.int64)
        inv = np.searchsorted(ukn, kn)
        cn = np.bincount(inv, weights=ew).astype(np.float64)
        ctxn = ukn // 256
        idx = np.flatnonzero(np.diff(ctxn)) + 1
        uctxn = ctxn[np.r_[0, idx]] if len(ctxn) else np.array([], np.int64)
        totn = np.add.reduceat(cn, np.r_[0, idx]) if len(cn) else np.array([], np.float64)
        ukd = np.unique(kd).astype(np.int64)
        invd = np.searchsorted(ukd, kd)
        cd = np.bincount(invd, weights=ew).astype(np.float64)
        ctxd = ukd // 256
        idxd = np.flatnonzero(np.diff(ctxd)) + 1
        uctxd = ctxd[np.r_[0, idxd]] if len(ctxd) else np.array([], np.int64)
        totd = np.add.reduceat(cd, np.r_[0, idxd]) if len(cd) else np.array([], np.float64)
        marg = np.bincount(cl, weights=ew, minlength=256).astype(np.float64)
        levels.append({'ukn': ukn, 'cn': cn, 'uctxn': uctxn, 'totn': totn,
                       'ukd': ukd, 'cd': cd, 'uctxd': uctxd, 'totd': totd,
                       'marg': marg, 'margs': marg.sum()})
    return levels


def lookup(keys, vals, query):
    pos = np.searchsorted(keys, query)
    pos = np.clip(pos, 0, max(0, len(keys) - 1))
    hit = (len(keys) > 0) & (keys[pos] == query) if len(keys) else np.zeros(len(query), bool)
    out = np.zeros(len(query))
    out[hit] = vals[pos[hit]]
    return out, hit


def ctx_total(ctx_keys, ctx_tot, query):
    pos = np.searchsorted(ctx_keys, query)
    pos = np.clip(pos, 0, max(0, len(ctx_keys) - 1))
    hit = (len(ctx_keys) > 0) & (ctx_keys[pos] == query) if len(ctx_keys) else np.zeros(len(query), bool)
    out = np.zeros(len(query))
    out[hit] = ctx_tot[pos[hit]]
    return out, hit


def score_pairs(levels, codes, A, B, alpha, eps):
    a1, a2, a3 = codes[A, 0], codes[A, 1], codes[A, 2]
    b1, b2, b3 = codes[B, 0], codes[B, 1], codes[B, 2]
    s = [a1, a1 * 256 + a2, (a1 * 256 + a2) * 256 + a3]
    t = [np.zeros_like(b1), b1, b1 * 256 + b2]
    cl = [b1, b2, b3]
    total = np.zeros(len(A))
    denom = alpha * 256.0
    for l in range(3):
        lv = levels[l]
        cn, _ = lookup(lv['ukn'], lv['cn'], (s[l] * 256 + t[l]) * 256 + cl[l])
        totn, hitn = ctx_total(lv['uctxn'], lv['totn'], s[l] * 256 + t[l])
        marg = lv['marg']; mtot = lv['margs']
        Pn = np.where(hitn, (cn + alpha) / (totn + denom),
                      (marg[cl[l]] + alpha) / (mtot + denom))
        cd, _ = lookup(lv['ukd'], lv['cd'], (t[l]) * 256 + cl[l])
        totd, hitd = ctx_total(lv['uctxd'], lv['totd'], t[l])
        Pd = np.where(hitd, (cd + alpha) / (totd + denom),
                      (marg[cl[l]] + alpha) / (mtot + denom))
        total += np.log((Pn + eps) / (Pd + eps))
    return total


def auc_rank(pos, neg):
    x = np.concatenate([pos, neg])
    r = stats.rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def main() -> None:
    ea, eb, ew = load_graph()
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    keys = ea * N + eb
    out_deg_all = np.bincount(ea, minlength=N)
    in_deg_all = np.bincount(eb, minlength=N)
    pos = np.searchsorted(keys, anchor * N + target)
    pos = np.clip(pos, 0, len(keys) - 1)
    seen = keys[pos] == anchor * N + target
    unseen = ~seen
    rng = np.random.default_rng(SEED)
    neg = np.empty(n, dtype=np.int64)
    for i, r in enumerate(va):
        excl = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        while True:
            cand = int(rng.integers(0, N))
            if cand not in excl:
                neg[i] = cand; break
    sample = rng.choice(n, 10000, replace=False)
    # SID list + codes
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    sids = []
    for g in table['groups']:
        for r in g['rows']:
            f = (r.get('features') or {}).get('_source')
            if not f or not Path(f).is_file():
                continue
            rec = json.load(open(f))
            sp = (rec.get('sid') or {}).get('path')
            if not sp or not Path(sp).is_file():
                continue
            raw = json.loads(Path(sp).read_text())
            codes = np.zeros((N, 4), dtype=np.int64)
            for k, v in raw.items():
                codes[int(k), :len(v)] = v[:4]
            sids.append({'sid_sha256': r['sid_sha256'], 'codes': codes[:N, :3],
                         'label': r.get('label_recall@10'), 'arms': [a.get('arm') for a in r.get('arms', [])][:2],
                         'strong': False})
            break  # one row per group
    # canonical baseline SID first for calibration
    cal_sids = ['025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059']
    high = sorted([s for s in sids if isinstance(s['label'], (int, float)) and s['label'] > 0.06],
                  key=lambda s: s['sid_sha256'])
    for s in high[:4]:
        if s['sid_sha256'] not in cal_sids:
            cal_sids.append(s['sid_sha256'])
    # source-anchored split of edges for calibration
    src_hash = (ea * 2654435761) % 5
    fit_mask = src_hash != 0
    cal_mask = ~fit_mask
    # calibration positives: 8k edges from the held-out side; negatives: 8k random non-edges
    cal_pos_idx = np.flatnonzero(cal_mask)
    cal_pos_idx = rng.choice(cal_pos_idx, min(8000, len(cal_pos_idx)), replace=False)
    ea_fit, eb_fit, ew_fit = ea[fit_mask], eb[fit_mask], ew[fit_mask]
    neg_pairs = []
    keyset = set(keys.tolist())
    while len(neg_pairs) < 8000:
        a = int(rng.integers(0, N)); b = int(rng.integers(0, N))
        if a == b or (a * N + b) in keyset:
            continue
        neg_pairs.append((a, b))
    neg_a = np.array([x[0] for x in neg_pairs]); neg_b = np.array([x[1] for x in neg_pairs])
    cal_pos_a, cal_pos_b = ea[cal_pos_idx], eb[cal_pos_idx]
    grid = []
    for s in sids:
        if s['sid_sha256'] not in cal_sids:
            continue
        tables = build_tables(s['codes'], ea_fit, eb_fit, ew_fit.astype(np.float64))
        for alpha in ALPHAS:
            for eps in EPSILONS:
                p = score_pairs(tables, s['codes'], cal_pos_a, cal_pos_b, alpha, eps)
                q = score_pairs(tables, s['codes'], neg_a, neg_b, alpha, eps)
                grid.append({'sid': s['sid_sha256'], 'alpha': alpha, 'eps': eps, 'auc': auc_rank(p, q)})
    best = {}
    for alpha in ALPHAS:
        for eps in EPSILONS:
            vals = [g['auc'] for g in grid if g['alpha'] == alpha and g['eps'] == eps and g['auc'] is not None]
            best[(alpha, eps)] = float(np.mean(vals)) if vals else -1
    (alpha_sel, eps_sel), best_auc = max(best.items(), key=lambda kv: kv[1])
    # final tables on all training edges for every SID
    od_all = np.clip(np.searchsorted(np.percentile(out_deg_all[anchor[unseen]], np.arange(10, 100, 10)),
                                     out_deg_all[anchor[unseen]]), 0, 9)
    idg_all = np.clip(np.searchsorted(np.percentile(in_deg_all[target[unseen]], np.arange(10, 100, 10)),
                                      in_deg_all[target[unseen]]), 0, 9)
    rows = []
    for s in sids:
        tabs = build_tables(s['codes'], ea, eb, ew.astype(np.float64))
        sc_pos = score_pairs(tabs, s['codes'], anchor[unseen], target[unseen], alpha_sel, eps_sel)
        sc_neg = score_pairs(tabs, s['codes'], anchor[unseen], neg[unseen], alpha_sel, eps_sel)
        sc_rev = score_pairs(tabs, s['codes'], target[unseen], anchor[unseen], alpha_sel, eps_sel)
        auc = auc_rank(sc_pos, sc_neg)
        # degree-matched cells (deciles precomputed once)
        num = den = 0.0
        cell = od_all * 10 + idg_all
        for o in range(10):
            for i in range(10):
                m = cell == (o * 10 + i)
                if m.sum() > 5:
                    num += (sc_pos[m].mean() - sc_neg[m].mean()) * m.sum(); den += m.sum()
        margin_dm = num / den if den else None
        rows.append({'sid_sha256': s['sid_sha256'], 'arms': s['arms'], 'label': s['label'],
                     'dec_auc_unseen': auc, 'dec_margin_degmatched': margin_dm,
                     'direction_accuracy': float(np.mean(sc_pos > sc_rev)),
                     'mean_pos': float(sc_pos.mean()), 'mean_neg': float(sc_neg.mean())})
    out = {'schema': 'genrec.issue280.decoder/1', 'n_unseen': int(unseen.sum()),
           'calibration': {'sids': cal_sids, 'alpha': alpha_sel, 'eps': eps_sel,
                           'grid': {f'{a},{e}': v for (a, e), v in best.items()}},
           'rows': rows}
    (P / 'DECODER.json').write_text(json.dumps(out, indent=1) + '\n')
    (P / 'SID_DECODER_FEATURES.json').write_text(json.dumps({'schema': 'genrec.issue280.sid_features/1',
                                                              'rows': rows}, indent=1) + '\n')
    aucs = [r['dec_auc_unseen'] for r in rows if r['dec_auc_unseen'] is not None]
    print(json.dumps({'n_sids': len(rows), 'n_unseen': int(unseen.sum()), 'alpha': alpha_sel, 'eps': eps_sel,
                      'calibration_auc': best_auc,
                      'auc_mean': float(np.mean(aucs)), 'auc_min': float(np.min(aucs)),
                      'auc_max': float(np.max(aucs)),
                      'direction_acc_mean': float(np.mean([r['direction_accuracy'] for r in rows]))}, indent=1))


if __name__ == '__main__':
    main()
