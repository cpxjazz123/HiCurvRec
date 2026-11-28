"""Issue #280 — low-capacity ordered-code interaction diagnostic (CONTRACT.md §4).

Hashed one-hot of ordered code pairs (a_l, b_l) per level, L2 logistic, trained on 10 fixed SIDs only.
Diagnostic; never used for selection or as the main model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

import importlib.util
spec = importlib.util.spec_from_file_location('d280', 'results/research/issue280/decoder280.py')
d280 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d280)
N = d280.N
P = Path('results/research/issue280')
SEED = 42
B = 4096


def _mix(x):
    """64-bit multiplicative mixing so that (a,b) pairs do not collapse to a target-only key."""
    x = np.asarray(x, dtype=np.uint64)
    return ((x * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(29)) % np.uint64(B)


def features_mixed(codes, A, Bp):
    a = codes[A]; b = codes[Bp]
    cols = [_mix(a[:, l] * 256 + b[:, l]) for l in range(3)]
    rows = np.repeat(np.arange(len(A)), 3)
    cols = np.concatenate(cols)
    return csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(A), 3 * B))


def features_target_only(codes, A, Bp):
    """Control: target prefixes only, no source information -> measures popularity/identity leakage."""
    a = codes[A]; b = codes[Bp]
    cols = [_mix(b[:, 0]), _mix(b[:, 0] * 256 + b[:, 1]), _mix((b[:, 0] * 256 + b[:, 1]) * 256 + b[:, 2])]
    rows = np.repeat(np.arange(len(A)), 3)
    cols = np.concatenate(cols)
    return csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(A), 3 * B))


def features_prefix(codes, A, Bp):
    """Prefix-only interaction: (a1,b1) and (a1a2,b1b2), hashed. No full-code pair, so an item pair
    cannot be memorised; this is the structural version of the diagnostic."""
    a = codes[A]; b = codes[Bp]
    cols = [(a[:, 0] * 256 + b[:, 0]) % B,
            ((a[:, 0] * 256 + a[:, 1]) * 65536 + (b[:, 0] * 256 + b[:, 1])) % B]
    rows = np.repeat(np.arange(len(A)), 2)
    cols = np.concatenate(cols)
    data = np.ones(len(cols), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(A), 2 * B))


def features(codes, A, Bp):
    a = codes[A]; b = codes[Bp]
    cols = [(l * B + (a[:, l] * 256 + b[:, l]) % B) for l in range(3)]
    rows = np.repeat(np.arange(len(A)), 3)
    cols = np.concatenate(cols)
    data = np.ones(len(cols), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(A), 3 * B))


def main() -> None:
    ea, eb, ew = d280.load_graph()
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    keys = ea * N + eb
    pos = np.searchsorted(keys, anchor * N + target); pos = np.clip(pos, 0, len(keys) - 1)
    unseen = keys[pos] != anchor * N + target
    rng = np.random.default_rng(SEED)
    neg = np.empty(n, dtype=np.int64)
    for i, r in enumerate(va):
        excl = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        while True:
            cand = int(rng.integers(0, N))
            if cand not in excl:
                neg[i] = cand; break
    # training sample: 20k edges + 20k non-edges
    idx = rng.choice(len(ea), min(20000, len(ea)), replace=False)
    tr_a, tr_b = ea[idx], eb[idx]
    keyset = set(keys.tolist())
    na, nb = [], []
    while len(na) < 20000:
        a = int(rng.integers(0, N)); b = int(rng.integers(0, N))
        if a == b or (a * N + b) in keyset:
            continue
        na.append(a); nb.append(b)
    na = np.array(na); nb = np.array(nb)
    # 10 fixed SIDs: calibration set + the next five by sha order
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    sids = []
    seen = set()
    for g in table['groups']:
        for r in g['rows']:
            f = (r.get('features') or {}).get('_source')
            if not f or not Path(f).is_file() or r['sid_sha256'] in seen:
                continue
            rec = json.load(open(f)); sp = (rec.get('sid') or {}).get('path')
            if not sp or not Path(sp).is_file():
                continue
            raw = json.loads(Path(sp).read_text())
            codes = np.zeros((N, 4), dtype=np.int64)
            for k, v in raw.items():
                codes[int(k), :len(v)] = v[:4]
            sids.append({'sid_sha256': r['sid_sha256'], 'codes': codes[:N, :3]})
            seen.add(r['sid_sha256'])
    cal = ['025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059']
    high = sorted([s for s in sids if s['sid_sha256'] in
                   {x['sid_sha256'] for x in json.load(open('results/research/issue276/SID_TABLE.json'))['groups'][0]['rows']}], key=lambda x: x['sid_sha256'])
    # explicit 10: canonical + 4 high-recall strong + 5 by sha order
    dec = {r['sid_sha256']: r for r in json.load(open(P / 'SID_DECODER_FEATURES.json'))['rows']}
    strong_high = sorted([s for s in sids if (dec.get(s['sid_sha256'], {}).get('label') or 0) > 0.06],
                         key=lambda s: s['sid_sha256'])[:4]
    chosen = cal + [s['sid_sha256'] for s in strong_high if s['sid_sha256'] != cal[0]]
    for s in sids:
        if len(chosen) >= 10:
            break
        if s['sid_sha256'] not in chosen:
            chosen.append(s['sid_sha256'])
    chosen = chosen[:10]
    out = {'schema': 'genrec.issue280.logistic_diag/1', 'n_sids': len(chosen), 'buckets': B, 'rows': []}
    for sha in chosen:
        s = next(x for x in sids if x['sid_sha256'] == sha)
        Xtr = features(s['codes'], np.r_[tr_a, na], np.r_[tr_b, nb])
        ytr = np.r_[np.ones(len(tr_a)), np.zeros(len(na))]
        clf = LogisticRegression(C=1.0, max_iter=300, solver='liblinear')
        clf.fit(Xtr, ytr)
        Xte = features(s['codes'], np.r_[anchor[unseen], anchor[unseen]], np.r_[target[unseen], neg[unseen]])
        p = clf.predict_proba(Xte)[:, 1]
        n1 = int(unseen.sum())
        auc = d280.auc_rank(p[:n1], p[n1:])
        # structural (prefix-only) variant
        Xtr2 = features_prefix(s['codes'], np.r_[tr_a, na], np.r_[tr_b, nb])
        clf2 = LogisticRegression(C=1.0, max_iter=300, solver='liblinear')
        clf2.fit(Xtr2, ytr)
        Xte2 = features_prefix(s['codes'], np.r_[anchor[unseen], anchor[unseen]], np.r_[target[unseen], neg[unseen]])
        p2 = clf2.predict_proba(Xte2)[:, 1]
        auc2 = d280.auc_rank(p2[:n1], p2[n1:])
        # corrected mixed-hash pair model + target-only leakage control
        Xtr3 = features_mixed(s['codes'], np.r_[tr_a, na], np.r_[tr_b, nb])
        clf3 = LogisticRegression(C=1.0, max_iter=300, solver='liblinear'); clf3.fit(Xtr3, ytr)
        Xte3 = features_mixed(s['codes'], np.r_[anchor[unseen], anchor[unseen]], np.r_[target[unseen], neg[unseen]])
        auc3 = d280.auc_rank(clf3.predict_proba(Xte3)[:n1, 1], clf3.predict_proba(Xte3)[n1:, 1])
        Xtr4 = features_target_only(s['codes'], np.r_[tr_a, na], np.r_[tr_b, nb])
        clf4 = LogisticRegression(C=1.0, max_iter=300, solver='liblinear'); clf4.fit(Xtr4, ytr)
        Xte4 = features_target_only(s['codes'], np.r_[anchor[unseen], anchor[unseen]], np.r_[target[unseen], neg[unseen]])
        auc4 = d280.auc_rank(clf4.predict_proba(Xte4)[:n1, 1], clf4.predict_proba(Xte4)[n1:, 1])
        out['rows'].append({'sid_sha256': sha, 'count_decoder_auc': dec.get(sha, {}).get('dec_auc_unseen'),
                            'logistic_fullpair_auc': auc, 'logistic_prefix_auc': auc2,
                            'logistic_mixed_auc': auc3, 'target_only_auc': auc4,
                            'note': 'full-pair hashed features can memorise an item pair (negative control); '
                                    'prefix-only is the structural diagnostic'})
    (P / 'LOGISTIC_DIAG.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
