"""Issue #279 — per-SID higher-order fidelity features (frozen candidates in CONTRACT.md §4).

For every SID in the #276 table with a readable SID file:
  * L1/L1+L2 h+/h- on the frozen validation rows (same legal negative definition),
  * `er_unseen_2hop`  = weighted ER over unseen transitions (w = 1 + log1p(two_hop)),
  * `dp_unseen_degmatch` = degree-matched high-minus-low sharing difference inside the unseen stratum,
  * descriptive: unseen uniform ER, seen/unseen macros.
Writes SID_FEATURES.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import importlib.util
spec = importlib.util.spec_from_file_location('rl', 'results/research/issue279/rellevel279.py')
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)

P = Path('results/research/issue279')
N = rl.N


def two_hop_all(anchor, target, succ, pred, c_from, c_to):
    out = np.zeros(len(anchor))
    for i, (a, b) in enumerate(zip(anchor.tolist(), target.tolist())):
        mids = succ.get(a, set()) & pred.get(b, set())
        if mids:
            ca = c_from.get(a, {}); ct = c_to.get(b, {})
            out[i] = sum(ca[m] * ct[m] for m in mids)
    return out


def main() -> None:
    A, B, C, succ, pred, c_from, c_to = rl.load_graph()
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    two_hop = two_hop_all(anchor, target, succ, pred, c_from, c_to)
    direct = np.array([c_from.get(int(a), {}).get(int(b), 0) for a, b in zip(anchor, target)])
    unseen = direct == 0
    out_deg = np.array([len(succ.get(int(a), set())) for a in anchor])
    in_deg = np.array([len(pred.get(int(b), set())) for b in target])
    od = np.clip(np.searchsorted(np.percentile(out_deg, np.arange(10, 100, 10)), out_deg), 0, 9)
    idg = np.clip(np.searchsorted(np.percentile(in_deg, np.arange(10, 100, 10)), in_deg), 0, 9)
    pos_th = two_hop[unseen & (two_hop > 0)]
    thresh = float(np.median(np.log1p(pos_th))) if len(pos_th) else None
    high = unseen & (two_hop > 0) & (np.log1p(two_hop) >= thresh) if thresh is not None else np.zeros(n, bool)
    low = unseen & (two_hop > 0) & (np.log1p(two_hop) < thresh) if thresh is not None else np.zeros(n, bool)
    w2 = 1.0 + np.log1p(two_hop)
    cid = od * 10 + idg

    # excluded items flattened (for h-)
    row_idx, item_idx, n_exc = [], [], np.zeros(n, dtype=np.int64)
    for i, r in enumerate(va):
        s = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        n_exc[i] = len(s)
        row_idx.extend([i] * len(s)); item_idx.extend(s)
    row_idx = np.asarray(row_idx); item_idx = np.asarray(item_idx)
    population = N - n_exc
    eff = population > 0

    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    seen = set()
    rows_out = []
    for g in table['groups']:
        for r in g['rows']:
            sha = r['sid_sha256']
            if sha in seen:
                continue
            seen.add(sha)
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
            item = {'sid_sha256': sha, 'arms': [a.get('arm') for a in r.get('arms', [])][:3],
                    'label_recall@10': r.get('label_recall@10'), 'label_strength': r.get('label_strength')}
            for level, prefix in (('L1', codes[:, 0]), ('L2', codes[:, 0] * 256 + codes[:, 1])):
                cnt = np.bincount(prefix, minlength=int(prefix.max()) + 1)
                hp = (prefix[anchor] == prefix[target]).astype(np.float64)
                same = (prefix[item_idx] == prefix[anchor][row_idx]).astype(np.float64)
                excl = np.bincount(row_idx, weights=same, minlength=n)
                hn = np.where(eff, (cnt[prefix[anchor]] - excl) / np.maximum(population, 1), 0.0)
                def wer(mask, w):
                    m = mask & eff & (w > 0)
                    sw = w[m].sum()
                    if sw <= 0:
                        return None, None, None
                    P = float((w[m] * hp[m]).sum() / sw); Nw = float((w[m] * hn[m]).sum() / sw)
                    return P, Nw, (P / Nw if Nw > 0 else None)
                Pw, Nw, ER = wer(unseen, w2)
                Pu, Nu, ERu = wer(unseen, np.ones(n))
                Pc, Nc, ERc = wer(np.ones(n, bool), np.ones(n))
                # degree-matched high-minus-low sharing difference in the unseen stratum
                hh = high & eff
                ll = low & eff
                ch = np.bincount(cid[hh], minlength=100)
                cl = np.bincount(cid[ll], minlength=100)
                both = (ch > 0) & (cl > 0)
                if both.any():
                    sh = np.bincount(cid[hh], weights=hp[hh], minlength=100)
                    sl = np.bincount(cid[ll], weights=hp[ll], minlength=100)
                    dcell = sh[both] / ch[both] - sl[both] / cl[both]
                    wts = (ch[both] + cl[both]).astype(float)
                    dp = float((dcell * wts).sum() / wts.sum())
                else:
                    dp = None
                item[level] = {'er_unseen_2hop': ER, 'P_unseen_2hop': Pw, 'N_unseen_2hop': Nw,
                               'er_unseen_uniform': ERu, 'er_all_uniform': ERc,
                               'dp_unseen_degmatch': dp,
                               'n_high': int(hh.sum()), 'n_low': int(ll.sum()),
                               'n_unseen': int((unseen & eff).sum())}
            rows_out.append(item)
    (P / 'SID_FEATURES.json').write_text(json.dumps({
        'schema': 'genrec.issue279.sid_features/1', 'threshold_log1p_median': thresh,
        'n_sids': len(rows_out), 'rows': rows_out}, indent=1) + '\n')
    print('sids', len(rows_out), 'threshold', thresh, 'n_unseen', int(unseen.sum()),
          'n_high', int(high.sum()), 'n_low', int(low.sum()))
    sample = rows_out[0]
    print(json.dumps({k: (sample[k] if k != 'L2' else '...') for k in sample if k != 'L2'}, indent=1)[:600])


if __name__ == '__main__':
    main()
