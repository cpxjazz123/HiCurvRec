"""Issue #282 — pair-level directed types and per-SID type-conditioned L1 fidelity (CONTRACT.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

P = Path('results/research/issue282')
N = 24587
SEED = 42
K_NEG = 5
MAX_ATTEMPT = 20
MIN_SUPPORT = 100


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
    A = (uk // N).astype(np.int64); B = (uk % N).astype(np.int64)
    from collections import defaultdict
    c_from = defaultdict(set); c_to = defaultdict(set)
    for a, b in zip(A.tolist(), B.tolist()):
        c_from[a].add(b); c_to[b].add(a)
    out = np.bincount(A, minlength=N); inn = np.bincount(B, minlength=N)
    return A, B, cnt.astype(np.int64), c_from, c_to, out, inn


def type_counts(a, b, c_from, c_to):
    sa, sb = c_from.get(a, set()), c_from.get(b, set())
    pa, pb = c_to.get(a, set()), c_to.get(b, set())
    return {'chain': len(sa & pb), 'revchain': len(sb & pa),
            'merge': len(sa & sb), 'fork': len(pa & pb)}


def main() -> None:
    A, B, C, c_from, c_to, out, inn = load_graph()
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    seen = np.array([t in c_from.get(int(a), set()) for a, t in zip(anchor, target)])
    od = np.clip(np.searchsorted(np.percentile(out[anchor], np.arange(10, 100, 10)), out[anchor]), 0, 9)
    idg = np.clip(np.searchsorted(np.percentile(inn[target], np.arange(10, 100, 10)), inn[target]), 0, 9)
    types = ['chain', 'revchain', 'merge', 'fork']
    tc = {t: np.zeros(n, dtype=np.int64) for t in types}
    for i in range(n):
        d = type_counts(int(anchor[i]), int(target[i]), c_from, c_to)
        for t in types:
            tc[t][i] = d[t]
    rng = np.random.default_rng(SEED)
    out_bounds = np.percentile(out[anchor], np.arange(10, 100, 10))
    in_bounds = np.percentile(inn[target], np.arange(10, 100, 10))
    neg = np.empty((n, K_NEG), dtype=np.int64)
    level = np.zeros((n, K_NEG), dtype=np.int8)
    for i, r in enumerate(va):
        excl = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        a = int(anchor[i])
        pts = {t for t in types if tc[t][i] > 0}
        pod, pid = int(od[i]), int(idg[i])
        for j in range(K_NEG):
            best = None
            for _ in range(MAX_ATTEMPT):
                cand = int(rng.integers(0, N))
                if cand in excl:
                    continue
                cod = int(np.clip(np.searchsorted(out_bounds, out[cand]), 0, 9))
                cid = int(np.clip(np.searchsorted(in_bounds, inn[cand]), 0, 9))
                if (cod, cid) != (pod, pid):
                    continue
                d = type_counts(a, cand, c_from, c_to)
                inter = {t for t in types if d[t] > 0} & pts
                if inter:
                    best = (cand, 2); break
                if best is None:
                    best = (cand, 1)
            if best is None:
                while True:
                    cand = int(rng.integers(0, N))
                    if cand not in excl:
                        best = (cand, 0); break
            neg[i, j] = best[0]; level[i, j] = best[1]

    # per-SID type-conditioned fidelity
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    seen_sha = set()
    rows = []
    for g in table['groups']:
        for r in g['rows']:
            sha = r['sid_sha256']
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            f = (r.get('features') or {}).get('_source')
            if not f or not Path(f).is_file():
                continue
            rec = json.load(open(f))
            sp = (rec.get('sid') or {}).get('path')
            if not sp or not Path(sp).is_file():
                continue
            raw = json.loads(Path(sp).read_text())
            p1 = np.zeros(N, dtype=np.int64)
            for k, v in raw.items():
                p1[int(k)] = int(v[0])
            hp = (p1[anchor] == p1[target]).astype(np.float64)
            hn = (p1[anchor][:, None] == p1[neg]).astype(np.float64)  # per (row, neg)
            row = {'sid_sha256': sha, 'arms': [a.get('arm') for a in r.get('arms', [])][:2],
                   'label': r.get('label_recall@10'), 'label_strength': r.get('label_strength')}
            for t in types:
                mask = tc[t] > 0
                sup = int(mask.sum())
                if sup == 0:
                    row[t] = {'support': 0}
                    continue
                Pv = float(hp[mask].mean())
                Nn = float(hn[mask].mean()) if mask.any() else None
                ER = (Pv / Nn) if (Nn not in (None, 0)) else None
                row[t] = {'support': sup, 'P': Pv, 'N': Nn, 'ER': ER, 'delta': (Pv - Nn if Nn is not None else None)}
            ms = (tc['merge'] > 0); fs = (tc['fork'] > 0); cs = (tc['chain'] > 0)
            def er_of(mask):
                if mask.sum() < 1:
                    return None
                Pv = float(hp[mask].mean()); Nn = float(hn[mask].mean())
                return Pv / Nn if Nn > 0 else None
            em, ef, ec = er_of(ms), er_of(fs), er_of(cs)
            row['F1_merge_minus_chain'] = (em - ec) if (em is not None and ec is not None) else None
            row['F2_fork_minus_chain'] = (ef - ec) if (ef is not None and ec is not None) else None
            row['support_ok'] = bool(ms.sum() >= MIN_SUPPORT and cs.sum() >= MIN_SUPPORT)
            rows.append(row)

    # aggregate audit with anchor-clustered CIs
    rngb = np.random.default_rng(SEED)
    uniq_anchor = np.unique(anchor)
    idx_by_anchor = {a: np.where(anchor == a)[0] for a in uniq_anchor}
    audit = {'schema': 'genrec.issue282.type_audit/1', 'n_rows': n, 'unseen_share': float(1 - seen.mean()),
             'level_distribution': {int(k): int((level == k).sum()) for k in range(3)},
             'types': {}}
    for t in types:
        mask = tc[t] > 0
        audit['types'][t] = {'support_rows': int(mask.sum())}
    # pooled P/N per type with anchor bootstrap (recompute hp/hn on pooled prefix data is SID-specific;
    # the audit uses the *count* of type membership and the per-row h+ of each SID is per-SID, so the
    # aggregate type statistics are reported as the mean of the per-SID values in the table below)
    table_payload = {'schema': 'genrec.issue282.type_table/1', 'types': types, 'min_support': MIN_SUPPORT,
                     'definitions': 'CONTRACT.md section 1', 'rows': rows}
    (P / 'TYPE_TABLE.json').write_text(json.dumps(table_payload, indent=1) + '\n')
    # aggregate per type over SIDs: mean P/N/ER of the per-SID values
    agg = {}
    for t in types:
        vals = [r[t] for r in rows if r[t].get('support', 0) > 0]
        agg[t] = {'n_sids': len(vals),
                  'mean_P': float(np.mean([v['P'] for v in vals])) if vals else None,
                  'mean_N': float(np.mean([v['N'] for v in vals])) if vals else None,
                  'mean_ER': float(np.mean([v['ER'] for v in vals if v['ER'] is not None])) if any(v['ER'] is not None for v in vals) else None,
                  'mean_support_rows': float(np.mean([v['support'] for v in vals])) if vals else None}
    audit['types'] = agg
    audit['row_level'] = {'chain_positive_rows': int((tc['chain'] > 0).sum()),
                          'merge_positive_rows': int((tc['merge'] > 0).sum()),
                          'fork_positive_rows': int((tc['fork'] > 0).sum()),
                          'revchain_positive_rows': int((tc['revchain'] > 0).sum()),
                          'closed_seen_rows': int(seen.sum())}
    (P / 'TYPE_AUDIT.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(json.dumps(audit, indent=1)[:1500])
    f1 = [r['F1_merge_minus_chain'] for r in rows if r['F1_merge_minus_chain'] is not None]
    f2 = [r['F2_fork_minus_chain'] for r in rows if r['F2_fork_minus_chain'] is not None]
    print('F1 n', len(f1), 'mean', float(np.mean(f1)) if f1 else None)
    print('F2 n', len(f2), 'mean', float(np.mean(f2)) if f2 else None)


if __name__ == '__main__':
    main()
