"""Issue #283: per-SID joint-pattern statistics over the full 28,391,948-row triplet asset.

Outputs JOIN_TABLE.json (per SID/class/level) and JOIN_SUMMARY.json (aggregate + null models).
All definitions follow CONTRACT.md; nothing here trains or modifies public infra.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, '.')
from results.research.issue283 import joint283 as J
from tools.graph_infra.assets import load_sparse

SHARDS = [f'shard_{i:03d}.npz' for i in range(13)]
NC = len(J.CLASS_NAMES)
GMAX = 1024  # largest observed codebook (cap_l1c512 / wide_l1 use 512/1024)
NL = 3
ALLOWED_TRANS = {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 4), (2, 2), (2, 4),
                 (3, 3), (3, 4), (4, 4)}
G = {}


def load_rows():
    a, m, b, cls, orb = [], [], [], [], []
    for name in SHARDS:
        z = np.load(J.ROWS / name)
        a.append(z['a']); m.append(z['m']); b.append(z['b'])
        cls.append(z['cls']); orb.append(z['orbit'])
    return (np.concatenate(a), np.concatenate(m), np.concatenate(b),
            np.concatenate(cls).astype(np.int8), np.concatenate(orb))


def strata_of_items():
    und = load_sparse('assets/train/visible_undirected.npz').tocsr()
    degree = np.diff(und.indptr).astype(np.int64)
    bounds = np.quantile(degree, np.linspace(0.1, 0.9, 9))
    return np.clip(np.searchsorted(bounds, degree, side='right'), 0, 9).astype(np.int8)


def setup():
    A, M, B, CLS, ORB = load_rows()
    strata = strata_of_items()
    SA, SM, SB = strata[A], strata[M], strata[B]
    G.update(A=A, M=M, B=B, CLS=CLS, ORB=ORB, SA=SA, SM=SM, SB=SB, N=J.N)
    G['support'] = np.bincount(CLS, minlength=NC).astype(np.int64)
    for name, s1, s2 in (('P2am', SA, SM), ('P2mb', SM, SB), ('P2ab', SA, SB)):
        G[name] = np.bincount(CLS.astype(np.int64) * 100 + s1.astype(np.int64) * 10 + s2,
                              minlength=NC * 100).reshape(NC, 10, 10)
    G['P3'] = np.bincount(CLS.astype(np.int64) * 1000 + SA.astype(np.int64) * 100 +
                          SM.astype(np.int64) * 10 + SB, minlength=NC * 1000).reshape(NC, 10, 10, 10)
    G['strata_item'] = strata


def p2_matrix(ngs):
    """E[share] for a pair of items in strata (s1,s2), within-stratum permutation."""
    n_s = ngs.sum(axis=1).astype(np.float64)
    within = np.array([float((g * (g - 1)).sum() / (n * (n - 1))) if n > 1 else 0.0
                       for g, n in zip(ngs, n_s)])
    frac = ngs / np.maximum(n_s, 1)[:, None]
    out = np.zeros((10, 10))
    for s1 in range(10):
        for s2 in range(10):
            out[s1, s2] = within[s1] if s1 == s2 else float(frac[s1] @ frac[s2])
    return out


def p3_tensor(ngs):
    n_s = ngs.sum(axis=1).astype(np.float64)
    frac = ngs / np.maximum(n_s, 1)[:, None]
    same3 = np.array([float((g * (g - 1) * (g - 2)).sum() / (n * (n - 1) * (n - 2)))
                      if n > 2 else 0.0 for g, n in zip(ngs, n_s)])
    same2 = np.zeros((10, GMAX))
    for s in range(10):
        g = ngs[s].astype(np.float64)
        n = n_s[s]
        same2[s] = (g * (g - 1)) / (n * (n - 1)) if n > 1 else 0.0
    out = np.zeros((10, 10, 10))
    for sa in range(10):
        for sm in range(10):
            for sb in range(10):
                if sa == sm == sb:
                    out[sa, sm, sb] = same3[sa]
                elif sa == sm:
                    out[sa, sm, sb] = float(same2[sa] @ frac[sb])
                elif sm == sb:
                    out[sa, sm, sb] = float(same2[sm] @ frac[sa])
                elif sa == sb:
                    out[sa, sm, sb] = float(same2[sa] @ frac[sm])
                else:
                    out[sa, sm, sb] = float(frac[sa] @ (frac[sm] * frac[sb]))
    return out


def null_degrees(codes_col):
    ngs = np.bincount(G['strata_item'].astype(np.int64) * GMAX + codes_col.astype(np.int64),
                     minlength=10 * GMAX).reshape(10, GMAX).astype(np.float64)
    ep = p2_matrix(ngs)
    et = p3_tensor(ngs)
    return ep, et


def per_sid(task):
    sha, path = task
    codes = J.codes_matrix(path)
    A, M, B, CLS, CTR = G['A'], G['M'], G['B'], G['CLS'], G['M']
    N = G['N']
    support = G['support']
    out = {}
    cum = [np.ones(len(A), dtype=bool) for _ in range(3)]
    lcp = [np.zeros(len(A), dtype=np.int8) for _ in range(3)]
    prev_pat = None
    for k in range(1, 4):
        col = codes[:, k - 1]
        ca, cm, cb = codes[A, k - 1], codes[M, k - 1], codes[B, k - 1]
        e_am, e_mb, e_ab = ca == cm, cm == cb, ca == cb
        for i, e in enumerate((e_am, e_mb, e_ab)):
            cum[i] = cum[i] & e
            lcp[i] += cum[i].astype(np.int8)
        pat = J.pattern_of(cum[0], cum[1], cum[2])
        single = J.pattern_of(e_am, e_mb, e_ab)
        counts = np.bincount(CLS.astype(np.int64) * 5 + pat, minlength=NC * 5).reshape(NC, 5)
        if not np.array_equal(counts.sum(axis=1), support):
            raise AssertionError(f'{sha} level {k}: partition counts do not sum to support')
        scounts = np.bincount(CLS.astype(np.int64) * 5 + single, minlength=NC * 5).reshape(NC, 5)
        keys = (CLS.astype(np.int64) * N + CTR) * 5 + pat
        bc = np.bincount(keys, minlength=NC * N * 5).reshape(NC, N, 5)
        sup = bc.sum(axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = bc / sup[..., None]
        macro = np.where(sup[..., None] > 0, rate, np.nan)
        macro_mean = np.nanmean(macro, axis=1)
        n_centres = (sup > 0).sum(axis=1)
        if prev_pat is not None:
            trans = np.bincount(CLS.astype(np.int64) * 25 + prev_pat * 5 + pat,
                                minlength=NC * 25).reshape(NC, 5, 5)
        n_g = np.bincount(col, minlength=GMAX)
        uni = J.null_rates(n_g, N)
        ep, et = null_degrees(col)
        for ci in range(NC):
            S = int(support[ci])
            if S == 0:
                continue
            c = counts[ci]
            name = J.CLASS_NAMES[ci]
            st = out.setdefault(name, {})
            lvl = {'counts': c.tolist(), 'single': scounts[ci].tolist(),
                   'macro': [None if not np.isfinite(v) else round(float(v), 6) for v in macro_mean[ci]],
                   'n_centres': int(n_centres[ci]),
                   'pair': {'am': round(float(c[0] + c[1]) / S, 6),
                            'mb': round(float(c[0] + c[2]) / S, 6),
                            'ab': round(float(c[0] + c[3]) / S, 6)},
                   'null_u': {'pair': round(uni['pair'], 6), 'triple': round(uni['triple'], 6)},
                   'null_d': {'am': round(float((G['P2am'][ci] / S * ep).sum()), 6),
                              'mb': round(float((G['P2mb'][ci] / S * ep).sum()), 6),
                              'ab': round(float((G['P2ab'][ci] / S * ep).sum()), 6),
                              'all': round(float((G['P3'][ci] / S * et).sum()), 6)}}
            lvl['lift'] = {'all_u': round(float(c[0]) / S - uni['triple'], 6),
                           'all_d': round(float(c[0]) / S - lvl['null_d']['all'], 6)}
            if J.SYMMETRIC[ci]:
                lvl['amb_merged'] = [int(c[0]), int(c[1] + c[2]), int(c[3]), int(c[4])]
            st[str(k)] = lvl
            if prev_pat is not None:
                cells = {}
                for a, b in sorted(ALLOWED_TRANS):
                    v = int(trans[ci, a, b])
                    if v:
                        cells[f'{J.PATTERN_NAMES[a]}>{J.PATTERN_NAMES[b]}'] = v
                st[str(k)]['trans'] = cells
                bad = int(trans[ci].sum() - sum(cells.values()))
                if bad:
                    raise AssertionError(f'{sha} class {name} level {k}: {bad} forbidden transitions')
        prev_pat = pat
    code_lcp = (lcp[0].astype(np.int64) * 16 + lcp[1].astype(np.int64) * 4 + lcp[2].astype(np.int64))
    lc = np.bincount(CLS.astype(np.int64) * 64 + code_lcp, minlength=NC * 64).reshape(NC, 64)
    for ci in range(NC):
        if not support[ci]:
            continue
        name = J.CLASS_NAMES[ci]
        d = {}
        for (x, y, z) in J.FEASIBLE_LCP:
            v = int(lc[ci, x * 16 + y * 4 + z])
            if v:
                d[f'{x},{y},{z}'] = v
        infeasible = int(lc[ci].sum() - sum(d.values()))
        if infeasible:
            raise AssertionError(f'{sha} class {name}: {infeasible} infeasible LCP combos')
        out[name]['lcp'] = d
    return sha, {'path': path, 'classes': out}


def worker(task):
    return per_sid(task)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    setup()
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    tasks = []
    seen = set()
    for g in table['groups']:
        for r in g['rows']:
            sha = r['sid_sha256']
            if sha in seen:
                continue
            seen.add(sha)
            src = (r.get('features') or {}).get('_source')
            rec = json.loads(open(src).read())
            tasks.append((sha, rec['sid']['path']))
    if limit:
        tasks = tasks[:limit]
    print(f'{len(tasks)} SIDs', flush=True)
    with mp.Pool(8) as pool:
        results = pool.map(worker, tasks, chunksize=1)
    entries = dict(results)
    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(type(o))
    (J.P / 'JOIN_TABLE.json').write_text(json.dumps(
        {'schema': 'genrec.issue283.join_table/1', 'classes': J.CLASS_NAMES,
         'levels': [1, 2, 3], 'support': {J.CLASS_NAMES[i]: int(G['support'][i]) for i in range(NC)},
         'entries': entries}, separators=(',', ':'), default=_default) + '\n')
    # aggregate summary: pooled counts/macro/transitions/LCP and the marginal identity check
    summary = {'schema': 'genrec.issue283.join_summary/1', 'classes': J.CLASS_NAMES,
               'support': {J.CLASS_NAMES[i]: int(G['support'][i]) for i in range(NC)},
               'levels': {}}
    for k in range(1, 4):
        per_class = {}
        for ci, name in enumerate(J.CLASS_NAMES):
            S = int(G['support'][ci])
            if not S:
                continue
            pooled = np.zeros(5, dtype=np.int64)
            macro = []
            pair = np.zeros(3)
            lift_u = []
            lift_d = []
            trans = np.zeros((5, 5), dtype=np.int64) if k > 1 else None
            lcp = {}
            for sha, rec in entries.items():
                st = rec['classes'].get(name)
                if not st:
                    continue
                lvl = st[str(k)]
                pooled += np.asarray(lvl['counts'])
                if lvl['macro'][0] is not None:
                    m = np.asarray([0 if v is None else v for v in lvl['macro']])
                    macro.append(m)
                pair += np.asarray([lvl['pair']['am'], lvl['pair']['mb'], lvl['pair']['ab']])
                lift_u.append(lvl['lift']['all_u'])
                lift_d.append(lvl['lift']['all_d'])
                if k > 1:
                    for key, v in lvl.get('trans', {}).items():
                        a, b = key.split('>')
                        trans[J.PATTERN_NAMES.index(a), J.PATTERN_NAMES.index(b)] += v
            S_all = pooled.sum()
            p = pooled / S_all
            macro_arr = np.asarray(macro) if macro else None
            per_class[name] = {'pooled_counts': pooled.tolist(),
                               'pooled_rate': [round(float(v), 6) for v in p],
                               'macro_mean': ([round(float(v), 6) for v in np.mean(macro_arr, axis=0)]
                                              if macro_arr is not None else [None] * 5),
                               'pair_mean': [round(float(v / len(entries)), 6) for v in pair],
                               'lift_u_mean': round(float(np.mean(lift_u)), 6),
                               'lift_d_mean': round(float(np.mean(lift_d)), 6)}
            if k > 1:
                per_class[name]['trans_pooled'] = {f'{J.PATTERN_NAMES[a]}>{J.PATTERN_NAMES[b]}': int(trans[a, b])
                                                   for a, b in sorted(ALLOWED_TRANS) if trans[a, b]}
        summary['levels'][str(k)] = per_class
    for ci, name in enumerate(J.CLASS_NAMES):
        lcp = {}
        for sha, rec in entries.items():
            st = rec['classes'].get(name)
            if st:
                for key, v in st['lcp'].items():
                    lcp[key] = lcp.get(key, 0) + v
        summary.setdefault('lcp', {})[name] = lcp
    (J.P / 'JOIN_SUMMARY.json').write_text(json.dumps(summary, indent=1, default=_default) + '\n')
    print(json.dumps({'entries': len(entries), 'classes': len(J.CLASS_NAMES)}, indent=1))


if __name__ == '__main__':
    main()
