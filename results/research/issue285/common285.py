"""Issue #285 shared helpers: frozen fold-local strata, gates, per-SID feature table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
import common284 as C

P = Path('results/research/issue285')
N_ROWS = None


def load_table():
    rows = C.table_rows()
    folds = C.table_rows.folds
    ch = json.load(open('results/research/issue284/CHAIN_TABLE.json'))['entries']
    out = {'rows': rows, 'folds': folds, 'chain': ch}
    out['y'] = np.array([r['y'] for r in rows])
    out['fold'] = np.array([folds['assignment'][r['group']] for r in rows])
    out['strong'] = np.array([r['strong'] for r in rows])
    out['N'] = np.array([r['x8']['l1_neg_hit'] for r in rows])
    out['C'] = np.array([r['x8']['l3_collision'] for r in rows])
    out['P'] = np.array([r['x8']['l1_pos_hit'] for r in rows])
    out['ER'] = np.array([r['x8']['l1_hard_er'] for r in rows])
    out['ALL'] = np.array([ch[r['sha']]['all1_chain'] for r in rows])
    out['ALL_fork'] = np.array([ch[r['sha']].get('all1_fork') or np.nan for r in rows], float)
    out['ALL_merge'] = np.array([ch[r['sha']].get('all1_merge') or np.nan for r in rows], float)
    out['X6'] = np.array([[r['x6'][k] for k in C.FEATS6] for r in rows])
    out['unit'] = C.clusters(rows)
    return out


def fold_edges(values, fold):
    """Per-fold (q1,q2) computed on the training rows only (frozen rule)."""
    edges = {}
    for f in sorted(set(fold.tolist())):
        tr = fold != f
        edges[int(f)] = tuple(float(x) for x in np.quantile(values[tr], [1 / 3, 2 / 3]))
    return edges


def bin_of(values, fold, edges):
    """Return (level, G_binary, G_ramp): level -1 low, 0 mid, +1 high; ramp 1 at the low edge -> 0 at q2."""
    level = np.zeros(len(values), dtype=int)
    ramp = np.zeros(len(values), dtype=float)
    for f, (q1, q2) in edges.items():
        m = fold == f
        level[m] = np.where(values[m] <= q1, -1, np.where(values[m] >= q2, 1, 0))
        if q2 > q1:
            ramp[m] = np.clip((q2 - values[m]) / (q2 - q1), 0.0, 1.0)
        else:
            ramp[m] = 0.0
    return level, ramp


def gates(d):
    """Binary low gates and continuous ramps for N, C from training-fold quantiles."""
    edges_n = fold_edges(d['N'], d['fold'])
    edges_c = fold_edges(d['C'], d['fold'])
    ln, ramp_n = bin_of(d['N'], d['fold'], edges_n)
    lc, ramp_c = bin_of(d['C'], d['fold'], edges_c)
    G_N = (ln == -1).astype(float)
    G_C = (lc == -1).astype(float)
    return {'edges_N': edges_n, 'edges_C': edges_c, 'level_N': ln, 'level_C': lc,
            'G_N': G_N, 'G_C': G_C, 'ramp_N': ramp_n, 'ramp_C': ramp_c}


def cell_labels(level_n, level_c):
    """Pre-registered cells (CONTRACT §2): LL, LN=lowN&~lowC, LC=~lowN&lowC, HH, else other."""
    cells = np.full(len(level_n), 'other', dtype=object)
    cells[(level_n == -1) & (level_c == -1)] = 'LL'
    cells[(level_n == -1) & (level_c != -1)] = 'LN'
    cells[(level_n != -1) & (level_c == -1)] = 'LC'
    cells[(level_n == 1) & (level_c == 1)] = 'HH'
    return cells


def gates_global(d):
    """Global-tercile cross-check only (#284 style); never used for the frozen models."""
    qn = np.quantile(d['N'], [1 / 3, 2 / 3]); qc = np.quantile(d['C'], [1 / 3, 2 / 3])
    ln = np.where(d['N'] <= qn[0], -1, np.where(d['N'] >= qn[1], 1, 0))
    lc = np.where(d['C'] <= qc[0], -1, np.where(d['C'] >= qc[1], 1, 0))
    return qn, qc, ln, lc
