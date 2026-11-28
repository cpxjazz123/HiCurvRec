"""Issue #287: how much information does the curvature screen add beyond the structural label?

The diagnostic works on the frozen Issue #274 pair index and on a documented sample of the frozen
triplet table (the first three shards).  It reports

  * the coverage of the frozen index (groups, centres, items, the hold-out share);
  * the curvature distributions of the closed and the open class (the cross-structure
    incomparability that Issue #274 measured);
  * the within-class Spearman correlations between the curvature and the structural variables
    (the pair's degree sum, the number of common neighbours, the reciprocal-edge flag);
  * the behavioural profile of the **selected** pairs against the class-wide profile, i.e. whether
    the curvature extreme picks pairs that differ from the rest of their class.

Nothing here trains anything; the result is a measurement, not a mechanism.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
from tools.graph_infra.assets import load_sparse

REPO = Path('.').resolve()
INDEX = REPO / 'results/research/issue274/pairs_index.npz'
SHARDS = ['shard_000.npz', 'shard_001.npz', 'shard_002.npz']
OUT = REPO / 'results/research/issue287'


def common_neighbours(graph, a, b, limit=200000):
    """The number of shared neighbours of two items (row intersection of the sparse graph)."""
    a = np.asarray(a); b = np.asarray(b)
    rows_a = graph[a]; rows_b = graph[b]
    counts = np.zeros(len(a), dtype=np.int64)
    for i in range(len(a)):
        sa = set(rows_a.indices[rows_a.indptr[i]:rows_a.indptr[i + 1]].tolist())
        sb = set(rows_b.indices[rows_b.indptr[i]:rows_b.indptr[i + 1]].tolist())
        counts[i] = len(sa & sb)
    return counts


def main() -> None:
    data = np.load(INDEX)
    forward = load_sparse(REPO / 'assets/train/visible_forward.npz').tocsr()
    reverse = load_sparse(REPO / 'assets/train/visible_reverse.npz').tocsr()
    undirected = ((forward + reverse) > 0).astype(np.int8).tocsr()
    degree = np.diff(undirected.indptr)
    out = {'schema': 'genrec.issue287.screen/1',
           'index': {'path': str(INDEX),
                     'groups': int(len(data['centre'])),
                     'centres': int(len(np.unique(data['centre']))),
                     'holdout_groups': int(data['is_holdout'].sum()),
                     'train_groups': int((~data['is_holdout']).sum()),
                     'items_touched': int(len(np.unique(np.concatenate(
                         [data['pos_b'], data['pos_c'], data['neg_d'], data['neg_e']]))))},
           'sample_shards': SHARDS}
    # the selected pairs' structural profile
    for label, (x, y) in (('selected_positive', (data['pos_b'], data['pos_c'])),
                          ('selected_negative', (data['neg_d'], data['neg_e']))):
        edges = np.asarray((undirected[x, y])).ravel() > 0
        cn = common_neighbours(undirected, x, y) if len(x) <= 200000 else np.zeros(len(x), dtype=np.int64)
        out.setdefault('selected_profile', {})[label] = {
            'pairs': int(len(x)),
            'edge_share': round(float(edges.mean()), 6),
            'degree_sum_mean': round(float((degree[x] + degree[y]).mean()), 4),
            'common_neighbours_mean': round(float(cn.mean()), 6),
            'common_neighbours_median': float(np.median(cn)),
            'curvature_median': round(float(np.median(
                data['pos_curvature'] if label == 'selected_positive' else data['neg_curvature'])), 6)}
    # the class-wide profile and the curvature's within-class correlations
    profile = {'closed': [], 'open': []}
    corr = {'closed': {'degree_sum': [], 'common_neighbours': [], 'reciprocal': []},
            'open': {'degree_sum': [], 'common_neighbours': [], 'reciprocal': []}}
    curvature = {'closed': [], 'open': []}
    reciprocal_flag = {'closed': [], 'open': []}
    for shard in SHARDS:
        blob = np.load(REPO / 'assets/shards' / shard)
        b = blob['b']; c = blob['c']; curv = blob['curvature'].astype(np.float64)
        dist = blob['bc_dist_sq']
        closed = dist == 1
        opened = dist == 4
        for name, mask in (('closed', closed), ('open', opened)):
            if not mask.any():
                continue
            bb = b[mask]; cc = c[mask]
            profile[name].append(np.stack([degree[bb] + degree[cc]], axis=1))
            curvature[name].append(curv[mask])
            # the reciprocal flag of the pair and a capped common-neighbour count
            fwd = np.asarray((forward[bb, cc])).ravel() > 0
            rev = np.asarray((reverse[bb, cc])).ravel() > 0
            reciprocal_flag[name].append((fwd & rev).astype(np.float64))
            sub = np.arange(0, len(bb), max(1, len(bb) // 20000))[:20000]
            cn = common_neighbours(undirected, bb[sub], cc[sub])
            corr[name]['common_neighbours'].append((cn.astype(np.float64), curv[mask][sub]))
    for name in ('closed', 'open'):
        if not profile[name]:
            continue
        deg = np.concatenate(profile[name])[:, 0]
        cur = np.concatenate(curvature[name])
        rec = np.concatenate(reciprocal_flag[name])
        entry = {'rows': int(len(cur)),
                 'curvature_median': round(float(np.median(cur)), 6),
                 'curvature_q25': round(float(np.quantile(cur, 0.25)), 6),
                 'curvature_q75': round(float(np.quantile(cur, 0.75)), 6),
                 'degree_sum_mean': round(float(deg.mean()), 4)}
        if np.std(cur) > 0 and np.std(deg) > 0:
            entry['spearman_curvature_vs_degree_sum'] = round(
                float(stats.spearmanr(cur, deg).correlation), 4)
        if np.std(rec) > 0:
            entry['spearman_curvature_vs_reciprocal'] = round(
                float(stats.spearmanr(cur, rec).correlation), 4)
        for label, chunks in corr[name].items():
            if label == 'common_neighbours':
                if chunks:
                    cn = np.concatenate([c[0] for c in chunks]); cv = np.concatenate([c[1] for c in chunks])
                    if np.std(cn) > 0 and np.std(cv) > 0:
                        entry['spearman_curvature_vs_common_neighbours'] = round(
                            float(stats.spearmanr(cn, cv).correlation), 4)
        out.setdefault('class_profile', {})[name] = entry
    out['interpretation'] = (
        'the curvature is comparable only inside a structure class; its within-class correlation with '
        'the degree sum, the common-neighbour count and the reciprocal flag says whether the screen '
        'adds information beyond the closed/open label, the direct edge, the common neighbours and '
        'the degrees')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'SCREEN_287.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:3000])


if __name__ == '__main__':
    main()
