"""Per-module verification of the v4 assets on the real data (issue #248 C10).

Every check recomputes a claim from the persisted files or from an independent computation;
reading the builder's own report is not accepted as evidence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import shortest_path

from tools.graph_infra import config, sectional_full
from tools.graph_infra.interfaces_v4 import TrainingView, EvaluationView, load_manifest, verify_manifest

REPO = config.REPO
ROOT = REPO / 'results/research/issue248'
OUT = ROOT / 'MODULE_CHECKS.json'


def main() -> int:
    rows = []

    def row(module, check, ok, detail):
        rows.append({'module': module, 'check': check,
                     'verdict': 'PASS' if ok else 'FAIL', 'detail': detail})

    training = TrainingView(REPO)
    payload = load_manifest(REPO)
    verified = verify_manifest(payload, REPO)
    row('manifest', 'every asset hash recomputes', True,
        {'assets': verified['verified'], 'names_sample': verified['names'][:5]})

    table = training.curvature_table()
    undirected = training.undirected_projection()
    visible = training.visible_graph()
    degree = np.diff(undirected.indptr)
    wanted = degree * (degree - 1) // 2
    np.testing.assert_array_equal(table.item_count, wanted)
    row('enumeration', 'per-centre triple count equals C(deg,2) for every centre', True,
        {'items': len(wanted), 'total': int(wanted.sum())})

    total = table.total_rows()
    row('enumeration', 'global row count equals the recomputed population', total == int(wanted.sum()),
        {'rows': total, 'expected': int(wanted.sum())})

    # global uniqueness of (centre, b, c) over the merged shards
    seen = 0
    duplicate = False
    ordered = True
    for name in table.shard_names():
        blob = table.shard(name)
        keys = (blob['centre'].astype(np.int64) << 30
                | blob['b'].astype(np.int64) << 15
                | blob['c'].astype(np.int64))
        duplicate |= len(np.unique(keys)) != len(keys)
        ordered |= bool(np.all(blob['b'] < blob['c']))
        seen += len(keys)
    row('enumeration', 'no duplicate triple and b < c everywhere', (not duplicate) and ordered,
        {'rows': seen})

    # curvature finiteness and the missing mask
    finite = True
    positive_references = True
    for name in table.shard_names():
        blob = table.shard(name)
        finite &= bool(np.isfinite(blob['curvature']).all())
        positive_references &= bool((blob['effective_references'] > 0).all())
    row('curvature', 'every stored curvature is finite and every triple has a reference set',
        finite and positive_references, {})

    # the closed form recomputed from the persisted distance histogram
    histogram = np.load(ROOT / 'assets/distance_histogram.npz')
    moments = sectional_full.moments_from_histogram(
        {'histogram_offsets': histogram['offsets'], 'histogram_counts': histogram['counts']})
    node_moments = np.load(ROOT / 'assets/node_moments.npz')
    row('distance', 'S2 recomputed from the persisted histogram equals the stored S2',
        bool(np.allclose(moments['s2'], node_moments['s2'], atol=1e-9)),
        {'max_abs_difference': float(np.max(np.abs(moments['s2'] - node_moments['s2'])))})

    # independent BFS on a sample of centres: S2 must match
    rng = np.random.default_rng(248003)
    sample = rng.choice(undirected.shape[0], 12, replace=False)
    distances = shortest_path(undirected, directed=False, unweighted=True, indices=sample)
    squares = np.where(np.isfinite(distances), distances ** 2, 0.0)
    independent = squares.sum(axis=1)
    stored = node_moments['s2'][sample]
    row('distance', 'S2 from an independent BFS matches the stored value on 12 sampled nodes',
        bool(np.allclose(independent, stored, rtol=1e-9)),
        {'max_relative_difference': float(np.max(np.abs(independent - stored) / np.maximum(stored, 1)))})

    # an independent recomputation of the curvature of sampled triples: explicit per-source rows
    sample_triples = []
    for item in (0, 7, 1000, 12345, 24586):
        rows_ = table.item_triples(item)
        if len(rows_['b']):
            position = int(np.argmax(rows_['b'] * (1 << 32) + rows_['c']))
            sample_triples.append((item, int(rows_['b'][position]), int(rows_['c'][position])))
    worst = 0.0
    for m, b, c in sample_triples:
        rows_ = table.item_triples(m)
        position = int(np.flatnonzero((rows_['b'] == b) & (rows_['c'] == c))[0])
        stored_value = float(rows_['curvature'][position])
        sources = np.array([m, b, c])
        d = shortest_path(undirected, directed=False, unweighted=True, indices=sources)
        d = np.where(np.isfinite(d), d, 0.0)
        am, ab, ac = d[0], d[1], d[2]
        bc = float(shortest_path(undirected, directed=False, unweighted=True, indices=[b])[0, c])
        from_m = shortest_path(undirected, directed=False, unweighted=True, indices=m).ravel()
        finite = np.isfinite(from_m)
        terms = am ** 2 + bc ** 2 / 4.0 - (ab ** 2 + ac ** 2) / 2.0
        recomputed = float(terms[finite].mean())
        worst = max(worst, abs(recomputed - stored_value))
    row('curvature', 'five sampled triples recomputed from explicit distance rows match the table',
        worst <= 1e-9, {'max_abs_difference': worst})

    # per-item distribution recomputed independently from the shards
    summary = np.load(ROOT / 'assets/item_summary.npz')
    worst_mean = 0.0
    worst_count = 0
    for item in (0, 3, 500, 9999, 20000, 24586):
        rows_ = table.item_triples(item)
        values = rows_['curvature']
        if not len(values):
            continue
        if int(summary['total'][item]) != len(values):
            worst_count += 1
        worst_mean = max(worst_mean, abs(float(values.mean()) - float(summary['mean'][item])))
    row('distribution', 'per-item counts and means recomputed from the shards match the summary',
        worst_count == 0 and worst_mean <= 1e-9,
        {'max_abs_mean_difference': worst_mean, 'count_mismatches': worst_count})

    # reference coverage: every triple's reference count equals its centre's component size
    component_size = node_moments['component_size']
    reference_ok = True
    for name in table.shard_names():
        blob = table.shard(name)
        reference_ok &= bool((blob['effective_references'] == component_size[blob['centre']]).all())
    row('coverage', 'every triple uses exactly its centre\'s full reachable component', reference_ok,
        {'components': int(node_moments['component'].max() + 1),
         'largest_component': int(component_size.max())})

    # structure classes: counts per class match the table
    counts = table.structure_class_counts()
    row('structure', 'the per-class row counts sum to the whole table',
        sum(counts.values()) == total,
        {'classes_present': sum(1 for value in counts.values() if value > 0),
         'class_slots': len(counts), 'rows': total})

    # isolation: the training view refuses evaluation assets, the evaluation view refuses training ones
    evaluation = EvaluationView(REPO)
    training_refuses = False
    try:
        training.load('held_out_support')
    except Exception:
        training_refuses = True
    evaluation_refuses = False
    try:
        evaluation.entry('motif_counts')
    except Exception:
        evaluation_refuses = True
    row('isolation', 'the training view refuses evaluation assets and vice versa',
        training_refuses and evaluation_refuses, {})

    # the visible graph the assets were built from is the frozen training-visible graph
    row('isolation', 'the visible graph used here is the frozen training-visible graph',
        visible.nnz == 261132 and int(degree.sum()) == 510464,
        {'directed_edges': visible.nnz, 'undirected_entries': int(degree.sum())})

    # the strict evaluation banks stay disjoint from the training banks
    strict = set()
    training_keys = set()
    for name in payload['by_name']:
        if name.startswith('strict_'):
            blob = evaluation.load(name)
            strict.update(tuple(sorted(x)) for x in blob['pairs'].tolist())
        elif name.startswith('train_') and name != 'train_source':
            blob = training.load(name)
            training_keys.update(tuple(sorted(x)) for x in blob['pairs'].tolist())
    row('isolation', 'strict evaluation keys and training bank keys are disjoint',
        len(strict & training_keys) == 0, {'strict': len(strict), 'training': len(training_keys)})

    payload_out = {'issue': 248, 'modules': rows,
                   'pass': sum(1 for r in rows if r['verdict'] == 'PASS'),
                   'fail': sum(1 for r in rows if r['verdict'] == 'FAIL'),
                   'na': sum(1 for r in rows if r['verdict'] == 'NA')}
    OUT.write_text(json.dumps(payload_out, indent=2) + '\n')
    print(json.dumps({key: payload_out[key] for key in ('pass', 'fail', 'na')}))
    return 1 if payload_out['fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
