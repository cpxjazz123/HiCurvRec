"""Build the v4 full-resolution assets (issue #248).

Everything is derived from the frozen training-visible graph. No training, no PPR, no sampling of
triples. The triple table is written in shards by centre range so that one item's rows are
contiguous and randomly accessible; the manifest records hashes, coverage, commands and costs.

    python3 -m tools.graph_infra.build_assets_v4            # build
    python3 -m tools.graph_infra.build_assets_v4 --verify   # verify an existing build
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np

from tools.graph_infra import assets, config, graph_build, graphlet, sectional, sectional_full
from tools.graph_infra.assets import sha256, save_sparse, load_sparse
from tools.graph_infra.sectional import projection

REPO = config.REPO
ROOT = REPO / 'results/research/issue248'
OUT = ROOT / 'assets'
SHARDS = OUT / 'shards'
TRAIN = OUT / 'train'
V2 = config.ASSET_ROOT
V3 = REPO / 'results/research/issue247/assets'
CENTRES_PER_SHARD = 2000
COMMAND = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.graph_infra.build_assets_v4'
SEED = 248001


def dump(path: Path, payload) -> None:
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(payload, indent=2, default=convert, allow_nan=False) + '\n')


def source_hashes() -> dict:
    names = ['sectional_full.py', 'full_assets.py', 'build_assets_v4.py', 'assets.py',
             'graphlet.py', 'sectional.py', 'interfaces_v3.py']
    return {f'tools/graph_infra/{name}': sha256(REPO / 'tools/graph_infra' / name)
            for name in names}


def audit_inputs() -> dict:
    """Every input is hash-checked and its role declared; nothing is overwritten."""
    frozen = json.loads((V2 / 'manifest.json').read_text())
    for entry in frozen['assets']:
        assert sha256(REPO / entry['path']) == entry['sha256'], ('v2 drift', entry['name'])
    historical = json.loads((V3 / 'manifest.json').read_text())
    for entry in historical['assets']:
        path = REPO / entry['path']
        if path.is_file():
            assert sha256(path) == entry['sha256'], ('v3 drift', entry['name'])
    return {'v2_assets_verified': len(frozen['assets']),
            'v3_historical_assets_verified': sum(1 for e in historical['assets']
                                                 if (REPO / e['path']).is_file()),
            'v2_manifest_sha256': sha256(V2 / 'manifest.json'),
            'v3_manifest_sha256': sha256(V3 / 'manifest.json')}


def build() -> dict:
    start = time.monotonic()
    audit = audit_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    # The training assets are copied so the training view is self-contained. The evaluation
    # assets are NOT copied: the frozen #246 banks are referenced by immutable hash, which keeps
    # the evaluation material physically outside this asset tree and avoids a duplicate upload.
    for directory in (SHARDS, TRAIN):
        directory.mkdir(parents=True, exist_ok=True)

    ids = graph_build.item_index()['ids']
    n = len(ids)
    visible = load_sparse(V2 / 'graph/visible_forward.npz')
    reverse = load_sparse(V2 / 'graph/reverse_visible.npz') if (V2 / 'graph/reverse_visible.npz').is_file() \
        else load_sparse(V2 / 'graph/visible_reverse.npz')
    undirected = projection(visible)
    save_sparse(TRAIN / 'visible_forward.npz', visible)
    save_sparse(TRAIN / 'visible_reverse.npz', reverse)
    save_sparse(TRAIN / 'visible_undirected.npz', undirected)
    np.save(TRAIN / 'item_ids.npy', ids)

    distance_seconds = time.monotonic() - start
    inter_start = time.monotonic()
    inter = sectional_full.distance_intermediates(undirected, chunk=2000)
    inter_seconds = time.monotonic() - inter_start
    np.savez_compressed(OUT / 'distance_histogram.npz',
                        offsets=inter['histogram_offsets'], counts=inter['histogram_counts'])
    np.savez_compressed(OUT / 'node_moments.npz', s1=inter['s1'], s2=inter['s2'],
                        eccentricity=inter['eccentricity'], component=inter['component'],
                        component_size=inter['component_size'],
                        degree_out=np.diff(visible.indptr),
                        degree_in=np.diff(reverse.indptr),
                        reciprocal_degree=np.asarray(visible.multiply(reverse).getnnz(axis=1)))

    projection_keys = sectional_full._directed_edge_keys(undirected)
    directed_keys = sectional_full._directed_edge_keys(visible)

    degree = np.diff(undirected.indptr)
    expected_pairs = degree * (degree - 1) // 2
    total_expected = int(expected_pairs.sum())

    item_shard = np.zeros(n, dtype=np.int32)
    item_start = np.zeros(n, dtype=np.int64)
    item_count = np.zeros(n, dtype=np.int64)
    # Counts are zero when there is no triple; every statistic is NaN, because a missing
    # distribution must never be read as a zero-valued one.
    summary = {key: np.zeros(n) for key in
               ('total', 'valid', 'invalid', 'positive', 'zero', 'negative')}
    for key in ('positive_ratio', 'zero_ratio', 'negative_ratio',
                'mean', 'variance', 'minimum', 'maximum'):
        summary[key] = np.full(n, np.nan)
    summary['quantiles'] = np.full((n, 5), np.nan)
    class_counts = np.zeros(64, dtype=np.int64)
    class_names = np.array([f'class_{i}' for i in range(64)])
    shard_names = []
    produced_total = 0
    shard_bounds = []

    for shard_index, start_centre in enumerate(range(0, n, CENTRES_PER_SHARD)):
        stop_centre = min(start_centre + CENTRES_PER_SHARD, n)
        centres = np.arange(start_centre, stop_centre, dtype=np.int64)
        chunks = sectional_full.triple_chunks(undirected, visible, inter, centres,
                                              projection_keys, directed_keys)
        # triples are generated centre by centre, so the concatenated rows are already grouped
        # by centre and sorted by (centre, b, c); the index is therefore positional.
        merged = {key: np.concatenate([chunk[key] for chunk in chunks])
                  for key in ('centre', 'b', 'c', 'bc_dist_sq', 'curvature',
                              'effective_references', 'valid', 'invalid_reason', 'structure_class')}
        name = f'shard_{shard_index:03d}.npz'
        np.savez_compressed(SHARDS / name, **merged)
        shard_names.append(name)
        shard_bounds.append({'shard': name, 'first_centre': int(start_centre),
                              'last_centre': int(stop_centre - 1), 'rows': int(len(merged['b']))})
        produced_total += len(merged['b'])
        offset = 0
        for chunk in chunks:
            centre = int(chunk['centre'][0]) if len(chunk['centre']) else None
            count = len(chunk['b'])
            if count:
                item_shard[centre] = shard_index
                item_start[centre] = offset
                item_count[centre] = count
                values = chunk['curvature']
                valid = chunk['valid'] & np.isfinite(values)
                finite = values[valid]
                summary['total'][centre] = count
                summary['valid'][centre] = int(valid.sum())
                summary['invalid'][centre] = count - int(valid.sum())
                summary['positive'][centre] = int((finite > sectional.ZERO_TOLERANCE).sum())
                summary['zero'][centre] = int((np.abs(finite) <= sectional.ZERO_TOLERANCE).sum())
                summary['negative'][centre] = int((finite < -sectional.ZERO_TOLERANCE).sum())
                if len(finite):
                    summary['positive_ratio'][centre] = summary['positive'][centre] / len(finite)
                    summary['zero_ratio'][centre] = summary['zero'][centre] / len(finite)
                    summary['negative_ratio'][centre] = summary['negative'][centre] / len(finite)
                    summary['mean'][centre] = finite.mean()
                    summary['variance'][centre] = finite.var()
                    summary['minimum'][centre] = finite.min()
                    summary['maximum'][centre] = finite.max()
                    summary['quantiles'][centre] = np.quantile(finite, [0, .25, .5, .75, 1])
                offset += count
            # centres with fewer than two neighbours produce no chunk at all; their
            # item_count stays 0 and their missing reason is set below.
        for chunk in chunks:
            if len(chunk['structure_class']):
                np.add.at(class_counts, chunk['structure_class'], 1)

    assert produced_total == total_expected, (produced_total, total_expected)

    missing_reason = np.full(n, '', dtype=object)
    for item in range(n):
        if item_count[item] == 0:
            missing_reason[item] = 'isolated' if degree[item] == 0 else 'no_triple'
    np.savez_compressed(OUT / 'item_summary.npz',
                        **{key: value for key, value in summary.items()},
                        missing_reason=np.array(missing_reason, dtype='<U16'))
    np.savez_compressed(OUT / 'index.npz',
                        item_shard=item_shard, item_start=item_start, item_count=item_count,
                        shard_names=np.array(shard_names), total_rows=np.int64(produced_total),
                        structure_class_counts=class_counts, structure_class_names=class_names,
                        expected_pairs_per_centre=expected_pairs,
                        expected_total_rows=np.int64(total_expected))

    # ---- reused assets: referenced by immutable hash, never re-uploaded
    reuse = [
        ('visible_forward_v2', V2 / 'graph/visible_forward.npz', 'csr', 'training'),
        ('visible_reverse_v2', V2 / 'graph/visible_reverse.npz', 'csr', 'training'),
        ('split_users', V2 / 'graph/split_users.npz', 'npz', 'evaluation'),
        ('held_out_support', V2 / 'graph/held_out_support.npz', 'csr', 'evaluation'),
        ('ego_full_1hop_visible', V2 / 'ego/ego_full_1hop_visible.npz', 'npz', 'training'),
        ('ego_sampled_visible', V2 / 'ego/ego_sampled_visible.npz', 'npz', 'training'),
        ('motif_counts', V2 / 'motifs/motif_counts.npz', 'npz', 'training'),
        ('motif_instances_3node', V2 / 'motifs/motif_instances_visible.npz', 'npz', 'training'),
        ('motif_instances_4node_sampled', V2 / 'motifs/motif_instances_4node_visible.npz', 'npz', 'training'),
        ('topology_features_v3', V3 / 'topology_features.npz', 'npz', 'training'),
        ('sectional_summary_v3_sampled', V3 / 'sectional_summary.npz', 'npz', 'training'),
        ('tiger_sid_v3', V3 / 'tiger.json', 'json', 'description'),
    ]
    for path in sorted((V2 / 'pairs').glob('*.npz')):
        if 'pairs' not in np.load(path).files:
            continue  # the v2 PPR cache is never read or referenced (issue #248 forbids PPR)
        reuse.append((path.stem, path, 'npz',
                      'training' if path.stem.startswith('train_') else 'evaluation'))
    reuse.append(('semantic_embeddings', config.EMBEDDINGS, 'npy', 'training'))
    reuse.append(('items_parquet', config.ITEMS, 'parquet', 'training'))

    coverage = {
        'items': n,
        'triples_expected': total_expected,
        'triples_produced': produced_total,
        'triples_per_centre_match': True,
        'centres_with_triples': int((item_count > 0).sum()),
        'items_without_triple': int((item_count == 0).sum()),
        'isolated_items': int((degree == 0).sum()),
        'degree_one_items': int((degree == 1).sum()),
        'eligible_pair_population': total_expected,
        'reference_set': 'every node reachable from the centre (its connected component)',
        'reference_exclusions': 'none: a = m, b, c are valid references, as in the paper sum over V',
        'unreachable_references': 'excluded from the mean and reported as component_outside; never zero-imputed',
        'components': int(inter['component_sizes_by_id'].size),
        'largest_component': int(inter['component_sizes_by_id'].max()),
        'zero_tolerance': sectional.ZERO_TOLERANCE,
        'shards': shard_bounds,
        'reference_sampling': 'none (full reachable reference set)',
        'structure_classes_present': int((class_counts > 0).sum()),
        'distance_histogram': 'exact per-node histogram; every moment is recomputed from it',
        'four_node_assets': 'sampled in #246 (500 centres) and referenced as such; NOT full coverage',
        'ego_assets': '#246 sampled and full 1-hop ego, referenced by hash',
        'ppr': 'not computed, not read, not referenced',
        'storage': {'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
                    'shard_bytes': sum(p.stat().st_size for p in SHARDS.glob('*.npz'))},
        'runtime_seconds': {'total': time.monotonic() - start, 'distance': distance_seconds,
                            'intermediates': inter_seconds},
    }
    dump(OUT / 'coverage.json', coverage)

    entries = []
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or path.name == 'manifest.json':
            continue  # the manifest cannot contain its own hash
        relative = path.relative_to(REPO)
        entries.append({'name': path.stem if path.parent != SHARDS else path.stem,
                        'path': str(relative), 'use': 'training', 'schema': path.suffix.lstrip('.'),
                        'bytes': path.stat().st_size, 'sha256': sha256(path),
                        'origin': 'built by issue #248', 'command': COMMAND})
    for name, path, schema, use in reuse:
        entries.append({'name': name, 'path': str(path.relative_to(REPO)), 'use': use,
                        'schema': schema, 'bytes': path.stat().st_size, 'sha256': sha256(path),
                        'origin': 'reused by immutable hash', 'command': 'none (reference)'})

    manifest = {'issue': 248, 'version': 'v4-full-sectional-248',
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'command': COMMAND, 'seed': SEED,
                'inputs': {'items': sha256(config.ITEMS), 'train': sha256(config.TRAIN),
                           'embeddings': sha256(config.EMBEDDINGS),
                           'visible_forward': sha256(V2 / 'graph/visible_forward.npz'),
                           'v2_manifest': sha256(V2 / 'manifest.json'),
                           'v3_manifest': sha256(V3 / 'manifest.json')},
                'source_hashes': source_hashes(),
                'audit': audit,
                'coverage_file': 'results/research/issue248/assets/coverage.json',
                'assets': entries}
    dump(OUT / 'manifest.json', manifest)
    return {'triples': produced_total, 'shards': len(shard_names),
            'seconds': time.monotonic() - start}


def verify() -> dict:
    start = time.monotonic()
    from tools.graph_infra.full_assets import FullAssets
    assets_api = FullAssets(REPO)
    verified = assets_api.verify()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['source_hashes'] != source_hashes():
        raise AssertionError('the production source changed after the assets were built')
    index = assets_api.index
    n = len(index['item_shard'])
    expected = int(index['expected_total_rows'])
    total = assets_api.total_rows()
    if total != expected:
        raise AssertionError((total, expected))
    # per-centre counts must equal C(deg,2) exactly
    degrees = np.diff(assets_api.undirected_projection().indptr)
    wanted = degrees * (degrees - 1) // 2
    got = index['item_count']
    if not np.array_equal(wanted, got):
        raise AssertionError('per-centre triple count mismatch')
    # global no-duplicate / no-omission check on the merged table
    seen = 0
    for name in assets_api.shard_names():
        blob = assets_api.shard(name)
        # a triple is (centre, b, c); the key must include the centre, otherwise two
        # different centres with the same neighbour pair look like a duplicate.
        keys = (blob['centre'].astype(np.int64) << 30
                | blob['b'].astype(np.int64) << 15
                | blob['c'].astype(np.int64))
        if len(np.unique(keys)) != len(keys):
            raise AssertionError('duplicate triples inside a shard')
        if np.any(blob['b'] >= blob['c']):
            raise AssertionError('b >= c in a stored triple')
        if not np.all(np.isfinite(blob['curvature'])):
            raise AssertionError('non-finite curvature')
        if not np.all(blob['effective_references'] > 0):
            raise AssertionError('a triple without a reachable reference')
        seen += len(keys)
    if seen != expected:
        raise AssertionError((seen, expected))
    report = {'status': 'PASS', 'verified_assets': verified['verified'], 'triples': seen,
              'items': n, 'shards': len(assets_api.shard_names()),
              'structure_classes_present': int((index['structure_class_counts'] > 0).sum()),
              'seconds': time.monotonic() - start}
    dump(ROOT / 'ACCEPTANCE.json', report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        print(json.dumps(build()), flush=True)
    print(json.dumps(verify()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
