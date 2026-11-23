"""Independent, CPU-only reconstruction of the frozen #246 visible graph.

No production graph builder is imported and no graph/target binary is written.
Run with the Stage2 interpreter: python -m tools.issue252.graph_audit.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

AUDIT_VERSION = 'issue252.raw-transitions.v1'
NAME = 'infra246-visible-v2'
OUTPUT = Path('assets/v7_graph_contract')


def _sequences(records):
    longest = {}
    lengths = {}
    for row in records:
        user = int(row['user'])
        sequence = tuple(int(x) for x in row['history']) + (int(row['target']),)
        # Even a shorter tied prefix is ambiguous; do not hide it behind a later row.
        key = (user, len(sequence))
        if key in lengths and lengths[key] != sequence:
            raise ValueError('conflicting equal-length sequences for user %d' % user)
        lengths[key] = sequence
        if user not in longest or len(sequence) > len(longest[user]):
            longest[user] = sequence
    for (user, length), sequence in lengths.items():
        if longest[user][:length] != sequence:
            raise ValueError('conflicting prefix sequence for user %d' % user)
    return longest


def reconstruct_edges(records, allowed_users=None):
    """Count ordered distinct adjacent transitions once per longest user sequence."""
    sequences = _sequences(records)
    allowed = None if allowed_users is None else set(int(x) for x in allowed_users)
    counts = Counter()
    for user, sequence in sequences.items():
        if allowed is not None and user not in allowed:
            continue
        counts.update((a, b) for a, b in zip(sequence, sequence[1:]) if a != b)
    return counts


def classify_difference(full, visible, struct_val, struct_test):
    """Disjoint missing-edge support classes, separately from weight differences."""
    missing = set(full) - set(visible)
    val, test = set(struct_val), set(struct_test)
    return {
        'val_only': len(missing & (val - test)),
        'test_only': len(missing & (test - val)),
        'both': len(missing & val & test),
        'unexplained': len(missing - val - test),
        'visible_only': len(set(visible) - set(full)),
        'weight_difference': int(sum(full.values()) - sum(visible.values())),
        'heldout_weight_on_visible': int(sum(struct_val[e] + struct_test[e] for e in visible)),
        'weight_residual': int(sum(abs(full[e] - visible[e] - struct_val[e] - struct_test[e])
                                   for e in set(full) | set(visible) | val | test)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(root):
    """Read raw inputs, compare every integer weight and freeze their exact identities."""
    import numpy as np
    import pandas as pd
    from scipy import sparse

    root = Path(root)
    bindings = {}

    def bind(relative):
        relative = str(relative)
        path = root / relative
        bindings[relative] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        return path

    def load_json(relative):
        return json.loads(bind(relative).read_text())

    train_path = bind('results/stage0_build_parquet/train.parquet')
    records = pd.read_parquet(train_path).to_dict('records')
    sequences = _sequences(records)
    full = reconstruct_edges(records)
    users = np.array(sorted(sequences), dtype=np.int64)
    permutation = np.random.default_rng(246001).permutation(len(users))
    first, second = round(len(users) * .8), round(len(users) * .9)
    split = {name: np.sort(users[index]) for name, index in (
        ('visible', permutation[:first]), ('struct_val', permutation[first:second]),
        ('struct_test', permutation[second:]))}
    graph_root = 'results/research/infra246/assets/graph/'
    with np.load(bind(graph_root + 'split_users.npz')) as stored:
        for name, values in split.items():
            if not np.array_equal(values, stored[name]):
                raise ValueError('split mismatch: ' + name)
    sets = [set(x.tolist()) for x in split.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)) or set.union(*sets) != set(users):
        raise ValueError('split is not disjoint/complete')
    counters = {name: reconstruct_edges(records, values) for name, values in split.items()}

    def load_csr(path):
        with np.load(bind(path)) as blob:
            return sparse.csr_matrix((blob['data'], blob['indices'], blob['indptr']),
                                     shape=tuple(blob['shape']))

    def sparse_counts(path):
        matrix = load_csr(path).tocoo()
        if not np.all(matrix.data == matrix.data.astype(np.int64)):
            raise ValueError('noninteger graph weights: ' + path)
        return Counter({(int(a), int(b)): int(w) for a, b, w in zip(matrix.row, matrix.col, matrix.data)})

    for name, expected in dict(counters, full_description=full).items():
        if sparse_counts(graph_root + name + '_forward.npz') != expected:
            raise ValueError('support or integer weight mismatch: ' + name)
    canonical = pd.read_parquet(bind('results/research/behavior_graph_directed/edges.parquet'))
    if Counter({(int(a), int(b)): int(w) for a, b, w in canonical.itertuples(index=False, name=None)}) != full:
        raise ValueError('canonical edge weights differ from raw train')
    with np.load(bind('results/research/behavior_graph_directed/graph.npz')) as blob:
        if Counter({(int(a), int(b)): int(w) for a, b, w in zip(blob['rows'], blob['cols'], blob['weight'])}) != full:
            raise ValueError('canonical binary differs from raw train')
    bind('results/research/behavior_graph_directed/metadata.json')
    difference = classify_difference(full, counters['visible'], counters['struct_val'], counters['struct_test'])
    if difference['unexplained'] or difference['visible_only'] or difference['weight_residual']:
        raise ValueError('unexplained graph difference')

    # Future recommendation rows are read only AFTER graph reconstruction, to
    # verify the Stage0 cutoff; they never enter any edge counter.
    valid = pd.read_parquet(bind('results/stage0_build_parquet/valid.parquet'))
    test = pd.read_parquet(bind('results/stage0_build_parquet/test.parquet'))
    valid_rows = {int(row['user']): row for row in valid.to_dict('records')}
    test_rows = {int(row['user']): row for row in test.to_dict('records')}
    for user, sequence in sequences.items():
        vr, tr = valid_rows[user], test_rows[user]
        if tuple(vr['history']) != sequence or tuple(tr['history']) != sequence + (int(vr['target']),):
            raise ValueError('Stage0 recommendation cutoff mismatch: %d' % user)
    if any(tuple(row['seen_history']) != tuple(row['history']) for row in records):
        raise ValueError('train seen_history mismatch')

    manifest = json.loads((root / 'assets/manifest.json').read_text())
    leaves = {}
    for entry in manifest['assets']:
        path = bind(entry['path'])
        if sha256(path) != entry['sha256']:
            raise ValueError('root manifest leaf mismatch: ' + entry['name'])
        leaves[entry['name']] = {key: entry[key] for key in ('path', 'sha256', 'use')}
    # Independently verify strict banks against all training banks and held-out support.
    old = load_json('results/research/infra246/assets/manifest.json')
    train_keys, evaluation = set(), {}
    held = load_csr(graph_root + 'held_out_support.npz')
    blocked = 0
    for entry in old['assets']:
        name = entry['name']
        if not (name.startswith('bank_train_') or name.startswith('bank_strict_')):
            continue
        with np.load(bind(entry['path'])) as blob:
            pairs = blob['pairs']
            keys = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs}
            if name.startswith('bank_train_'):
                train_keys |= keys
                blocked += sum(bool(held[a, b] or held[b, a]) for a, b in pairs)
            else:
                evaluation[name] = keys
    overlaps = {name: len(keys & train_keys) for name, keys in evaluation.items()}
    if blocked or any(overlaps.values()):
        raise ValueError('strict structural bank leakage')
    # Graph lineage is evidenced by leaf hashes, versioned build sources and
    # original manifests, not by importing a builder or recomputing its targets.
    for relative in ('tools/issue252/graph_audit.py', 'tools/graph_infra/graph_build.py',
                     'tools/graph_infra/splits.py', 'tools/graph_infra/config.py',
                     'tools/graph_infra/isolation.py', 'tools/graph_infra/build_assets_v2.py',
                     'tools/graph_infra/build_assets_v4.py', 'tools/graph_infra/build_assets_v5.py',
                     'tools/graph_infra/sectional_full.py',
                     'stage0_build_parquet/build_recbole_parquet.py',
                     'analysis_sid_recall/build_directed_behavior_graph.py'):
        bind(relative)
    # The loader implementation is itself a registered training dependency.
    bind('tools/graph_infra/ego_gnn.py')
    bind('tools/graph_infra/assets.py')
    for relative in ('assets/v5_gnn_geometry/ego_batching_index_report.json',
                     'results/research/issue248/assets/manifest.json'):
        bind(relative)
    family = {}
    for suffix in ('forward', 'reverse', 'undirected'):
        current = load_csr('assets/train/visible_' + suffix + '.npz')
        frozen = load_csr(graph_root + 'visible_' + suffix + '.npz')
        if current.shape != frozen.shape or (current != frozen).nnz:
            raise ValueError('mixed visible graph family: ' + suffix)
        family[suffix] = {'exact_values_equal': True, 'training_dtype': str(current.dtype),
                          'frozen_dtype': str(frozen.dtype)}
    visible = load_csr('assets/train/visible_forward.npz')
    reverse = load_csr('assets/train/visible_reverse.npz')
    undirected = load_csr('assets/train/visible_undirected.npz')
    if (reverse != visible.T).nnz or (undirected != ((visible + visible.T) > 0)).nnz:
        raise ValueError('visible reverse/projection ancestry mismatch')
    indegree, outdegree = np.diff(reverse.indptr), np.diff(visible.indptr)
    reciprocal = np.asarray(visible.multiply(reverse).getnnz(axis=1))
    features = np.column_stack((indegree, outdegree, reciprocal, np.log1p(indegree),
                                np.log1p(outdegree), (indegree + outdegree) == 0)).astype(np.float64)
    with np.load(root / leaves['node_features']['path']) as blob:
        if not np.array_equal(features, blob['features']):
            raise ValueError('node features are not derived from visible graph')
    with np.load(root / leaves['ego_full_1hop_visible']['path']) as blob:
        ego = {key: blob[key] for key in ('nodes', 'edges', 'node_offsets', 'edge_offsets', 'root_local')}
    for centre in range(visible.shape[0]):
        start, stop = ego['node_offsets'][centre:centre + 2]
        nodes = ego['nodes'][start:stop]
        expected_nodes = set(undirected.indices[undirected.indptr[centre]:undirected.indptr[centre + 1]]) | {centre}
        if set(nodes) != expected_nodes or nodes[ego['root_local'][centre]] != centre:
            raise ValueError('ego node ancestry mismatch: %d' % centre)
        start, stop = ego['edge_offsets'][centre:centre + 2]
        block = visible[nodes][:, nodes].tocoo()
        if set(map(tuple, ego['edges'][start:stop])) != set(zip(block.row, block.col)):
            raise ValueError('ego induced-edge ancestry mismatch: %d' % centre)
    with np.load(root / leaves['ego_batching_index']['path']) as blob:
        if not np.array_equal(blob['one_hop_sizes'], np.diff(ego['node_offsets'])):
            raise ValueError('ego batching index ancestry mismatch')
    with np.load(root / leaves['node_moments']['path']) as blob:
        if not np.array_equal(blob['degree_in'], indegree) or not np.array_equal(blob['degree_out'], outdegree):
            raise ValueError('curvature moment graph ancestry mismatch')
    report = {
        'schema': AUDIT_VERSION, 'status': 'PASS', 'name': NAME,
        'split_seed': 246001, 'split_sizes': {k: len(v) for k, v in split.items()},
        'split_disjoint_complete': True,
        'edges': {'full': len(full), **{k: len(v) for k, v in counters.items()}},
        'transition_weights': {'full': sum(full.values()), **{k: sum(v.values()) for k, v in counters.items()}},
        'difference': difference,
        'recommendation_cutoff': {'checked_train_users': len(sequences), 'future_rows_graph_inputs': False},
        'isolation': {'blocked_train_pairs': blocked, 'strict_eval_train_overlaps': overlaps},
        'ancestry': {'features': 'visible forward/reverse degree and reciprocity only',
                     'ego': 'frozen infra246 visible one-hop; visible-only induced two-hop',
                     'curvature': 'build_assets_v4 -> sectional_full: visible undirected full-component references'},
        'visible_copy_comparison': family,
        'ancestry_checks': {'all_visible_ego_nodes_and_induced_edges': visible.shape[0],
                            'node_feature_rows': len(features), 'curvature_moment_degrees': True},
        'bindings': bindings, 'manifest_leaves': leaves,
    }
    return report


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
    report = audit(root)
    directory = root / OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'audit.json'
    if receipt.exists():
        raise FileExistsError('immutable audit already exists: ' + str(receipt))
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    contract = {'schema': 'genrec.named_graph.v1', 'name': NAME, 'audit_version': AUDIT_VERSION,
                'audit': str(OUTPUT / 'audit.json'), 'audit_sha256': sha256(receipt),
                'bindings': report['bindings'], 'manifest_leaves': report['manifest_leaves']}
    (directory / 'contract.json').write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'edges', 'transition_weights', 'difference')}))
    print('contract_sha256=' + sha256(directory / 'contract.json'))


if __name__ == '__main__':
    main()
