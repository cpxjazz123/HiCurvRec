"""Read-only, hash-bound catalogue structure and observed-history diagnostics.

No graph, triples, model parameters, or training assets are constructed here.
CSV/NPZ outputs are evaluation-only and live below issue254/tables.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue254'
N_ITEMS = 24587
TOPOLOGY = 'results/research/issue247/assets/'
MOTIFS = 'results/research/infra246/assets/motifs/'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class BoundAssets:
    """Check the frozen manifest before trusting any individual asset hashes."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.hashes = {}
        evidence_path = self.root / 'results/research/issue254/EVIDENCE.json'
        self.evidence = json.loads(evidence_path.read_text())
        self.hashes['results/research/issue254/EVIDENCE.json'] = sha256(evidence_path)
        manifest_path = self.root / 'assets/manifest.json'
        actual = sha256(manifest_path)
        if actual != self.evidence['asset_manifest']['sha256']:
            raise ValueError('Frozen assets/manifest.json hash mismatch')
        self.hashes['assets/manifest.json'] = actual
        self.manifest = json.loads(manifest_path.read_text())
        self.expected = {x['path']: x['sha256'] for x in self.manifest['assets']}
        for entry in self.evidence['assets']:
            path = entry['path']
            expected = entry['actual_sha256']
            if path in self.expected and self.expected[path] != expected:
                raise ValueError('Evidence/manifest disagreement: ' + path)
            self.expected[path] = expected

    def checked(self, path, expected=None):
        full = Path(path)
        if not full.is_absolute():
            full = self.root / full
        key = str(full.relative_to(self.root))
        expected = expected or self.expected.get(key)
        if not expected:
            raise ValueError('Asset has no frozen hash: ' + key)
        actual = self.hashes.get(key)
        if actual is None:
            actual = sha256(full)
        if actual != expected:
            raise ValueError('Asset hash mismatch: ' + key)
        self.hashes[key] = actual
        return full

    def npz(self, path):
        with np.load(self.checked(path), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}


def validate_item_ids(item_ids, catalogue_ids, expected_size=N_ITEMS):
    """CSR row indices must equal explicit, zero-based global catalogue item IDs."""
    ids = np.asarray(item_ids)
    catalogue = np.asarray(catalogue_ids)
    expected = np.arange(expected_size)
    if ids.dtype.kind not in 'iu' or not np.array_equal(ids, expected):
        raise ValueError('Graph item_ids are not the ordered zero-based catalogue')
    if catalogue.dtype.kind not in 'iu' or not np.array_equal(np.sort(catalogue), expected):
        raise ValueError('Catalogue is missing, duplicated, or has extra item IDs')


def _load_csr(bound, path):
    arrays = bound.npz(path)
    matrix = csr_matrix((arrays['data'], arrays['indices'], arrays['indptr']),
                        shape=tuple(arrays['shape']), copy=False)
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format or not matrix.has_sorted_indices:
        raise ValueError('Frozen CSR has duplicates or unsorted indices: ' + path)
    if not np.isfinite(matrix.data).all() or (matrix.data <= 0).any():
        raise ValueError('Frozen CSR contains nonpositive or invalid weights: ' + path)
    if matrix.diagonal().any():
        raise ValueError('Frozen graph contains self edges: ' + path)
    return matrix


def load_graphs(root=ROOT):
    """Return forward/reverse/undirected CSR, explicit item_ids and checked hashes.

    CSR row indices are global item IDs, proven against both checkpoint-owned
    SID catalogues. The returned dictionary is also accepted by context_features.
    """
    bound = BoundAssets(root)
    item_ids = np.load(bound.checked('assets/train/item_ids.npy'), allow_pickle=False)
    for generation in bound.evidence['generations'].values():
        entry = generation['stage2_sid']
        catalogue = json.loads(bound.checked(entry['path'], entry['sha256']).read_text())
        validate_item_ids(item_ids, [int(key) for key in catalogue])
    forward = _load_csr(bound, 'assets/train/visible_forward.npz')
    reverse = _load_csr(bound, 'assets/train/visible_reverse.npz')
    undirected = _load_csr(bound, 'assets/train/visible_undirected.npz')
    if any(graph.shape != (N_ITEMS, N_ITEMS) for graph in (forward, reverse, undirected)):
        raise ValueError('Graph shape does not match catalogue')
    if (forward.T != reverse).nnz or (undirected != undirected.T).nnz:
        raise ValueError('Frozen reverse or undirected symmetry mismatch')
    if forward.nnz != 261132 or undirected.nnz != 510464:
        raise ValueError('Frozen graph population mismatch')
    return {'forward': forward, 'reverse': reverse, 'undirected': undirected,
            'item_ids': item_ids, 'hashes': bound.hashes, '_bound_assets': bound}


def _item(item, n):
    if not isinstance(item, (int, np.integer)) or not 0 <= int(item) < n:
        raise ValueError('Item must be an integer catalogue ID: ' + repr(item))
    return int(item)


def _neighbors(graph, item):
    return graph.indices[graph.indptr[item]:graph.indptr[item + 1]]


def _weight(graph, source, target):
    start, stop = graph.indptr[source:source + 2]
    position = int(np.searchsorted(graph.indices[start:stop], target)) + int(start)
    if position < stop and graph.indices[position] == target:
        return float(graph.data[position])
    return 0.


def relation_features(forward, undirected, source, target):
    """Observed directed weights and open-neighbour overlap; no inferred edges.

    two_step_walk means a shared undirected neighbour (also possible if adjacent).
    exact_two_hop additionally excludes adjacency and source==target. within_two_hop
    is the closed radius-two relation, so an item is within two hops of itself.
    Jaccard of two empty open neighbour sets is undefined (NaN), not zero.
    """
    n = forward.shape[0]
    source, target = _item(source, n), _item(target, n)
    left, right = _neighbors(undirected, source), _neighbors(undirected, target)
    common = int(np.intersect1d(left, right, assume_unique=True).size)
    union = len(left) + len(right) - common
    st, ts = _weight(forward, source, target), _weight(forward, target, source)
    adjacent = _weight(undirected, source, target) > 0
    return {'source_to_target_edge': st > 0, 'source_to_target_weight': st,
            'target_to_source_edge': ts > 0, 'target_to_source_weight': ts,
            'undirected_adjacent': adjacent, 'common_neighbor_count': common,
            'neighbor_jaccard': common / union if union else float('nan'),
            'undirected_two_step_walk': common > 0,
            'undirected_exact_two_hop': source != target and not adjacent and common > 0,
            'undirected_within_two_hop': source == target or adjacent or common > 0}


def context_features(graphs, history, target):
    """Observed support over full raw history; repeated occurrences stay explicit.

    support_count/weight count history occurrences; unique_count/unique_weight
    count distinct history items. No future or test transition is inserted in CSR.
    """
    forward, undirected = graphs['forward'], graphs['undirected']
    target = _item(target, forward.shape[0])
    history = [_item(item, forward.shape[0]) for item in history]
    unique = set(history)
    weights = {item: _weight(forward, item, target) for item in unique}
    last = history[-1] if history else None
    result = {'history_length': len(history), 'history_unique_length': len(unique),
              'last20_history_length': min(len(history), 20),
              'history_truncated_to_last20': len(history) > 20,
              'last_history_item': last,
              'history_to_target_support_count': sum(weights[item] > 0 for item in history),
              'history_to_target_weight': sum(weights[item] for item in history),
              'history_to_target_support_unique_count': sum(w > 0 for w in weights.values()),
              'history_to_target_unique_weight': sum(weights.values())}
    if last is None:
        result.update(last_to_target_edge=None, last_to_target_weight=float('nan'),
                      target_to_last_edge=None, target_to_last_weight=float('nan'),
                      last_target_jaccard=float('nan'), last_target_common_neighbor_count=None,
                      last_target_adjacent=None, last_target_two_hop=None,
                      last_target_within_two_hop=None)
    else:
        relation = relation_features(forward, undirected, last, target)
        result.update(last_to_target_edge=relation['source_to_target_edge'],
                      last_to_target_weight=relation['source_to_target_weight'],
                      target_to_last_edge=relation['target_to_source_edge'],
                      target_to_last_weight=relation['target_to_source_weight'],
                      last_target_jaccard=relation['neighbor_jaccard'],
                      last_target_common_neighbor_count=relation['common_neighbor_count'],
                      last_target_adjacent=relation['undirected_adjacent'],
                      last_target_two_hop=relation['undirected_exact_two_hop'],
                      last_target_within_two_hop=relation['undirected_within_two_hop'])
    return result


def graph_item_features(forward, reverse, undirected):
    out_degree, in_degree = np.diff(forward.indptr), np.diff(reverse.indptr)
    degree = np.diff(undirected.indptr)
    reciprocal = np.asarray(forward.multiply(reverse).getnnz(axis=1)).ravel()
    if not np.array_equal(degree, out_degree + in_degree - reciprocal):
        raise ValueError('Undirected support disagrees with directed graph')
    ratio = np.divide(reciprocal, degree, out=np.full(len(degree), np.nan), where=degree > 0)
    density = np.full(len(degree), np.nan)
    for item in np.flatnonzero(degree >= 2):
        neighbors = _neighbors(undirected, item)
        # Each undirected edge appears twice. Do not include the ego root.
        density[item] = undirected[neighbors][:, neighbors].nnz / (degree[item] * (degree[item] - 1))
    return {'degree_out': out_degree, 'degree_in': in_degree,
            'degree_undirected': degree, 'reciprocal_degree': reciprocal,
            'reciprocal_ratio': ratio, 'induced_neighbor_density': density,
            'one_hop_size': degree, 'one_hop_closed_size': degree + 1}


def curvature_features(summary):
    """Preserve true zero counts but mask every undefined distribution/moment."""
    missing = np.asarray(summary['valid']) == 0
    result = {}
    counts = ('total', 'valid', 'invalid', 'positive', 'zero', 'negative')
    for key in counts:
        values = np.asarray(summary[key])
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError('Invalid stored curvature count: ' + key)
        result['curvature_' + key] = values.astype(np.int64)
    if not np.array_equal(summary['valid'] + summary['invalid'], summary['total']):
        raise ValueError('Curvature count conservation failed')
    if not np.array_equal(summary['positive'] + summary['zero'] + summary['negative'], summary['valid']):
        raise ValueError('Curvature sign conservation failed')
    for key in ('positive_ratio', 'zero_ratio', 'negative_ratio', 'mean', 'variance', 'minimum', 'maximum', 'quantiles'):
        values = np.array(summary[key], dtype=float, copy=True)
        values[missing] = np.nan
        result['curvature_' + key] = values
    reason = np.asarray(summary['missing_reason']).astype('<U32')
    reason = reason.copy()
    reason[missing & (reason == '')] = 'no_valid_triple'
    classes = np.full(len(missing), 'zero/other', dtype='<U16')
    positive, negative = summary['positive'] > 0, summary['negative'] > 0
    classes[positive & ~negative] = 'positive-only'
    classes[negative & ~positive] = 'negative-only'
    classes[positive & negative] = 'mixed-sign'
    classes[missing] = 'missing'
    result.update(curvature_missing=missing, curvature_missing_reason=reason, curvature_class=classes)
    return result


def sampled_coverage(instances, n, vertices):
    centers, nodes = instances['centers'], instances['nodes']
    if nodes.shape != (len(centers), vertices) or (nodes < 0).any() or (nodes >= n).any():
        raise ValueError('Invalid sampled motif node IDs')
    if (centers < 0).any() or (centers >= n).any() or not (nodes == centers[:, None]).any(axis=1).all():
        raise ValueError('Invalid sampled motif center IDs')
    center_count = np.bincount(centers, minlength=n)
    participation = np.bincount(nodes.ravel(), minlength=n)
    prefix = 'sampled_' + str(vertices) + 'node_'
    return {prefix + 'center_count': center_count, prefix + 'center_covered': center_count > 0,
            prefix + 'participation_count': participation,
            prefix + 'participation_covered': participation > 0}


def _load_vectors(graphs):
    bound = graphs['_bound_assets']
    n = len(graphs['item_ids'])
    moments = bound.npz('assets/node_moments.npz')
    summary = bound.npz('assets/item_summary.npz')
    topology = bound.npz(TOPOLOGY + 'topology_features.npz')
    counts = bound.npz(MOTIFS + 'motif_counts.npz')
    instances3 = bound.npz(MOTIFS + 'motif_instances_visible.npz')
    instances4 = bound.npz(MOTIFS + 'motif_instances_4node_visible.npz')
    topology_ids = np.load(bound.checked(TOPOLOGY + 'item_ids.npy',
                                        bound.expected['assets/train/item_ids.npy']), allow_pickle=False)
    if not np.array_equal(topology_ids, graphs['item_ids']):
        raise ValueError('Topology item ordering mismatch')
    # The small schema is independently reconstructible from hash-bound instances.
    # Verify its historical manifest as well as that exact derived equality.
    supplemental_path = bound.root / TOPOLOGY / 'manifest.json'
    supplemental = json.loads(supplemental_path.read_text())
    bound.hashes[TOPOLOGY + 'manifest.json'] = sha256(supplemental_path)
    supplemental_entries = {x['path']: x['sha256'] for x in supplemental['assets']}
    if supplemental_entries[TOPOLOGY + 'topology_features.npz'] != bound.expected[TOPOLOGY + 'topology_features.npz']:
        raise ValueError('Supplemental topology manifest disagrees with frozen manifest')
    schema_path = bound.checked(TOPOLOGY + 'orbit_schema.npz', supplemental_entries[TOPOLOGY + 'orbit_schema.npz'])
    with np.load(schema_path, allow_pickle=False) as schema:
        orbit_keys = schema['graphlet_orbit_keys']
    observed_keys = np.column_stack((instances3['graphlets'], instances3['orbits']))
    if not np.array_equal(orbit_keys, np.unique(observed_keys, axis=0)):
        raise ValueError('Sampled orbit schema disagrees with bound instances')
    vectors = {'item': graphs['item_ids']}
    vectors.update(graph_item_features(graphs['forward'], graphs['reverse'], graphs['undirected']))
    for key in ('degree_in', 'degree_out', 'reciprocal_degree'):
        if not np.array_equal(vectors[key], moments[key]) or not np.array_equal(vectors[key], topology[key]):
            raise ValueError('Stored degree mismatch: ' + key)
    vectors.update(curvature_features(summary))
    offsets = np.load(bound.checked('assets/v8_runtime_layout/two_hop_offsets.npy'), allow_pickle=False)
    nodes = np.load(bound.checked('assets/v8_runtime_layout/two_hop_nodes.npy'), mmap_mode='r', allow_pickle=False)
    root_local = np.load(bound.checked('assets/v8_runtime_layout/two_hop_root_local.npy'), allow_pickle=False)
    if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(nodes):
        raise ValueError('Invalid frozen two-hop offsets')
    sizes = np.diff(offsets)
    if (sizes < vectors['one_hop_closed_size']).any() or (root_local < 0).any() or (root_local >= sizes).any():
        raise ValueError('Invalid full two-hop population sizes/root positions')
    if not np.array_equal(nodes[offsets[:-1] + root_local], vectors['item']):
        raise ValueError('Two-hop root IDs disagree with catalogue')
    vectors['two_hop_closed_size'] = sizes
    vectors['two_hop_size'] = sizes - 1
    vectors['exact_two_hop_size'] = sizes - vectors['one_hop_closed_size']
    target_offsets = np.load(bound.checked('assets/v8_runtime_layout/target_offsets.npy'), allow_pickle=False)
    if not np.array_equal(np.diff(target_offsets), vectors['curvature_valid']):
        raise ValueError('Full stored target counts disagree with item summary')
    bound.checked('assets/v8_runtime_layout/targets.npy')
    if int(vectors['curvature_total'].sum()) != 28391948:
        raise ValueError('Full curvature population mismatch')
    if not np.array_equal(counts['counts'], topology['family_counts']):
        raise ValueError('Exact family vectors disagree across frozen assets')
    family_counts = topology['family_counts']
    family_total = family_counts.sum(axis=1)
    valid = family_total > 0
    if not np.array_equal(valid, topology['family_valid']):
        raise ValueError('Exact family validity mismatch')
    distribution = np.divide(family_counts, family_total[:, None],
                             out=np.full(family_counts.shape, np.nan), where=valid[:, None])
    if not np.allclose(distribution[valid], topology['family_distribution'][valid], rtol=0, atol=1e-14):
        raise ValueError('Exact family distribution mismatch')
    if not np.array_equal(family_total, vectors['degree_undirected'] * (vectors['degree_undirected'] - 1) // 2):
        raise ValueError('Exact three-node neighbour-pair family coverage mismatch')
    dominant = family_counts.argmax(axis=1)
    names = counts['names']
    vectors.update(family_counts=family_counts, family_distribution=distribution, family_valid=valid,
                   family_total=family_total, dominant_family=np.where(valid, names[dominant], ''),
                   dominant_family_index=np.where(valid, dominant, -1),
                   dominant_family_count=np.where(valid, family_counts[np.arange(n), dominant], 0),
                   dominant_family_ratio=np.where(valid, distribution[np.arange(n), dominant], np.nan))
    orbit_counts = topology['sampled_orbit_counts']
    orbit_total = orbit_counts.sum(axis=1)
    orbit_valid = orbit_total > 0
    if not np.array_equal(orbit_valid, topology['sampled_orbit_valid']):
        raise ValueError('Sampled orbit validity mismatch')
    key_to_column = {tuple(key): col for col, key in enumerate(orbit_keys.tolist())}
    independently_counted = np.zeros_like(orbit_counts)
    columns = np.array([key_to_column[tuple(key)] for key in observed_keys.tolist()])
    np.add.at(independently_counted, (instances3['centers'], columns), 1)
    if not np.array_equal(independently_counted, orbit_counts):
        raise ValueError('Sampled orbit counts disagree with stored instances')
    orbit_distribution = np.divide(orbit_counts, orbit_total[:, None],
                                  out=np.full(orbit_counts.shape, np.nan), where=orbit_valid[:, None])
    if not np.allclose(orbit_distribution[orbit_valid], topology['sampled_orbit_distribution'][orbit_valid], rtol=0, atol=1e-14):
        raise ValueError('Sampled orbit distribution mismatch')
    vectors.update(sampled_orbit_counts=orbit_counts, sampled_orbit_distribution=orbit_distribution,
                   sampled_orbit_valid=orbit_valid, sampled_orbit_total=orbit_total,
                   sampled_orbit_role_coverage=(orbit_counts > 0).sum(axis=1))
    vectors.update(sampled_coverage(instances3, n, 3))
    vectors.update(sampled_coverage(instances4, n, 4))
    return vectors, names, orbit_keys


def _flat_columns(vectors, family_names, orbit_keys):
    columns = {}
    quantile_names = ('q00', 'q25', 'q50', 'q75', 'q100')
    for key, values in vectors.items():
        if values.ndim == 1:
            columns[key] = values
        elif key.startswith('family_'):
            for j, name in enumerate(family_names):
                columns[key + '_' + str(name)] = values[:, j]
        elif key.startswith('sampled_orbit_'):
            for j, (graphlet, orbit) in enumerate(orbit_keys):
                columns[key + '_g' + str(graphlet) + '_o' + str(orbit)] = values[:, j]
        elif key == 'curvature_quantiles':
            for j, name in enumerate(quantile_names):
                columns['curvature_' + name] = values[:, j]
        else:
            raise ValueError('Unlabelled vector: ' + key)
    return columns


def _write_item_csv(path, columns, n):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    with path.open(newline='') as stream:
        rows = csv.DictReader(stream)
        ids = np.array([int(row['item']) for row in rows], dtype=np.int64)
    validate_item_ids(ids, ids, n)


def _write_context(path, ledger, graphs):
    count = 0
    seen_rows, seen_users = set(), set()
    with ledger.open(newline='') as source, path.open('w', newline='') as destination:
        reader = csv.DictReader(source)
        required = {'global_row', 'user', 'target_item', 'history'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Context ledger lacks full-history identity columns')
        writer = None
        for row in reader:
            global_row, user = int(row['global_row']), int(row['user'])
            if global_row in seen_rows or user in seen_users:
                raise ValueError('Duplicate context identity')
            seen_rows.add(global_row); seen_users.add(user)
            history = json.loads(row['history'])
            if not isinstance(history, list):
                raise ValueError('Context history must be a JSON list')
            target = int(row['target_item'])
            result = {'global_row': global_row, 'user': user, 'target_item': target}
            if 'sample_sha256' in row:
                result['sample_sha256'] = row['sample_sha256']
            result.update(context_features(graphs, history, target))
            if writer is None:
                writer = csv.DictWriter(destination, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            count += 1
    if count != 57439 or seen_rows != set(range(57439)):
        raise ValueError('Context ledger does not cover the frozen paired population')
    with path.open(newline='') as stream:
        reloaded = sum(1 for _ in csv.DictReader(stream))
    if reloaded != count:
        raise ValueError('Context CSV reload count mismatch')
    return count


def main():
    graphs = load_graphs()
    vectors, family_names, orbit_keys = _load_vectors(graphs)
    tables = OUTPUT / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = tables / 'structure_items.csv', tables / 'structure_vectors.npz'
    columns = _flat_columns(vectors, family_names, orbit_keys)
    _write_item_csv(csv_path, columns, N_ITEMS)
    np.savez_compressed(npz_path, **vectors, family_names=family_names,
                        sampled_orbit_graphlet_role_keys=orbit_keys)
    with np.load(npz_path, allow_pickle=False) as reloaded:
        for key, expected in vectors.items():
            actual = reloaded[key]
            if expected.dtype.kind == 'f':
                matches = np.array_equal(actual, expected, equal_nan=True)
            else:
                matches = np.array_equal(actual, expected)
            if not matches:
                raise ValueError('Structure NPZ reload mismatch: ' + key)
    ledger = tables / 'paired_user_ledger.csv'
    context_path = tables / 'context_graph.csv'
    context = {'requested_input': str(ledger.relative_to(ROOT)), 'computed': False,
               'reason': 'paired_user_ledger.csv absent; no optional context analysis claimed'}
    if ledger.exists():
        ledger_hash = sha256(ledger)
        rows = _write_context(context_path, ledger, graphs)
        if sha256(ledger) != ledger_hash:
            raise ValueError('Paired ledger changed during context calculation')
        graphs['hashes'][str(ledger.relative_to(ROOT))] = ledger_hash
        context = {'computed': True, 'rows': rows, 'full_history': True,
                   'path': str(context_path.relative_to(ROOT)), 'sha256': sha256(context_path),
                   'reload_verified': True}
    coverage = {key: int(values.sum()) for key, values in vectors.items()
                if values.dtype.kind == 'b'}
    result = {'schema': 'genrec.issue254.structure/1', 'purpose': 'evaluation', 'items': N_ITEMS,
              'input_sha256': graphs['hashes'], 'catalogue_order_verified': True,
              'graph_nnz': {key: int(graphs[key].nnz) for key in ('forward', 'reverse', 'undirected')},
              'coverage': coverage, 'full_stored_triples': int(vectors['curvature_total'].sum()),
              'curvature_valid_triples': int(vectors['curvature_valid'].sum()),
              'family_names': family_names.tolist(), 'family_scope': 'exact all neighbour pairs per center, three nodes',
              'sampled_orbit_graphlet_role_keys': orbit_keys.tolist(),
              'sampled_3node_instances': int(vectors['sampled_3node_center_count'].sum()),
              'sampled_4node_instances': int(vectors['sampled_4node_center_count'].sum()),
              'sampled_scope': 'stored sampled instances only; not exhaustive three/four-node roles',
              'definitions': {'reciprocal_ratio': 'reciprocal_degree / undirected_degree; undefined if degree=0',
                              'induced_neighbor_density': 'undirected edges among open neighbours / choose(degree,2); undefined below degree2',
                              'one_hop_size': 'open neighbours; closed adds root',
                              'two_hop_size': 'full frozen closed radius-two population minus root',
                              'exact_two_hop_size': 'two_hop_size minus one_hop_size',
                              'curvature_quantiles': [0, .25, .5, .75, 1],
                              'missing': 'NaN for undefined ratios/moments/vectors; explicit validity masks; counts remain actual counts',
                              'dominant_family': 'first schema family in a count tie; index=-1 and name empty when invalid',
                              'curvature_class': 'positive/negative-only permit zero-valued triples; mixed-sign requires both signs'},
              'vectors': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in vectors.items()},
              'outputs': {'items_csv': {'path': str(csv_path.relative_to(ROOT)), 'sha256': sha256(csv_path),
                                        'rows': N_ITEMS, 'columns': list(columns)},
                          'vectors_npz': {'path': str(npz_path.relative_to(ROOT)), 'sha256': sha256(npz_path)}},
              'reload_verified': True, 'context': context}
    (OUTPUT / 'STRUCTURE_SUMMARY.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'items': N_ITEMS, 'summary': str((OUTPUT / 'STRUCTURE_SUMMARY.json').relative_to(ROOT)),
                      'context_computed': context['computed']}, sort_keys=True))


if __name__ == '__main__':
    main()
