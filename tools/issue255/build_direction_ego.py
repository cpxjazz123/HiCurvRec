"""Build the frozen direction-aware ego layouts for issue #255.

For every item the asset stores the induced subgraph "centre + all
in-neighbours" and "centre + all out-neighbours" of the frozen training-visible
directed graph, in the same array format as the v9 ego operators so the
existing kernels can consume it.  The population is exact and complete: no
sampling, no truncation, no pruning.

Usage (no arguments, project Stage2 interpreter):
    python3.9 -m tools.issue255.build_direction_ego
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
BASE = ROOT / 'assets/v10_direction_ego'
GRAPH_PATH = 'assets/train/visible_forward.npz'
ITEMS_PATH = 'assets/train/item_ids.npy'
GRAPH_SHA256 = 'ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029'
ITEMS_SHA256 = '262f7ce84b271f6232e7c59893580988ae0049606d09263a5f86e3c37e14804e'
VARIANTS = ('in_only', 'out_only')
ARRAY_NAMES = ('nodes', 'offsets', 'node_counts', 'in_ptr', 'in_col', 'out_ptr', 'out_col',
               'in_edge_counts', 'out_edge_counts')


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_csr(path: Path) -> csr_matrix:
    with np.load(path, allow_pickle=False) as archive:
        matrix = csr_matrix((archive['data'], archive['indices'], archive['indptr']),
                            shape=tuple(archive['shape']), copy=False)
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format or not matrix.has_sorted_indices:
        raise ValueError('the frozen graph has duplicate or unsorted indices')
    # A duplicated directed edge or a self loop would make the induced
    # support ambiguous, so the frozen graph must have neither.
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    if len(matrix.indices) and np.any(matrix.indices == rows):
        raise ValueError('the frozen graph contains a self loop')
    return matrix


def local_csr(count: int, edges: np.ndarray, incoming: bool):
    """The v9 local CSR: sorted (row, col) support over the token's local ids."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    row, col = (edges[:, 1], edges[:, 0]) if incoming else (edges[:, 0], edges[:, 1])
    if len(edges) and (edges.min() < 0 or edges.max() >= count):
        raise ValueError('local ego endpoint outside the node population')
    order = np.lexsort((col, row))
    row, col = row[order], col[order]
    if len(row) > 1 and np.any((row[1:] == row[:-1]) & (col[1:] == col[:-1])):
        raise ValueError('duplicate directed edge in the induced support')
    ptr = np.empty(count + 1, dtype=np.int64)
    ptr[0] = 0
    np.cumsum(np.bincount(row, minlength=count), out=ptr[1:])
    return ptr, col.astype(np.int32)


def variant_population(item: int, neighbours: np.ndarray, graph: csr_matrix):
    """The token node set (root first) and the induced directed edges."""
    nodes = np.concatenate(([item], np.sort(neighbours))).astype(np.int32)
    position = {int(node): index for index, node in enumerate(nodes)}
    rows, cols = [], []
    for node in nodes:
        start, stop = graph.indptr[node], graph.indptr[node + 1]
        for target in graph.indices[start:stop]:
            index = position.get(int(target))
            if index is not None:
                rows.append(position[int(node)])
                cols.append(index)
    edges = np.column_stack((np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)))
    return nodes, edges


def build_variant(graph: csr_matrix, reverse: csr_matrix, incoming: bool):
    """Return the nine arrays of one direction variant, plus the audit counters."""
    count = graph.shape[0]
    node_blocks, in_ptr_blocks, in_col_blocks = [], [], []
    out_ptr_blocks, out_col_blocks = [], []
    in_counts, out_counts = [], []
    total_nodes = total_in = total_out = 0
    reciprocal_items = overlap_items = 0
    reciprocal_pairs = reciprocal_pair_tokens = 0
    overlap_incidences = 0
    for item in range(count):
        if incoming:
            neighbours = reverse.indices[reverse.indptr[item]:reverse.indptr[item + 1]]
        else:
            neighbours = graph.indices[graph.indptr[item]:graph.indptr[item + 1]]
        nodes, edges = variant_population(item, neighbours, graph)
        in_ptr, in_col = local_csr(len(nodes), edges, incoming=True)
        out_ptr, out_col = local_csr(len(nodes), edges, incoming=False)
        node_blocks.append(nodes)
        in_ptr_blocks.append(in_ptr)
        in_col_blocks.append(in_col)
        out_ptr_blocks.append(out_ptr)
        out_col_blocks.append(out_col)
        in_counts.append(int(in_ptr[-1]))
        out_counts.append(int(out_ptr[-1]))
        total_nodes += len(nodes)
        total_in += len(in_col)
        total_out += len(out_col)
        # A reciprocal pair is a mutual directed edge pair inside the token's
        # own node set; each unordered pair is counted once.
        support = {(int(row), int(col)) for row, col in edges}
        mutual = sum(1 for row, col in support
                     if row < col and (col, row) in support)
        reciprocal_pairs += mutual
        if mutual:
            reciprocal_pair_tokens += 1
        # The token's neighbour set overlaps the other direction's neighbour
        # set exactly when the item is both an in- and an out-neighbour.
        other = (graph.indices[graph.indptr[item]:graph.indptr[item + 1]]
                 if incoming else
                 reverse.indices[reverse.indptr[item]:reverse.indptr[item + 1]])
        shared = np.intersect1d(neighbours, other)
        overlap_incidences += int(len(shared))
        if len(shared):
            overlap_items += 1
    offsets = np.zeros(count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(block) for block in node_blocks])
    # The v9 operator format indexes the edge pointers by the NODE position,
    # so every item's local row pointer is shifted by its own node and edge
    # base exactly like tools/issue252/ego_operators.py::_write_layout.
    in_ptr = np.zeros(total_nodes + 1, dtype=np.int64)
    out_ptr = np.zeros(total_nodes + 1, dtype=np.int64)
    edge_base = 0
    for index, block in enumerate(node_blocks):
        start = int(offsets[index])
        size = len(block)
        in_ptr[start:start + size + 1] = in_ptr_blocks[index] + edge_base
        out_ptr[start:start + size + 1] = out_ptr_blocks[index] + edge_base
        edge_base += int(in_counts[index])
    arrays = {
        'nodes': np.concatenate(node_blocks).astype(np.int32),
        'offsets': offsets,
        'node_counts': np.asarray([len(block) for block in node_blocks], dtype=np.int64),
        'in_ptr': in_ptr,
        'in_col': np.concatenate(in_col_blocks).astype(np.int32) if total_in else np.zeros(0, np.int32),
        'out_ptr': out_ptr,
        'out_col': np.concatenate(out_col_blocks).astype(np.int32) if total_out else np.zeros(0, np.int32),
        'in_edge_counts': np.asarray(in_counts, dtype=np.int64),
        'out_edge_counts': np.asarray(out_counts, dtype=np.int64),
    }
    audit = {'items': int(count), 'nodes': int(total_nodes), 'in_edges': int(total_in),
             'out_edges': int(total_out),
             'empty_neighbourhood_items': int(sum(1 for c in arrays['node_counts'] if c == 1)),
             'reciprocal_pairs_in_induced_sets': int(reciprocal_pairs),
             'tokens_with_reciprocal_pairs': int(reciprocal_pair_tokens),
             'overlap_items': int(overlap_items),
             'overlap_incidences': int(overlap_incidences)}
    return arrays, audit


def write_variant(variant: str, arrays) -> dict:
    directory = BASE / variant
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ARRAY_NAMES:
        path = directory / (name + '.npy')
        np.save(path, arrays[name], allow_pickle=False)
        written[name] = {'path': str(path.relative_to(ROOT)), 'dtype': str(arrays[name].dtype),
                         'shape': list(arrays[name].shape), 'bytes': path.stat().st_size,
                         'sha256': sha256(path)}
    return written


def audit_variant(variant: str, arrays, graph: csr_matrix, reverse: csr_matrix) -> dict:
    """Independent per-item reconstruction from the CSR, compared with the arrays."""
    offsets = arrays['offsets']
    failures = []
    for item in range(graph.shape[0]):
        neighbours = (reverse.indices[reverse.indptr[item]:reverse.indptr[item + 1]]
                      if variant == 'in_only'
                      else graph.indices[graph.indptr[item]:graph.indptr[item + 1]])
        nodes = np.concatenate(([item], np.sort(neighbours)))
        stored = arrays['nodes'][offsets[item]:offsets[item + 1]]
        if not np.array_equal(stored, nodes):
            failures.append({'item': item, 'reason': 'node set'})
            continue
        position = {int(node): index for index, node in enumerate(nodes)}
        edges = set()
        for node in nodes:
            start, stop = graph.indptr[node], graph.indptr[node + 1]
            for target in graph.indices[start:stop]:
                if int(target) in position:
                    edges.add((position[int(node)], position[int(target)]))
        stored_in = set()
        stored_out = set()
        base = int(offsets[item])
        for row in range(len(nodes)):
            lo = int(arrays['in_ptr'][base + row])
            hi = int(arrays['in_ptr'][base + row + 1])
            for column in arrays['in_col'][lo:hi]:
                stored_in.add((row, int(column)))
            lo = int(arrays['out_ptr'][base + row])
            hi = int(arrays['out_ptr'][base + row + 1])
            for column in arrays['out_col'][lo:hi]:
                stored_out.add((row, int(column)))
        # The incoming direction stores the transposed local edges.
        if stored_in != {(column, row) for row, column in edges}:
            failures.append({'item': item, 'reason': 'induced edges (incoming)'})
        elif stored_out != edges:
            failures.append({'item': item, 'reason': 'induced edges (outgoing)'})
        elif int(arrays['in_edge_counts'][item]) != len(edges):
            failures.append({'item': item, 'reason': 'in-edge count'})
        elif int(arrays['out_edge_counts'][item]) != len(edges):
            failures.append({'item': item, 'reason': 'out-edge count'})
    # The global population must be the exact sum over every token.
    total_edges = int(arrays['in_edge_counts'].sum())
    if total_edges != int(arrays['out_edge_counts'].sum()):
        failures.append({'item': -1, 'reason': 'total edge counts disagree'})
    if total_edges != len(arrays['in_col']) or total_edges != len(arrays['out_col']):
        failures.append({'item': -1, 'reason': 'total edge count disagrees with the columns'})
    if int(arrays['node_counts'].sum()) != len(arrays['nodes']):
        failures.append({'item': -1, 'reason': 'total node count disagrees with the nodes'})
    return {'variant': variant, 'items_checked': int(graph.shape[0]),
            'nodes_checked': int(len(arrays['nodes'])),
            'edges_checked': total_edges,
            'mismatches': len(failures), 'examples': failures[:5],
            'verdict': 'PASS' if not failures else 'FAIL'}


def register_manifest(entries) -> None:
    """Append the new entries; refresh only this asset's own stale registrations."""
    manifest_path = ROOT / 'assets/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    index = {entry['path']: entry for entry in manifest['assets']}
    for entry in entries:
        current = index.get(entry['path'])
        if current is None:
            manifest['assets'].append(entry)
            index[entry['path']] = entry
        else:
            # This asset's own registration is always refreshed from the
            # current builder and arrays; no other asset's entry is touched.
            current.update(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def main() -> int:
    if sha256(ROOT / GRAPH_PATH) != GRAPH_SHA256:
        raise SystemExit('the frozen visible graph hash changed')
    if sha256(ROOT / ITEMS_PATH) != ITEMS_SHA256:
        raise SystemExit('the frozen item id array hash changed')
    items = np.load(ROOT / ITEMS_PATH, allow_pickle=False)
    if not np.array_equal(items, np.arange(len(items))):
        raise SystemExit('the item id array is not the ordered zero-based catalogue')
    graph = load_csr(ROOT / GRAPH_PATH)
    reverse = graph.T.tocsr()
    reverse.check_format(full_check=True)
    BASE.mkdir(parents=True, exist_ok=True)
    written, audits = {}, []
    for variant in VARIANTS:
        arrays, counters = build_variant(graph, reverse, incoming=variant == 'in_only')
        written[variant] = write_variant(variant, arrays)
        audits.append(dict(counters, **audit_variant(variant, arrays, graph, reverse)))
    # Graph-level facts that both variants share: an item with no visible
    # edge at all is isolated, and an item that is both an in- and an
    # out-neighbour puts the same item into both token populations.
    out_degree = np.diff(graph.indptr)
    in_degree = np.diff(reverse.indptr)
    graph_isolated = int(((out_degree == 0) & (in_degree == 0)).sum())
    shared = []
    for item in range(graph.shape[0]):
        incoming = reverse.indices[reverse.indptr[item]:reverse.indptr[item + 1]]
        outgoing = graph.indices[graph.indptr[item]:graph.indptr[item + 1]]
        shared.append(len(np.intersect1d(incoming, outgoing)))
    shared = np.asarray(shared, dtype=np.int64)
    graph_stats = {
        'isolated_items': graph_isolated,
        'items_both_in_and_out_neighbour': int((shared > 0).sum()),
        'in_out_overlap_incidences': int(shared.sum()),
    }
    metadata = {
        'schema': 'genrec.assets.v10_direction_ego/1',
        'issue': 255,
        'use': 'training',
        'graph': {'path': GRAPH_PATH, 'sha256': GRAPH_SHA256},
        'item_ids': {'path': ITEMS_PATH, 'sha256': ITEMS_SHA256},
        'items': int(len(items)),
        'variants': list(VARIANTS),
        'semantics': {
            'node_set': 'the root item first, then the variant neighbours in ascending global id order',
            'edges': 'every real directed visible edge whose both endpoints are inside the token node set',
            'in_ptr/in_col': 'the local CSR of the induced edges in the incoming direction (row=edge target, col=edge source); the pointer is indexed by the NODE position like the v9 operator layouts',
            'out_ptr/out_col': 'the local CSR of the induced edges in the outgoing direction (row=edge source, col=edge target); the pointer is indexed by the NODE position like the v9 operator layouts',
            'empty_neighbourhood': 'the token contains the root only and has no edges; no neighbour is invented',
            'overlap': 'the in-only and out-only node sets are not mutually exclusive',
        },
        'arrays': {variant: written[variant] for variant in VARIANTS},
    }
    (BASE / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    coverage = {
        'schema': 'genrec.assets.v10_direction_ego.coverage/1',
        'items': int(len(items)),
        'population': 'exact and complete: every item, every induced edge, no sampling or truncation',
        'graph': graph_stats,
        'variants': {variant: {key: value for key, value in audit.items() if key != 'examples'}
                      for variant, audit in zip(VARIANTS, audits)},
    }
    (BASE / 'coverage.json').write_text(json.dumps(coverage, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    report = {
        'schema': 'genrec.assets.v10_direction_ego.audit/1',
        'verdict': 'PASS' if all(audit['verdict'] == 'PASS' for audit in audits) else 'FAIL',
        'graph': graph_stats,
        'checks': audits,
        'independence': 'every node set and induced edge set was recomputed from the frozen CSR and compared with the stored arrays',
    }
    (BASE / 'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    entries = []
    for variant in VARIANTS:
        for name, record in written[variant].items():
            entries.append({
                'name': 'direction_%s_%s' % (variant, name),
                'path': record['path'],
                'use': 'training',
                'graph_name': 'infra246-visible-v2',
                'graph_sha256': GRAPH_SHA256,
                'schema': 'npy',
                'dtype': record['dtype'],
                'shape': record['shape'],
                'bytes': record['bytes'],
                'sha256': record['sha256'],
                'origin': 'built by issue #255',
                'command': '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue255.build_direction_ego',
                'inputs': {GRAPH_PATH: GRAPH_SHA256, ITEMS_PATH: ITEMS_SHA256},
                'source_hashes': {'tools/issue255/build_direction_ego.py': sha256(Path(__file__))},
                'content': 'exact induced %s ego layout for every item' % variant,
            })
    for name, description in (('metadata', 'the array semantics and hashes'),
                              ('coverage', 'the exact population and overlap statistics'),
                              ('audit', 'the independent per-item reconstruction verdict')):
        path = BASE / (name + '.json')
        entries.append({
            'name': 'direction_ego_' + name,
            'path': str(path.relative_to(ROOT)),
            'use': 'description',
            'schema': 'json',
            'bytes': path.stat().st_size,
            'sha256': sha256(path),
            'origin': 'built by issue #255',
            'command': '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue255.build_direction_ego',
            'inputs': {GRAPH_PATH: GRAPH_SHA256, ITEMS_PATH: ITEMS_SHA256},
            'content': description,
        })
    register_manifest(entries)
    readme = ROOT / 'assets/README.md'
    section = ('\n## v10 direction-aware ego layouts (issue #255)\n\n'
               '`assets/v10_direction_ego/{in_only,out_only}/` holds, for every one of the 24587 items, the '
               'induced subgraph "centre + all in-neighbours" and "centre + all out-neighbours" of the frozen '
               'training-visible directed graph: `nodes.npy` (the token node set, root first), `offsets.npy` '
               'and `node_counts.npy` (the per-token geometry), `in_ptr.npy`/`in_col.npy` and '
               '`out_ptr.npy`/`out_col.npy` (the local CSR of the induced edges in both directions) and the '
               'matching edge counts. The two populations are exact and complete, are not mutually exclusive '
               'and never invent a neighbour. `metadata.json` states the semantics, `coverage.json` the '
               'population and `audit.json` the independent per-item reconstruction verdict.\n')
    text = readme.read_text(encoding='utf-8')
    if 'v10 direction-aware ego layouts' not in text:
        readme.write_text(text.rstrip('\n') + '\n' + section, encoding='utf-8')
    print(json.dumps({'verdict': report['verdict'], 'variants': {variant: written[variant]['nodes']['sha256']
                                                              for variant in VARIANTS},
                      'checks': [{k: v for k, v in audit.items() if k != 'examples'} for audit in audits]},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
