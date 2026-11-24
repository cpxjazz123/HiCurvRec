"""Lossless full-ego CSR layouts and vectorized block-diagonal runtime gathers.

No learned values are persisted. Production reads only prebuilt arrays; only the
explicit rootless synthetic adapter constructs layouts at runtime. Run this module
without arguments to build and audit all egos from registered training leaves.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
DESTINATION = REPO / 'assets/v9_ego_operators'
GRAPH_NAME = 'infra246-visible-v2'
GRAPH_SHA256 = 'ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029'
SOURCE_PATHS = ('tools/issue252/ego_operators.py', 'tools/graph_infra/assets.py')
VERSION = 'issue252.full-ego-csr.v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_csr(n, edges, incoming=False):
    """Sorted integer support; degree normalization is deferred to model dtype."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    row, col = (edges[:, 1], edges[:, 0]) if incoming else (edges[:, 0], edges[:, 1])
    if n > np.iinfo(np.int32).max or (len(edges) and (edges.min() < 0 or edges.max() >= n)):
        raise ValueError('local ego endpoint outside int32 node population')
    order = np.lexsort((col, row))
    row, col = row[order], col[order]
    if len(row) > 1 and np.any((row[1:] == row[:-1]) & (col[1:] == col[:-1])):
        raise ValueError('duplicate directed edge in static graph support')
    ptr = np.empty(n + 1, dtype=np.int64)
    ptr[0] = 0
    np.cumsum(np.bincount(row, minlength=n), out=ptr[1:])
    return ptr, col.astype(np.int32)


def pack_synthetic(store, hops):
    """Tiny rootless stores use the identical packed representation, once only."""
    batch = store.collate(list(range(store.n)), hops=hops, standardized=False)
    nodes = np.asarray(batch['nodes'], dtype=np.int32)
    groups = np.asarray(batch['batch_index'], dtype=np.int64)
    mask = np.asarray(batch['node_mask'], dtype=bool)
    edges = np.asarray(batch['edges'], dtype=np.int64).reshape(-1, 2)
    edges = edges[np.asarray(batch['edge_mask'], dtype=bool)]
    edges = edges[((edges >= 0) & (edges < len(nodes))).all(1)]
    edges = edges[mask[edges[:, 0]] & mask[edges[:, 1]] & (groups[edges[:, 0]] == groups[edges[:, 1]])]
    keep = np.flatnonzero(mask)
    mapping = np.full(len(nodes), -1, dtype=np.int64)
    mapping[keep] = np.arange(len(keep))
    edges = mapping[edges]
    groups = groups[keep]
    if np.any(np.diff(groups) < 0):
        raise ValueError('synthetic collate nodes must be grouped in item order')
    offsets = np.r_[0, np.cumsum(np.bincount(groups, minlength=store.n))].astype(np.int64)
    result = {'nodes': nodes[keep], 'offsets': offsets}
    for direction, incoming in (('in', True), ('out', False)):
        ptr, col = local_csr(len(keep), edges, incoming)
        # Convert global concatenated columns back into local block columns.
        rows = np.repeat(np.arange(len(keep)), np.diff(ptr))
        col = (col - offsets[groups[rows]]).astype(np.int32)
        result[direction + '_ptr'], result[direction + '_col'] = ptr, col
    return result


class DeviceLayout:
    """Static integer topology resident once per store/device (no activations)."""
    def __init__(self, arrays, device):
        for name, array in arrays.items():
            # Own tensor storage: read-only memmaps must never back writable tensors.
            tensor = torch.tensor(np.asarray(array), device=device,
                                  dtype=torch.int32 if name.endswith('_col') or name == 'nodes' else torch.int64)
            setattr(self, name, tensor)


def _ranges(starts, lengths):
    ends = lengths.cumsum(0)
    bases = ends - lengths
    if len(starts) == 0:
        return starts.new_empty(0), bases
    return (torch.repeat_interleave(starts - bases, lengths)
            + torch.arange(ends[-1], device=starts.device)), bases


def batch_operators(layout, ids, dtype):
    """Gather complete blocks, without item loops or CPU node/edge expansion."""
    starts = layout.offsets[ids]
    sizes = layout.offsets[ids + 1] - starts
    indices, bases = _ranges(starts, sizes)
    groups = torch.repeat_interleave(torch.arange(len(ids), device=ids.device), sizes)
    nodes = layout.nodes[indices].long()
    operators = []
    for direction in ('in', 'out'):
        ptr, col = getattr(layout, direction + '_ptr'), getattr(layout, direction + '_col')
        degree = ptr[indices + 1] - ptr[indices]
        edge_indices, _ = _ranges(ptr[indices], degree)
        crow = torch.cat((degree.new_zeros(1), degree.cumsum(0)))
        columns = col[edge_indices].long()
        columns += torch.repeat_interleave(bases[groups], degree)
        values = torch.repeat_interleave(degree.to(dtype).clamp_min(1).reciprocal(), degree)
        operators.append(torch.sparse_csr_tensor(crow, columns, values,
                                                size=(len(indices), len(indices)), device=ids.device))
    return nodes, groups, sizes, *operators


def load_layouts(root):
    """Load frozen layouts only; missing production assets fail closed."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {entry['name']: entry for entry in manifest['assets']}
    metadata = json.loads((root / 'v9_ego_operators/metadata.json').read_text())
    if (metadata.get('version') != VERSION or metadata.get('graph_name') != GRAPH_NAME
            or metadata.get('graph_sha256') != GRAPH_SHA256 or metadata.get('use') != 'training'):
        raise ValueError('wrong full-ego operator graph identity')
    expected = {'ego%d_%s_%s' % (hops, direction, part)
                for hops in (1, 2) for direction in ('in', 'out') for part in ('ptr', 'col')}
    if {entry['name'] for entry in metadata['assets']} != expected:
        raise ValueError('incomplete full-ego operator inventory')
    for entry in metadata['assets']:
        registered = entries.get(entry['name'])
        if registered is None or any(registered.get(k) != entry[k] for k in ('path', 'sha256', 'bytes')):
            raise ValueError('unregistered full-ego operator: ' + entry['name'])
    for path, digest in metadata['inputs'].items():
        if not any(entry['path'] == path and entry['sha256'] == digest and entry.get('use') == 'training'
                   for entry in entries.values()):
            raise ValueError('unregistered full-ego source input: ' + path)
    for path, digest in metadata['source_hashes'].items():
        if sha256(root.parent / path) != digest:
            raise ValueError('changed full-ego operator source: ' + path)
    frozen_entry = entries['ego_full_1hop_visible']
    with np.load(root.parent / frozen_entry['path'], allow_pickle=False) as frozen:
        one_nodes, one_offsets = frozen['nodes'], frozen['node_offsets']
    arrays = []
    for hops in (1, 2):
        layout = {'nodes': one_nodes if hops == 1 else np.load(root.parent / entries['two_hop_nodes']['path'], mmap_mode='r'),
                  'offsets': one_offsets if hops == 1 else np.load(root.parent / entries['two_hop_offsets']['path'], mmap_mode='r')}
        for direction in ('in', 'out'):
            for part in ('ptr', 'col'):
                key = direction + '_' + part
                layout[key] = np.load(root.parent / entries['ego%d_%s' % (hops, key)]['path'], mmap_mode='r')
        arrays.append(layout)
    return arrays


def _inputs():
    entries = {entry['name']: entry for entry in json.loads((REPO / 'assets/manifest.json').read_text())['assets']}
    names = ('visible_forward', 'ego_full_1hop_visible', 'two_hop_nodes', 'two_hop_offsets', 'item_ids')
    selected = {name: entries[name] for name in names}
    for name, entry in selected.items():
        path = REPO / entry['path']
        if entry.get('use') != 'training' or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('changed or non-training input: ' + name)
    if selected['visible_forward']['sha256'] != GRAPH_SHA256:
        raise ValueError('wrong visible graph')
    return selected, {entry['path']: entry['sha256'] for entry in selected.values()}


def _populations(entries):
    with np.load(REPO / entries['ego_full_1hop_visible']['path'], allow_pickle=False) as frozen:
        one = {key: frozen[key] for key in ('nodes', 'node_offsets', 'edges', 'edge_offsets')}
    two = {key: np.load(REPO / entries['two_hop_' + key]['path'], mmap_mode='r', allow_pickle=False)
           for key in ('nodes', 'offsets')}
    return one, two


def _blocks(hops, one, two, graph):
    offsets = one['node_offsets'] if hops == 1 else two['offsets']
    nodes = one['nodes'] if hops == 1 else two['nodes']
    for item, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        population = nodes[start:stop]
        if hops == 1:
            first, last = one['edge_offsets'][item:item + 2]
            edges = one['edges'][first:last]
        else:
            block = graph[population][:, population].tocoo(copy=False)
            edges = np.column_stack((block.row, block.col))
        yield int(start), int(stop), edges


def _write_layout(hops, one, two, graph):
    offsets = one['node_offsets'] if hops == 1 else two['offsets']
    # First pass counts only: bound peak RAM to one complete ego.
    total_edges = sum(len(edges) for _, _, edges in _blocks(hops, one, two, graph))
    arrays = {}
    for direction in ('in', 'out'):
        for part, dtype, size in (('ptr', np.int64, int(offsets[-1]) + 1), ('col', np.int32, total_edges)):
            key = direction + '_' + part
            arrays[key] = np.lib.format.open_memmap(DESTINATION / ('ego%d_%s.npy' % (hops, key)),
                                                    mode='w+', dtype=dtype, shape=(size,))
    cursor = 0
    for start, stop, edges in _blocks(hops, one, two, graph):
        for direction, incoming in (('in', True), ('out', False)):
            ptr, col = local_csr(stop - start, edges, incoming)
            arrays[direction + '_ptr'][start:stop + 1] = ptr + cursor
            arrays[direction + '_col'][cursor:cursor + len(edges)] = col
        cursor += len(edges)
    for array in arrays.values():
        array.flush()


def audit():
    """All egos: compare every sorted local edge and row pointer to frozen inputs."""
    from tools.graph_infra.assets import load_sparse
    entries, inputs = _inputs()
    metadata = json.loads((DESTINATION / 'metadata.json').read_text())
    sources = {path: sha256(REPO / path) for path in SOURCE_PATHS}
    if metadata.get('inputs') != inputs or metadata.get('source_hashes') != sources:
        raise ValueError('operator input/source identity changed')
    for entry in metadata['assets']:
        path = REPO / entry['path']
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('operator asset changed: ' + entry['path'])
    one, two = _populations(entries)
    graph = load_sparse(REPO / entries['visible_forward']['path'])
    coverage = {}
    for hops in (1, 2):
        arrays = {direction + '_' + part: np.load(DESTINATION / ('ego%d_%s_%s.npy' % (hops, direction, part)), mmap_mode='r')
                  for direction in ('in', 'out') for part in ('ptr', 'col')}
        count = edges_total = nodes_total = 0
        for start, stop, edges in _blocks(hops, one, two, graph):
            for direction, incoming in (('in', True), ('out', False)):
                ptr, col = local_csr(stop - start, edges, incoming)
                actual_ptr = arrays[direction + '_ptr'][start:stop + 1]
                first, last = int(actual_ptr[0]), int(actual_ptr[-1])
                if not np.array_equal(actual_ptr - first, ptr) or not np.array_equal(arrays[direction + '_col'][first:last], col):
                    raise ValueError('full ego CSR mismatch: hop%d item%d %s' % (hops, count, direction))
            count += 1
            nodes_total += stop - start
            edges_total += len(edges)
        coverage[str(hops)] = dict(egos=count, nodes=nodes_total, directed_edges=edges_total)
    report = dict(status='PASS', coverage=coverage, inputs=inputs, source_hashes=sources,
                  metadata_sha256=sha256(DESTINATION / 'metadata.json'))
    (DESTINATION / 'audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    from tools.graph_infra.assets import load_sparse
    entries, inputs = _inputs()
    if DESTINATION.exists():
        audit()
        return 0
    one, two = _populations(entries)
    if len(one['node_offsets']) != len(two['offsets']):
        raise ValueError('one/two-hop catalogue coverage differs')
    graph = load_sparse(REPO / entries['visible_forward']['path'])
    DESTINATION.mkdir()
    for hops in (1, 2):
        _write_layout(hops, one, two, graph)
    common = dict(graph_name=GRAPH_NAME, graph_sha256=GRAPH_SHA256, use='training',
                  inputs=inputs, source_hashes={path: sha256(REPO / path) for path in SOURCE_PATHS},
                  command='python -m tools.issue252.ego_operators',
                  coverage=dict(items=len(one['node_offsets']) - 1, scales=[1, 2],
                                population='every full induced ego; all directed support; no sampling',
                                normalization='reciprocal row degree computed in model dtype at runtime'))
    assets = []
    for path in sorted(DESTINATION.glob('*.npy')):
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        assets.append(dict(common, name=path.stem, path=str(path.relative_to(REPO)), schema='npy',
                           sha256=sha256(path), bytes=path.stat().st_size, shape=list(array.shape),
                           dtype=array.dtype.str, data_bytes=int(array.nbytes)))
    metadata = dict(common, version=VERSION, assets=assets,
                    references=[dict(name=name, path=entry['path'], sha256=entry['sha256']) for name, entry in entries.items()])
    (DESTINATION / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    audit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
