"""Lossless, static views of registered visible-graph assets; never rebuild inputs.

Build (no arguments): python -m tools.issue252.runtime_layout
Audit (no arguments): python -c 'from tools.issue252.runtime_layout import audit; audit()'
The separate metadata.json is for selective registration by the caller. An interrupted
or different destination is an error, not permission to overwrite immutable evidence.
"""
from __future__ import annotations

import json
import operator
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from tools.graph_infra.assets import sha256
from tools.graph_infra.ego_gnn import EgoStore

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / 'stage2_RQ-VAE').is_dir())
DESTINATION = REPO / 'assets/v8_runtime_layout'
GRAPH_NAME = 'infra246-visible-v2'
GRAPH_SHA256 = 'ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029'
EXPECTED_ITEMS = 24587
EXPECTED_VALID_ROWS = 28391948
COMMAND = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue252.runtime_layout'
AUDIT_COMMAND = ("/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -c "
                 "'from tools.issue252.runtime_layout import audit; audit()'")
TARGET_DTYPE = np.dtype([('b', '<i4'), ('c', '<i4'), ('target', '<f8')])
SOURCE_PATHS = ('tools/issue252/runtime_layout.py', 'tools/graph_infra/ego_gnn.py',
                'tools/graph_infra/assets.py')
OUTPUT_FILES = ('visible_forward_bits.npy', 'two_hop_nodes.npy',
                'two_hop_offsets.npy', 'two_hop_root_local.npy',
                'targets.npy', 'target_offsets.npy')


def pack_directed(matrix) -> np.ndarray:
    """Pack nonzero row support, column j at byte j//8, bit j%8."""
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('directed adjacency must be square')
    n = matrix.shape[0]
    if not sparse.issparse(matrix):
        return np.packbits(np.asarray(matrix) != 0, axis=1, bitorder='little')
    csr = matrix.tocsr(copy=False)
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    packed = np.zeros((n, (n + 7) // 8), dtype=np.uint8)
    for i in range(n):
        start, stop = csr.indptr[i:i + 2]
        columns = csr.indices[start:stop][csr.data[start:stop] != 0]
        np.bitwise_or.at(packed[i], columns // 8,
                         np.left_shift(np.uint8(1), (columns % 8).astype(np.uint8)))
    return packed


def _item_count(n):
    try:
        n = operator.index(n)
    except TypeError as error:
        raise ValueError('item count must be an integer') from error
    if not 0 <= n <= np.iinfo(np.int32).max:
        raise ValueError('item count outside int32 range')
    return n


def _valid_columns(shard, n):
    columns = [np.asarray(shard[k]) for k in ('centre', 'b', 'c', 'curvature', 'valid')]
    if any(x.ndim != 1 or len(x) != len(columns[0]) for x in columns):
        raise ValueError('target columns must be aligned one-dimensional arrays')
    centre, b, c, target, valid = columns
    if valid.dtype != np.dtype(bool):
        raise ValueError('valid mask must be boolean')
    selected = []
    for name, values in (('centre', centre), ('b', b), ('c', c)):
        if values.dtype.kind not in 'iu':
            raise ValueError(name + ' must contain integer IDs')
        values = values[valid]
        if np.any(values < 0) or np.any(values >= n):
            raise ValueError(name + ' contains an out-of-range valid ID')
        selected.append(values.astype(np.int64, copy=False))
    if target.dtype.kind not in 'fiu':
        raise ValueError('targets must be real numeric values')
    # Copying stored float64 values preserves signed zero and NaN payload bits.
    return (*selected, target[valid].astype(np.float64, copy=False))


def _group_rows(shard, n):
    centre, b, c, target = _valid_columns(shard, n)
    order = np.argsort(centre, kind='stable')
    centres = centre[order]
    rows = np.empty(len(order), dtype=TARGET_DTYPE)
    rows['b'], rows['c'], rows['target'] = b[order], c[order], target[order]
    boundaries = np.r_[0, np.flatnonzero(centres[1:] != centres[:-1]) + 1, len(centres)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start != stop:
            yield int(centres[start]), rows[start:stop]


def pack_targets(shards, n):
    """Stable centre-major rows; duplicates and original valid float64 bits survive.

    This small-input API consumes an iterable once. The production writer below uses
    two streaming passes and a memmap instead of retaining all packed shards.
    """
    n = _item_count(n)
    groups = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=np.int64)
    for shard in shards:
        for centre, rows in _group_rows(shard, n):
            groups[centre].append(rows)
            counts[centre] += len(rows)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    result = np.empty(int(offsets[-1]), dtype=TARGET_DTYPE)
    for centre, blocks in enumerate(groups):
        position = int(offsets[centre])
        for block in blocks:
            result[position:position + len(block)] = block
            position += len(block)
    return result, offsets


def closed_two_hop_nodes(store: EgoStore, node: int):
    """Full sorted closed population using only EgoStore's frozen 1-hop/CSR.

    Avoid constructing induced edges/features just to persist node IDs. The audit
    separately compares every population and root to EgoStore.ego(hops=2).
    """
    node = operator.index(node)
    if not 0 <= node < store.n:
        raise IndexError('item id out of range')
    frozen = store._frozen_ego
    start, stop = frozen['node_offsets'][node:node + 2]
    one_hop = frozen['nodes'][start:stop]
    nodes = np.unique(np.concatenate([one_hop] + [store._neighbours(int(i)) for i in one_hop]))
    return nodes, int(np.searchsorted(nodes, node))


def _write_target_layout(source, n, destination):
    """source() opens a fresh bounded, one-shard-at-a-time iterator per pass."""
    n = _item_count(n)
    counts = np.zeros(n, dtype=np.int64)
    for shard in source():
        centres, _, _, _ = _valid_columns(shard, n)
        counts += np.bincount(centres, minlength=n)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    np.save(destination / 'target_offsets.npy', offsets, allow_pickle=False)
    rows = np.lib.format.open_memmap(destination / 'targets.npy', mode='w+',
                                     dtype=TARGET_DTYPE, shape=(int(offsets[-1]),))
    cursors = offsets[:-1].copy()
    for shard in source():
        for centre, block in _group_rows(shard, n):
            start = int(cursors[centre])
            stop = start + len(block)
            if stop > offsets[centre + 1]:
                raise ValueError('target input changed between passes')
            rows[start:stop] = block
            cursors[centre] = stop
    if not np.array_equal(cursors, offsets[1:]):
        raise ValueError('target input changed between passes')
    rows.flush()
    return int(offsets[-1])


def _write_two_hop(store, destination):
    offsets = np.zeros(store.n + 1, dtype=np.int64)
    roots = np.empty(store.n, dtype=np.int32)
    for node in range(store.n):
        population, root = closed_two_hop_nodes(store, node)
        offsets[node + 1] = offsets[node] + len(population)
        roots[node] = root
    np.save(destination / 'two_hop_offsets.npy', offsets, allow_pickle=False)
    np.save(destination / 'two_hop_root_local.npy', roots, allow_pickle=False)
    nodes = np.lib.format.open_memmap(destination / 'two_hop_nodes.npy', mode='w+',
                                      dtype=np.int32, shape=(int(offsets[-1]),))
    for node in range(store.n):
        population, _ = closed_two_hop_nodes(store, node)
        nodes[offsets[node]:offsets[node + 1]] = population
    nodes.flush()


def _json_once(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError('different immutable evidence: ' + str(path))
    else:
        with path.open('x') as output:
            output.write(text)


def _registered_inputs():
    manifest = json.loads((REPO / 'assets/manifest.json').read_text())
    entries = {}
    for entry in manifest['assets']:
        if entry['name'] in entries:
            raise ValueError('duplicate registered asset name: ' + entry['name'])
        entries[entry['name']] = entry
    names = ['visible_forward', 'visible_reverse', 'visible_undirected', 'item_ids',
             'ego_full_1hop_visible', 'node_features', 'node_features_report', 'index']
    with np.load(REPO / entries['index']['path'], allow_pickle=False) as index:
        shard_files = [str(x) for x in index['shard_names']]
    if len(shard_files) != len(set(shard_files)):
        raise ValueError('duplicate curvature shards in frozen index')
    names += [Path(filename).stem for filename in shard_files]
    selected = {name: entries[name] for name in names}
    # EgoStore intentionally uses these existing canonical asset locations.
    for name in ('visible_forward', 'visible_reverse', 'visible_undirected', 'item_ids'):
        expected = 'assets/train/' + name + ('.npy' if name == 'item_ids' else '.npz')
        if selected[name]['path'] != expected:
            raise ValueError('EgoStore input path differs from registration: ' + name)
    if selected['index']['path'] != 'assets/index.npz':
        raise ValueError('unexpected frozen curvature index path')
    for filename in shard_files:
        if selected[Path(filename).stem]['path'] != 'assets/shards/' + filename:
            raise ValueError('curvature shard registration differs from frozen index')
    for name, entry in selected.items():
        path = REPO / entry['path']
        if entry.get('use') != 'training' or sha256(path) != entry['sha256']:
            raise ValueError('unregistered, changed or non-training asset: ' + name)
        if path.stat().st_size != entry['bytes']:
            raise ValueError('registered asset byte size changed: ' + name)
    if selected['visible_forward']['sha256'] != GRAPH_SHA256:
        raise ValueError('visible graph identity changed')
    inputs = {entry['path']: entry['sha256'] for entry in selected.values()}
    sources = {path: sha256(REPO / path) for path in SOURCE_PATHS}
    return selected, shard_files, inputs, sources


def _shards(entries, shard_files):
    for filename in shard_files:
        path = REPO / entries[Path(filename).stem]['path']
        with np.load(path, allow_pickle=False) as blob:
            yield {key: blob[key] for key in ('centre', 'b', 'c', 'curvature', 'valid')}


def _store():
    store = EgoStore(REPO)
    if store.n != EXPECTED_ITEMS or not np.array_equal(store.ids, np.arange(store.n)):
        raise ValueError('frozen global item IDs are missing, reordered or changed')
    if store.forward.shape != (store.n, store.n):
        raise ValueError('visible adjacency shape differs from global item IDs')
    return store


def _array_description(array):
    dtype = array.dtype.descr if array.dtype.names else array.dtype.str
    return {'shape': list(array.shape), 'dtype': dtype, 'data_bytes': int(array.nbytes)}


def _npz_headers(path):
    """Describe NPZ members without loading arrays or executing object pickles."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                raise ValueError('unexpected NPZ member: ' + name)
            with archive.open(name) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError('unsupported NPZ header version: ' + str(version))
            result[name[:-4]] = {
                'shape': list(shape), 'dtype': dtype.descr if dtype.names else dtype.str,
                'data_bytes': None if dtype.hasobject else math.prod(shape) * dtype.itemsize}
    return result


def _metadata(entries, inputs, sources, store):
    coverage = {'items': store.n, 'valid_targets': EXPECTED_VALID_ROWS,
                'centres': 'all global IDs in item_ids order',
                'target_order': 'centre, then frozen index shard order, then stored row order',
                'duplicates': 'preserved', 'target_comparison': 'float64 bit-for-bit',
                'two_hop': 'complete closed populations; no truncation or sampling'}
    common = {'graph_name': GRAPH_NAME, 'graph_sha256': GRAPH_SHA256, 'use': 'training',
              'inputs': inputs, 'source_hashes': sources, 'coverage': coverage, 'command': COMMAND}
    assets = []
    for filename in OUTPUT_FILES:
        path = DESTINATION / filename
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        entry = dict(common, name=path.stem, path=str(path.relative_to(REPO)),
                     schema='npy', bytes=path.stat().st_size, sha256=sha256(path),
                     **_array_description(array))
        if filename == 'visible_forward_bits.npy':
            entry['bit_order'] = 'little: row[source, target//8] bit target%8; support only'
            entry['logical_shape'] = [store.n, store.n]
        assets.append(entry)
    for name in ('ego_full_1hop_visible', 'node_features', 'node_features_report'):
        original = entries[name]
        path = REPO / original['path']
        description = {}
        if path.suffix == '.npz':
            description = _npz_headers(path)
        else:
            description = json.loads(path.read_text())
        assets.append(dict(common, name=name, path=original['path'], schema=original['schema'],
                           sha256=original['sha256'], bytes=original['bytes'],
                           shape={k: v['shape'] for k, v in description.items()} if path.suffix == '.npz' else [],
                           dtype={k: v['dtype'] for k, v in description.items()} if path.suffix == '.npz' else 'json',
                           reused=True, content=description))
    return dict(common, version='issue252.lossless-runtime.v1', assets=assets,
                root_ids={'path': entries['item_ids']['path'],
                          'sha256': entries['item_ids']['sha256']},
                feature_standardization='reuse node_features and frozen node_features_report statistics',
                audit_command=AUDIT_COMMAND)


def _load_verified_metadata(inputs, sources):
    path = DESTINATION / 'metadata.json'
    if not path.is_file():
        raise FileExistsError('incomplete runtime layout: metadata.json missing')
    metadata = json.loads(path.read_text())
    for field, expected in (('version', 'issue252.lossless-runtime.v1'),
                            ('graph_name', GRAPH_NAME), ('graph_sha256', GRAPH_SHA256),
                            ('inputs', inputs), ('source_hashes', sources)):
        if metadata.get(field) != expected:
            raise ValueError('runtime layout identity changed: ' + field)
    generated = {Path(entry['path']).name for entry in metadata['assets'] if not entry.get('reused')}
    if generated != set(OUTPUT_FILES):
        raise ValueError('incomplete runtime layout output inventory')
    for entry in metadata['assets']:
        path = REPO / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('runtime layout content changed: ' + str(path))
    return metadata


def _check_offsets(offsets, n, total, name):
    if (offsets.dtype != np.dtype('int64') or offsets.shape != (n + 1,)
            or offsets[0] != 0 or offsets[-1] != total or np.any(np.diff(offsets) < 0)):
        raise ValueError('invalid ' + name + ' CSR offsets')


def audit():
    """No-argument all-item, full-row audit; persist deterministic audit.json evidence."""
    entries, shard_files, inputs, sources = _registered_inputs()
    metadata = _load_verified_metadata(inputs, sources)
    store = _store()
    arrays = {name: np.load(DESTINATION / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('visible_forward_bits', 'two_hop_nodes', 'two_hop_offsets',
                           'two_hop_root_local', 'targets', 'target_offsets')}
    bits = arrays['visible_forward_bits']
    if bits.dtype != np.uint8 or bits.shape != (store.n, (store.n + 7) // 8):
        raise ValueError('invalid packed directed support shape/dtype')
    for node in range(store.n):
        decoded = np.unpackbits(bits[node], bitorder='little')
        columns = np.flatnonzero(decoded[:store.n])
        row = store.forward.getrow(node)
        expected = np.sort(row.indices[row.data != 0])
        if not np.array_equal(columns, expected) or np.any(decoded[store.n:]):
            raise ValueError('directed support mismatch at item ' + str(node))
    nodes, offsets, roots = (arrays[key] for key in ('two_hop_nodes', 'two_hop_offsets', 'two_hop_root_local'))
    _check_offsets(offsets, store.n, len(nodes), 'two-hop')
    if nodes.dtype != np.int32 or roots.dtype != np.int32 or roots.shape != (store.n,):
        raise ValueError('invalid two-hop nodes/root dtype or shape')
    for node in range(store.n):
        reference = store.ego(node, hops=2)
        population = nodes[offsets[node]:offsets[node + 1]]
        if (not np.array_equal(population, reference['nodes'])
                or roots[node] != reference['root_local']
                or population[roots[node]] != node):
            raise ValueError('full EgoStore two-hop mismatch at item ' + str(node))
    rows, offsets = arrays['targets'], arrays['target_offsets']
    if rows.dtype != TARGET_DTYPE or rows.shape != (EXPECTED_VALID_ROWS,):
        raise ValueError('invalid valid-target table shape/dtype')
    _check_offsets(offsets, store.n, len(rows), 'target')
    cursors = offsets[:-1].copy()
    valid_rows = 0
    original_rows = 0
    for shard in _shards(entries, shard_files):
        original_rows += len(shard['valid'])
        for centre, expected in _group_rows(shard, store.n):
            start = int(cursors[centre])
            stop = start + len(expected)
            if stop > offsets[centre + 1]:
                raise ValueError('extra original target rows at centre ' + str(centre))
            actual = rows[start:stop]
            if (not np.array_equal(actual['b'], expected['b'])
                    or not np.array_equal(actual['c'], expected['c'])
                    or not np.array_equal(actual['target'].view(np.uint64),
                                          expected['target'].view(np.uint64))):
                raise ValueError('original target bits/order mismatch at centre ' + str(centre))
            cursors[centre] = stop
            valid_rows += len(expected)
    if not np.array_equal(cursors, offsets[1:]) or valid_rows != EXPECTED_VALID_ROWS:
        raise ValueError('missing original valid target occurrences')
    report = {'verdict': 'PASS', 'graph_name': GRAPH_NAME, 'graph_sha256': GRAPH_SHA256,
              'command': AUDIT_COMMAND, 'inputs': inputs, 'source_hashes': sources,
              'metadata_sha256': sha256(DESTINATION / 'metadata.json'),
              'outputs': {entry['path']: entry['sha256'] for entry in metadata['assets']},
              'coverage': {'directed_rows': store.n, 'two_hop_populations': store.n,
                           'root_positions': store.n, 'original_rows': original_rows,
                           'valid_target_occurrences': valid_rows,
                           'excluded_invalid_rows': original_rows - valid_rows,
                           'zero_valid_row_centres': int(np.sum(np.diff(offsets) == 0)),
                           'target_comparison': 'all stored float64 bits and b/c IDs, in order',
                           'two_hop_comparison': 'every full EgoStore.ego(hops=2) population/root'}}
    _json_once(DESTINATION / 'audit.json', report)
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    """Build once from source constants; verify rather than overwrite an existing layout."""
    entries, shard_files, inputs, sources = _registered_inputs()
    if DESTINATION.exists():
        _load_verified_metadata(inputs, sources)
        if not (DESTINATION / 'audit.json').is_file():
            raise FileExistsError('incomplete runtime layout: audit.json missing')
        audit()
        return 0
    store = _store()
    DESTINATION.mkdir()
    np.save(DESTINATION / 'visible_forward_bits.npy', pack_directed(store.forward), allow_pickle=False)
    _write_two_hop(store, DESTINATION)
    count = _write_target_layout(lambda: _shards(entries, shard_files), store.n, DESTINATION)
    if count != EXPECTED_VALID_ROWS:
        raise ValueError('valid frozen target count changed: ' + str(count))
    # Detect changed inputs/sources across generation before issuing any receipt.
    _, _, final_inputs, final_sources = _registered_inputs()
    if inputs != final_inputs or sources != final_sources:
        raise ValueError('registered input/source changed during layout generation')
    _json_once(DESTINATION / 'metadata.json', _metadata(entries, inputs, sources, store))
    audit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
