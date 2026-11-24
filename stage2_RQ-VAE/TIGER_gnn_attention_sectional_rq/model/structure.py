"""Two independent full induced ego readouts, sharing one directed GNN.

Features and normalization are frozen #249 assets; only rootless synthetic stores
may fit local statistics. Global IDs select features, never message endpoints.
"""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
import torch
from torch import nn
from tools.issue252.ego_operators import DeviceLayout, batch_operators, load_layouts, pack_synthetic
from tools.issue252.host_transfer import host_int64
from tools.issue252 import ego_kernel
from tools.issue252.host_transfer import host_int64
from tools.issue252.sparse_adjoint import directional_means
from tools.issue252.static_features import StaticFeatureMeans
from tools.issue252.sparse_adjoint import normalized_directional_means


def _verify_store_assets(store):
    """Check bytes once before any registered feature or layout array is loaded.

    The original CSR producer is frozen in the asset provenance; runtime
    integrity belongs to its training consumer, not a regenerated asset copy.
    """
    repo, root = Path(store.repo).resolve(), Path(store.root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {entry['name']: entry for entry in manifest['assets']}
    if len(entries) != len(manifest['assets']):
        raise ValueError('duplicate runtime asset identity')
    names = ['ego_full_1hop_visible', 'two_hop_nodes', 'two_hop_offsets',
             'node_features', 'node_features_report', 'ego_operator_metadata']
    names += ['ego%d_%s_%s' % (hops, direction, part)
              for hops in (1, 2) for direction in ('in', 'out') for part in ('ptr', 'col')]
    for name in names:
        entry = entries.get(name, {})
        relative = Path(entry.get('path', ''))
        path = (repo / relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts or repo not in path.parents
                or entry.get('use') != ('description' if name == 'ego_operator_metadata' else 'training')
                or not path.is_file() or path.stat().st_size != entry.get('bytes')):
            raise ValueError('runtime asset identity changed: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != entry.get('sha256'):
            raise ValueError('runtime asset bytes changed: ' + name)


def _edge_operators(edges, n, dtype, device):
    """Explicit ragged inputs; production gathers already-persisted CSR blocks."""
    operators = []
    for row, col in ((edges[:, 1], edges[:, 0]), (edges[:, 0], edges[:, 1])):
        order = torch.argsort(row * n + col)
        degree = torch.bincount(row, minlength=n)
        ptr = torch.cat((degree.new_zeros(1), degree.cumsum(0)))
        values = degree[row[order]].to(dtype).clamp_min(1).reciprocal()
        operators.append(torch.sparse_csr_tensor(ptr, col[order], values, size=(n, n), device=device))
    return operators


class DirectedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_linear = nn.Linear(32, 32)
        self.incoming = nn.Linear(32, 32, bias=False)
        self.outgoing = nn.Linear(32, 32, bias=False)

    def forward(self, x, edges):
        return self.forward_sparse(x, *_edge_operators(edges, len(x), x.dtype, x.device))

    def forward_sparse(self, x, incoming, outgoing):
        return torch.tanh(self.self_linear(x)
                          + self.incoming(torch.sparse.mm(incoming, x))
                          + self.outgoing(torch.sparse.mm(outgoing, x)))


def _affine_layer(features, mean_in, mean_out, nonempty_in, nonempty_out, affine):
    """The first ego layer: the frozen six-feature means through the live affine."""
    self_weight, self_bias, in_weight, in_bias, out_weight, out_bias = affine
    return torch.tanh(features @ self_weight.T + self_bias
                      + mean_in @ in_weight.T + nonempty_in[:, None] * in_bias
                      + mean_out @ out_weight.T + nonempty_out[:, None] * out_bias)


@lru_cache(maxsize=1)
def _compiled_affine_layer():
    """The fused first layer: one kernel chain instead of six full-size passes."""
    from tools.issue252.cuda_math import compile_dense
    return compile_dense(_affine_layer)


class SubgraphTokens(nn.Module):
    """forward(EgoStore.collate batch) -> [B,32]; tokens_for_items -> [N,2,32]."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(6, 32)
        self.layers = nn.ModuleList([DirectedMean(), DirectedMean()])
        self._feature_cache = weakref.WeakKeyDictionary()

    def forward(self, batch):
        device, dtype = self.input.weight.device, self.input.weight.dtype
        def tensor(value, kind):
            return torch.as_tensor(value, dtype=kind, device=device)
        features = tensor(batch['features'], dtype).reshape(-1, 6)
        groups = tensor(batch['batch_index'], torch.long)
        mask = tensor(batch['node_mask'], torch.bool)
        edges = tensor(batch['edges'], torch.long).reshape(-1, 2)
        edge_mask = tensor(batch['edge_mask'], torch.bool)
        # Masked padding may use sentinel endpoints: filter before indexing.
        edges = edges[edge_mask]
        in_range = ((edges >= 0) & (edges < len(features))).all(dim=1)
        edges = edges[in_range]
        edges = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
        edges = edges[groups[edges[:, 0]] == groups[edges[:, 1]]]
        incoming, outgoing = _edge_operators(edges, len(features), dtype, device)
        x = self.input(features.masked_fill(~mask[:, None], 0))
        for layer in self.layers:
            x = layer.forward_sparse(x, incoming, outgoing).masked_fill(~mask[:, None], 0)
        count = len(batch['items'])
        readout = x.new_zeros((count, 32))
        sizes = x.new_zeros((count, 1))
        readout.index_add_(0, groups[mask], x[mask])
        sizes.index_add_(0, groups[mask], x.new_ones((len(groups[mask]), 1)))
        return readout / sizes.clamp_min(1)

    def _cached_store(self, store):
        cached = self._feature_cache.get(store)
        if cached is None:
            if hasattr(store, 'repo'):
                _verify_store_assets(store)
            if hasattr(store, 'root'):
                root = Path(store.root)
                direct = root / 'node_features.npz'
                feature_path = direct if direct.is_file() else root / 'v5_gnn_geometry' / 'node_features.npz'
                with np.load(feature_path, allow_pickle=False) as asset:
                    features = asset['features'].copy()
                report = json.loads(feature_path.with_name('node_features_report.json').read_text())
                stats = report['standardization']
            else:
                features = store.node_features()
                std = features.std(axis=0)
                stats = {'mean': features.mean(axis=0), 'std': np.where(std < 1e-12, 1., std)}
            mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
            if features.shape != (store.n, 6) or mean.shape != (6,) or std.shape != (6,):
                raise ValueError('frozen six-feature asset shape mismatch')
            if not np.isfinite(features).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError('invalid frozen feature normalization')
            normalized = (features - mean) / std
            if hasattr(store, 'repo'):
                layouts = load_layouts(store.root)
            else:
                view = copy.copy(store)
                view.node_features = lambda: features
                layouts = [pack_synthetic(view, hops) for hops in (1, 2)]
            cached = {'normalized': normalized, 'layouts': layouts, 'devices': {}}
            self._feature_cache[store] = cached
        device, dtype = self.input.weight.device, self.input.weight.dtype
        device_key = str(device)
        if device_key not in cached['devices']:
            cached['devices'][device_key] = {
                'layouts': [DeviceLayout(layout, device) for layout in cached['layouts']],
                'features': {}, 'means': {}}
        resident = cached['devices'][device_key]
        if dtype not in resident['features']:
            resident['features'][dtype] = torch.tensor(cached['normalized'], device=device, dtype=dtype)
        if dtype not in resident['means']:
            resident['means'][dtype] = [
                StaticFeatureMeans(layout, resident['features'][dtype])
                for layout in resident['layouts']]
        return resident['layouts'], resident['features'][dtype]

    def _first_affine(self):
        """Compose live weights/biases; only six-feature means are static."""
        layer = self.layers[0]
        weight, bias = self.input.weight, self.input.bias
        return (layer.self_linear.weight @ weight,
                layer.self_linear.weight @ bias + layer.self_linear.bias,
                layer.incoming.weight @ weight, layer.incoming.weight @ bias,
                layer.outgoing.weight @ weight, layer.outgoing.weight @ bias)

    def _read_sparse(self, features, groups, sizes, static, affine, operator):
        mean_in, mean_out, nonempty_in, nonempty_out = static
        layer_function = (_compiled_affine_layer() if features.is_cuda
                         else _affine_layer)
        x = layer_function(features, mean_in, mean_out, nonempty_in, nonempty_out,
                         affine)
        layer = self.layers[1]
        mean_in, mean_out = operator(x)
        x = torch.tanh(layer.self_linear(x) + layer.incoming(mean_in) + layer.outgoing(mean_out))
        readout = x.new_zeros((len(sizes), 32))
        readout.index_add_(0, groups, x)
        return readout / sizes.to(x.dtype).clamp_min(1)[:, None]

    def tokens_for_items(self, store, item_ids, device, chunk=1024):
        """Deduplicate global items, run each scale separately, then restore order.

        chunk bounds ego batch size, never the size of an individual full ego.
        """
        if chunk < 1:
            raise ValueError('chunk must be positive')
        device = torch.device(device)
        if device != self.input.weight.device:
            raise ValueError('requested device differs from GNN device')
        host = np.asarray(item_ids.detach().cpu().numpy() if torch.is_tensor(item_ids)
                          else item_ids, dtype=np.int64).reshape(-1)
        if len(host) == 0:
            return self.input.weight.new_empty((0, 2, 32)), torch.empty((0, 2), dtype=torch.bool, device=device)
        if ((host < 0) | (host >= store.n)).any():
            raise IndexError('item id out of range')
        # The unique order and every ego size are known on the host, so no
        # gather below has to read a device value to size its output.
        unique, restore = np.unique(host, return_inverse=True)
        layouts, normalized = self._cached_store(store)
        means = self._feature_cache[store]['devices'][str(self.input.weight.device)]['means'][normalized.dtype]
        affine = self._first_affine()
        outputs, validity = [], []
        for begin in range(0, len(unique), chunk):
            block = unique[begin:begin + chunk]
            items = host_int64(block, device)
            scales, masks = [], []
            for layout, static in zip(layouts, means):
                plan = layout.plan(block)
                # The fused kernel reads the frozen layout directly: the batch's
                # block-diagonal CSR is never materialised.
                geometry = ego_kernel.batch_geometry(layout, block, device)
                nodes = layout.nodes[geometry['layout_rows']].long()
                groups = geometry['row_ego'].long()
                sizes = host_int64(layout.node_counts[block], device)
                operators = None
                if device.type != 'cuda':
                    operators = batch_operators(layout, items, normalized.dtype, plan)[3:]
                operator = (lambda values, part=layout, batch=geometry:
                            directional_means(values, part, batch, operators))
                # Retain the live activations; backward must not replay the GNN.
                value = self._read_sparse(normalized[nodes], groups, sizes,
                                          static.gather(layout, items, plan), affine,
                                          operator)
                scales.append(value)
                masks.append(sizes > 0)
            outputs.append(torch.stack(scales, dim=1))
            validity.append(torch.stack(masks, dim=1))
        tokens = torch.cat(outputs)
        valid = torch.cat(validity)
        if len(unique) != len(host) or not np.array_equal(unique, host):
            index = host_int64(restore, device)
            tokens, valid = tokens[index], valid[index]
        return tokens, valid
