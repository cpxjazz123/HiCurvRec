"""Two independent full induced ego readouts, sharing one directed GNN.

Features and normalization are frozen #249 assets; only rootless synthetic stores
may fit local statistics. Global IDs select features, never message endpoints.
"""
import copy
import json
from pathlib import Path
import weakref

import numpy as np
import torch
from torch import nn


class DirectedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_linear = nn.Linear(32, 32)
        self.incoming = nn.Linear(32, 32, bias=False)
        self.outgoing = nn.Linear(32, 32, bias=False)

    def forward(self, x, edges):
        source, target = edges.unbind(1)
        incoming, outgoing = torch.zeros_like(x), torch.zeros_like(x)
        counts_in, counts_out = x.new_zeros((len(x), 1)), x.new_zeros((len(x), 1))
        incoming.index_add_(0, target, x[source])
        outgoing.index_add_(0, source, x[target])
        ones = x.new_ones((len(source), 1))
        counts_in.index_add_(0, target, ones)
        counts_out.index_add_(0, source, ones)
        return torch.tanh(self.self_linear(x) + self.incoming(incoming / counts_in.clamp_min(1))
                          + self.outgoing(outgoing / counts_out.clamp_min(1)))


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
        x = self.input(features.masked_fill(~mask[:, None], 0))
        for layer in self.layers:
            x = layer(x, edges).masked_fill(~mask[:, None], 0)
        count = len(batch['items'])
        readout = x.new_zeros((count, 32))
        sizes = x.new_zeros((count, 1))
        readout.index_add_(0, groups[mask], x[mask])
        sizes.index_add_(0, groups[mask], x.new_ones((int(mask.sum()), 1)))
        return readout / sizes.clamp_min(1)

    def _cached_store(self, store):
        cached = self._feature_cache.get(store)
        if cached is None:
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
            cached = (features, normalized)
            self._feature_cache[store] = cached
        features, normalized = cached
        # EgoStore calls these methods repeatedly inside ego/collate. A private
        # shallow view reuses graph assets without mutating the caller's store.
        view = copy.copy(store)
        view.node_features = lambda: features
        return view, normalized

    def tokens_for_items(self, store, item_ids, device, chunk=32):
        """Deduplicate global items, run each scale separately, then restore order.

        chunk bounds ego batch size, never the size of an individual full ego.
        """
        if chunk < 1:
            raise ValueError('chunk must be positive')
        if torch.device(device) != self.input.weight.device:
            raise ValueError('requested device differs from GNN device')
        ids = item_ids.detach().cpu().numpy() if torch.is_tensor(item_ids) else np.asarray(item_ids)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            return self.input.weight.new_empty((0, 2, 32)), torch.empty((0, 2), dtype=torch.bool, device=device)
        unique, inverse = np.unique(ids, return_inverse=True)
        view, normalized = self._cached_store(store)
        outputs, validity = [], []
        for start in range(0, len(unique), chunk):
            items = unique[start:start + chunk].tolist()
            scales, masks = [], []
            for hops in (1, 2):
                batch = view.collate(items, hops=hops, standardized=False)
                batch['features'] = normalized[batch['nodes']]
                scales.append(self(batch))
                groups = np.asarray(batch['batch_index'])[np.asarray(batch['node_mask'], dtype=bool)]
                masks.append(torch.as_tensor(np.bincount(groups, minlength=len(items)) > 0, device=device))
            outputs.append(torch.stack(scales, dim=1))
            validity.append(torch.stack(masks, dim=1))
        restore = torch.as_tensor(inverse, dtype=torch.long, device=device)
        return torch.cat(outputs)[restore], torch.cat(validity)[restore]
