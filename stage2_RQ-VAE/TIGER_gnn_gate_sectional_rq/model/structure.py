"""Training-visible, item-global directed structure descriptors and curvature gates.

T = [h0(root), mean_{closed radius 1} h1, mean_{closed radius 2} h2].
The incoming and outgoing channels follow the actual (source, target) arcs in
EgoStore batches; edge_direction is metadata, not a second set of edges.
"""
from __future__ import annotations

import weakref
import json

import numpy as np
import torch
from torch import nn

from tools.graph_infra.ego_gnn import EgoStore


class _DirectedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_linear = nn.Linear(32, 32)
        self.incoming = nn.Linear(32, 32, bias=False)
        self.outgoing = nn.Linear(32, 32, bias=False)

    def forward(self, hidden, source, target, in_count, out_count):
        incoming = torch.zeros_like(hidden).index_add(0, target, hidden[source])
        outgoing = torch.zeros_like(hidden).index_add(0, source, hidden[target])
        return torch.tanh(self.self_linear(hidden)
                          + self.incoming(incoming / in_count)
                          + self.outgoing(outgoing / out_count))


class _FixedFeatureView(EgoStore):
    """Read-only view retaining EgoStore's exact untruncated ego/collate code.

    The shared store is never patched. Its original node_features implementation
    otherwise rebuilds the *entire* feature matrix once for every requested ego.
    Only fixed inputs/statistics are cached, never trainable representations.
    """
    def __init__(self, source, features, stats):
        self._source = source
        self._features = features
        self._stats = stats

    def __getattr__(self, name):
        return getattr(self._source, name)

    def node_features(self):
        return self._features

    def standardization(self):
        return self._stats


def _mean_pool(hidden, batch_index, selected, batch_size):
    index = batch_index[selected]
    total = hidden.new_zeros((batch_size, 32)).index_add(0, index, hidden[selected])
    count = torch.bincount(index, minlength=batch_size).clamp_min(1).to(hidden.dtype)
    return total / count[:, None]


class DirectedGate(nn.Module):
    """Six fixed features -> two width-32 directed layers -> five expert weights.

    ``forward`` accepts an EgoStore ``collate(..., hops=2)`` dictionary, with
    either NumPy arrays or tensors. All outputs retain autograd. Closed radii
    include the root, so isolates have well-defined, finite pooled descriptors.
    ``input``/``layers`` own the GNN parameters; ``gate`` owns only the MLP.
    """
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(6, 32)
        self.layers = nn.ModuleList([_DirectedMean(), _DirectedMean()])
        self.gate = nn.Sequential(nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 5))
        self._feature_cache = weakref.WeakKeyDictionary()

    def forward(self, batch, behavior_on=True):
        if int(batch.get('hops', 2)) < 2:
            raise ValueError('DirectedGate requires untruncated two-hop egos')
        parameter = self.input.weight
        device, dtype = parameter.device, parameter.dtype
        features = torch.as_tensor(batch['features'], dtype=dtype, device=device)
        roots = torch.as_tensor(batch['root_global'], dtype=torch.long, device=device)
        batch_size = roots.numel()
        if not batch_size:
            return parameter.new_empty((0, 96)), parameter.new_empty((0, 5))
        if features.ndim != 2 or features.shape[1] != 6:
            raise ValueError('structure input must contain exactly six fixed features')
        batch_index = torch.as_tensor(batch['batch_index'], dtype=torch.long, device=device)
        edges = torch.as_tensor(batch['edges'], dtype=torch.long, device=device).reshape(-1, 2)
        node_mask = torch.as_tensor(batch.get('node_mask', np.ones(len(features), dtype=bool)),
                                    dtype=torch.bool, device=device)
        edge_mask = torch.as_tensor(batch.get('edge_mask', np.ones(len(edges), dtype=bool)),
                                    dtype=torch.bool, device=device)
        edge_mask = edge_mask & node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
        source, target = edges[edge_mask].unbind(dim=1)
        in_count = torch.bincount(target, minlength=len(features)).clamp_min(1).to(dtype)[:, None]
        out_count = torch.bincount(source, minlength=len(features)).clamp_min(1).to(dtype)[:, None]

        # Recover each radius from adjacency and the global batched centre index,
        # not from labels, node-id ordering, or the size of the complete ego.
        centre = torch.zeros(len(features), dtype=torch.bool, device=device)
        centre[roots] = True
        one_hop = centre.clone()
        one_hop[target[centre[source]]] = True
        one_hop[source[centre[target]]] = True
        two_hop = one_hop.clone()
        two_hop[target[one_hop[source]]] = True
        two_hop[source[one_hop[target]]] = True

        hidden0 = torch.tanh(self.input(features))
        hidden1 = self.layers[0](hidden0, source, target, in_count, out_count)
        hidden2 = self.layers[1](hidden1, source, target, in_count, out_count)
        descriptor = torch.cat((hidden0[roots],
                                _mean_pool(hidden1, batch_index, one_hop & node_mask, batch_size),
                                _mean_pool(hidden2, batch_index, two_hop & node_mask, batch_size)), dim=1)
        weights = torch.softmax(self.gate(descriptor), dim=-1) if behavior_on \
            else descriptor.new_full((batch_size, 5), 1. / 5.)
        return descriptor, weights

    def _fixed_view(self, store):
        cached = self._feature_cache.get(store)
        if cached is None:
            if hasattr(store, 'root'):
                # Production consumes the already-frozen features/statistics;
                # only synthetic in-memory test graphs fit their own statistics.
                asset_root = store.root / 'v5_gnn_geometry'
                with np.load(asset_root / 'node_features.npz') as saved:
                    features = saved['features']
                stats = json.loads((asset_root / 'node_features_report.json').read_text())['standardization']
                cached = (features, stats)
            else:
                features = store.node_features()
                mean = features.mean(axis=0)
                std = features.std(axis=0)
                std = np.where(std < 1e-12, 1., std)
                cached = (features, {'mean': mean.tolist(), 'std': std.tolist()})
            self._feature_cache[store] = cached
        return _FixedFeatureView(store, *cached)

    def gates_for_items(self, store, item_ids, device, chunk=32, behavior_on=True):
        """Compute each distinct item's full two-hop ego once, then restore order.

        Chunking bounds ego memory only; it never samples/truncates neighbours.
        Repeated ids share the exact output row and accumulate their gradients.
        The module must already reside on ``device``, as for a normal forward.
        """
        if chunk <= 0:
            raise ValueError('chunk must be positive')
        requested_device = torch.device(device)
        model_device = self.input.weight.device
        if requested_device.type != model_device.type or (requested_device.index is not None
                                                          and requested_device.index != model_device.index):
            raise ValueError('DirectedGate must be moved to the requested device before use')
        if isinstance(item_ids, torch.Tensor):
            item_ids = item_ids.detach().cpu().numpy()
        item_ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        if not len(item_ids):
            return self.input.weight.new_empty((0, 96)), self.input.weight.new_empty((0, 5))
        unique, inverse = np.unique(item_ids, return_inverse=True)
        view = self._fixed_view(store)
        descriptors, weights = [], []
        for start in range(0, len(unique), chunk):
            batch = view.collate(unique[start:start + chunk], hops=2, standardized=True)
            descriptor, weight = self(batch, behavior_on=behavior_on)
            descriptors.append(descriptor)
            weights.append(weight)
        inverse = torch.as_tensor(inverse, dtype=torch.long, device=model_device)
        return torch.cat(descriptors, dim=0)[inverse], torch.cat(weights, dim=0)[inverse]
