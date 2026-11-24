"""Independent dense full-ego oracle; deliberately imports no production module."""
from types import SimpleNamespace

import numpy as np
import torch


def packed_layout(blocks, device):
    """Pack explicit (global_nodes, directed_local_edges), retaining multiplicity."""
    offsets = [0]
    nodes = []
    pointers = {'in': [0], 'out': [0]}
    columns = {'in': [], 'out': []}
    for block_nodes, edges in blocks:
        nodes.extend(block_nodes)
        n = len(block_nodes)
        offsets.append(offsets[-1] + n)
        for direction in ('in', 'out'):
            rows = [[] for _ in range(n)]
            for source, target in edges:
                if not (0 <= source < n and 0 <= target < n):
                    raise ValueError('reference edge outside ego')
                row, col = (target, source) if direction == 'in' else (source, target)
                rows[row].append(col)
            for row in rows:
                columns[direction].extend(sorted(row))
                pointers[direction].append(len(columns[direction]))
    offsets = np.asarray(offsets, dtype=np.int64)
    layout = SimpleNamespace(
        offsets_np=offsets, node_counts=np.diff(offsets),
        offsets=torch.tensor(offsets, device=device),
        nodes=torch.tensor(nodes, dtype=torch.int32, device=device),
        reference_blocks=blocks)
    for direction in ('in', 'out'):
        ptr = np.asarray(pointers[direction], dtype=np.int64)
        setattr(layout, direction + '_ptr', torch.tensor(ptr, device=device))
        setattr(layout, direction + '_col', torch.tensor(
            columns[direction], dtype=torch.int32, device=device))
        setattr(layout, direction + '_edge_counts', ptr[offsets[1:]] - ptr[offsets[:-1]])
    return layout


def occurrence_operators(layout, ids, dtype, device):
    """Independent CSR adapter retaining every repeated directed occurrence.

    Dense-to-CSR conversion would merge duplicate support and produce unequal
    weights, outside the paired-adjoint's uniform-per-stored-edge contract.
    """
    total = sum(len(layout.reference_blocks[item][0]) for item in ids)
    operators = []
    for direction in ('in', 'out'):
        pointer, columns, weights = [0], [], []
        base = 0
        for item in ids:
            nodes, edges = layout.reference_blocks[item]
            rows = [[] for _ in nodes]
            for source, target in edges:
                row, col = (target, source) if direction == 'in' else (source, target)
                rows[row].append(base + col)
            for row in rows:
                columns.extend(sorted(row))
                weights.extend([1 / len(row)] * len(row) if row else [])
                pointer.append(len(columns))
            base += len(nodes)
        operators.append(torch.sparse_csr_tensor(
            torch.tensor(pointer, dtype=torch.int64, device=device),
            torch.tensor(columns, dtype=torch.int64, device=device),
            torch.tensor(weights, dtype=dtype, device=device),
            size=(total, total), check_invariants=False))
    return tuple(operators)


def batch_reference(layout, ids, dtype, device):
    """Enumerate a block-diagonal dense operator independently of CSR consumers."""
    sizes = [len(layout.reference_blocks[item][0]) for item in ids]
    total = sum(sizes)
    adjacency = torch.zeros((total, total), dtype=dtype, device=device)
    groups, nodes, rows = [], [], []
    base = 0
    for group, item in enumerate(ids):
        block_nodes, edges = layout.reference_blocks[item]
        nodes.extend(block_nodes)
        groups.extend([group] * len(block_nodes))
        rows.extend(range(int(layout.offsets_np[item]), int(layout.offsets_np[item + 1])))
        for source, target in edges:
            adjacency[base + source, base + target] += 1
        base += len(block_nodes)
    outgoing = adjacency / adjacency.sum(1, keepdim=True).clamp_min(1)
    incoming = adjacency.T / adjacency.T.sum(1, keepdim=True).clamp_min(1)
    return SimpleNamespace(
        incoming=incoming, outgoing=outgoing,
        nodes=torch.tensor(nodes, dtype=torch.long, device=device),
        groups=torch.tensor(groups, dtype=torch.long, device=device),
        sizes=torch.tensor(sizes, dtype=torch.long, device=device),
        layout_rows=torch.tensor(rows, dtype=torch.long, device=device))


def second_layer_readout(x, incoming, outgoing, groups, sizes, layer):
    """Explicit affine, directed means, tanh and all-node mean, without fusion."""
    value = torch.tanh(
        x @ layer.self_linear.weight.T + layer.self_linear.bias
        + (incoming @ x) @ layer.incoming.weight.T
        + (outgoing @ x) @ layer.outgoing.weight.T)
    # Per-ego slicing, not the production scatter/index_add readout.
    outputs = []
    start = 0
    for size in sizes:
        count = int(size)
        outputs.append(value[start:start + count].mean(0) if count
                       else value.new_zeros(value.shape[1]))
        start += count
    return torch.stack(outputs) if outputs else x.new_empty((0, x.shape[1]))


class FullDirectedStore:
    """Weak closed-hop node sets with EVERY directed induced edge, plus isolate."""
    n = 11
    edges = ((0, 1), (0, 2), (2, 1), (1, 3), (3, 4), (4, 2),
             (2, 5), (5, 6), (6, 3), (6, 7), (7, 8), (8, 0),
             (8, 9), (9, 4), (3, 3))

    def node_features(self):
        row = np.arange(self.n, dtype=np.float64)[:, None]
        col = np.arange(6, dtype=np.float64)[None, :]
        return np.sin((row + 1) * (col + 1) * .37) + row * (col + 1) / 17

    def block(self, item, hops):
        selected = {int(item)}
        for _ in range(hops):
            previous = set(selected)
            for source, target in self.edges:
                if source in previous or target in previous:
                    selected.update((source, target))
        nodes = sorted(selected)
        local = {node: index for index, node in enumerate(nodes)}
        edges = [(local[source], local[target]) for source, target in self.edges
                 if source in local and target in local]
        return nodes, edges

    def collate(self, items, hops=1, standardized=False):
        nodes, groups, edges = [], [], []
        for group, item in enumerate(items):
            block_nodes, block_edges = self.block(item, hops)
            base = len(nodes)
            nodes.extend(block_nodes)
            groups.extend([group] * len(block_nodes))
            edges.extend((base + source, base + target) for source, target in block_edges)
        features = self.node_features()
        if standardized:
            features = (features - features.mean(0)) / features.std(0)
        return dict(items=list(items), nodes=np.asarray(nodes, dtype=np.int64),
                    features=features[nodes], batch_index=np.asarray(groups, dtype=np.int64),
                    node_mask=np.ones(len(nodes), dtype=bool),
                    edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                    edge_mask=np.ones(len(edges), dtype=bool))


def dense_tokens(model, store, ids, features):
    """Original uncomposed live projection and two GNN layers, per complete ego."""
    scales, masks = [], []
    for hops in (1, 2):
        blocks = [store.block(item, hops) for item in ids]
        layout = packed_layout(blocks, features.device)
        reference = batch_reference(layout, list(range(len(ids))), features.dtype, features.device)
        local = features[reference.nodes]
        value = local @ model.input.weight.T + model.input.bias
        first = model.layers[0]
        value = torch.tanh(value @ first.self_linear.weight.T + first.self_linear.bias
                           + (reference.incoming @ value) @ first.incoming.weight.T
                           + (reference.outgoing @ value) @ first.outgoing.weight.T)
        scales.append(second_layer_readout(
            value, reference.incoming, reference.outgoing,
            reference.groups, [len(nodes) for nodes, _ in blocks], model.layers[1]))
        masks.append(reference.sizes > 0)
    return torch.stack(scales, 1), torch.stack(masks, 1)
