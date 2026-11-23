"""GNN-ready ego subgraphs and deterministic batching (issue #249, CONTRACT.md §5).

Only the training-visible graph is read. The full 1-hop ego is reused from the frozen #246 asset;
the 2-hop ego is produced on demand from the visible CSR, deterministically and never truncated.

Nothing here trains a GNN and nothing here produces an embedding: the interface only hands a model
the inputs (nodes, local positions, directed edges, masks and constant node features).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import load_sparse

#: The constant features the interface offers. Structure labels and the 768-dim semantic
#: embedding are deliberately absent: they must not become mandatory GNN inputs.
FEATURE_NAMES = ('degree_in', 'degree_out', 'reciprocal_degree',
                 'log_degree_in', 'log_degree_out', 'isolated')


class EgoStore:
    """Ego subgraphs of the training-visible graph, with a deterministic 1-hop / 2-hop builder."""

    def __init__(self, repo: Path, assets_root: Path | None = None):
        self.repo = Path(repo)
        root = Path(assets_root or self.repo / 'assets')
        self.root = root
        self.train = root / 'train'
        self.forward = load_sparse(self.train / 'visible_forward.npz')
        self.reverse = load_sparse(self.train / 'visible_reverse.npz')
        self.undirected = load_sparse(self.train / 'visible_undirected.npz')
        self.ids = np.load(self.train / 'item_ids.npy')
        self.n = len(self.ids)
        self._node_degree = np.diff(self.undirected.indptr).astype(np.int64)
        self._in_degree = np.diff(self.reverse.indptr).astype(np.int64)
        self._out_degree = np.diff(self.forward.indptr).astype(np.int64)
        self._reciprocal = np.asarray(self.forward.multiply(self.reverse).getnnz(axis=1)).ravel()
        manifest = json.loads((root / 'manifest.json').read_text())
        entry = next(e for e in manifest['assets'] if e['name'] == 'ego_full_1hop_visible')
        with np.load(self.repo / entry['path']) as frozen:
            self._frozen_ego = {key: frozen[key] for key in
                                ('nodes', 'edges', 'node_offsets', 'edge_offsets', 'root_local')}

    # ------------------------------------------------------------------ features
    def node_features(self) -> np.ndarray:
        """The constant per-node feature matrix, in the frozen column order."""
        in_degree = self._in_degree.astype(np.float64)
        out_degree = self._out_degree.astype(np.float64)
        reciprocal = self._reciprocal.astype(np.float64)
        isolated = ((in_degree + out_degree) == 0).astype(np.float64)
        return np.column_stack([in_degree, out_degree, reciprocal,
                                 np.log1p(in_degree), np.log1p(out_degree), isolated])

    def standardization(self) -> dict:
        """Mean and standard deviation, fitted on the training-visible items only."""
        features = self.node_features()
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return {'feature_names': list(FEATURE_NAMES), 'mean': mean.tolist(), 'std': std.tolist(),
                'fitted_on': 'training-visible items only',
                'items': int(self.n)}

    def standardize(self, features: np.ndarray) -> np.ndarray:
        stats = self.standardization()
        mean = np.asarray(stats['mean'])
        std = np.asarray(stats['std'])
        return (features - mean) / std

    # ------------------------------------------------------------------ ego
    def _neighbours(self, node: int) -> np.ndarray:
        start, end = self.undirected.indptr[node], self.undirected.indptr[node + 1]
        return self.undirected.indices[start:end].astype(np.int64)

    def ego(self, node: int, hops: int = 1) -> dict:
        """The induced ego subgraph of one item, without truncation.

        The returned block holds the global node ids, the local index of every node, the directed
        edges (source, target, local) with their direction, the node and edge masks, and the
        centre's local position.
        """
        node = int(node)
        if not 0 <= node < self.n:
            raise IndexError('item id out of range')
        if hops == 1:
            frozen = self._frozen_ego
            ns, ne = frozen['node_offsets'][node:node + 2]
            es, ee = frozen['edge_offsets'][node:node + 2]
            nodes = frozen['nodes'][ns:ne]
            edges = frozen['edges'][es:ee]
            centre = int(frozen['root_local'][node])
            return {
                'root': node, 'root_local': centre, 'nodes': nodes,
                'local_of': np.arange(len(nodes), dtype=np.int64),
                'edges': edges, 'edge_direction': np.ones(len(edges), dtype=np.int8),
                'node_mask': np.ones(len(nodes), dtype=bool),
                'edge_mask': np.ones(len(edges), dtype=bool),
                'features': self.node_features()[nodes],
                'is_centre': np.arange(len(nodes)) == centre,
            }
        visited = {node}
        frontier = {node}
        for _ in range(int(hops)):
            nxt = set()
            for current in frontier:
                nxt.update(int(x) for x in self._neighbours(current))
            nxt -= visited
            visited |= nxt
            frontier = nxt
        nodes = np.array(sorted(visited), dtype=np.int64)
        local = {int(g): index for index, g in enumerate(nodes.tolist())}
        sources, targets, directions = [], [], []
        for g in nodes.tolist():
            out_start, out_end = self.forward.indptr[g], self.forward.indptr[g + 1]
            for target in self.forward.indices[out_start:out_end].tolist():
                if target in local:
                    sources.append(local[g])
                    targets.append(local[target])
                    directions.append(1)
            in_start, in_end = self.reverse.indptr[g], self.reverse.indptr[g + 1]
            for source in self.reverse.indices[in_start:in_end].tolist():
                if source in local and not self._has_forward_edge(int(source), g):
                    sources.append(local[source])
                    targets.append(local[g])
                    directions.append(-1)
        edges = np.column_stack([np.array(sources, dtype=np.int64),
                                  np.array(targets, dtype=np.int64)])
        edge_direction = np.array(directions, dtype=np.int8)
        if edges.size == 0:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_direction = np.zeros(0, dtype=np.int8)
        order = np.lexsort((edges[:, 1], edges[:, 0])) if len(edges) else np.zeros(0, dtype=np.int64)
        edges = edges[order]
        edge_direction = edge_direction[order]
        return {
            'root': node,
            'root_local': int(local[node]),
            'nodes': nodes,
            'local_of': np.array([local[int(g)] for g in nodes.tolist()], dtype=np.int64),
            'edges': edges,
            'edge_direction': edge_direction,
            'node_mask': np.ones(len(nodes), dtype=bool),
            'edge_mask': np.ones(len(edges), dtype=bool),
            'features': self.node_features()[nodes],
            'is_centre': np.arange(len(nodes), dtype=np.int64) == local[node],
        }

    def _has_forward_edge(self, source: int, target: int) -> bool:
        start, end = self.forward.indptr[source], self.forward.indptr[source + 1]
        indices = self.forward.indices[start:end]
        return bool(np.any(indices == target))

    # ------------------------------------------------------------------ batching
    def collate(self, items, hops: int = 1, standardized: bool = True) -> dict:
        """Compose several egos into one batch, preserving the item order.

        The result holds the concatenated nodes, the per-node batch index, the batched edges with
        their direction, the masks, the centred features and the recovery indices.
        """
        items = [int(item) for item in items]
        blocks = [self.ego(item, hops=hops) for item in items]
        nodes, batch_index, edges, edge_direction, edge_batch = [], [], [], [], []
        local_offset = 0
        for index, block in enumerate(blocks):
            nodes.append(block['nodes'])
            batch_index.append(np.full(len(block['nodes']), index, dtype=np.int64))
            if len(block['edges']):
                edges.append(block['edges'] + local_offset)
                edge_direction.append(block['edge_direction'])
                edge_batch.append(np.full(len(block['edges']), index, dtype=np.int64))
            local_offset += len(block['nodes'])
        node_matrix = np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int64)
        features = np.concatenate([block['features'] for block in blocks]) if blocks \
            else np.zeros((0, len(FEATURE_NAMES)))
        if standardized and len(features):
            features = self.standardize(features)
        roots_global = []
        offset = 0
        for block in blocks:
            roots_global.append(offset + int(block['root_local']))
            offset += len(block['nodes'])
        return {
            'items': items,
            'batch_index': np.concatenate(batch_index) if batch_index else np.zeros(0, dtype=np.int64),
            'nodes': node_matrix,
            'edges': np.concatenate(edges) if edges else np.zeros((0, 2), dtype=np.int64),
            'edge_direction': np.concatenate(edge_direction) if edge_direction else np.zeros(0, dtype=np.int8),
            'edge_batch': np.concatenate(edge_batch) if edge_batch else np.zeros(0, dtype=np.int64),
            'node_mask': np.ones(len(node_matrix), dtype=bool),
            'edge_mask': np.ones(sum(len(edge) for edge in edges), dtype=bool),
            'features': features,
            'root_local': np.array([block['root_local'] for block in blocks], dtype=np.int64),
            'root_global': np.array(roots_global, dtype=np.int64),
            'node_counts': np.array([len(block['nodes']) for block in blocks], dtype=np.int64),
            'hops': int(hops),
        }

    def collate_sparse(self, items, hops: int = 1, chunk: int = 512,
                       standardized: bool = True) -> dict:
        """The same batch, computed block by block so the peak memory stays bounded."""
        items = [int(item) for item in items]
        blocks = []
        for start in range(0, len(items), chunk):
            blocks.append(self.collate(items[start:start + chunk], hops=hops,
                                       standardized=standardized))
        if not blocks:
            return self.collate(items, hops=hops, standardized=standardized)
        nodes, batch_index, edges, edge_direction, edge_batch = [], [], [], [], []
        offset = 0
        item_offset = 0
        roots_global = []
        for block in blocks:
            nodes.append(block['nodes'])
            batch_index.append(block['batch_index'] + item_offset)
            roots_global.append(block['root_global'] + offset)
            if len(block['edges']):
                edges.append(block['edges'] + offset)
                edge_direction.append(block['edge_direction'])
                edge_batch.append(block['edge_batch'] + item_offset)
            offset += len(block['nodes'])
            item_offset += len(block['items'])
        return {
            'items': items,
            'batch_index': np.concatenate(batch_index),
            'nodes': np.concatenate(nodes),
            'edges': np.concatenate(edges) if edges else np.zeros((0, 2), dtype=np.int64),
            'edge_direction': np.concatenate(edge_direction) if edge_direction else np.zeros(0, dtype=np.int8),
            'edge_batch': np.concatenate(edge_batch) if edge_batch else np.zeros(0, dtype=np.int64),
            'node_mask': np.ones(offset, dtype=bool),
            'edge_mask': np.ones(sum(len(edge) for edge in edges), dtype=bool),
            'features': np.concatenate([block['features'] for block in blocks]),
            'root_local': np.concatenate([block['root_local'] for block in blocks]),
            'root_global': np.concatenate(roots_global),
            'node_counts': np.concatenate([block['node_counts'] for block in blocks]),
            'hops': int(hops),
        }


def write_batching_index(store: EgoStore, path: Path) -> dict:
    """Persist the per-item ego sizes and the standardization statistics.

    The node lists themselves are not stored: they are recomputed deterministically from the
    visible CSR, which keeps the persisted footprint small and removes any chance of a stale copy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = np.zeros(store.n, dtype=np.int64)
    for node in range(store.n):
        sizes[node] = len(store.ego(node, hops=1)['nodes'])
    np.savez_compressed(path, one_hop_sizes=sizes)
    report = {'path': str(path), 'items': int(store.n),
              'one_hop_mean': float(sizes.mean()), 'one_hop_max': int(sizes.max()),
              'isolated_items': int((sizes == 1).sum()),
              'standardization': store.standardization()}
    (path.parent / (path.stem + '_report.json')).write_text(json.dumps(report, indent=2) + '\n')
    return report
