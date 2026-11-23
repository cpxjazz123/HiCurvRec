"""Loader and query interface for the v4 full-resolution assets (issue #248).

The loader never rebuilds anything: it reads the persisted files, verifies their hashes against
the manifest and answers queries by item, by triple and by structure class.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import load_sparse, sha256


class FullAssets:
    """Read-only access to the persisted v4 assets."""

    def __init__(self, repo: Path, root: Path | None = None):
        self.repo = Path(repo)
        self.root = Path(root or self.repo / 'results/research/issue248/assets')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.entries = {entry['name']: entry for entry in self.manifest['assets']}
        self.index = np.load(self.root / 'index.npz')
        self.shards = self.index['shard_names'].tolist()
        self.item_shard = self.index['item_shard']
        self.item_start = self.index['item_start']
        self.item_count = self.index['item_count']
        self._cache = {}

    # ---------------------------------------------------------------- verification
    def verify(self) -> dict:
        """Recompute every hash in the manifest; raise on any drift."""
        checked = []
        for entry in self.manifest['assets']:
            path = self.repo / entry['path']
            if not path.is_file():
                raise FileNotFoundError(entry['path'])
            digest = sha256(path)
            if digest != entry['sha256']:
                raise ValueError(f"hash mismatch for {entry['name']}")
            checked.append(entry['name'])
        return {'verified': len(checked), 'names': checked}

    # ---------------------------------------------------------------- shard access
    def shard(self, name: str) -> dict:
        if name not in self._cache:
            blob = np.load(self.root / 'shards' / name)
            self._cache[name] = {key: blob[key] for key in blob.files}
        return self._cache[name]

    def shard_names(self) -> list[str]:
        return list(self.shards)

    def total_rows(self) -> int:
        return int(self.index['total_rows'])

    # ---------------------------------------------------------------- queries
    def item_triples(self, item: int) -> dict:
        """Every triple of one item, with its curvature and structure class."""
        if not 0 <= int(item) < len(self.item_shard):
            raise IndexError('item index out of range')
        shard_name = self.shards[int(self.item_shard[item])]
        blob = self.shard(shard_name)
        start = int(self.item_start[item]); stop = start + int(self.item_count[item])
        return {'centre': blob['centre'][start:stop], 'b': blob['b'][start:stop],
                'c': blob['c'][start:stop], 'curvature': blob['curvature'][start:stop],
                'effective_references': blob['effective_references'][start:stop],
                'valid': blob['valid'][start:stop], 'invalid_reason': blob['invalid_reason'][start:stop],
                'structure_class': blob['structure_class'][start:stop],
                'bc_dist_sq': blob['bc_dist_sq'][start:stop], 'shard': shard_name,
                'start': start, 'stop': stop}

    def item_summary(self, item: int) -> dict:
        blob = np.load(self.root / 'item_summary.npz')
        return {key: blob[key][int(item)] for key in blob.files}

    def triple(self, m: int, b: int, c: int) -> dict:
        """Exact lookup of one triple; raises KeyError when the triple does not exist."""
        if b > c:
            b, c = c, b
        rows = self.item_triples(m)
        order = np.lexsort((rows['c'], rows['b']))
        keys = rows['b'][order] * (1 << 32) + rows['c'][order]
        position = int(np.searchsorted(keys, b * (1 << 32) + c))
        if position >= len(keys) or int(keys[position]) != b * (1 << 32) + c:
            raise KeyError((int(m), int(b), int(c)))
        row = int(order[position])
        return {'centre': int(m), 'b': int(b), 'c': int(c),
                'curvature': float(rows['curvature'][row]),
                'effective_references': int(rows['effective_references'][row]),
                'valid': bool(rows['valid'][row]),
                'invalid_reason': int(rows['invalid_reason'][row]),
                'structure_class': int(rows['structure_class'][row]),
                'bc_dist_sq': int(rows['bc_dist_sq'][row])}

    def structure_class_counts(self) -> dict:
        counts = self.index['structure_class_counts']
        names = self.index['structure_class_names']
        return {str(names[i]): int(counts[i]) for i in range(len(names))}

    def rows_of_structure_class(self, class_id: int) -> dict:
        """Every row of one structure class, read shard by shard."""
        rows = {'centre': [], 'b': [], 'c': [], 'curvature': []}
        for name in self.shards:
            blob = self.shard(name)
            mask = blob['structure_class'] == class_id
            if mask.any():
                rows['centre'].append(blob['centre'][mask])
                rows['b'].append(blob['b'][mask])
                rows['c'].append(blob['c'][mask])
                rows['curvature'].append(blob['curvature'][mask])
        return {key: (np.concatenate(value) if value else np.zeros(0, dtype=np.int64))
                for key, value in rows.items()}

    # ---------------------------------------------------------------- graph assets
    def visible_graph(self):
        return load_sparse(self.repo / self.entries['visible_forward']['path'])

    def undirected_projection(self):
        return load_sparse(self.repo / self.entries['visible_undirected']['path'])
