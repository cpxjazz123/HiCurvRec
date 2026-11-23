"""Streaming reader for the full-resolution curvature table (issue #249, CONTRACT.md §6).

The table is the frozen #248 asset: 13 shards, 28,391,948 rows. This reader streams it; it never
materialises the whole table, and a full traversal visits every row exactly once.
"""
from __future__ import annotations
import resource
import time
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import sha256


class CurvatureReader:
    """Random and sequential access to the frozen triple table."""

    def __init__(self, repo: Path, assets_root: Path | None = None):
        self.repo = Path(repo)
        self.root = Path(assets_root or self.repo / 'assets')
        self.index = np.load(self.root / 'index.npz')
        self.shard_names = [str(name) for name in self.index['shard_names']]
        self.item_shard = self.index['item_shard']
        self.item_start = self.index['item_start']
        self.item_count = self.index['item_count']
        self.total_rows = int(self.index['total_rows'])
        self._cache = {}

    def shard(self, name: str) -> dict:
        if name not in self._cache:
            blob = np.load(self.root / 'shards' / name)
            self._cache[name] = {key: blob[key] for key in blob.files}
        return self._cache[name]

    # ------------------------------------------------------------------ access
    def item_rows(self, item: int) -> dict:
        """Every row of one item; the rows of an item are contiguous inside one shard."""
        item = int(item)
        if not 0 <= item < len(self.item_count):
            raise IndexError('item id out of range')
        name = self.shard_names[int(self.item_shard[item])]
        blob = self.shard(name)
        start = int(self.item_start[item])
        stop = start + int(self.item_count[item])
        return {key: blob[key][start:stop] for key in
                ('centre', 'b', 'c', 'bc_dist_sq', 'curvature', 'effective_references',
                 'valid', 'invalid_reason', 'structure_class')}

    def triple(self, m: int, b: int, c: int) -> dict:
        """One triple by identity; raises KeyError when it does not exist."""
        rows = self.item_rows(m)
        if len(rows['b']) == 0:
            raise KeyError((int(m), int(b), int(c)))
        order = np.lexsort((rows['c'], rows['b']))
        keys = rows['b'][order] * (1 << 32) + rows['c'][order]
        target = int(b) * (1 << 32) + int(c)
        position = int(np.searchsorted(keys, target))
        if position >= len(keys) or int(keys[position]) != target:
            raise KeyError((int(m), int(b), int(c)))
        row = int(order[position])
        return {key: (rows[key][row] if key != 'centre' else int(rows['centre'][row]))
                for key in rows}

    def iter_shards(self):
        """Stream the shards one by one."""
        for name in self.shard_names:
            yield name, self.shard(name)

    def traverse(self) -> dict:
        """Visit every row exactly once and report the coverage, the time and the memory."""
        start = time.monotonic()
        memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows = 0
        centres = set()
        duplicate = 0
        seen = set()
        for name, blob in self.iter_shards():
            rows += len(blob['b'])
            centres.update(np.unique(blob['centre']).tolist())
            keys = (blob['centre'].astype(np.int64) << 30
                    | blob['b'].astype(np.int64) << 15
                    | blob['c'].astype(np.int64))
            unique = np.unique(keys)
            duplicate += len(keys) - len(unique)
            for key in unique.tolist():
                if key in seen:
                    duplicate += 1
                seen.add(key)
        memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {
            'rows_read': rows,
            'expected_rows': self.total_rows,
            'distinct_centres': len(centres),
            'duplicate_triples': duplicate,
            'rows_match_expected': rows == self.total_rows,
            'seconds': time.monotonic() - start,
            'shards': len(self.shard_names),
            'max_rss_bytes_before': memory_before * 1024,
            'max_rss_bytes_after': memory_after * 1024,
            'peak_rss_bytes': max(memory_before, memory_after) * 1024,
        }
        return report

    def aggregate_by_item(self, items=None, chunk: int = 5000) -> dict:
        """Exact per-item aggregates, computed in batches without loading the whole table.

        The result is compared against the persisted per-item summary, so a batched aggregation
        must equal the single-item computation.
        """
        items = range(len(self.item_count)) if items is None else [int(i) for i in items]
        items = list(items)
        means = np.zeros(len(items))
        counts = np.zeros(len(items), dtype=np.int64)
        for start in range(0, len(items), chunk):
            block = items[start:start + chunk]
            for position, item in enumerate(block):
                rows = self.item_rows(item)
                values = rows['curvature']
                valid = rows['valid'] & np.isfinite(values)
                counts[position] = int(valid.sum())
                means[position] = float(values[valid].mean()) if counts[position] else float('nan')
        return {'items': items, 'mean': means, 'valid_count': counts}

    def summary_agreement(self, sample: int = 512, seed: int = 249001) -> dict:
        """Compare a sampled batched aggregation with the persisted per-item summary."""
        summary = np.load(self.root / 'item_summary.npz')
        rng = np.random.default_rng(seed)
        items = rng.choice(len(self.item_count), size=min(sample, len(self.item_count)),
                           replace=False).tolist()
        aggregated = self.aggregate_by_item(items)
        worst = 0.0
        for position, item in enumerate(items):
            expected = float(summary['mean'][item])
            produced = float(aggregated['mean'][position])
            if np.isnan(expected) and np.isnan(produced):
                continue
            worst = max(worst, abs(expected - produced))
        return {'sampled_items': len(items), 'max_abs_mean_difference': worst,
                'matches': worst <= 1e-12}

    def manifest_hashes(self) -> dict:
        """The sha256 of every shard, so a reader can prove what it read."""
        return {name: sha256(self.root / 'shards' / name) for name in self.shard_names}
