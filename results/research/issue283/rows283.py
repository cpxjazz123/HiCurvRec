"""Issue #283: build the per-row role/class/orbit table from the frozen triplet shards."""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, '.')
from results.research.issue283 import joint283 as J
from tools.graph_infra.assets import load_sparse, sha256

SHARDS = [f'shard_{i:03d}.npz' for i in range(13)]


def main() -> None:
    J.ROWS.mkdir(parents=True, exist_ok=True)
    graph_path = 'assets/train/visible_forward.npz'
    graph = load_sparse(graph_path).tocsr()
    n = graph.shape[0]
    coo = graph.tocoo()
    keys = np.unique(coo.row.astype(np.int64) * n + coo.col.astype(np.int64))

    def has_edge(src, dst):
        probe = src.astype(np.int64) * n + dst.astype(np.int64)
        pos = np.searchsorted(keys, probe)
        pos = np.clip(pos, 0, len(keys) - 1)
        return keys[pos] == probe

    manifest = {'schema': 'genrec.issue283.rows_manifest/1',
                'graph': {'path': graph_path, 'sha256': sha256(graph_path), 'nodes': int(n),
                          'directed_edges': int(graph.nnz)},
                'order': 'a=row.b, m=row.centre, b=row.c; code bits m->a(32),m->b(16),a->m(8),a->b(4),b->m(2),b->a(1)',
                'classes': J.CLASS_NAMES, 'symmetric': [J.CLASS_NAMES[i] for i in sorted(J.SYMMETRIC_IDS)],
                'families': J.FAMILY, 'shards': {}}
    total = 0
    class_counts = np.zeros(len(J.CLASS_NAMES), dtype=np.int64)
    family_counts = {}
    orbit_counts = np.zeros((3, 3), dtype=np.int64)
    for name in SHARDS:
        path = f'assets/shards/{name}'
        blob = np.load(path)
        m = blob['centre'].astype(np.int32)
        b = blob['b'].astype(np.int32)
        c = blob['c'].astype(np.int32)
        m2a = has_edge(m, b); m2b = has_edge(m, c)
        a2m = has_edge(b, m); a2b = has_edge(b, c)
        b2m = has_edge(c, m); b2a = has_edge(c, b)
        code = ((m2a.astype(np.int64) << 5) | (m2b.astype(np.int64) << 4) |
                (a2m.astype(np.int64) << 3) | (a2b.astype(np.int64) << 2) |
                (b2m.astype(np.int64) << 1) | b2a.astype(np.int64))
        stored = blob['structure_class'].astype(np.int64)
        canonical = J.CANONICAL_CODE[code].astype(np.int64)
        if not np.array_equal(canonical, stored):
            bad = int((canonical != stored).sum())
            raise AssertionError(f'{name}: canonical class mismatch on {bad} rows')
        cls = J.CLASS_OF_CODE[code]
        if (cls < 0).any():
            raise AssertionError(f'{name}: infeasible code present')
        orbit = J.ORBIT_OF_CODE[code]
        out = J.ROWS / name
        np.savez(out, a=b, m=m, b=c, cls=cls.astype(np.int8), orbit=orbit.astype(np.int8),
                 code=code.astype(np.uint8))
        rows = len(m)
        total += rows
        class_counts += np.bincount(cls, minlength=len(J.CLASS_NAMES))
        for pos in range(3):
            orbit_counts[pos] += np.bincount(orbit[:, pos], minlength=3)
        manifest['shards'][name] = {'rows': int(rows), 'sha256': sha256(path), 'rows_sha256': sha256(out),
                                     'class_counts': {J.CLASS_NAMES[i]: int(v) for i, v in
                                                      enumerate(np.bincount(cls, minlength=len(J.CLASS_NAMES))) if v},
                                     'orbit_class_counts': {f'{J.CLASS_NAMES[i]}': int(v) for i, v in
                                                            enumerate(np.bincount(cls, minlength=len(J.CLASS_NAMES))) if v}}
        print(f'{name}: {rows} rows, classes {int((np.bincount(cls, minlength=len(J.CLASS_NAMES)) > 0).sum())}',
              flush=True)
    if total != 28391948:
        raise AssertionError(f'total rows {total} != 28391948')
    manifest['total_rows'] = int(total)
    manifest['class_counts'] = {J.CLASS_NAMES[i]: int(v) for i, v in enumerate(class_counts) if v}
    manifest['family_counts'] = {}
    for ci, name in enumerate(J.CLASS_NAMES):
        fam = J.FAMILY[ci]
        manifest['family_counts'][fam] = manifest['family_counts'].get(fam, 0) + int(class_counts[ci])
    manifest['orbit_counts'] = {f'{prefix}_{k}': int(orbit_counts[pos, k])
                                for pos, prefix in enumerate(('m', 'a', 'b')) for k in range(3)}
    manifest['definition_sha256'] = sha256('results/research/issue283/joint283.py')
    (J.P / 'ROWS_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'total': total, 'class_counts': manifest['class_counts'],
                      'family_counts': manifest['family_counts']}, indent=1))


if __name__ == '__main__':
    main()
