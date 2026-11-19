"""Frozen triangle/open-wedge pairs (degree matched) for structural measurements."""
import numpy as np
from pathlib import Path
from scipy import sparse

OUT = Path('results/research/sid_utilization/triangle_wedge_pairs.npz')
GRAPH = Path('results/research/behavior_graph/graph.npz')
PAIRS = 20000
BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def bucket(value):
    for limit in BUCKETS:
        if value <= limit:
            return limit
    return 10_000


def main():
    payload = np.load(GRAPH)
    rows, cols = payload['rows'], payload['cols']
    n = int(payload['nodes'])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n)
    adjacency = sparse.csr_matrix((np.ones(len(rows) * 2, dtype=bool),
                                   (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                                  shape=(n, n))
    rng = np.random.default_rng(42)
    buckets = {}
    for node in range(n):
        if degree[node] > 0:
            buckets.setdefault(bucket(int(degree[node])), []).append(node)
    buckets = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}

    order = rng.permutation(len(rows))
    triangles, wedges = [], []
    for index in order:
        if len(triangles) >= PAIRS:
            break
        a, b = int(rows[index]), int(cols[index])
        common = np.intersect1d(adjacency[a].indices, adjacency[b].indices)
        if not len(common):
            continue
        c = int(common[rng.integers(0, len(common))])
        members = [a, b, c]
        centre = max(members, key=lambda node: degree[node])
        leaves = [node for node in members if node != centre]
        pool = buckets.get(bucket(int(degree[centre])))
        if pool is None:
            continue
        for _ in range(200):
            candidate = int(pool[rng.integers(0, len(pool))])
            neighbours = adjacency[candidate].indices
            if len(neighbours) < 2:
                continue
            picks = neighbours[rng.integers(0, len(neighbours), size=8)]
            pairs = [(int(x), int(y)) for i, x in enumerate(picks) for y in picks[i + 1:] if x != y]
            if not pairs:
                continue
            left, right = pairs[rng.integers(0, len(pairs))]
            if adjacency[left, right]:
                continue
            if bucket(int(degree[left])) != bucket(int(degree[leaves[0]])):
                continue
            if bucket(int(degree[right])) != bucket(int(degree[leaves[1]])):
                continue
            triangles.append((centre, leaves[0], leaves[1]))
            wedges.append((candidate, left, right))
            break
    np.savez_compressed(OUT, triangles=np.asarray(triangles, dtype=np.int64),
                        wedges=np.asarray(wedges, dtype=np.int64))
    print('pairs', len(triangles), '->', OUT)


if __name__ == '__main__':
    main()
