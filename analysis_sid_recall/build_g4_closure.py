"""Issue #216: 4-cycle closure vs branch, matched on degrees and on AF3.

For a centre a and two neighbours b, c with (a,b), (a,c) edges, (b,c) not an edge,
and both edges having triangle count 0 (so AF3 = 4 - d_u - d_v is fully determined
by the degrees):

    y = 1  (square)  exists d != a with (b,d) and (c,d): a-b-d-c-a closes
    y = 0  (branch)  no such d: b and c meet only at a

Both classes carry the same edge labels (two edges + one non-edge); the only
difference is the second path.  Pairs are matched on the exact degree triple
(d_a, d_b, d_c); with t = 0 that also matches both AF3 values.

Outputs (results/research/g4_closure/):
    pairs_train.npz / pairs_heldout.npz   centre, b, c, y, degrees
    coverage.json                          counts, matching depth, hashes

Split: the stable hash of the unordered endpoint pair {b, c} decides train/heldout,
so the same target pair never appears on both sides; a triple produced by another
centre is assigned by the same hash.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
OUT = REPO / "results" / "research" / "g4_closure"
PER_CENTRE_CAP = 40
HELDOUT_FRACTION = 0.2


def verify_graph() -> dict:
    metadata = json.loads(METADATA.read_text())
    record = (metadata.get("outputs") or {}).get("graph.npz")
    digest = hashlib.sha256(GRAPH.read_bytes()).hexdigest()
    if not record or digest != record.get("sha256"):
        raise RuntimeError("graph.npz does not match metadata.json; rebuild the graph")
    if (metadata.get("policy") or {}).get("single_undirected_graph") is not True:
        raise RuntimeError("the graph policy block is missing")
    return {"sha256": digest, "counts": metadata.get("counts")}


def load():
    payload = np.load(GRAPH, allow_pickle=False)
    rows = payload["rows"].astype(np.int64)
    cols = payload["cols"].astype(np.int64)
    triangles = payload["triangles"].astype(np.int64)
    n = int(payload["nodes"])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n)
    neighbours = defaultdict(set)
    edge_t = {}
    for a, b, t in zip(rows.tolist(), cols.tolist(), triangles.tolist()):
        neighbours[a].add(b)
        neighbours[b].add(a)
        edge_t[(a, b)] = t
        edge_t[(b, a)] = t
    return rows, cols, degree, neighbours, edge_t, n


def split_of(pair) -> str:
    first, second = sorted(pair)
    digest = hashlib.sha256(f"{first}-{second}".encode()).hexdigest()
    return "heldout" if int(digest[:8], 16) % 100 < HELDOUT_FRACTION * 100 else "train"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = verify_graph()
    rows, cols, degree, neighbours, edge_t, n = load()
    triples = []
    seen = set()
    per_centre = defaultdict(int)
    for a in sorted(neighbours):
        if degree[a] < 2:
            continue
        members = sorted(neighbours[a])
        for index, b in enumerate(members):
            if per_centre[a] >= PER_CENTRE_CAP:
                break
            for c in members[index + 1:]:
                if c in neighbours[b]:
                    continue
                if edge_t.get((a, b), 0) != 0 or edge_t.get((a, c), 0) != 0:
                    continue
                key = (a, min(b, c), max(b, c))
                if key in seen:
                    continue
                seen.add(key)
                common = (neighbours[b] & neighbours[c]) - {a}
                label = 1 if common else 0
                triples.append((a, min(b, c), max(b, c), label))
                per_centre[a] += 1
                if per_centre[a] >= PER_CENTRE_CAP:
                    break
    data = np.asarray(triples, dtype=np.int64).reshape(-1, 4)
    degrees = np.stack([degree[data[:, 0]], degree[data[:, 1]], degree[data[:, 2]]], axis=1)

    # Exact degree-triple matching: every positive gets one negative with identical
    # (d_a, d_b, d_c); the pools are paired so both classes have equal support.
    buckets = defaultdict(lambda: {0: [], 1: []})
    for index, (deg_triple, label) in enumerate(zip(map(tuple, degrees.tolist()), data[:, 3].tolist())):
        buckets[deg_triple][int(label)].append(index)
    matched = []
    unmatched_positive = 0
    for deg_triple, both in buckets.items():
        positives, negatives = both[1], both[0]
        if not positives or not negatives:
            unmatched_positive += len(positives)
            continue
        take = min(len(positives), len(negatives))
        rng = np.random.default_rng(abs(hash(deg_triple)) % (2 ** 32))
        pos_pick = rng.choice(positives, size=take, replace=False)
        neg_pick = rng.choice(negatives, size=take, replace=False)
        for p, q in zip(pos_pick, neg_pick):
            matched.append((int(p), int(q)))
    matched = np.asarray(matched, dtype=np.int64).reshape(-1, 2)
    pairs = []
    for p, q in matched:
        pairs.append(data[p])
        pairs.append(data[q])
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 4)
    degrees_paired = np.stack([degree[pairs[:, 0]], degree[pairs[:, 1]], degree[pairs[:, 2]]], axis=1)
    split = np.array([split_of((row[1], row[2])) for row in pairs])
    for name in ("train", "heldout"):
        mask = split == name
        np.savez_compressed(OUT / f"pairs_{name}.npz",
                            triples=pairs[mask], degrees=degrees_paired[mask],
                            labels=pairs[mask][:, 3])
    counts = {}
    for name in ("train", "heldout"):
        mask = split == name
        labels = pairs[mask][:, 3]
        counts[name] = {"pairs": int(mask.sum()), "positives": int(labels.sum()),
                        "negatives": int((labels == 0).sum()),
                        "degree_triples": int(len({tuple(d) for d in degrees_paired[mask].tolist()}))}
    report = {"graph": audit, "kept_positive_without_negative": unmatched_positive,
              "counts": counts, "per_centre_cap": PER_CENTRE_CAP,
              "heldout_fraction": HELDOUT_FRACTION,
              "pairs_sha": {name: hashlib.sha256((OUT / f"pairs_{name}.npz").read_bytes()).hexdigest()
                            for name in ("train", "heldout")}}
    (OUT / "coverage.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
