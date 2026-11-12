"""Issue #197 acceptance 4: the two regimes' shares on the real TRAIN graph, and acceptance 2's independent recomputation.

The shares decide whether this arm is worth dispatching at all: a Delta that is positive for 96% of items makes the mechanism the baseline with a curvature
swap on everything, and the issue says that case has to be reported as "this arm equals X" rather than trained as if it were a finding.

The recomputation is a second, deliberately naive implementation of the same definition for 200 items -- it reads the parquet again, rebuilds the groups by
BFS instead of union-find, and averages edge weights by explicit loops -- so agreement is evidence and not a tautology.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_sign_rq"
SAMPLE = 200
SEED = 42


def load_sign():
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location("arm_sign", TREE / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["arm_sign"] = arm
    spec.loader.exec_module(arm)
    from model import sign_metric
    return arm, sign_metric


def edges_of(frame: pd.DataFrame):
    return sorted({(int(h[-1]), int(t)) for h, t in
                   zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                   if h is not None and len(h) > 0})


def naive_delta(edges, items):
    """A second implementation: BFS components over the triangle-closed edges, explicit weight averaging."""
    adjacency = defaultdict(set)
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    shared = {(i, j): len(adjacency[i] & adjacency[j]) for i, j in edges}
    closed = [(i, j) for (i, j) in edges if shared[(i, j)] >= 1]
    closed_adjacency = defaultdict(set)
    for i, j in closed:
        closed_adjacency[i].add(j)
        closed_adjacency[j].add(i)
    group: dict[int, int] = {}
    for node in list(closed_adjacency):
        if node in group:
            continue
        queue = deque([node])
        group[node] = node
        while queue:
            current = queue.popleft()
            for neighbour in closed_adjacency[current]:
                if neighbour not in group:
                    group[neighbour] = node
                    queue.append(neighbour)
    degrees = defaultdict(int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    af3 = {(i, j): 4.0 - degrees[i] - degrees[j] + 3.0 * shared[(i, j)] for i, j in edges}
    ordered = sorted(af3.values())
    q05 = ordered[int(0.05 * (len(ordered) - 1))]
    q95 = ordered[int(0.95 * (len(ordered) - 1))]
    span = max(q95 - q05, 1e-12)
    weight = {e: 1.0 + min(max((v - q05) / span, 0.0), 1.0) for e, v in af3.items()}
    inside, outside = defaultdict(list), defaultdict(list)
    for (i, j), w in weight.items():
        same = group.get(i) is not None and group.get(i) == group.get(j)
        for node in (i, j):
            (inside if same else outside)[node].append(w)
    out = {}
    for node in items:
        a, b = inside.get(node, []), outside.get(node, [])
        out[node] = None if not a or not b else sum(a) / len(a) - sum(b) / len(b)
    return out


def main() -> int:
    arm, sign = load_sign()
    frame = pd.read_parquet(arm.TRAIN_FILE)
    edges = edges_of(frame)
    groups = sign.components(sign.closed_subgraph(edges))
    weights = sign.af3_weights(edges)
    delta = sign.delta_of(edges, weights, groups)

    known = {k: v for k, v in delta.items() if v is not None}
    positive = [k for k, v in known.items() if v > 0.0]
    negative = [k for k, v in known.items() if v < 0.0]
    n_items = int(max(max(e) for e in edges)) + 1

    rng = np.random.default_rng(SEED)
    sample = [int(x) for x in rng.choice(n_items, size=min(SAMPLE, n_items), replace=False)]
    naive = naive_delta(edges, sample)
    mismatches = []
    for item in sample:
        a = delta.get(item)
        b = naive.get(item)
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
            mismatches.append({"item": item, "union_find": a, "bfs": b})

    report = {
        "train_file": str(arm.TRAIN_FILE.relative_to(REPO_ROOT)),
        "edges": len(edges),
        "closed_edges": len(sign.closed_subgraph(edges)),
        "groups": len(set(groups.values())),
        "items_in_groups": len(groups),
        "items_total": n_items,
        "delta_defined": len(known),
        "delta_missing": len(delta) - len(known),
        "positive": len(positive),
        "negative": len(negative),
        "positive_share_of_defined": len(positive) / max(len(known), 1),
        "missing_share_of_touched": (len(delta) - len(known)) / max(len(delta), 1),
        "degenerate_single_regime": bool(known and (len(positive) / len(known) > 0.95 or len(positive) / len(known) < 0.05)),
        "recompute_sample": SAMPLE,
        "recompute_mismatches": len(mismatches),
        "recompute_first_mismatches": mismatches[:5],
        "delta_quantiles": {
            "q05": float(np.quantile(list(known.values()), 0.05)),
            "q50": float(np.quantile(list(known.values()), 0.50)),
            "q95": float(np.quantile(list(known.values()), 0.95)),
            "mean": float(np.mean(list(known.values()))),
        } if known else None,
    }
    (HERE / "sign_shares.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
