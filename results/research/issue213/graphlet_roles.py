"""Structural roles from ORCA orbit counts, then how well each role's behaviour edges survive the SID.

Two questions, in this order:

  1. does the commerce graph's items fall into distinguishable local-structure roles -- dense community, hub, chain, bridge -- when the role comes from exact
     small-subgraph orbit counts rather than from a hand-written rule?
  2. does an existing RQ-VAE SID keep the behaviour relation of an edge equally well for every role, or is some role systematically penalised?

Method, and what it does *not* claim:

  * orbits come from ORCA (``orca node 4``, 15 node orbits of graphlets up to 4 nodes) on the undirected TRAIN graph, exact for every node;
  * roles are NMF over the orbit matrix plus four cheap invariants -- RolX's core idea: soft memberships, no hand labels, k fixed by ``ROLES``;
  * "retention" is the share of behaviour edges whose endpoints share at least one SID token, and separately all three, reported beside a matched random
    baseline at the same degree decile so a high share cannot be confused with "these items are simply high degree";
  * both SIDs are resolved by their recorded sha256, because a concurrent reorganisation moved the arm result directories; a path that does not hash to the
    recorded value is skipped and said so.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import NMF

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRAIN = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
ORCA_NODE4 = HERE / "orca" / "node4.out"
ORCA_NODE5 = HERE / "orca" / "node5.out"

#: The two SIDs used for the retention columns, by their recorded sha256.
SIDS = {
    "F0": ["results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json",
           "results/research/stage3_queue/_stage/TIGER_mixed_rq_F0/item_sids.json"],
    "TIGER_RQ-VAE": ["results/research/stage3_queue/_stage/TIGER_RQ-VAE/item_sids.json",
                     "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json"],
}
SID_SHA = {
    "F0": "afa89a10b1b197a7b56da4ad322c6ddd96a3c26c04238af326d88747378bb848",
    "TIGER_RQ-VAE": "53273cb605dc3210c20ad2ae5da778ce6fe7fd26bd8b3d6e356c2c7e0e7eed85",
}

ROLES = 6
SEED = 42
ORBITS_4 = [f"o4_{i:02d}" for i in range(15)]
ORBITS_5 = [f"o5_{i:02d}" for i in range(73)]


def edges_of() -> list[tuple[int, int]]:
    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        pairs.add((int(history[-1]), int(target)))
    return sorted(pairs)


def orca_orbits(path: Path, width: int) -> tuple[np.ndarray, str | None]:
    """Exact node-orbit counts as ORCA wrote them; ``(empty, None)`` when that size has not been computed."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, width), dtype=np.int64), None
    counts = np.loadtxt(path, dtype=np.int64)
    assert counts.ndim == 2 and counts.shape[1] == width, counts.shape
    return counts, str(path.relative_to(REPO_ROOT))


def sid_tokens(name: str) -> dict[int, tuple]:
    """The SID that carries the recorded sha256; a path that does not hash to it is skipped and said so."""
    for relative in SIDS[name]:
        path = REPO_ROOT / relative
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SID_SHA[name]:
            print(f"skipping {relative}: sha256 {digest[:12]} is not the recorded {SID_SHA[name][:12]}", flush=True)
            continue
        payload = json.loads(path.read_text())
        print(f"{name}: {relative} ({len(payload)} items, sha256 verified)", flush=True)
        return {int(k): tuple(v[:3]) for k, v in payload.items()}
    raise SystemExit(f"no copy of the {name} SID matches its recorded sha256")


def features_and_roles(graph, degrees, adj, orbits4, orbits5):
    """RolX's core idea on this graph: standardised per-node features, NMF, argmax membership.

    Returned as ``(nodes, role_of, profiles)``; ``nodes`` is ascending, so two callers that pass the
    same graph and orbits get the same role for the same item.
    """
    clustering = nx.clustering(graph)
    core = nx.core_number(graph)
    mean_neighbour_degree = {
        node: float(np.mean([degrees[n] for n in adj[node]])) if adj[node] else 0.0 for node in graph.nodes()
    }
    overlap = {}
    for node in graph.nodes():
        neigh = list(adj[node])
        if len(neigh) < 2:
            overlap[node] = 1.0
            continue
        total = 0.0
        for a, b in itertools.combinations(neigh, 2):
            union = len(adj[a] | adj[b])
            total += len(adj[a] & adj[b]) / union if union else 0.0
        overlap[node] = total / (len(neigh) * (len(neigh) - 1) / 2)

    nodes = sorted(graph.nodes())
    rows = []
    for node in nodes:
        rows.append([
            np.log1p(degrees[node]), clustering[node], core[node], np.log1p(mean_neighbour_degree[node]), overlap[node],
            *[np.log1p(v) for v in orbits4[node]],
            *([np.log1p(v) for v in orbits5[node]] if orbits5.size else []),
        ])
    features = ["log_degree", "clustering", "core", "log_mean_neighbour_degree", "overlap", *ORBITS_4]
    if orbits5.size:
        features = features + ORBITS_5
    X = np.asarray(rows, dtype=np.float64)
    X = (X - X.mean(0)) / X.std(0).clip(min=1e-9)

    model = NMF(n_components=ROLES, init="nndsvda", random_state=SEED, max_iter=800)
    W = model.fit_transform(X.clip(min=0.0))
    role_of = {node: int(np.argmax(W[i])) for i, node in enumerate(nodes)}

    profiles = []
    for r in range(ROLES):
        mass = W[:, r]
        cent = (mass[:, None] * X).sum(0) / max(mass.sum(), 1e-9)
        top = sorted(zip(features, cent.tolist()), key=lambda kv: -kv[1])[:5]
        profiles.append({"role": r, "members": int((np.asarray(list(role_of.values())) == r).sum()),
                         "top_features": [{"name": n, "value": round(v, 3)} for n, v in top]})
    return nodes, role_of, profiles


def main() -> int:
    rng = np.random.default_rng(SEED)
    edges = edges_of()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges", flush=True)

    orbits4, used4 = orca_orbits(ORCA_NODE4, 15)
    # ``--only-4`` reproduces the first pass, which had no five-node orbits yet:
    # same graph, same roles constant, 73 fewer feature columns.  It is the
    # stability check for "does the picture depend on the larger graphlets".
    only4 = "--only-4" in sys.argv
    orbits5, used5 = ((np.zeros((0, 73), dtype=np.int64), None) if only4
                      else orca_orbits(ORCA_NODE5, 73))
    print(f"orbits: 4-node {used4 or 'NOT_RUN'} | 5-node {used5 or 'NOT_RUN'}", flush=True)
    _, role_of, profiles = features_and_roles(graph, degrees, adj, orbits4, orbits5)

    def retention(name: str) -> dict:
        sid = sid_tokens(name)
        per_role = defaultdict(lambda: {"edges": 0, "share1": 0, "share3": 0})
        overall = {"edges": 0, "share1": 0, "share3": 0}
        for a, b in edges:
            if a not in sid or b not in sid:
                continue
            shared = len(set(sid[a]) & set(sid[b]))
            overall["edges"] += 1
            overall["share1"] += shared >= 1
            overall["share3"] += shared == 3
            role = role_of.get(a)
            if role is not None:
                per_role[role]["edges"] += 1
                per_role[role]["share1"] += shared >= 1
                per_role[role]["share3"] += shared == 3

        deciles = np.quantile([degrees[n] for n in sid if n in degrees], np.linspace(0, 1, 11))
        by_decile = defaultdict(list)
        for n in sid:
            if n in degrees:
                d = int(np.searchsorted(deciles, degrees[n], side="right")) - 1
                by_decile[min(max(d, 0), 9)].append(n)
        pairs = share1 = share3 = 0
        for members in by_decile.values():
            if len(members) < 2:
                continue
            for _ in range(min(4000, len(members) * 4)):
                a, b = rng.choice(members, size=2, replace=False).tolist()
                shared = len(set(sid[a]) & set(sid[b]))
                pairs += 1
                share1 += shared >= 1
                share3 += shared == 3

        for v in per_role.values():
            v["share1"] = v["share1"] / max(v["edges"], 1)
            v["share3"] = v["share3"] / max(v["edges"], 1)
        overall["share1"] = overall["share1"] / max(overall["edges"], 1)
        overall["share3"] = overall["share3"] / max(overall["edges"], 1)
        return {"overall": overall, "per_role": {int(k): v for k, v in sorted(per_role.items())},
                "matched_random": {"pairs": pairs, "share1": share1 / max(pairs, 1), "share3": share3 / max(pairs, 1)}}

    report = {
        "graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "roles": ROLES,
                  "orbit_sources": {"4-node": used4, "5-node": used5},
                  "orbits": 15 + (73 if orbits5.size else 0)},
        "role_profiles": profiles,
        "retention": {"F0": retention("F0"), "TIGER_RQ-VAE": retention("TIGER_RQ-VAE")},
        "sid_sha256": SID_SHA,
        "limits": [
            "orbits are ORCA's node orbits of graphlets up to 4 nodes (15) and, when node5.out is present, up to 5 nodes (73) as well",
            "roles are NMF over standardised features; the role count is the script's constant, not selected",
            "retention is the share of behaviour edges whose endpoints share at least one / all three SID tokens",
        ],
    }
    (HERE / ("graphlet_roles_4node.json" if only4 else "graphlet_roles.json")).write_text(
        json.dumps(report, indent=1, default=float))
    print(json.dumps({"graph": report["graph"], "profiles": profiles}, indent=1, default=float)[:2500])
    for name, block in report["retention"].items():
        print(f"--- {name}: overall {block['overall']} random {block['matched_random']}")
        for role, v in block["per_role"].items():
            print(f"    role {role}: edges {v['edges']:>7} share>=1 {v['share1']:.4f} share3 {v['share3']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
