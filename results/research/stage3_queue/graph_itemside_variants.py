"""Same question, several ways of measuring it.

The graph is always the original item graph and SIDs only label full-prefix
groups.  What changes is how the graph is built and how the readings are taken:

  graphs   consecutive pairs, pairs within a window of two, and session
           co-occurrence (pairs drawn inside one user history)
  closure  global transitivity inside groups, mean local clustering inside
           groups, multiplicity-free weighted transitivity inside groups
  curvature AF3 intra mean, AF3 gap (intra - inter), normalised gap, plain
           Forman gap (4 - d_u - d_v, no triangle term), median gap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "analysis_sid_recall" / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "graph_itemside_variants.json"
CAP = 200_000


def histories():
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        yield np.asarray([int(x) for x in history])


def pairs(seqs, window: int):
    out = []
    for items in seqs:
        for shift in range(1, window + 1):
            if len(items) > shift:
                out.append(np.stack([items[:-shift], items[shift:]], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.concatenate(out, axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges[edges[:, 0] != edges[:, 1]]


def session_pairs(seqs, per_session: int = 8, seed: int = 42):
    """All pairs inside one session, sampled uniformly so 397k sessions fit."""
    rng = np.random.default_rng(seed)
    out = []
    for items in seqs:
        n = len(items)
        if n < 2:
            continue
        wanted = min(per_session, n * (n - 1) // 2)
        left = rng.integers(0, n, size=wanted)
        right = rng.integers(0, n, size=wanted)
        keep = left != right
        if keep.any():
            out.append(np.stack([items[left[keep]], items[right[keep]]], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.sort(np.concatenate(out, axis=0), axis=1)
    return np.unique(edges, axis=0)


def subsample(edges, cap=CAP):
    if len(edges) > cap:
        rng = np.random.default_rng(42)
        edges = edges[rng.choice(len(edges), cap, replace=False)]
    return edges


def graph(edges):
    nodes, index = np.unique(edges, return_inverse=True)
    index = index.reshape(-1, 2)
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(len(nodes), len(nodes))).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    S = A @ A
    tri = np.asarray(S[index[:, 0], index[:, 1]]).ravel()
    af3 = 4.0 - deg[index[:, 0]] - deg[index[:, 1]] + 3.0 * tri
    plain = 4.0 - deg[index[:, 0]] - deg[index[:, 1]]
    return nodes, index, A, deg, tri, af3, plain


def readings(index, A, deg, tri, af3, plain, groups):
    same = groups[index[:, 0]] == groups[index[:, 1]]
    out = {}
    spread = float(af3.std()) or 1.0
    out["curv_af3_intra"] = float(af3[same].mean()) if same.any() else None
    if same.any() and (~same).any():
        out["curv_af3_gap"] = float(af3[same].mean() - af3[~same].mean())
        out["curv_af3_gap_z"] = out["curv_af3_gap"] / spread
        out["curv_plain_gap"] = float(plain[same].mean() - plain[~same].mean())
        out["curv_med_gap"] = float(np.median(af3[same]) - np.median(af3[~same]))
    intra = index[same]
    if len(intra):
        M = sparse.coo_matrix((np.ones(len(intra)), (intra[:, 0], intra[:, 1])),
                              shape=A.shape).tocsr()
        M = M + M.T
        M.data[:] = 1.0
        d = np.asarray(M.sum(axis=1)).ravel()
        S = M @ M
        tri3 = np.asarray(S.multiply(M).sum()) / 6.0
        triples = (d * (d - 1) / 2.0).sum()
        out["clos_global"] = float(3.0 * tri3 / triples) if triples else None
        diag3 = np.asarray(S.multiply(M).sum(axis=1)).ravel()   # (M^3)_vv = 2 * triangles at v
        mask = d >= 2
        if mask.any():
            local = diag3[mask] / (d[mask] * (d[mask] - 1.0))
            out["clos_local"] = float(local.mean())
    return out


def main():
    ledger = json.loads(LEDGER.read_text())
    numbers = sorted((k for k, e in ledger.items() if k.isdigit() and not e.get("retired")), key=int)
    seqs = list(histories())
    graphs = {
        "consecutive": subsample(pairs(seqs, 1)),
        "window2": subsample(pairs(seqs, 2)),
        "session_sample": subsample(pairs(seqs, 1), cap=100_000),
        "session_all_pairs": subsample(session_pairs(seqs), cap=CAP),
    }
    built = {name: graph(edges) for name, edges in graphs.items()}
    rows = []
    for number in numbers:
        sid = MIRROR / number / "item_sids.json"
        if not sid.is_file():
            continue
        table = json.loads(sid.read_text())
        recall = None
        rpath = MIRROR / number / "recall.json"
        if rpath.is_file():
            value = json.loads(rpath.read_text()).get("test_recall@10")
            recall = float(value) if isinstance(value, (int, float)) else None
        for name, (nodes, index, A, deg, tri, af3, plain) in built.items():
            position = {int(node): i for i, node in enumerate(nodes)}
            for level in (1, 2, 3):
                groups = np.full(len(nodes), -1, dtype=np.int64)
                for key, tokens in table.items():
                    pos = position.get(int(key))
                    if pos is None or not isinstance(tokens, (list, tuple)) or len(tokens) < level:
                        continue
                    groups[pos] = hash(tuple(int(x) for x in tokens[:level])) & 0x7FFFFFFF
                rows.append({"number": number, "graph": name, "level": level,
                             "recall@10": recall,
                             **readings(index, A, deg, tri, af3, plain, groups)})
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    fields = [k for k in rows[0] if k.startswith(("clos_", "curv_"))]
    scores = []
    for name in built:
        for level in (1, 2, 3):
            subset = [r for r in rows if r["graph"] == name and r["level"] == level
                      and r["recall@10"] is not None]
            for field in fields:
                usable = [r for r in subset if r.get(field) is not None]
                if len(usable) < 10:
                    continue
                xs = [r[field] for r in usable]; ys = [r["recall@10"] for r in usable]
                rho, p_rho = spearmanr(xs, ys); r, p_r = pearsonr(xs, ys)
                scores.append({"graph": name, "level": level, "metric": field, "n": len(usable),
                               "spearman": rho, "p_spearman": p_rho, "pearson": r, "p_pearson": p_r})
    scores.sort(key=lambda s: -abs(s["spearman"]))
    print("%-15s %-5s %-16s %3s %8s %8s %8s %8s" % ("graph", "level", "metric", "n", "rho", "p_rho", "r", "p_r"))
    for s in scores[:15]:
        print("%-15s %-5d %-16s %3d %8.3f %8.4f %8.3f %8.4f" % (s["graph"], s["level"], s["metric"],
              s["n"], s["spearman"], s["p_spearman"], s["pearson"], s["p_pearson"]))


if __name__ == "__main__":
    main()
