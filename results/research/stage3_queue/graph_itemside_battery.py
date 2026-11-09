"""More readings, same question: does a SID keep the item graph's local structure?

Adds density, density ratio, count-weighted clustering, group-macro averages and
the rank separation (AUC) between intra-group and cross-group Forman-Ricci
values, on top of the gap readings already tried.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "analysis_sid_recall" / "r2"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
OUT = Path(__file__).resolve().parent / "graph_itemside_battery.json"
CAP = 60_000  # the three levels share one graph; 60k edges keeps the sweep under ten minutes


SESSIONS = 60_000


def seqs():
    """A fixed random 80k-session sample: the full 397k makes np.unique the bottleneck."""
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    histories = [h for h in frame["history"] if h is not None and len(h) >= 2]
    rng = np.random.default_rng(42)
    if len(histories) > SESSIONS:
        pick = rng.choice(len(histories), SESSIONS, replace=False)
        histories = [histories[i] for i in pick]
    for history in histories:
        yield np.asarray([int(x) for x in history])


def build(kind: str):
    rows = []
    for items in seqs():
        n = len(items)
        if kind == "window2":
            for shift in (1, 2):
                if n > shift:
                    rows.append(np.stack([items[:-shift], items[shift:]], axis=1))
        else:  # session_all_pairs: sample pairs per session so 397k sessions stay bounded
            m = min(n, 40)
            left, right = np.triu_indices(m, k=1)
            if len(left) > 20:
                rng = np.random.default_rng(42 + n)
                pick = rng.choice(len(left), 20, replace=False)
                left, right = left[pick], right[pick]
            rows.append(np.stack([items[left], items[right]], axis=1))
    edges = np.sort(np.concatenate(rows, axis=0), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    if len(uniq) > CAP:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(uniq), CAP, replace=False)
        uniq, counts = uniq[pick], counts[pick]
    nodes, inverse = np.unique(uniq, return_inverse=True)
    index = inverse.reshape(-1, 2)
    return nodes, index, counts.astype(np.float64)


def core(index, counts, n):
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    W = sparse.coo_matrix((counts, (index[:, 0], index[:, 1])), shape=(n, n)).tocsr()
    W = W + W.T
    deg = np.asarray(A.sum(axis=1)).ravel()
    S = A @ A
    tri = np.asarray(S[index[:, 0], index[:, 1]]).ravel()
    af3 = 4.0 - deg[index[:, 0]] - deg[index[:, 1]] + 3.0 * tri
    wdeg = np.asarray(W.sum(axis=1)).ravel()
    SW = W @ W
    wtri = np.asarray(SW[index[:, 0], index[:, 1]]).ravel()
    return A, W, deg, tri, af3, wdeg, wtri


def readings(index, counts, A, W, deg, tri, af3, wdeg, wtri, groups):
    same = groups[index[:, 0]] == groups[index[:, 1]]
    intra_idx, inter_idx = index[same], index[~same]
    out = {"edges": int(len(index)), "intra_share": float(same.mean())}
    # density: intra edges over all intra pairs, both within the observed item set
    sizes = np.bincount(groups[groups >= 0])
    possible = (sizes * (sizes - 1) / 2.0).sum()
    out["density_intra"] = float(len(intra_idx) / possible) if possible else None
    cross_pairs = (len(groups[groups >= 0]) ** 2 - (sizes ** 2).sum()) / 2.0
    out["density_ratio"] = (out["density_intra"] / (len(inter_idx) / cross_pairs)
                            if cross_pairs and len(inter_idx) else None)
    # count-weighted clustering (Onnela) inside groups
    if len(intra_idx):
        M = sparse.coo_matrix((counts[same], (intra_idx[:, 0], intra_idx[:, 1])), shape=A.shape).tocsr()
        M = M + M.T
        k = np.asarray((M > 0).sum(axis=1)).ravel()
        strength = np.asarray(M.sum(axis=1)).ravel()
        # Onnela's cube-of-weights form is quadratic in nnz and dominated the run;
        # the strength-weighted triangle count below is the cheap surrogate.
        tri_w = np.asarray((M @ M).multiply(M).sum(axis=1)).ravel()
        den = strength * (k - 1.0)
        ok = (den > 0) & (k >= 2)
        out["clos_strength_mean"] = float((tri_w[ok] / den[ok]).mean()) if ok.any() else None
        # macro average of per-group clustering, equal weight per group
        G = sparse.coo_matrix((np.ones(len(intra_idx)), (intra_idx[:, 0], intra_idx[:, 1])), shape=A.shape).tocsr()
        G = G + G.T
        d = np.asarray(G.sum(axis=1)).ravel()
        SG = G @ G
        diag3 = np.asarray(SG.multiply(G).sum(axis=1)).ravel()
        mask = d >= 2
        if mask.any():
            local = diag3[mask] / (d[mask] * (d[mask] - 1.0))
            group_of = groups[mask]
            per_group = pd.Series(local).groupby(group_of).mean()
            out["clos_macro"] = float(per_group.mean())
            out["clos_micro"] = float(local.mean())
    if len(intra_idx) and len(inter_idx):
        a, b = af3[same], af3[~same]
        stat = mannwhitneyu(a, b, alternative="greater")
        out["af3_auc"] = float(stat.statistic / (len(a) * len(b)))
        out["af3_gap_z_intra"] = float((a.mean() - b.mean()) / (a.std() or 1.0))
        out["af3_med_gap"] = float(np.median(a) - np.median(b))
        out["af3_intra"] = float(a.mean())
    return out


def main():
    ledger = json.loads(LEDGER.read_text())
    numbers = sorted((k for k, e in ledger.items() if k.isdigit() and not e.get("retired")), key=int)
    built = {kind: build(kind) for kind in ("window2",)}
    prepared = {}
    for kind, (nodes, index, counts) in built.items():
        prepared[kind] = (nodes, index, counts, *core(index, counts, len(nodes)))
    rows = []
    for index_number, number in enumerate(numbers):
        if index_number % 10 == 0:
            print("arm %d/%d" % (index_number, len(numbers)), flush=True)
        sid = MIRROR / number / "item_sids.json"
        if not sid.is_file():
            continue
        table = json.loads(sid.read_text())
        recall = None
        rpath = MIRROR / number / "recall.json"
        if rpath.is_file():
            value = json.loads(rpath.read_text()).get("test_recall@10")
            recall = float(value) if isinstance(value, (int, float)) else None
        for kind, (nodes, index, counts, A, W, deg, tri, af3, wdeg, wtri) in prepared.items():
            position = {int(node): i for i, node in enumerate(nodes)}
            for level in (1, 2, 3):
                groups = np.full(len(nodes), -1, dtype=np.int64)
                for key, tokens in table.items():
                    pos = position.get(int(key))
                    if pos is None or not isinstance(tokens, (list, tuple)) or len(tokens) < level:
                        continue
                    groups[pos] = hash(tuple(int(x) for x in tokens[:level])) & 0x7FFFFFFF
                rows.append({"number": number, "graph": kind, "level": level, "recall@10": recall,
                             **readings(index, counts, A, W, deg, tri, af3, wdeg, wtri, groups)})
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    fields = [k for k in rows[0] if k.startswith(("clos_", "af3_", "density_"))]
    scores = []
    for kind in prepared:
        for level in (1, 2, 3):
            subset = [r for r in rows if r["graph"] == kind and r["level"] == level
                      and r["recall@10"] is not None]
            for field in fields:
                usable = [r for r in subset if r.get(field) is not None]
                if len(usable) < 10:
                    continue
                xs = [r[field] for r in usable]; ys = [r["recall@10"] for r in usable]
                rho, p_rho = spearmanr(xs, ys); r, p_r = pearsonr(xs, ys)
                scores.append({"graph": kind, "level": level, "metric": field, "n": len(usable),
                               "spearman": rho, "p_spearman": p_rho, "pearson": r, "p_pearson": p_r})
    scores.sort(key=lambda s: -abs(s["spearman"]))
    print("%-17s %-5s %-20s %3s %8s %8s %8s %8s" % ("graph", "level", "metric", "n", "rho", "p_rho", "r", "p_r"))
    for s in scores[:14]:
        print("%-17s %-5d %-20s %3d %8.3f %8.4f %8.3f %8.4f" % (s["graph"], s["level"], s["metric"],
              s["n"], s["spearman"], s["p_spearman"], s["pearson"], s["p_pearson"]))


if __name__ == "__main__":
    main()
