"""Step 5, RQ4: Euclidean vs Poincare metric-embedding distortion of the behaviour graph.

Both geometries are trained under an identical protocol -- same node set, same
dimension, the SAME training pairs with the SAME ground-truth shortest-path distances,
the same evaluation pairs, the same optimiser, learning rate, step count, batch size
and seed. The only difference is the distance function. Each side learns one global
positive scale fitted on the EVALUATION pairs only.

Evaluation pairs are drawn from a node sample DISJOINT from the one the training pairs
are drawn from, so nothing is scored on pairs it was fitted on. A second experiment
holds out 10% of the edges, trains on the reduced graph's distances and scores the
removed edges against degree-matched non-edges; the true distances of the held-out
edges are never needed, only the ranking, so nothing leaks through them.

Run: /home/wlia0047/ar57/scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/step5_embedding.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import embed_lib as E  # noqa: E402
import graph_lib as L  # noqa: E402

SOURCE_SAMPLE = 8000
TRAIN_PAIRS = 300_000
EVAL_PAIRS = 100_000
HELDOUT_EDGE_FRACTION = 0.10
NEG_PER_EDGE = 1


def build_graph() -> tuple[nx.Graph, np.ndarray]:
    lcc = np.load(C.OUT_DIR / "lcc_edges.npz")
    nodes = lcc["nodes"]
    edges = lcc["edges"]
    g = nx.Graph()
    g.add_nodes_from(range(len(nodes)))
    g.add_edges_from(map(tuple, edges))
    return g, nodes


def source_sample(n_nodes: int, seed: int) -> np.ndarray:
    """The node sample the metric embedding is fitted and scored on.

    An earlier version drew a DISJOINT node sample for the evaluation pairs. That is
    wrong: those nodes are almost never touched by a training pair, so their
    embeddings stay near their initialisation and the evaluation measures an
    untrained model. The separation belongs at the PAIR level, so one node sample
    serves both and the pair sets are made disjoint instead.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n_nodes)[:SOURCE_SAMPLE])


def distances_to_sources(g: nx.Graph, sources: np.ndarray) -> np.ndarray:
    adj = nx.to_scipy_sparse_array(g, format="csr")
    from scipy.sparse.csgraph import shortest_path
    D = shortest_path(adj, method="D", unweighted=True, directed=False,
                      indices=sources.astype(np.int64))
    return np.asarray(D, dtype=np.float64)


def make_pairs(rows: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random pairs among the source SAMPLE, with their true distances taken from
    the corresponding rows. Distinct endpoints, positive distance."""
    n = rows.size
    out = []
    got = 0
    while got < k:
        m = int((k - got) * 1.4) + 16
        a = rows[rng.integers(0, n, m)]
        b = rows[rng.integers(0, n, m)]
        keep = a != b
        out.append(np.stack([a[keep], b[keep]], axis=1))
        got += int(keep.sum())
    return np.concatenate(out)[:k]


def group_masks(pairs: np.ndarray, true_d: np.ndarray, degrees: np.ndarray,
                comm: np.ndarray) -> dict:
    masks = {}
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 9)):
        m = (true_d >= lo) & (true_d <= hi)
        masks[f"path_len_{lo}_{hi}"] = m
    deg = np.maximum(degrees[pairs[:, 0]], degrees[pairs[:, 1]])
    for lo, hi in ((1, 5), (6, 15), (16, 60), (61, 10 ** 9)):
        masks[f"max_degree_{lo}_{hi}"] = (deg >= lo) & (deg <= hi)
    same = comm[pairs[:, 0]] == comm[pairs[:, 1]]
    masks["within_community"] = same
    masks["cross_community"] = ~same
    return masks


def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[t={time.time()-t0:.1f}s] device={device}", flush=True)

    g, nodes = build_graph()
    n = g.number_of_nodes()
    degrees = np.array([d for _, d in g.degree()], dtype=np.int64)
    print(f"[t={time.time()-t0:.1f}s] LCC n={n} m={g.number_of_edges()}", flush=True)

    comm = np.full(n, -1, dtype=np.int64)
    parts = nx.community.louvain_communities(g, weight=None, resolution=1.0, seed=C.SEED)
    for i, p in enumerate(parts):
        for v in p:
            comm[v] = i
    print(f"[t={time.time()-t0:.1f}s] {len(parts)} Louvain communities", flush=True)

    rows = source_sample(n, C.SEED)
    D = distances_to_sources(g, rows)
    pos_of = np.searchsorted(rows, np.arange(n))
    in_sample = np.isin(np.arange(n), rows)
    # Pin the distance matrix against networkx before anything depends on it.
    _chk = np.random.default_rng(0)
    for _ in range(5):
        _i = int(_chk.integers(0, rows.size))
        _tgt = int(_chk.integers(0, n))
        _ref = int(nx.single_source_shortest_path_length(g, int(rows[_i]))[_tgt])
        if int(D[_i, _tgt]) != _ref:
            raise AssertionError(
                f"distance matrix is mis-indexed: D[{_i},{_tgt}]={int(D[_i,_tgt])} "
                f"but networkx says {_ref} for (row {int(rows[_i])}, node {_tgt})")
    print(f"[t={time.time()-t0:.1f}s] distances done ({rows.size} sampled nodes), "
          f"matrix pinned against networkx", flush=True)

    rng = np.random.default_rng(C.SEED)
    train_pairs = make_pairs(rows, TRAIN_PAIRS, rng)
    # Isolation at the PAIR level via an int64 key min*N+max and np.isin, instead of
    # a Python set of tuples per candidate: the set version spent minutes on this loop.
    NKEY = int(rows.size)
    def pkey(p):
        lo = np.minimum(p[:, 0], p[:, 1]).astype(np.int64)
        hi = np.maximum(p[:, 0], p[:, 1]).astype(np.int64)
        return lo * NKEY + hi
    seen_keys = np.unique(pkey(train_pairs))
    eval_pairs = np.zeros((0, 2), dtype=np.int64)
    for _ in range(20):
        need = EVAL_PAIRS - len(eval_pairs)
        if need <= 0:
            break
        cand = make_pairs(rows, int(need * 1.6) + 64, rng)
        fresh = ~np.isin(pkey(cand), seen_keys)
        cand = cand[fresh]
        if len(cand) == 0:
            continue
        k = pkey(cand)
        _, first = np.unique(k, return_index=True)
        cand = cand[np.sort(first)][:need]
        eval_pairs = np.concatenate([eval_pairs, cand])
        seen_keys = np.union1d(seen_keys, pkey(cand))
    eval_pairs = eval_pairs[:EVAL_PAIRS]
    # distances_to_sources returns (n_sources, n_NODES): the ROW index is the
    # position of a sampled source inside `rows`, while the COLUMN index is the node
    # id itself. Pair endpoints are drawn from `rows`, so the row of a pair comes from
    # searchsorted and the column is the raw node id. The sanity check above pins this.
    train_d = D[np.searchsorted(rows, train_pairs[:, 0]), train_pairs[:, 1]]
    eval_d = D[np.searchsorted(rows, eval_pairs[:, 0]), eval_pairs[:, 1]]
    ok = (train_d > 0) & np.isfinite(train_d)
    train_pairs, train_d = train_pairs[ok], train_d[ok]
    ok = (eval_d > 0) & np.isfinite(eval_d)
    eval_pairs, eval_d = eval_pairs[ok], eval_d[ok]
    overlap = len(set(map(tuple, train_pairs)) & set(map(tuple, eval_pairs)))
    print(f"[t={time.time()-t0:.1f}s] train pairs {len(train_pairs)} "
          f"(d {train_d.min():.0f}-{train_d.max():.0f}), eval pairs {len(eval_pairs)} "
          f"(d {eval_d.min():.0f}-{eval_d.max():.0f}), overlapping pairs {overlap}", flush=True)

    # neighbourhood ground truth over the sampled nodes only
    # torch.topk on the GPU replaces 8000 separate argsort calls over an 8000-long row
    sub_nodes = rows
    kmax = max(C.NEIGHBOURHOOD_K)
    Dt = torch.as_tensor(D, dtype=torch.float32, device=device)
    Dt.fill_diagonal_(float("inf"))
    top = torch.topk(Dt, kmax, dim=1, largest=False).indices.cpu().numpy()
    # columns of D are node ids, so a top-k column index IS a node id
    neighbours = {int(rows[i]): [int(j) for j in top[i]] for i in range(rows.size)}
    del Dt

    masks = group_masks(eval_pairs, eval_d, degrees, comm)
    masks_eval = {k: v for k, v in masks.items()}

    rows_csv = []
    detail = {}
    for dim in C.EMBED_DIMS:
        for kind in ("euclid", "poincare"):
            X, scale, hist = E.train_metric_embedding(
                kind, n, dim, train_pairs, train_d, steps=C.EMBED_STEPS, lr=C.EMBED_LR,
                batch=C.EMBED_BATCH_PAIRS, seed=C.SEED, device=device, log_every=1000)
            ev = E.evaluate(kind, X, eval_pairs, eval_d, C.NEIGHBOURHOOD_K,
                            neighbours, n, device=device)
            grp = E.grouped_distortion(kind, X, eval_pairs, eval_d, masks_eval,
                                      "", ev["fitted_scale"], device=device)
            key = f"{kind}_d{dim}"
            detail[key] = {"learned_scale_during_training": scale, "history": hist,
                           "evaluation": ev, "grouped": grp}
            rows_csv.append({
                "geometry": kind, "dim": dim,
                "normalised_stress": ev["normalised_stress"],
                "mean_relative_distortion": ev["mean_relative_distortion"],
                "spearman": ev["spearman"],
                **{f"neighbourhood_preservation@{k}": v for k, v in
                   ev["neighbourhood_preservation"].items()},
                "radius_median": ev["radius_median"],
                "radius_p99": ev["radius_p99"],
                "share_radius_gt_0.99": ev["share_radius_gt_0.99"],
                "all_finite": ev["all_finite"],
                # the eval-fitted scale can absorb a divergence, so the training loss
                # trajectory is reported next to the evaluation numbers
                "train_loss_final": hist[-1]["loss"] if hist else None,
                "train_loss_first": hist[0]["loss"] if hist else None,
            })
            print(f"[t={time.time()-t0:.1f}s] {key}: stress={ev['normalised_stress']:.4f} "
                  f"MRD={ev['mean_relative_distortion']:.4f} rho={ev['spearman']:.4f}"
                  + (f" r_med={ev['radius_median']:.4f} r>.99={ev['share_radius_gt_0.99']:.4f}"
                     if kind == "poincare" else ""), flush=True)

    # ---- held-out edge experiment -------------------------------------------------
    edges = np.array(list(g.edges()), dtype=np.int64)
    rng2 = np.random.default_rng(C.SEED)
    perm = rng2.permutation(edges.shape[0])
    k = int(HELDOUT_EDGE_FRACTION * edges.shape[0])
    held = edges[perm[:k]]
    kept = edges[perm[k:]]
    g_red = nx.Graph()
    g_red.add_nodes_from(range(n))
    g_red.add_edges_from(map(tuple, kept))
    print(f"[t={time.time()-t0:.1f}s] reduced graph: {g_red.number_of_edges()} edges "
          f"({k} held out)", flush=True)

    tr_rows_r = source_sample(n, C.SEED)
    Dtr_r = distances_to_sources(g_red, tr_rows_r)
    rng3 = np.random.default_rng(C.SEED + 1)
    hpairs = make_pairs(tr_rows_r, 200_000, rng3)
    hd = Dtr_r[np.searchsorted(tr_rows_r, hpairs[:, 0]), hpairs[:, 1]]
    ok = (hd > 0) & np.isfinite(hd)
    hpairs, hd = hpairs[ok], hd[ok]
    red_deg = np.array([d for _, d in g_red.degree()], dtype=np.int64)

    edges_out = []
    for dim in C.EMBED_DIMS:
        for kind in ("euclid", "poincare"):
            X, scale, hist = E.train_metric_embedding(
                kind, n, dim, hpairs, hd, steps=C.EMBED_STEPS, lr=C.EMBED_LR,
                batch=C.EMBED_BATCH_PAIRS, seed=C.SEED, device=device, log_every=1000)
            Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12) if kind == "poincare" else \
                X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            # degree-matched negatives: for each held-out edge, nodes not adjacent in the
            # reduced graph, matched on the endpoint degree decile
            # Degree-decile matched negatives, drawn in batches instead of a Python
            # loop over every held-out edge.
            dec = np.searchsorted(np.sort(red_deg), red_deg).astype(np.int64)
            kept_key = np.unique(np.minimum(kept[:, 0], kept[:, 1]).astype(np.int64) * n
                                 + np.maximum(kept[:, 0], kept[:, 1]).astype(np.int64))
            a_all = np.repeat(held[:, 0], NEG_PER_EDGE)
            b_all = np.repeat(held[:, 1], NEG_PER_EDGE)
            neg_dst = np.full(a_all.size, -1, dtype=np.int64)
            TRIES = 128
            for t in range(TRIES):
                need = neg_dst < 0
                if not need.any():
                    break
                cand = rng3.integers(0, n, size=int(need.sum()))
                aa, bb = a_all[need], b_all[need]
                cand_key = np.minimum(aa, cand) * n + np.maximum(aa, cand)
                ok = (cand != aa) & ~np.isin(cand_key, kept_key) \
                    & (np.abs(dec[cand] - dec[bb]) <= 1)
                idx = np.where(need)[0][ok]
                neg_dst[idx] = cand[ok]
            got = neg_dst >= 0
            neg_src, neg_dst = a_all[got], neg_dst[got]
            all_a = np.concatenate([held[:, 0], neg_src.astype(np.int64)])
            all_b = np.concatenate([held[:, 1], neg_dst.astype(np.int64)])
            all_lab = np.concatenate([np.ones(len(held), dtype=np.int64),
                                      np.zeros(len(neg_src), dtype=np.int64)])
            d = E.paired_distance(kind, Xn, all_a, all_b)
            auc = E.mean_auc(-d, all_lab)   # closer = higher score
            edges_out.append({"geometry": kind, "dim": dim, "heldout_auc": auc,
                              "train_loss_final": hist[-1]["loss"] if hist else None,
                              "n_positive": int(len(held)), "n_negative": int(len(neg_src))})
            print(f"[t={time.time()-t0:.1f}s] held-out edges {kind}_d{dim}: AUC={auc:.4f} "
                  f"(pos {len(held)}, neg {len(neg_src)})", flush=True)

    out = {"protocol": {
               "device": device, "nodes": int(n), "edges": int(g.number_of_edges()),
               "source_sample": int(rows.size),
               "pair_isolation": "unordered pair sets of train and eval are disjoint; "
                                 "the SAME nodes are used, otherwise the evaluation "
                                 "would score untrained embeddings",
               "train_pairs": int(len(train_pairs)), "eval_pairs": int(len(eval_pairs)),
               "overlapping_train_eval_pairs": int(overlap),
               "dims": list(C.EMBED_DIMS), "steps": C.EMBED_STEPS, "lr": C.EMBED_LR,
               "batch": C.EMBED_BATCH_PAIRS, "seed": C.SEED,
               "optimiser": "AdamW on both sides, identical hyper-parameters",
               "scale": "one global positive scale per geometry, fitted on evaluation pairs only",
               "held_out_edges": int(k), "held_out_fraction": HELDOUT_EDGE_FRACTION},
           "distortion": detail, "rows": rows_csv, "heldout_edge": edges_out}
    (C.OUT_DIR / "embedding_distortion_comparison.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    with open(C.OUT_DIR / "embedding_distortion_comparison.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dims = list(C.EMBED_DIMS)
    xs = np.arange(len(dims))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, title in ((axes[0], "normalised_stress", "normalised stress (lower better)"),
                              (axes[1], "mean_relative_distortion", "mean relative distortion (lower better)"),
                              (axes[2], "spearman", "Spearman rho (higher better)")):
        for kind, c in (("euclid", "C0"), ("poincare", "C1")):
            ys = [r[metric] for r in rows_csv if r["geometry"] == kind]
            ax.plot(xs, ys, "o-", color=c, label=kind)
        ax.set_xticks(xs); ax.set_xticklabels([f"d={d}" for d in dims])
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("RQ4: Euclidean vs Poincare embedding of the real behaviour graph "
                 "(identical protocol; scale fitted on evaluation pairs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / "embedding_distortion.png", dpi=C.FIG_DPI)

    print(f"\nwritten: {C.OUT_DIR/'embedding_distortion_comparison.csv'}")
    print(f"written: {C.OUT_DIR/'embedding_distortion_comparison.json'}")
    print(f"written: {C.OUT_DIR/'embedding_distortion.png'}")
    print(f"[t={time.time()-t0:.1f}s] total", flush=True)


if __name__ == "__main__":
    main()