"""Experiment 0: locate the behaviour-edge loss across x -> z -> q -> x_hat.

The four representations are built from the frozen Stage2 TIGER checkpoint already used by the
RQ2 diagnostics:

* x       sentence-T5 768-d text embedding (``sentence_t5.npy``)
* z       encoder output, 32-d (``tiger_vectors.npz: z``)
* q       exported-SID codebook sum, 32-d (``tiger_vectors.npz: q``)
* x_hat   decoder(q), 768-d

For every representation the same K=10 global kNN rule is applied (symmetrised union), with
the metric that matches the earlier reports: cosine for the 768-d text/reconstruction, Euclidean
for the 32-d latent/quantised representation.  K=5/15/20, an all-cosine and an all-Euclidean
variant, and a random-tie-break ensemble are reported as sensitivity checks.

The two recovery rates requested for Stage2 are reported:

1. overall recovery: all TRAIN behaviour edges found in the kNN graph;
2. conditional retention: of the behaviour edges already found by x, how many survive each later
   stage (x->z, x->z->q, x->z->q->x_hat).

No training, no behaviour supervision, no Stage3.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/quantization_loss_localization/experiment0_quantization_localization.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TREE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "tiger_vectors.npz"
VECTORS_META = REPO / "results" / "research" / "tiger_cluster_graph" / "tiger_vectors_meta.json"
BEHAVIOUR = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
PGD_MICRO = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior.micro"
K13 = REPO / "results" / "research" / "tiger_cluster_graph" / "k13_edges.csv"

OUT_JSON = HERE / "experiment0_metrics.json"
OUT_MD = HERE / "experiment0_metrics.md"
OUT_REPORT = HERE / "REPORT_QUANTIZATION_LOSS_LOCALIZATION.md"

K_VALUES = (5, 10, 15, 20)
K_MAIN = 10
K_MAX = max(K_VALUES)
SEEDS = 8
XHAT_SEEDS = 4
BATCH = 4096

# behaviour-edge Graphlet labels, copied from the RQ2 analysis so the per-type table is comparable
PGD_COLUMNS = {"triangle": 2, "cycle": 7, "chain": 9}
GRAPHLETS = ("triangle", "star", "chain", "cycle")
SIDECAR_TYPES = ("mixed", "low")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_behavior() -> np.ndarray:
    rows = []
    with BEHAVIOUR.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = (int(value) for value in line.replace(",", " ").split()[:2])
            rows.append((left, right) if left < right else (right, left))
    edges = np.asarray(rows, dtype=np.int64)
    # a union of user-session consecutive pairs can repeat a pair across sessions
    return np.unique(edges, axis=0)


def edge_codes(pairs: np.ndarray, n: int) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.unique(low * np.int64(n) + high)


def neighbors_to_codes(neighbors: np.ndarray, k: int, n: int) -> np.ndarray:
    left = np.repeat(np.arange(n, dtype=np.int64), k)
    right = neighbors[:, :k].reshape(-1).astype(np.int64)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    return np.unique(low * np.int64(n) + high)


def load_model():
    """Load the frozen TIGER checkpoint with the baseline package, exactly as build_vectors.py."""
    import sys
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location(
        "tiger_rqvae", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["tiger_rqvae"] = package
    spec.loader.exec_module(package)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.RQVAE(config, in_dim=768)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    return model, payload


def build_xhat(q: np.ndarray) -> np.ndarray:
    model, payload = load_model()
    tensor = torch.from_numpy(np.asarray(q, dtype=np.float32))
    with torch.no_grad():
        output = model.decoder(tensor)
    return output.cpu().numpy().astype(np.float32), payload


def jitter_scale(vectors: np.ndarray) -> float:
    norms = np.linalg.norm(vectors, axis=1)
    return max(float(np.median(norms)) * 1e-6, 1e-9)


def knn(vectors: np.ndarray, metric: str, kmax: int = K_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Return per-item neighbour indices with the self row removed, plus raw distances.

    The last K+1 neighbours are enough to detect a tie at every K boundary.  For duplicated
    quantised vectors the tie is broken by adding a 1e-6-relative Gaussian jitter at the call
    site; this function itself is deterministic.
    """
    n = len(vectors)
    model = NearestNeighbors(n_neighbors=kmax + 1, metric=metric, algorithm="brute", n_jobs=-1)
    model.fit(vectors)
    distances, indices = model.kneighbors(vectors, return_distance=True)
    neighbors = np.empty((n, kmax), dtype=np.int64)
    clean_distances = np.empty((n, kmax), dtype=np.float64)
    for index in range(n):
        row = indices[index]
        dist = distances[index]
        keep = row != index
        row = row[keep]
        dist = dist[keep]
        if len(row) < kmax:
            raise ValueError(f"only {len(row)} non-self neighbours for item {index}")
        neighbors[index] = row[:kmax]
        clean_distances[index] = dist[:kmax]
    return neighbors, clean_distances


def graph_for(vectors: np.ndarray, metric: str, seed: int | None = None,
              kmax: int = K_MAX) -> tuple[dict[int, np.ndarray], dict[int, float], dict]:
    """Build edge sets for each K; jitter only when seed is not None."""
    work = vectors
    if seed is not None:
        rng = np.random.default_rng(seed)
        work = vectors + rng.normal(0.0, jitter_scale(vectors), vectors.shape).astype(np.float32)
    neighbors, distances = knn(np.asarray(work, dtype=np.float32), metric, kmax)
    n = len(vectors)
    edges = {k: neighbors_to_codes(neighbors, k, n) for k in K_VALUES if k <= kmax}
    ties = {}
    for k in K_VALUES:
        if k < kmax:
            ties[k] = float(np.mean(distances[:, k - 1] == distances[:, k]))
        else:
            ties[k] = float("nan")
    info = {
        "items": n,
        "metric": metric,
        "jitter_seed": seed,
        "unique_rows": int(len(np.unique(vectors, axis=0))),
        "duplicate_items": int(n - len(np.unique(vectors, axis=0))),
    }
    return edges, ties, info


def hit_count(b_codes: np.ndarray, e_codes: np.ndarray) -> int:
    return int(np.intersect1d(b_codes, e_codes, assume_unique=True).size)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(len(array)),
    }


def load_graphlet_table(edges: np.ndarray) -> pd.DataFrame:
    lookup, inverse = {}, []
    with BEHAVIOUR.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = (int(value) for value in line.replace(",", " ").split()[:2])
            for vertex in (left, right):
                if vertex not in lookup:
                    lookup[vertex] = len(inverse)
                    inverse.append(vertex)
    micro = pd.read_csv(
        PGD_MICRO, skiprows=1, header=None,
        names=["src", "dst", "triangle", "2-star", "4-clique", "4-chordal-cycle",
               "4-tailed-triangle", "4-cycle", "3-star", "4-path"],
    )
    mapped = {}
    for values in micro.itertuples(index=False, name=None):
        left = inverse[int(values[0]) - 1]
        right = inverse[int(values[1]) - 1]
        pair = (left, right) if left < right else (right, left)
        mapped[pair] = {key: int(values[index]) for key, index in PGD_COLUMNS.items()}
    exact = pd.read_csv(K13)
    exact_map = {
        (int(a), int(b)): int(v)
        for a, b, v in exact[["src", "dst", "k13"]].itertuples(index=False)
    }
    rows = []
    for left, right in edges:
        pair = (int(left), int(right))
        values = mapped.get(pair)
        if values is None:
            raise ValueError(f"missing PGD edge {pair}")
        rows.append({
            "left": int(left), "right": int(right),
            "triangle": values["triangle"], "star": exact_map.get(pair, 0),
            "chain": values["chain"], "cycle": values["cycle"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    vectors = np.load(VECTORS)
    x = np.load(EMBEDDING).astype(np.float32)
    z = vectors["z"].astype(np.float32)
    q = vectors["q"].astype(np.float32)
    tokens = vectors["tokens"].astype(np.int64)
    if not (len(x) == len(z) == len(q) == len(tokens)):
        raise SystemExit("representation length mismatch")
    n = len(x)
    xhat, payload = build_xhat(q)
    meta = json.loads(VECTORS_META.read_text())

    behavior = read_behavior()
    b_codes = edge_codes(behavior, n)
    print(f"behaviour edges: raw={len(behavior)} unique={len(b_codes)}", flush=True)

    # --- duplicates, norms, reconstruction distortion -------------------------------------
    distortions = {
        "x_norm": {"min": float(np.linalg.norm(x, axis=1).min()),
                   "median": float(np.median(np.linalg.norm(x, axis=1))),
                   "max": float(np.linalg.norm(x, axis=1).max())},
        "z_norm": {"min": float(np.linalg.norm(z, axis=1).min()),
                   "median": float(np.median(np.linalg.norm(z, axis=1))),
                   "max": float(np.linalg.norm(z, axis=1).max())},
        "q_norm": {"min": float(np.linalg.norm(q, axis=1).min()),
                   "median": float(np.median(np.linalg.norm(q, axis=1))),
                   "max": float(np.linalg.norm(q, axis=1).max())},
        "xhat_norm": {"min": float(np.linalg.norm(xhat, axis=1).min()),
                      "median": float(np.median(np.linalg.norm(xhat, axis=1))),
                      "max": float(np.linalg.norm(xhat, axis=1).max())},
        "z_to_q_mse": float(np.mean((z.astype(np.float64) - q.astype(np.float64)) ** 2)),
        "x_to_xhat_mse": float(np.mean((x.astype(np.float64) - xhat.astype(np.float64)) ** 2)),
        "x_to_xhat_cosine_mean": float(np.mean(
            np.sum(x * xhat, axis=1) /
            np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(xhat, axis=1), 1e-12)
        )),
    }

    # --- deterministic graphs and tie accounting -------------------------------------------
    stages = {
        "x": {"vectors": x, "metric": "cosine", "seeds": 0},
        "z": {"vectors": z, "metric": "euclidean", "seeds": SEEDS},
        "q": {"vectors": q, "metric": "euclidean", "seeds": SEEDS},
        "xhat": {"vectors": xhat, "metric": "cosine", "seeds": XHAT_SEEDS},
    }
    deterministic = {}
    for name, spec in stages.items():
        edges, ties, info = graph_for(spec["vectors"], spec["metric"], seed=None)
        deterministic[name] = {"edges": edges, "ties": ties, "info": info}
        print(f"{name}: edges K10={len(edges[K_MAIN])} ties={ties}", flush=True)

    # --- tie-broken ensembles (z/q/xhat); x is unique except 22 duplicate rows ------------
    ensembles = {}
    for name in ("z", "q", "xhat"):
        spec = stages[name]
        count = spec["seeds"]
        per_seed = []
        for seed in range(count):
            edges, _, info = graph_for(spec["vectors"], spec["metric"], seed=seed)
            per_seed.append(edges)
        ensembles[name] = per_seed
        print(f"{name}: {count} jittered graphs", flush=True)

    x_edges = deterministic["x"]["edges"]
    bx = hit_count(b_codes, x_edges[K_MAIN])
    bx_codes = np.intersect1d(b_codes, x_edges[K_MAIN], assume_unique=True)
    denom_x = len(bx_codes)

    def overall(name: str, k: int = K_MAIN) -> dict:
        if name == "x":
            hits = hit_count(b_codes, deterministic[name]["edges"][k])
            rates = [hits / len(b_codes)]
        else:
            rates = [hit_count(b_codes, graph[k]) / len(b_codes) for graph in ensembles[name]]
        return {"hits": summarize([r * len(b_codes) for r in rates]), "rate": summarize(rates)}

    def conditional(name: str, k: int = K_MAIN) -> dict:
        rates = [hit_count(bx_codes, graph[k]) / denom_x for graph in ensembles[name]]
        return {"hits": summarize([r * denom_x for r in rates]), "rate": summarize(rates)}

    def chain_conditional(names: tuple[str, ...], k: int = K_MAIN) -> dict:
        rates = []
        for seed in range(SEEDS):
            current = bx_codes
            for name in names:
                graph = deterministic[name]["edges"][k] if name == "x" else ensembles[name][seed % len(ensembles[name])][k]
                current = np.intersect1d(current, graph, assume_unique=True)
            rates.append(len(current) / denom_x)
        return {"hits": summarize([r * denom_x for r in rates]), "rate": summarize(rates)}

    primary = {}
    for name in ("x", "z", "q", "xhat"):
        primary[name] = overall(name)
    conditional_stage = {}
    for name in ("z", "q", "xhat"):
        conditional_stage[name] = conditional(name)

    paths = {
        "x_to_z": chain_conditional(("z",)),
        "x_to_q_independent": chain_conditional(("q",)),
        "x_to_xhat_independent": chain_conditional(("xhat",)),
        "x_to_z_to_q": chain_conditional(("z", "q")),
        "x_to_z_to_q_to_xhat": chain_conditional(("z", "q", "xhat")),
    }

    # conditional on the previous stage rather than always on x
    z_to_q_rates = []
    for seed in range(SEEDS):
        bz = np.intersect1d(bx_codes, ensembles["z"][seed][K_MAIN], assume_unique=True)
        bzq = np.intersect1d(bz, ensembles["q"][seed][K_MAIN], assume_unique=True)
        ratio = len(bzq) / len(bz) if len(bz) else float("nan")
        z_to_q_rates.append(ratio)
    q_to_xhat_rates = []
    for seed in range(SEEDS):
        bzq = bx_codes
        for name in ("z", "q"):
            bzq = np.intersect1d(bzq, ensembles[name][seed][K_MAIN], assume_unique=True)
        bzqx = np.intersect1d(bzq, ensembles["xhat"][seed % XHAT_SEEDS][K_MAIN], assume_unique=True)
        ratio = len(bzqx) / len(bzq) if len(bzq) else float("nan")
        q_to_xhat_rates.append(ratio)

    # --- K sensitivity (one jitter seed for z/q/xhat, deterministic x) --------------------
    k_sensitivity = {}
    for k in K_VALUES:
        row = {}
        row["x"] = {"overall": hit_count(b_codes, x_edges[k]) / len(b_codes),
                    "conditional": hit_count(bx_codes, x_edges[k]) / denom_x}
        for name in ("z", "q", "xhat"):
            row[name] = {
                "overall": hit_count(b_codes, ensembles[name][0][k]) / len(b_codes),
                "conditional": hit_count(bx_codes, ensembles[name][0][k]) / denom_x,
            }
        path_rates = []
        for seed in range(SEEDS):
            current = bx_codes
            for name in ("z", "q"):
                current = np.intersect1d(current, ensembles[name][seed][k], assume_unique=True)
            path_rates.append(len(current) / denom_x)
        row["x_to_z_to_q"] = {"mean": float(np.mean(path_rates)),
                              "sd": float(np.std(path_rates))}
        k_sensitivity[str(k)] = row

    # --- metric sensitivity at K=10 --------------------------------------------------------
    metric_sensitivity = {}
    for metric in ("cosine", "euclidean"):
        row = {}
        for name, values in (("x", x), ("xhat", xhat), ("z", z), ("q", q)):
            edges, _, _ = graph_for(values, metric, seed=0 if name in ("z", "q", "xhat") else None)
            row[name] = {
                "overall": hit_count(b_codes, edges[K_MAIN]) / len(b_codes),
                "conditional": hit_count(bx_codes, edges[K_MAIN]) / denom_x,
            }
        metric_sensitivity[metric] = row

    # --- Graphlet type recovery at K=10 ----------------------------------------------------
    graphlet_rows = []
    try:
        labels = load_graphlet_table(behavior)
        # exact same definition as rq2_graphlet_semantic_quantization.py:
        # average rank with ties inside each dmax quintile, top-quartile type, near-tie mixed.
        degree = np.bincount(behavior.ravel(), minlength=n)
        dmax = np.maximum(degree[labels["left"].to_numpy()], degree[labels["right"].to_numpy()])
        bins = pd.qcut(dmax, 5, labels=False, duplicates="drop")
        bins = np.asarray(bins)
        ranks = {}
        for name in GRAPHLETS:
            values = labels[name].to_numpy(dtype=float)
            rank = np.zeros(len(values), dtype=float)
            for b in np.unique(bins):
                mask = bins == b
                order = np.argsort(values[mask], kind="mergesort")
                sorted_values = values[mask][order]
                starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1])
                ends = np.concatenate([starts[1:], [len(sorted_values)]]) - 1
                local = np.empty(mask.sum(), dtype=float)
                for start, end in zip(starts, ends):
                    local[order[start:end + 1]] = (start + end) / 2.0
                rank[mask] = local / max(len(local) - 1, 1)
            ranks[name] = rank
        matrix = np.stack([ranks[name] for name in GRAPHLETS], axis=1)
        best = matrix.argmax(axis=1)
        best_value = matrix.max(axis=1)
        second = np.sort(matrix, axis=1)[:, -2]
        main_type = np.array(["low"] * len(labels), dtype=object)
        for index, name in enumerate(GRAPHLETS):
            main_type[(best == index) & (best_value >= 0.75)] = name
        main_type[(best_value >= 0.75) & ((best_value - second) < 0.03)] = "mixed"
        labels["main_type"] = main_type
        for name in (*GRAPHLETS, *SIDECAR_TYPES):
            subset = behavior[labels["main_type"].to_numpy() == name]
            if not len(subset):
                continue
            subset_codes = edge_codes(subset, n)
            # denominator is the part of this graphlet that x already recovers
            bx_subset = np.intersect1d(subset_codes, x_edges[K_MAIN], assume_unique=True)
            row = {"type": name, "n": int(len(subset)),
                   "x": hit_count(subset_codes, x_edges[K_MAIN]) / len(subset)}
            for stage_name in ("z", "q", "xhat"):
                row[stage_name] = float(np.mean([
                    hit_count(subset_codes, ensembles[stage_name][seed][K_MAIN]) / len(subset)
                    for seed in range(len(ensembles[stage_name]))
                ]))
                row[f"{stage_name}_conditional"] = (
                    hit_count(bx_subset, ensembles[stage_name][0][K_MAIN]) / len(bx_subset)
                    if len(bx_subset) else float("nan")
                )
            graphlet_rows.append(row)
    except Exception as error:  # the core experiment must still report if labels are unavailable
        graphlet_rows = [{"error": repr(error)}]

    # ties for each stage and K (raw, deterministic)
    tie_table = {
        name: {str(k): value for k, value in deterministic[name]["ties"].items()}
        for name in deterministic
    }
    duplicate_table = {
        name: {
            "items": deterministic[name]["info"]["items"],
            "unique_rows": deterministic[name]["info"]["unique_rows"],
            "duplicate_rows": deterministic[name]["info"]["duplicate_items"],
            "duplicate_share": deterministic[name]["info"]["duplicate_items"] / n,
        }
        for name in deterministic
    }
    q_values, q_counts = np.unique(q, axis=0, return_counts=True)
    top_duplicates = sorted((int(v) for v in q_counts), reverse=True)[:10]

    # Behaviour edges that are trivially close because both endpoints share the exact same
    # vector.  This is not geometry learning: it is the KNN tie caused by quantisation.
    duplicate_behaviour = {}
    for name, values in (("x", x), ("z", z), ("q", q), ("xhat", xhat)):
        _, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        left_group = inverse[behavior[:, 0]]
        right_group = inverse[behavior[:, 1]]
        same_row = left_group == right_group
        both_duplicated = (counts[left_group] > 1) & (counts[right_group] > 1)
        duplicate_behaviour[name] = {
            "duplicate_groups": int((counts > 1).sum()),
            "items_in_duplicate_groups": int(counts[counts > 1].sum()),
            "max_group_size": int(counts.max()),
            "same_vector_edges": int(same_row.sum()),
            "same_vector_share": float(same_row.mean()),
            "both_in_duplicate_groups_edges": int(both_duplicated.sum()),
            "both_in_duplicate_groups_share": float(both_duplicated.mean()),
        }
        if same_row.any():
            same_codes = edge_codes(behavior[same_row], n)
            if name == "x":
                hit_rate = hit_count(same_codes, x_edges[K_MAIN]) / len(same_codes)
            else:
                hit_rate = float(np.mean([
                    hit_count(same_codes, graph[K_MAIN]) / len(same_codes)
                    for graph in ensembles[name]
                ]))
            duplicate_behaviour[name]["same_vector_hits"] = hit_rate * len(same_codes)
            duplicate_behaviour[name]["same_vector_hit_rate"] = hit_rate


    result = {
        "scope": "offline reconstruction and kNN; no training, no Stage3",
        "n_items": n,
        "n_behaviour_edges": int(len(behavior)),
        "n_behaviour_codes": int(len(b_codes)),
        "k_main": K_MAIN,
        "k_values": list(K_VALUES),
        "ensemble": {"z": SEEDS, "q": SEEDS, "xhat": XHAT_SEEDS,
                     "note": "Gaussian jitter 1e-6 x median vector norm breaks exact duplicate ties; x deterministic"},
        "sources": {
            "embedding": str(EMBEDDING.relative_to(REPO)),
            "vectors": str(VECTORS.relative_to(REPO)),
            "vectors_meta": meta,
            "checkpoint": str(CHECKPOINT.relative_to(REPO)),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "sid_json": str(SID_JSON.relative_to(REPO)),
            "sid_sha256": sha256(SID_JSON),
            "behaviour_edges": str(BEHAVIOUR.relative_to(REPO)),
            "tokens_match_export": meta.get("tokens_match_export"),
            "q_source": meta.get("q_source"),
        },
        "distortion": distortions,
        "duplicates": duplicate_table,
        "duplicate_behaviour": duplicate_behaviour,
        "q_duplicate_top_group_sizes": top_duplicates,
        "tie_boundary_share": tie_table,
        "overall_recovery": primary,
        "conditional_on_x": conditional_stage,
        "path_recovery": paths,
        "z_to_q_conditional": summarize(z_to_q_rates),
        "q_to_xhat_conditional": summarize(q_to_xhat_rates),
        "k_sensitivity": k_sensitivity,
        "metric_sensitivity": metric_sensitivity,
        "graphlet_type": graphlet_rows,
        "stage_diagnostics": {
            "x_edges_K10": int(len(x_edges[K_MAIN])),
            "x_behaviour_edges": int(bx),
            "x_behaviour_precision": bx / len(x_edges[K_MAIN]),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    def fmt(stats: dict) -> str:
        return f"{stats['mean']:.6f} ± {stats['sd']:.6f}"

    lines = [
        "# Experiment 0: quantization loss localization",
        "",
        "Offline reconstruction and kNN only. No training, no Stage3.",
        "",
        f"- items: **{n}**; TRAIN behaviour edges: **{len(behavior)}**; K main: **{K_MAIN}**",
        f"- x: sentence-T5 768-d, cosine; z/q: 32-d, Euclidean; x_hat: decoder(q), cosine",
        f"- q source: `{meta.get('q_source')}`; tokens_match_export: `{meta.get('tokens_match_export')}`",
        "",
        "## Distortion",
        "",
        "| measure | value |",
        "|---|---:|",
        f"| z→q MSE | {distortions['z_to_q_mse']:.8f} |",
        f"| x→x_hat MSE | {distortions['x_to_xhat_mse']:.8f} |",
        f"| x→x_hat cosine mean | {distortions['x_to_xhat_cosine_mean']:.6f} |",
        "",
        "## Duplicates and KNN ties",
        "",
        "| stage | unique rows | duplicate rows | duplicate share | tie@K=10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("x", "z", "q", "xhat"):
        d = duplicate_table[name]
        lines.append(f"| {name} | {d['unique_rows']} | {d['duplicate_rows']} | {d['duplicate_share']:.4f} | {tie_table[name]['10']:.4f} |")
    lines += [
        "",
        "### Duplicate-vector behaviour edges",
        "",
        "| stage | duplicate groups | items | max group | same-vector edges | hit rate | both-in-duplicate-group edges | share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("x", "z", "q", "xhat"):
        d = duplicate_behaviour[name]
        hit_rate = d.get("same_vector_hit_rate", float("nan"))
        lines.append(
            f"| {name} | {d['duplicate_groups']} | {d['items_in_duplicate_groups']} | "
            f"{d['max_group_size']} | {d['same_vector_edges']} | {hit_rate:.4f} | "
            f"{d['both_in_duplicate_groups_edges']} | {d['both_in_duplicate_groups_share']:.5f} |"
        )
    lines += [
        "",
        "## Overall behaviour-edge recovery",
        "",
        "| stage | kNN edges | behaviour hits | recovery |",
        "|---|---:|---:|---:|",
    ]
    for name in ("x", "z", "q", "xhat"):
        stats = primary[name]["rate"]
        hits = primary[name]["hits"]["mean"]
        edges = len(x_edges[K_MAIN]) if name == "x" else len(ensembles[name][0][K_MAIN])
        lines.append(f"| {name} | {edges} | {hits:.1f} | {fmt(stats)} |")
    lines += [
        "",
        "## Conditional on the edges x already recovers",
        "",
        f"x behaviour edges: **{bx}** (precision of x kNN graph: **{bx / len(x_edges[K_MAIN]):.4f}**)",
        "",
        "| path | retained | rate |",
        "|---|---:|---:|",
    ]
    for label, key in (("x → z", "x_to_z"), ("x → q (independent)", "x_to_q_independent"),
                       ("x → x_hat (independent)", "x_to_xhat_independent"),
                       ("x → z → q", "x_to_z_to_q"),
                       ("x → z → q → x_hat", "x_to_z_to_q_to_xhat")):
        item = paths[key]
        lines.append(f"| {label} | {item['hits']['mean']:.1f} | {fmt(item['rate'])} |")
    lines += [
        "",
        f"- z → q conditional on x→z: **{fmt(result['z_to_q_conditional'])}**",
        f"- q → x_hat conditional on x→z→q: **{fmt(result['q_to_xhat_conditional'])}**",
        "",
        "## K sensitivity (one tie-break seed for z/q/x_hat)",
        "",
        "| K | x overall | z overall | q overall | x_hat overall | x→q | x→z→q |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in K_VALUES:
        row = k_sensitivity[str(k)]
        lines.append(
            f"| {k} | {row['x']['overall']:.5f} | {row['z']['overall']:.5f} | "
            f"{row['q']['overall']:.5f} | {row['xhat']['overall']:.5f} | "
            f"{row['q']['conditional']:.5f} | "
            f"{row['x_to_z_to_q']['mean']:.5f} |"
        )
    lines += [
        "",
        "## Metric sensitivity at K=10",
        "",
        "| metric | x overall | z overall | q overall | x_hat overall | q conditional |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ("cosine", "euclidean"):
        row = metric_sensitivity[metric]
        lines.append(
            f"| {metric} | {row['x']['overall']:.5f} | {row['z']['overall']:.5f} | "
            f"{row['q']['overall']:.5f} | {row['xhat']['overall']:.5f} | "
            f"{row['q']['conditional']:.5f} |"
        )
    if graphlet_rows and "error" not in graphlet_rows[0]:
        lines += [
            "",
            "## Graphlet-type recovery (K=10)",
            "",
            "| type | n | x | z | q | x_hat | q conditional-on-x |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in graphlet_rows:
            lines.append(
                f"| {row['type']} | {row['n']} | {row['x']:.5f} | {row['z']:.5f} | "
                f"{row['q']:.5f} | {row['xhat']:.5f} | {row['q_conditional']:.5f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(OUT_JSON)
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
