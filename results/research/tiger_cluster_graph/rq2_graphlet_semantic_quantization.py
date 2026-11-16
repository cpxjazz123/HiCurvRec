"""RQ2: behaviour-edge recovery by Graphlet type and semantic relation.

For each real behaviour edge we attach:
  * Graphlet membership on the original behaviour graph: triangle, K1,3 star,
    4-path chain, 4-cycle;
  * semantic relation from the Amazon category paths: same fine, same coarse
    different fine, cross coarse.

Then build global K=10 (and K=5/15/20 for stability) kNN graphs from:
  * original sentence-T5 text embedding (cosine);
  * pure-Euclidean TIGER quantized q (Euclidean).

Recovery = fraction of behaviour edges in the subgroup that appear in the
symmetrised kNN union.  No training, no behaviour supervision, no Stage3.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/rq2_graphlet_semantic_quantization.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
Q_VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "tiger_vectors.npz"
BEHAVIOUR_CSV = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
PGD_MICRO = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior.micro"
K13 = REPO / "results" / "research" / "tiger_cluster_graph" / "k13_edges.csv"
LABELS = REPO / "results" / "research" / "semantic_tree_hypothesis" / "semantic_tree_labels.csv"
OUT_DIR = REPO / "results" / "research" / "rq2_graphlet_semantic_quantization"
OUT_JSON = OUT_DIR / "rq2_graphlet_semantic_quantization.json"
OUT_MD = OUT_DIR / "rq2_graphlet_semantic_quantization.md"
OUT_PAIRS = OUT_DIR / "cross_coarse_behaviour_sample.csv"
K_VALUES = (5, 10, 15, 20)
K_MAIN = 10
K_MAX = max(K_VALUES)
RELATIONS = ("same_fine", "same_coarse_diff_fine", "cross_coarse")
GRAPHLETS = ("triangle", "star", "chain", "cycle")
PGD_COLUMNS = {
    "triangle": "triangle",
    "cycle": "4-cycle",
    "chain": "4-path",
}


def normalise_title(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def read_behavior() -> np.ndarray:
    edges = []
    with BEHAVIOUR_CSV.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = (int(value) for value in line.replace(",", " ").split()[:2])
            edges.append((left, right))
    array = np.asarray(edges, dtype=np.int64)
    return np.sort(array, axis=1)


def map_pgd_edges() -> dict:
    """PGD relabels vertices; recover the original pair for every micro row."""
    pairs = []
    lookup, inverse = {}, []
    with BEHAVIOUR_CSV.open() as handle:
        for line in handle:
            left, right = (int(value) for value in line.replace(",", " ").split()[:2])
            for vertex in (left, right):
                if vertex not in lookup:
                    lookup[vertex] = len(inverse)
                    inverse.append(vertex)
    micro = pd.read_csv(PGD_MICRO, skiprows=1, header=None,
                        names=["src", "dst", "triangle", "2-star", "4-clique",
                               "4-chordal-cycle", "4-tailed-triangle", "4-cycle",
                               "3-star", "4-path"])
    mapped = {}
    for values in micro.itertuples(index=False, name=None):
        left = inverse[int(values[0]) - 1]
        right = inverse[int(values[1]) - 1]
        pair = (left, right) if left < right else (right, left)
        mapped[pair] = {
            "triangle": int(values[2]),
            "cycle": int(values[7]),
            "chain": int(values[9]),
        }
    return mapped


def load_graphlet_table(edges: np.ndarray) -> pd.DataFrame:
    pgd = map_pgd_edges()
    exact = pd.read_csv(K13)
    exact_map = {(int(a), int(b)): int(v) for a, b, v in exact[["src", "dst", "k13"]].itertuples(index=False)}
    rows = []
    for left, right in edges:
        pair = (int(left), int(right)) if left < right else (int(right), int(left))
        values = pgd.get(pair)
        if values is None:
            raise ValueError(f"missing PGD edge {pair}")
        rows.append({
            "left": int(left), "right": int(right),
            "triangle": int(values["triangle"]),
            "star": int(exact_map.get(pair, 0)),
            "chain": int(values["chain"]),
            "cycle": int(values["cycle"]),
        })
    return pd.DataFrame(rows)


def rank_within_bins(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    for value in np.unique(bins):
        mask = bins == value
        order = np.argsort(values[mask], kind="mergesort")
        sorted_values = values[mask][order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1])
        ends = np.concatenate([starts[1:], [len(sorted_values)]]) - 1
        ranks = np.empty(mask.sum(), dtype=float)
        for start, end in zip(starts, ends):
            ranks[order[start:end + 1]] = (start + end) / 2.0
        output[mask] = ranks / max(len(ranks) - 1, 1)
    return output


def add_dominant_graphlet(edges: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Assign one primary graphlet by the highest degree-binned count rank.

    Presence is nearly universal in this graph (star/chain/cycle > 0 for almost
    every edge), so a presence label cannot distinguish types.  We therefore use
    the top-quartile rank within the degree-max bin and assign the highest-ranked
    graphlet; below 0.75 is `low`.
    """
    degree = np.bincount(edges.ravel(), minlength=int(edges.max()) + 1)
    dmax = np.maximum(degree[edges[:, 0]], degree[edges[:, 1]])
    bins = pd.qcut(dmax, 5, labels=False, duplicates="drop")
    ranks = {name: rank_within_bins(table[name].to_numpy(dtype=float), np.asarray(bins))
             for name in GRAPHLETS}
    rank_matrix = np.column_stack([ranks[name] for name in GRAPHLETS])
    best = rank_matrix.argmax(axis=1)
    best_value = rank_matrix.max(axis=1)
    second = np.sort(rank_matrix, axis=1)[:, -2]
    dominant = np.array(["low"] * len(table), dtype=object)
    for index, name in enumerate(GRAPHLETS):
        dominant[(best == index) & (best_value >= 0.75)] = name
    # Near-ties are not strong single-type evidence.
    dominant[(best_value >= 0.75) & ((best_value - second) < 0.03)] = "mixed"
    out = table.copy()
    out["dominant"] = dominant
    for name in GRAPHLETS:
        out[f"{name}_high"] = ranks[name] >= 0.75
    return out


def load_labels() -> dict:
    labels = pd.read_csv(LABELS).drop_duplicates("item_id")
    return {
        int(row.item_id): (str(row.path), str(row.coarse), str(row.title))
        for row in labels.itertuples(index=False)
    }


def relation_of(left: int, right: int, labels: dict) -> str | None:
    if left not in labels or right not in labels:
        return None
    left_path, left_coarse, _ = labels[left]
    right_path, right_coarse, _ = labels[right]
    if left_path == right_path:
        return "same_fine"
    if left_coarse == right_coarse:
        return "same_coarse_diff_fine"
    return "cross_coarse"


def build_text_knn(embedding: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=K_MAX + 1, metric="cosine", algorithm="auto", n_jobs=-1)
    model.fit(embedding)
    raw = model.kneighbors(embedding, return_distance=False)
    neighbors = np.empty((len(embedding), K_MAX), dtype=np.int64)
    for index, row in enumerate(raw):
        row = row[row != index]
        if len(row) < K_MAX:
            raise ValueError(f"only {len(row)} non-self text neighbours for {index}")
        neighbors[index] = row[:K_MAX]
    return neighbors


def build_q_knn(q: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=K_MAX + 1, metric="euclidean", algorithm="auto", n_jobs=-1)
    model.fit(q)
    raw = model.kneighbors(q, return_distance=False)
    neighbors = np.empty((len(q), K_MAX), dtype=np.int64)
    for index, row in enumerate(raw):
        row = row[row != index]
        if len(row) < K_MAX:
            raise ValueError(f"only {len(row)} non-self q neighbours for {index}")
        neighbors[index] = row[:K_MAX]
    return neighbors


def undirected_edges(neighbors: np.ndarray, k: int) -> set[tuple[int, int]]:
    left = np.repeat(np.arange(len(neighbors), dtype=np.int64), k)
    right = neighbors[:, :k].reshape(-1)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    return {(int(a), int(b)) for a, b in zip(low, high)}


def rates(mask: np.ndarray, text_edges: set, q_edges: set,
          edges: np.ndarray) -> dict:
    selected = edges[mask]
    if not len(selected):
        return {"n": 0, "text": float("nan"), "q": float("nan"), "delta": float("nan")}
    text_hits = sum((int(min(a, b)), int(max(a, b))) in text_edges for a, b in selected)
    q_hits = sum((int(min(a, b)), int(max(a, b))) in q_edges for a, b in selected)
    text_rate = text_hits / len(selected)
    q_rate = q_hits / len(selected)
    return {
        "n": int(len(selected)),
        "text": float(text_rate),
        "q": float(q_rate),
        "delta": float(q_rate - text_rate),
        "relative_change": float((q_rate - text_rate) / text_rate) if text_rate else float("nan"),
    }


def subgroup_table(edges: np.ndarray, graphlets: pd.DataFrame,
                   relations: np.ndarray, labelled: np.ndarray,
                   text_edges: set, q_edges: set) -> dict:
    table = {}
    for graphlet in (*GRAPHLETS, "mixed", "low"):
        graph_mask = (graphlets["dominant"].to_numpy() == graphlet) & labelled
        table[graphlet] = {
            "all": rates(graph_mask, text_edges, q_edges, edges),
        }
        for relation_name in RELATIONS:
            mask = graph_mask & (relations == relation_name)
            table[graphlet][relation_name] = rates(mask, text_edges, q_edges, edges)
    table["all_edges"] = {"all": rates(labelled, text_edges, q_edges, edges)}
    for relation_name in RELATIONS:
        table["all_edges"][relation_name] = rates(
            relations == relation_name, text_edges, q_edges, edges)
    return table


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = read_behavior()
    graphlets = add_dominant_graphlet(edges, load_graphlet_table(edges))
    labels = load_labels()
    relations = np.array([relation_of(int(left), int(right), labels) for left, right in edges],
                         dtype=object)
    labelled = np.array([value is not None for value in relations])
    print("behaviour", len(edges), "labelled", int(labelled.sum()),
          "coverage", float(labelled.mean()), flush=True)
    print("graphlet type counts", Counter(graphlets["dominant"]), flush=True)
    print("top-quartile overlaps", {
        f"{first}+{second}": int((graphlets[f"{first}_high"].to_numpy() & graphlets[f"{second}_high"].to_numpy()).sum())
        for first, second in (("triangle", "star"), ("triangle", "chain"),
                              ("star", "chain"), ("triangle", "cycle"),
                              ("star", "cycle"), ("chain", "cycle"))
    }, flush=True)

    embedding = np.load(EMBEDDING).astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    q = np.load(Q_VECTORS)["q"].astype(np.float32)
    if q.shape != (len(embedding), 32):
        raise ValueError(f"q shape {q.shape}")

    text_neighbors = build_text_knn(embedding)
    q_neighbors = build_q_knn(q)
    print("kNN built", flush=True)

    stability = {}
    main_table = None
    for k in K_VALUES:
        text_edges = undirected_edges(text_neighbors, k)
        q_edges = undirected_edges(q_neighbors, k)
        table = subgroup_table(edges, graphlets, relations, labelled, text_edges, q_edges)
        stability[k] = table
        if k == K_MAIN:
            main_table = table
        print("k", k, "overall", table["all_edges"]["all"], flush=True)

    # Cross-coarse behaviour-edge sample for the follow-up semantic audit.
    cross_mask = labelled & (relations == "cross_coarse")
    cross_indices = np.flatnonzero(cross_mask)
    rng = np.random.default_rng(20261120)
    chosen = cross_indices if len(cross_indices) <= 30 else rng.choice(cross_indices, 30, replace=False)
    sample = []
    text10 = undirected_edges(text_neighbors, K_MAIN)
    q10 = undirected_edges(q_neighbors, K_MAIN)
    for index in chosen:
        left, right = int(edges[index, 0]), int(edges[index, 1])
        pair = (min(left, right), max(left, right))
        sample.append({
            "left_item_id": left, "left_title": labels[left][2],
            "left_path": labels[left][0],
            "right_item_id": right, "right_title": labels[right][2],
            "right_path": labels[right][0],
            "graphlet_type": str(graphlets["dominant"].iloc[index]),
            "text_recovered": pair in text10,
            "q_recovered": pair in q10,
        })
    pd.DataFrame(sample).to_csv(OUT_PAIRS, index=False)

    result = {
        "behaviour_edges": int(len(edges)),
        "labelled_edges": int(labelled.sum()),
        "label_coverage": float(labelled.mean()),
        "label_source": "semantic_tree_labels.csv from Amazon Musical Instruments categories",
        "text_embedding": "stage1_GeneEmbedding/output/sentence_t5.npy; cosine kNN",
        "q_embedding": "results/research/tiger_cluster_graph/tiger_vectors.npz q; pure Euclidean TIGER; Euclidean kNN",
        "graphlet_type_counts": {str(name): int(value) for name, value in Counter(graphlets["dominant"]).items()},
        "graphlet_source": "PGD behavior micro for triangle/4-cycle/4-path; exact K1,3 from k13_edges.csv; dominant type is the highest top-quartile degree-binned rank",
        "graphlet_overlap_note": "Top-quartile membership is multi-label; dominant assigns one primary type, with `mixed` for near-ties and `low` below the top quartile.",
        "k_values": list(K_VALUES), "k_main": K_MAIN,
        "main_k10": main_table,
        "stability": stability,
        "cross_coarse_behaviour_sample": str(OUT_PAIRS.relative_to(REPO)),
        "cross_coarse_behaviour_n": int(cross_mask.sum()),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# RQ2: Graphlet × semantic relation × quantization edge recovery",
        "",
        "Behaviour graph: **315,391** real TRAIN edges; semantic labels cover "
        f"**{int(labelled.sum())}** edges (**{labelled.mean():.2%}**).",
        "Primary Graphlet type: highest top-quartile degree-binned rank among triangle, exact K1,3 star,",
        "PGD 4-path chain, and PGD 4-cycle; `mixed` = near-tie, `low` = below top quartile in all four.",
        "Top-quartile memberships remain multi-label; the primary type is an explicit exclusive summary.",
        "",
        "Text kNN: sentence-T5 cosine. q kNN: pure-Euclidean TIGER q, Euclidean.",
        "Recovery = behaviour edge present in the symmetrised KNN union.",
        "",
        "## K=10",
        "",
        "| graphlet | relation | n | text | q | delta (q-text) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for graphlet in (*GRAPHLETS, "mixed", "low", "all_edges"):
        for relation_name in ("all", *RELATIONS):
            value = main_table[graphlet][relation_name]
            if not value["n"]:
                continue
            lines.append(f"| {graphlet} | {relation_name} | {value['n']} | "
                         f"{value['text']:.4f} | {value['q']:.4f} | {value['delta']:+.4f} |")
    lines += [
        "",
        "## K stability (overall and graphlet all-edge subsets)",
        "",
        "| K | graph | n | text | q | delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for k in K_VALUES:
        for graphlet in ("all_edges", *GRAPHLETS, "mixed", "low"):
            value = stability[k][graphlet]["all"]
            lines.append(f"| {k} | {graphlet} | {value['n']} | {value['text']:.4f} | "
                         f"{value['q']:.4f} | {value['delta']:+.4f} |")
    lines += [
        "",
        "## Cross-coarse behaviour sample",
        "",
        f"Cross-coarse behaviour edges: **{int(cross_mask.sum())}**; "
        f"30 sampled with titles and recovery flags in `{OUT_PAIRS.name}`.",
        "",
        "This sample is for the next semantic audit; cross-coarse is not automatically complementarity.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
