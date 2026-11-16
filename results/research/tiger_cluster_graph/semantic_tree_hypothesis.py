"""Single semantic hierarchy tree hypothesis test on sentence-T5 item embeddings.

Labels come from the original Amazon Reviews 2023 `meta_Musical_Instruments.jsonl`
category paths, matched to the local items by normalised title.  The parquet
`categories` field is empty, so this is the independent semantic hierarchy used here.
Only unambiguous title matches are kept.

Experiments:
  1. As K grows, which relation enters: same fine / same coarse different fine /
     cross coarse?  Compare text kNN with an ideal hierarchy-tree kNN.
  2. Compare graphlet distributions, hierarchy-order violations, and
     ultrametric/four-point residuals.

This is an offline diagnostic; it does not train a model or run Stage 3.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/semantic_tree_hypothesis.py
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
LABELS_CSV = REPO / "results" / "research" / "semantic_tree_hypothesis" / "semantic_tree_labels_metadata.csv"
PGD = REPO / "tools" / "research" / "third_party" / "PGD" / "pgd"
OUT_DIR = REPO / "results" / "research" / "semantic_tree_hypothesis"
OUT_JSON = OUT_DIR / "semantic_tree_hypothesis.json"
OUT_MD = OUT_DIR / "semantic_tree_hypothesis.md"
SAMPLES_CSV = OUT_DIR / "label_validation_samples.csv"
EXCLUDED_COARSE = {
    "20% back on select brands", "20% off select Musical Instruments",
    "Musical Instruments Low Return Rate Alert", "Musical Instruments Beginner Store",
    "Musical Instruments Savings", "Musical Instruments Outlet", "Lick Library",
    "Musican's Lifestyle Store", "Pre-Owned Musical Instruments & Accessories",
    "Vintage Musical Instruments", "Acoustic Guitars: Beginners",
    "V4 Customers Keep It - 50%", "Presidents Day Savings",
}
K_VALUES = (5, 10, 15, 20)
K_MAX = max(K_VALUES)
TREE_SEEDS = 20
BANDS = ((1, 5), (6, 10), (11, 15), (16, 20))
RELATIONS = ("same_fine", "same_coarse_diff_fine", "cross_coarse")
GRAPHLETS = ("K4", "diamond", "C4", "paw", "K1,3", "P4")
PGD_NAMES = {
    "4_clique": "K4", "4_chordcycle": "diamond", "4_cycle": "C4",
    "4_tailed_tris": "paw", "3_star": "K1,3", "4_path": "P4",
}


def write_edges(path: Path, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        for left, right in edges.tolist():
            handle.write(f"{int(left)},{int(right)}\n")


def run_pgd(edges: np.ndarray, name: str) -> dict:
    path = OUT_DIR / f"{name}.csv"
    log = OUT_DIR / f"{name}.pgd.log"
    write_edges(path, edges)
    with log.open("w") as handle:
        subprocess.run([str(PGD), "-f", str(path), "-w", "8", "-v"],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    text = log.read_text()
    counts = {key: int(value) for key, value in re.findall(r"total_(\w+) = (\d+)", text)}
    required = ("4_clique", "4_chordcycle", "4_cycle", "4_tailed_tris",
                "3_star", "4_path", "3_tris", "2_star")
    missing = [key for key in required if key not in counts]
    if missing:
        raise ValueError(f"PGD {name} missing {missing}")
    return counts


def graphlet_summary(counts: dict) -> dict:
    values = {name: int(counts[key]) for key, name in PGD_NAMES.items()}
    total = sum(values.values())
    return {
        "counts": values,
        "shares": {name: (value / total if total else float("nan")) for name, value in values.items()},
        "total_connected_4node": int(total),
        "triangles": int(counts["3_tris"]),
        "wedges": int(counts["2_star"]),
        "triangle_share_3node": float(counts["3_tris"] / (counts["3_tris"] + counts["2_star"]))
        if counts["3_tris"] + counts["2_star"] else float("nan"),
    }


def build_text_knn(embedding: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=K_MAX + 1, metric="cosine", algorithm="auto", n_jobs=-1)
    model.fit(embedding)
    raw = model.kneighbors(embedding, return_distance=False)
    neighbors = np.empty((len(embedding), K_MAX), dtype=np.int64)
    for index, row in enumerate(raw):
        row = row[row != index]
        if len(row) < K_MAX:
            raise ValueError(f"only {len(row)} non-self neighbours for item {index}")
        neighbors[index] = row[:K_MAX]
    return neighbors


def build_tree_knn(fine: np.ndarray, coarse: np.ndarray, seed: int = 0) -> np.ndarray:
    """Top-K under d(same fine)=1, d(same coarse)=2, d(cross coarse)=3.

    All candidates at the same tree distance are exact ties.  Ties are broken
    uniformly at random, not by item id: choosing by id makes a few low-id items
    into artificial hubs and inflates K1,3.  Different seeds give the tie
    ensemble reported by main().
    """
    rng = np.random.default_rng(seed)
    nodes = len(fine)
    fine_groups: dict[str, list[int]] = {}
    coarse_groups: dict[str, list[int]] = {}
    for index in range(nodes):
        fine_groups.setdefault(str(fine[index]), []).append(index)
        coarse_groups.setdefault(str(coarse[index]), []).append(index)
    for values in (*fine_groups.values(), *coarse_groups.values()):
        values.sort()
    cross_groups = {
        coarse_name: [index for index in range(nodes) if str(coarse[index]) != coarse_name]
        for coarse_name in coarse_groups
    }
    neighbors = np.empty((nodes, K_MAX), dtype=np.int64)
    for index in range(nodes):
        chosen: list[int] = []
        same_fine = [candidate for candidate in fine_groups[str(fine[index])] if candidate != index]
        same_coarse = [candidate for candidate in coarse_groups[str(coarse[index])]
                       if candidate != index and str(fine[candidate]) != str(fine[index])]
        cross = cross_groups[str(coarse[index])]
        for candidates in (same_fine, same_coarse, cross):
            need = K_MAX - len(chosen)
            if need <= 0:
                break
            if len(candidates) <= need:
                chosen.extend(candidates)
            else:
                chosen.extend(rng.choice(candidates, size=need, replace=False).tolist())
        if len(chosen) < K_MAX:
            raise ValueError(f"not enough tree candidates for item {index}")
        neighbors[index] = chosen
    return neighbors


def relation(source: int, target: int, fine: np.ndarray, coarse: np.ndarray) -> str:
    if fine[source] == fine[target]:
        return "same_fine"
    if coarse[source] == coarse[target]:
        return "same_coarse_diff_fine"
    return "cross_coarse"


def band_mix(neighbors: np.ndarray, fine: np.ndarray, coarse: np.ndarray) -> dict:
    output = {}
    for low, high in BANDS:
        counts = Counter()
        total = 0
        for source in range(len(neighbors)):
            for rank in range(low - 1, high):
                counts[relation(source, int(neighbors[source, rank]), fine, coarse)] += 1
                total += 1
        output[f"{low}-{high}"] = {
            "total": total,
            "counts": {name: int(counts[name]) for name in RELATIONS},
            "rates": {name: float(counts[name] / total) if total else float("nan")
                      for name in RELATIONS},
        }
    return output


def first_ranks(neighbors: np.ndarray, fine: np.ndarray, coarse: np.ndarray) -> dict:
    first_fine = np.full(len(neighbors), 1_000, dtype=np.int64)
    first_coarse = np.full(len(neighbors), 1_000, dtype=np.int64)
    first_diff_fine = np.full(len(neighbors), 1_000, dtype=np.int64)
    first_cross = np.full(len(neighbors), 1_000, dtype=np.int64)
    for source in range(len(neighbors)):
        fi, ci = fine[source], coarse[source]
        for rank, target in enumerate(neighbors[source], start=1):
            target = int(target)
            if fine[target] == fi:
                first_fine[source] = min(first_fine[source], rank)
            if coarse[target] == ci:
                first_coarse[source] = min(first_coarse[source], rank)
                if fine[target] != fi:
                    first_diff_fine[source] = min(first_diff_fine[source], rank)
            else:
                first_cross[source] = min(first_cross[source], rank)
    return {
        "same_fine": first_fine, "same_coarse": first_coarse,
        "diff_fine_same_coarse": first_diff_fine, "cross": first_cross,
    }


def violation_summary(neighbors: np.ndarray, fine: np.ndarray, coarse: np.ndarray) -> dict:
    ranks = first_ranks(neighbors, fine, coarse)
    same_coarse, cross = ranks["same_coarse"], ranks["cross"]
    same_fine, diff_fine = ranks["same_fine"], ranks["diff_fine_same_coarse"]
    has_same_coarse, has_cross = same_coarse < 1_000, cross < 1_000
    comparable_cross = has_same_coarse & has_cross
    cross_violation = comparable_cross & (cross < same_coarse)
    comparable_fine = (same_fine < 1_000) & (diff_fine < 1_000)
    fine_violation = comparable_fine & (diff_fine < same_fine)
    return {
        "cross_before_same_coarse": {
            "comparable": int(comparable_cross.sum()),
            "violations": int(cross_violation.sum()),
            "rate": float(cross_violation.sum() / comparable_cross.sum())
            if comparable_cross.sum() else float("nan"),
        },
        "diff_fine_before_same_fine": {
            "comparable": int(comparable_fine.sum()),
            "violations": int(fine_violation.sum()),
            "rate": float(fine_violation.sum() / comparable_fine.sum())
            if comparable_fine.sum() else float("nan"),
        },
        "first_cross_rank": {
            "n": int(has_cross.sum()),
            "median": float(np.median(cross[has_cross])) if has_cross.any() else float("nan"),
            "p90": float(np.quantile(cross[has_cross], 0.9)) if has_cross.any() else float("nan"),
            "fraction_le_5": float(np.mean(cross[has_cross] <= 5)) if has_cross.any() else float("nan"),
            "fraction_le_10": float(np.mean(cross[has_cross] <= 10)) if has_cross.any() else float("nan"),
            "fraction_le_15": float(np.mean(cross[has_cross] <= 15)) if has_cross.any() else float("nan"),
            "fraction_le_20": float(np.mean(cross[has_cross] <= 20)) if has_cross.any() else float("nan"),
        },
    }


def undirected(neighbors: np.ndarray, k: int) -> np.ndarray:
    left = np.repeat(np.arange(len(neighbors), dtype=np.int64), k)
    right = neighbors[:, :k].reshape(-1)
    return np.unique(np.sort(np.column_stack([left, right]), axis=1), axis=0)


def _tree_distances(fine: np.ndarray, coarse: np.ndarray, indices: np.ndarray) -> np.ndarray:
    fine_sample = fine[indices]
    coarse_sample = coarse[indices]
    distance = np.where(fine_sample[:, :, None] == fine_sample[:, None, :], 1.0,
                        np.where(coarse_sample[:, :, None] == coarse_sample[:, None, :], 2.0, 3.0))
    diagonal = np.arange(indices.shape[1])
    distance[:, diagonal, diagonal] = 0.0
    return distance


def _text_distances(embedding: np.ndarray, indices: np.ndarray) -> np.ndarray:
    vectors = embedding[indices]
    dots = np.einsum("ijk,ilk->ijl", vectors, vectors)
    # The embedding is L2-normalised, so this is Euclidean distance, not the
    # non-metric 1-cosine quantity.
    distance = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    diagonal = np.arange(indices.shape[1])
    distance[:, diagonal, diagonal] = 0.0
    return distance


def four_point(embedding: np.ndarray, fine: np.ndarray, coarse: np.ndarray,
               samples: int = 200_000) -> dict:
    rng = np.random.default_rng(20261117)
    indices = rng.integers(0, len(embedding), size=(samples * 2, 4))
    unique = np.array([len(set(row)) == 4 for row in indices])
    indices = indices[unique][:samples]
    text = _text_distances(embedding, indices)
    tree = _tree_distances(fine, coarse, indices)

    def summarize(distance: np.ndarray) -> dict:
        sums = np.stack([
            distance[:, 0, 1] + distance[:, 2, 3],
            distance[:, 0, 2] + distance[:, 1, 3],
            distance[:, 0, 3] + distance[:, 1, 2],
        ], axis=1)
        ordered = np.sort(sums, axis=1)[:, ::-1]
        residual = (ordered[:, 0] - ordered[:, 1]) / np.maximum(ordered[:, 0], 1e-12)
        return {
            "n": int(len(residual)),
            "mean": float(residual.mean()),
            "median": float(np.median(residual)),
            "p95": float(np.quantile(residual, 0.95)),
            "fraction_gt_0_05": float(np.mean(residual > 0.05)),
            "fraction_gt_0_10": float(np.mean(residual > 0.10)),
        }

    return {"text_euclidean": summarize(text), "ideal_tree": summarize(tree)}


def ultrametric(embedding: np.ndarray, fine: np.ndarray, coarse: np.ndarray,
                samples: int = 200_000) -> dict:
    rng = np.random.default_rng(20261118)
    indices = rng.integers(0, len(embedding), size=(samples * 2, 3))
    unique = np.array([len(set(row)) == 3 for row in indices])
    indices = indices[unique][:samples]
    text = _text_distances(embedding, indices)
    tree = _tree_distances(fine, coarse, indices)

    def summarize(distance: np.ndarray) -> dict:
        # Use the three distinct pair distances, not the 3x3 diagonal.
        pairs = np.stack([distance[:, 0, 1], distance[:, 0, 2], distance[:, 1, 2]], axis=1)
        ordered = np.sort(pairs, axis=1)[:, ::-1]
        residual = (ordered[:, 0] - ordered[:, 1]) / np.maximum(ordered[:, 0], 1e-12)
        return {
            "n": int(len(residual)),
            "mean": float(residual.mean()),
            "median": float(np.median(residual)),
            "p95": float(np.quantile(residual, 0.95)),
            "fraction_gt_0_05": float(np.mean(residual > 0.05)),
            "fraction_gt_0_10": float(np.mean(residual > 0.10)),
        }

    return {"text_euclidean": summarize(text), "ideal_tree": summarize(tree)}


def average_bands(bands: list[dict]) -> dict:
    output = {}
    for key in bands[0]:
        rates = {relation: np.array([band[key]["rates"][relation] for band in bands])
                 for relation in RELATIONS}
        output[key] = {
            "total": int(np.mean([band[key]["total"] for band in bands])),
            "rates_mean": {name: float(values.mean()) for name, values in rates.items()},
            "rates_sd": {name: float(values.std(ddof=1)) if len(values) > 1 else 0.0
                         for name, values in rates.items()},
        }
    return output


def average_violations(values: list[dict]) -> dict:
    def scalar(path):
        return np.array([item[path[0]][path[1]] for item in values], dtype=float)

    return {
        "cross_before_same_coarse": {
            "rate_mean": float(scalar(("cross_before_same_coarse", "rate")).mean()),
            "rate_sd": float(scalar(("cross_before_same_coarse", "rate")).std(ddof=1))
            if len(values) > 1 else 0.0,
        },
        "diff_fine_before_same_fine": {
            "rate_mean": float(scalar(("diff_fine_before_same_fine", "rate")).mean()),
            "rate_sd": float(scalar(("diff_fine_before_same_fine", "rate")).std(ddof=1))
            if len(values) > 1 else 0.0,
        },
        "first_cross_rank": {
            "n_mean": float(scalar(("first_cross_rank", "n")).mean()),
            "median_mean": float(np.nanmean([
                item["first_cross_rank"]["median"] for item in values
            ])),
            "fraction_le_5_mean": float(np.nanmean([
                item["first_cross_rank"]["fraction_le_5"] for item in values
            ])),
        },
    }


def average_graphlets(entries: dict[str, list[dict]]) -> dict:
    output = {}
    for key, summaries in entries.items():
        shares = {name: np.array([summary["shares"][name] for summary in summaries])
                  for name in GRAPHLETS}
        output[key] = {
            "shares_mean": {name: float(values.mean()) for name, values in shares.items()},
            "shares_sd": {name: float(values.std(ddof=1)) if len(values) > 1 else 0.0
                          for name, values in shares.items()},
            "triangle_share_3node_mean": float(np.mean([
                summary["triangle_share_3node"] for summary in summaries
            ])),
        }
    return output


def cross_pairs(neighbors: np.ndarray, fine: np.ndarray, coarse: np.ndarray,
                embedding: np.ndarray, labels: pd.DataFrame, sample: int = 100) -> dict:
    pairs = []
    for source in range(len(neighbors)):
        for rank in range(5):
            target = int(neighbors[source, rank])
            if coarse[source] == coarse[target]:
                continue
            pairs.append((source, target, rank + 1,
                          float(np.dot(embedding[source], embedding[target]))))
    category_counts = Counter(tuple(sorted((str(coarse[source]), str(coarse[target]))))
                              for source, target, _, _ in pairs)
    rng = np.random.default_rng(20261119)
    chosen = pairs if len(pairs) <= sample else [
        pairs[index] for index in rng.choice(len(pairs), sample, replace=False)
    ]
    rows = []
    for source, target, rank, cosine in chosen:
        source_row, target_row = labels.iloc[source], labels.iloc[target]
        rows.append({
            "source_item_id": int(source_row.item_id),
            "source_title": source_row.title,
            "source_coarse": source_row.coarse,
            "source_path": source_row.path,
            "target_item_id": int(target_row.item_id),
            "target_title": target_row.title,
            "target_coarse": target_row.coarse,
            "target_path": target_row.path,
            "rank": int(rank),
            "cosine": cosine,
            "category_pair": " <> ".join(sorted((str(source_row.coarse), str(target_row.coarse)))),
        })
    frame = pd.DataFrame(rows).sort_values(["category_pair", "rank"]).reset_index(drop=True)
    frame.to_csv(OUT_DIR / "cross_coarse_top5_pairs.csv", index=False)
    return {
        "total_pairs": len(pairs),
        "sampled_pairs": len(frame),
        "category_pair_counts": {
            " <> ".join(key): int(value) for key, value in category_counts.most_common()
        },
        "file": "cross_coarse_top5_pairs.csv",
    }


def label_samples(labels: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(20261117)
    rows = []
    for level in ("coarse", "fine"):
        for value, group in labels.groupby(level):
            chosen = group if len(group) <= count else group.iloc[
                rng.choice(len(group), count, replace=False)]
            for row in chosen.itertuples(index=False):
                rows.append({"level": level, "label": value,
                             "item_id": int(row.item_id), "title": row.title})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[~labels["coarse"].isin(EXCLUDED_COARSE)]
    labels = labels.drop_duplicates("item_id").reset_index(drop=True)
    item_ids = labels["item_id"].to_numpy(dtype=np.int64)
    # Leaf names repeat under different coarse branches (e.g. Stands under
    # Instrument Accessories and Microphones).  The full path is the tree node.
    fine = labels["path"].astype(str).to_numpy()
    coarse = labels["coarse"].astype(str).to_numpy()
    embedding = np.load(EMBEDDING).astype(np.float32)[item_ids]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    labels[["item_id", "title", "coarse", "fine", "path"]].to_csv(OUT_DIR / "semantic_tree_labels.csv", index=False)
    label_samples(labels).to_csv(SAMPLES_CSV, index=False)

    text_neighbors = build_text_knn(embedding)
    print("labeled", len(labels), "coarse", len(set(coarse)), "fine", len(set(fine)), flush=True)

    text_bands = average_bands([band_mix(text_neighbors, fine, coarse)])
    text_violations = average_violations([violation_summary(text_neighbors, fine, coarse)])
    text_graphlet_entries = {f"k{k}": [
        graphlet_summary(run_pgd(undirected(text_neighbors, k), f"text_k{k}"))
    ] for k in K_VALUES}

    tree_band_entries, tree_violation_entries = [], []
    tree_graphlet_entries = {f"k{k}": [] for k in K_VALUES}
    for seed in range(TREE_SEEDS):
        tree_neighbors = build_tree_knn(fine, coarse, seed=seed)
        tree_band_entries.append(band_mix(tree_neighbors, fine, coarse))
        tree_violation_entries.append(violation_summary(tree_neighbors, fine, coarse))
        for k in K_VALUES:
            tree_graphlet_entries[f"k{k}"].append(
                graphlet_summary(run_pgd(undirected(tree_neighbors, k), f"tree_k{k}_s{seed}"))
            )
        print("tree seed", seed, "done", flush=True)

    result = {
        "items_total": 24587,
        "labeled_items": int(len(labels)),
        "coverage": float(len(labels) / 24587),
        "label_source": "Amazon Reviews 2023 meta_Musical_Instruments categories, exact normalised title match; ambiguous titles dropped",
        "coarse_counts": {name: int(value) for name, value in Counter(coarse).items()},
        "fine_counts": {name: int(value) for name, value in Counter(fine).items()},
        "tree_tie_policy": f"uniform random within each exact-distance tier; {TREE_SEEDS} seeds",
        "tree_seeds": TREE_SEEDS,
        "bands": {
            "text": text_bands,
            "ideal_tree": average_bands(tree_band_entries),
        },
        "violations": {
            "text": text_violations,
            "ideal_tree": average_violations(tree_violation_entries),
        },
        "graphlets": {
            "text": average_graphlets(text_graphlet_entries),
            "ideal_tree": average_graphlets(tree_graphlet_entries),
        },
        "four_point": four_point(embedding, fine, coarse),
        "ultrametric": ultrametric(embedding, fine, coarse),
        "cross_coarse_pairs": cross_pairs(text_neighbors, fine, coarse, embedding, labels),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Single semantic hierarchy tree hypothesis test",
        "",
        "Labels: Amazon Reviews 2023 `meta_Musical_Instruments` category paths, matched to",
        "the local items by normalised title. Coarse = category level 2; fine = full",
        "category path (the leaf name alone can repeat under different coarse branches).",
        "Ambiguous and missing titles were dropped; this is an independent semantic hierarchy,",
        "not KMeans on the embedding.",
        "",
        f"Labeled items: **{len(labels)} / 24,587**; coarse categories: **{len(set(coarse))}**; fine categories: **{len(set(fine))}**.",
        "",
        "Ideal tree distance: same fine = 1, same coarse different fine = 2, cross coarse = 3.",
        "",
        "## New neighbours by K band (text kNN)",
        "",
        "| ranks | same fine | same coarse, diff fine | cross coarse |",
        "|---|---:|---:|---:|",
    ]
    for key, value in result["bands"]["text"].items():
        lines.append(f"| {key} | {value['rates_mean']['same_fine']:.4f} | "
                     f"{value['rates_mean']['same_coarse_diff_fine']:.4f} | "
                     f"{value['rates_mean']['cross_coarse']:.4f} |")
    lines += [
        "",
        "## New neighbours by K band (ideal tree)",
        "",
        "| ranks | same fine | same coarse, diff fine | cross coarse |",
        "|---|---:|---:|---:|",
    ]
    for key, value in result["bands"]["ideal_tree"].items():
        lines.append(f"| {key} | {value['rates_mean']['same_fine']:.4f} | "
                     f"{value['rates_mean']['same_coarse_diff_fine']:.4f} | "
                     f"{value['rates_mean']['cross_coarse']:.4f} |")
    lines += [
        "",
        "## Hierarchy violations",
        "",
        "| graph | cross before same coarse | diff-fine before same-fine | first cross n | first cross median | first cross <=5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("text", "ideal_tree"):
        value = result["violations"][name]
        lines.append(f"| {name} | {value['cross_before_same_coarse']['rate_mean']:.4f} | "
                     f"{value['diff_fine_before_same_fine']['rate_mean']:.4f} | "
                     f"{value['first_cross_rank']['n_mean']:.1f} | "
                     f"{value['first_cross_rank']['median_mean']:.1f} | "
                     f"{value['first_cross_rank']['fraction_le_5_mean']:.4f} |")
    lines += [
        "",
        "## Graphlets",
        "",
        "| graph | K1,3 | P4 | paw | diamond | K4 | C4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("text", "ideal_tree"):
        prefix = "text" if name == "text" else "tree"
        for k in K_VALUES:
            value = result["graphlets"][name][f"k{k}"]["shares_mean"]
            lines.append(f"| {name} k={k} | {value['K1,3']:.4f} | {value['P4']:.4f} | "
                         f"{value['paw']:.4f} | {value['diamond']:.4f} | "
                         f"{value['K4']:.4f} | {value['C4']:.4f} |")
    lines += [
        "",
        "## Cross-coarse top-5 pairs",
        "",
        f"Total cross-coarse pairs in text top-5: **{result['cross_coarse_pairs']['total_pairs']}**; "
        f"sampled **{result['cross_coarse_pairs']['sampled_pairs']}** for manual audit in "
        f"`{result['cross_coarse_pairs']['file']}`.",
        "",
        "| category pair | pairs |",
        "|---|---:|",
    ]
    for pair, count in list(result["cross_coarse_pairs"]["category_pair_counts"].items())[:15]:
        lines.append(f"| {pair} | {count} |")
    lines += [
        "",
        "## Ultrametric / four-point residuals",
        "",
        "| test | data | mean | median | p95 | >0.05 | >0.10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for test_name, test in (("four-point", result["four_point"]), ("ultrametric", result["ultrametric"])):
        for data_name in ("text_euclidean", "ideal_tree"):
            value = test[data_name]
            lines.append(f"| {test_name} | {data_name} | {value['mean']:.4f} | "
                         f"{value['median']:.4f} | {value['p95']:.4f} | "
                         f"{value['fraction_gt_0_05']:.4f} | {value['fraction_gt_0_10']:.4f} |")
    lines += [
        "",
        "Residuals are relative: (largest two-point sum minus second largest) / largest.",
        "The ideal tree is exact by construction; sampling gives 0.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
