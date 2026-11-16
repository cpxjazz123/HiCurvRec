"""Audited PGD 4-node motifs vs q-kNN recovery, with grouped CV.

Audit fixes:
  * PGD 4-tailed-triangle and 3-star are excluded: PGD leaves local_4_tailed_tris at zero,
    so its 3-star is over-counted. A synthetic K3+pendant graph reproduces the bug.
  * CPU kNN explicitly removes the query's own index; GPU already sets self-distance to inf.
  * PGD ID relabeling is inverted and every edge must map exactly once.
  * Matching coverage (degree+dmin+semantic cell with a non-class control) is recorded.
  * Partial controls use leave-target-out overall recovery, not the total that contains the
    target class.
  * CV is GroupShuffleSplit by model series, not random folds.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_pgd_audited.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
MICRO = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior.micro"
INPUT_CSV = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT_JSON = HERE / "results" / "qknn_pgd_audited.json"
OUT_MD = HERE / "results" / "qknn_pgd_audited.md"
K = 10
N_ITEMS = 24587
# Exclude PGD's broken 4-tailed-triangle and the 3-star that depends on it.
MOTIFS = ("triangle", "4-clique", "4-chordal-cycle", "4-cycle", "4-path")
ALL_COLUMNS = ("triangle", "2-star", "4-clique", "4-chordal-cycle",
               "4-tailed-triangle", "4-cycle", "3-star", "4-path")


def series_of(tree: str) -> str:
    for marker in ("curvature-prefix-ranking", "community", "RQ-VAE", "r48k",
                   "struct_mixed", "behaviour-pairs", "curvature"):
        if marker in tree:
            return marker
    return tree


def cv_grouped(features: np.ndarray, target: np.ndarray, groups: np.ndarray,
               repeats: int = 30, test_size: float = 0.2, seed: int = 42) -> dict:
    usable = np.isfinite(features).all(axis=1) & np.isfinite(target)
    features, target, groups = features[usable], target[usable], groups[usable]
    if len(target) < 10:
        return {"r2_mean": float("nan"), "mae_mean": float("nan"), "folds": 0}
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=test_size, random_state=seed)
    r2_values, mae_values = [], []
    for train, test in splitter.split(features, target, groups):
        if len(np.unique(groups[train])) < 2:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], target[train])
        prediction = model.predict(features[test])
        r2_values.append(r2_score(target[test], prediction))
        mae_values.append(mean_absolute_error(target[test], prediction))
    return {
        "r2_mean": float(np.mean(r2_values)) if r2_values else float("nan"),
        "r2_sd": float(np.std(r2_values)) if r2_values else float("nan"),
        "mae_mean": float(np.mean(mae_values)) if mae_values else float("nan"),
        "mae_sd": float(np.std(mae_values)) if mae_values else float("nan"),
        "folds": len(r2_values),
    }


def correlation(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first, second, control = values[usable], recall[usable], controls[usable]
    if len(first) < 5:
        return {"n": int(len(first)), "spearman": None}
    design = np.column_stack([np.ones(len(first)), control])
    res_x = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    res_y = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    rank_x = analysis.average_ranks(first)
    rank_y = analysis.average_ranks(second)
    rank_control = np.column_stack([analysis.average_ranks(control[:, i]) for i in range(control.shape[1])])
    rank_design = np.column_stack([np.ones(len(first)), rank_control])
    rank_res_x = rank_x - rank_design @ np.linalg.lstsq(rank_design, rank_x, rcond=None)[0]
    rank_res_y = rank_y - rank_design @ np.linalg.lstsq(rank_design, rank_y, rcond=None)[0]
    return {
        "n": int(len(first)),
        "mean": float(first.mean()),
        "spearman": analysis.spearman(first, second),
        "pearson": analysis.pearson(first, second),
        "partial_spearman": analysis.pearson(rank_res_x, rank_res_y),
        "partial_pearson": analysis.pearson(res_x, res_y),
    }


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


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    edge_map = {(int(a), int(b)): i for i, (a, b) in enumerate(behavior_edges.tolist())}
    raw = pd.read_csv(MICRO, skiprows=1, header=None, names=["src", "dst", *ALL_COLUMNS])
    lookup, internal_to_original = {}, []
    with INPUT_CSV.open() as handle:
        for line in handle:
            left, right = (int(x) for x in line.strip().split(","))
            for original in (left, right):
                if original not in lookup:
                    lookup[original] = len(internal_to_original)
                    internal_to_original.append(original)
    motif = np.zeros((len(behavior_edges), len(ALL_COLUMNS)), dtype=np.float64)
    seen = np.zeros(len(behavior_edges), dtype=bool)
    for left, right, value in zip(raw["src"].to_numpy(), raw["dst"].to_numpy(), raw[list(ALL_COLUMNS)].to_numpy()):
        original_left = internal_to_original[int(left) - 1]
        original_right = internal_to_original[int(right) - 1]
        key = (original_left, original_right) if original_left < original_right else (original_right, original_left)
        index = edge_map.get(key)
        if index is None or seen[index]:
            raise ValueError(f"PGD edge mapping fails at {key}")
        motif[index] = value
        seen[index] = True
    if not seen.all():
        raise ValueError(f"PGD mapping missing {int((~seen).sum())} edges")
    print("PGD one-to-one mapping verified; 4-tailed-triangle nonzeros:", int((motif[:, 4] != 0).sum()), flush=True)

    degree = np.array([len(x) for x in adjacency], dtype=np.int64)
    left, right = behavior_edges[:, 0], behavior_edges[:, 1]
    du, dv = degree[left], degree[right]
    dmax, dmin = np.maximum(du, dv), np.minimum(du, dv)
    dmax_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    dmin_bin = np.asarray(pd.qcut(dmin, 5, labels=False, duplicates="drop"), dtype=np.int64)
    embedding = np.load(EMBEDDING).astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    semantic = np.einsum("ij,ij->i", embedding[left], embedding[right])
    semantic_bin = np.asarray(pd.qcut(semantic, 5, labels=False, duplicates="drop"), dtype=np.int64)
    cell = dmax_bin * 25 + dmin_bin * 5 + semantic_bin

    classes = {}
    class_info = {}
    for name in MOTIFS:
        ranks = rank_within_bins(motif[:, ALL_COLUMNS.index(name)], dmax_bin)
        mask = ranks >= 0.75
        classes[name] = mask
        class_info[name] = {
            "edges": int(mask.sum()),
            "share": float(mask.mean()),
            "median_count": float(np.median(motif[mask, ALL_COLUMNS.index(name)])),
            "dmax_median": float(np.median(dmax[mask])),
        }
    print("classes", json.dumps(class_info, indent=2), flush=True)

    rows = []
    skipped = []
    for directory in sorted((path for path in R2.iterdir() if path.is_dir()), key=lambda path: path.name):
        checkpoint = directory / "rqvae_best.pth"
        sid_path = directory / "item_sids.json"
        recall_path = directory / "recall.json"
        if not (checkpoint.is_file() and sid_path.is_file() and recall_path.is_file()):
            continue
        packed = checkpoint_vq(checkpoint)
        if packed is None:
            skipped.append({"number": directory.name, "reason": "not standard 3x(256,32) vq_layers"})
            continue
        codebooks, payload = packed
        tokens = analysis.tokens_of(sid_path, nodes)
        if (tokens < 0).any() or (tokens > 255).any():
            skipped.append({"number": directory.name, "reason": "invalid SID"})
            continue
        q = sum(codebooks[level][tokens[:, level]] for level in range(3)).astype(np.float32)
        edges = qknn_edges(q)
        edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
        recovered = np.array([(int(u), int(v)) in edge_set for u, v in behavior_edges], dtype=bool)
        recall = json.loads(recall_path.read_text()).get("test_recall@10")
        if recall is None:
            skipped.append({"number": directory.name, "reason": "no recall"})
            continue
        tree = next((child for child in directory.iterdir() if child.is_dir()), directory).name
        row = {"number": directory.name, "series": series_of(tree), "tree": tree,
               "recall": float(recall), "q_edges": int(len(edges)),
               "collision": analysis.collision_of(tokens), "overall": float(recovered.mean())}
        for name, mask in classes.items():
            indices = np.flatnonzero(mask)
            observed = float(recovered[indices].mean())
            cell_rate = np.full(125, np.nan)
            for value in np.unique(cell):
                control = (~mask) & (cell == value)
                if control.any():
                    cell_rate[value] = recovered[control].mean()
            covered = np.isfinite(cell_rate[cell[indices]])
            expected = float(np.nanmean(cell_rate[cell[indices]])) if covered.any() else float("nan")
            row[f"{name}_rate"] = observed
            row[f"{name}_excess"] = observed - expected if covered.any() else float("nan")
            row[f"{name}_coverage"] = float(covered.mean())
            # Leave-target-out overall recovery as the control, not total including target.
            remaining = len(behavior_edges) - len(indices)
            row[f"{name}_overall_loo"] = float((recovered.sum() - recovered[indices].sum()) / remaining)
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} series {row['series']}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    groups = np.array([row["series"] for row in rows])
    results = {}
    for name in MOTIFS:
        rate = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        controls = np.column_stack([
            np.array([row[f"{name}_overall_loo"] for row in rows]), q_edges, collision
        ])
        coverage = np.array([row[f"{name}_coverage"] for row in rows])
        results[name] = {
            "coverage_mean": float(coverage.mean()),
            "coverage_min": float(coverage.min()),
            "correlation": correlation(rate, recall, controls),
            "excess": correlation(excess, recall, controls),
            "cv_base": cv_grouped(controls, recall, groups),
            "cv_rate": cv_grouped(np.column_stack([controls, rate]), recall, groups),
            "cv_excess": cv_grouped(np.column_stack([controls, excess]), recall, groups),
        }

    result = {
        "nodes": int(nodes), "behavior_edges": int(len(behavior_edges)), "k": K,
        "arms": len(rows), "motifs": list(MOTIFS),
        "excluded_pgd_columns": ["3-star", "4-tailed-triangle"],
        "class_info": class_info, "results": results, "rows": rows, "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Audited PGD motifs vs q-kNN recovery and recall",
        "",
        "PGD micro edge motifs; **3-star and 4-tailed-triangle are excluded** because PGD's",
        "`local_4_tailed_tris` is never populated and its 3-star correction therefore over-counts.",
        "A synthetic K3+pendant graph reproduces: tailed=0 on all edges, 3-star=1 on the pendant edge.",
        "Reliable columns used: triangle, 4-clique, 4-chordal-cycle, 4-cycle, 4-path.",
        "",
        "CPU kNN now explicitly removes each query's own index. PGD relabeling is inverted and",
        "every behaviour edge must map exactly once. Matching coverage is reported per class.",
        "",
        f"Behaviour edges **{len(behavior_edges)}**; arms **{len(rows)}**; grouped CV by model series.",
        "",
        "## Classes",
        "",
        "| motif | edges | share | count median | dmax median |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in class_info.items():
        lines.append(f"| {name} | {value['edges']} | {value['share']:.4f} | "
                     f"{value['median_count']:.1f} | {value['dmax_median']:.1f} |")
    lines += [
        "",
        "## Recall relations",
        "",
        "Partial controls leave-target-out overall recovery, q-kNN edge count and L3 collision. "
        "CV is GroupShuffleSplit by model series, 30 splits, Ridge(alpha=1).",
        "",
        "| motif | coverage | mean rate | mean excess | direct rho | direct r | partial rho | partial r | CV +rate | CV +excess |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MOTIFS:
        value = results[name]
        corr = value["correlation"]
        excess = value["excess"]
        base = value["cv_base"]["r2_mean"]
        lines.append(
            f"| {name} | {value['coverage_mean']:.4f} | {corr['mean']:.5f} | {excess['mean']:+.5f} | "
            f"{corr['spearman']:+.4f} | {corr['pearson']:+.4f} | {corr['partial_spearman']:+.4f} | "
            f"{corr['partial_pearson']:+.4f} | {value['cv_rate']['r2_mean'] - base:+.4f} | "
            f"{value['cv_excess']['r2_mean'] - base:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
