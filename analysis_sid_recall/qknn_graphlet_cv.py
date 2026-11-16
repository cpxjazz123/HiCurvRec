"""Robustness: fixed q-kNN edge budget, high-collision exclusion, larger graphlet sampling,
and cross-validated incremental prediction of Stage3 recall.

Only standard 3x(256,32) VQ arms.  q is rebuilt from codebooks + exported SIDs.  Every retained
graph is a k=20 candidate graph trimmed to the same global edge budget (the median k=10 edge count
of the 44 arms with L3 collision < 0.90).  Thus q edge count is identical across arms.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_graphlet_cv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from qknn_graphlet_recall import edge_key, exact_frontier  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
OLD_GRAPHLETS = REPO / "results" / "research" / "tiger_cluster_graph" / "graphlet_instances.npz"
NEW_GRAPHLETS = REPO / "results" / "research" / "tiger_cluster_graph" / "graphlet_instances_3000.npz"
OUT_JSON = HERE / "results" / "qknn_graphlet_cv.json"
OUT_MD = HERE / "results" / "qknn_graphlet_cv.md"
K_CANDIDATE = 20
N_ITEMS = 24587
COLLISION_MAX = 0.90
SAMPLED = ("K1_4", "P5", "T5", "C5")
STRUCTURES = ("K1_4", "P5", "T5", "C5", "triangle", "K4", "inside", "cross")


def sampled_edges(path: Path, name: str) -> set[tuple[int, int]]:
    archive = np.load(path)
    if name not in archive.files:
        return set()
    edges = set()
    for instance in archive[name]:
        for left in range(5):
            for right in range(left + 1, 5):
                edges.add(edge_key(int(instance[left]), int(instance[right])))
    return edges


def trim_edges(q: np.ndarray, edges: np.ndarray, target: int) -> np.ndarray:
    if len(edges) < target:
        raise ValueError(f"candidate graph has {len(edges)} edges < target {target}")
    distance = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1)
    return edges[np.argsort(distance)[:target]]


def recovery(values: np.ndarray, indices: np.ndarray) -> float:
    return float(values[indices].mean()) if len(indices) else float("nan")


def cv_scores(features: np.ndarray, target: np.ndarray, seed: int = 42) -> dict:
    folds = list(RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed).split(features))
    r2_values = []
    mae_values = []
    for train, test in folds:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], target[train])
        prediction = model.predict(features[test])
        r2_values.append(r2_score(target[test], prediction))
        mae_values.append(mean_absolute_error(target[test], prediction))
    return {
        "r2_mean": float(np.mean(r2_values)),
        "r2_sd": float(np.std(r2_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_sd": float(np.std(mae_values)),
        "folds": len(folds),
    }


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    edge_to_index = {edge_key(int(left), int(right)): index
                     for index, (left, right) in enumerate(behavior_edges.tolist())}
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    comm = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        comm[list(members)] = index
    within = comm[behavior_edges[:, 0]] == comm[behavior_edges[:, 1]]

    triangle, k4 = exact_frontier(behavior_edges, adjacency)
    old_structures = {
        "K1_4": sampled_edges(OLD_GRAPHLETS, "K1_4"),
        "P5": sampled_edges(OLD_GRAPHLETS, "P5_centre") | sampled_edges(OLD_GRAPHLETS, "P5_hops"),
        "T5": sampled_edges(OLD_GRAPHLETS, "T5"),
        "C5": sampled_edges(OLD_GRAPHLETS, "C5"),
    }
    new_structures = {
        "K1_4": sampled_edges(NEW_GRAPHLETS, "K1_4"),
        "P5": sampled_edges(NEW_GRAPHLETS, "P5_centre") | sampled_edges(NEW_GRAPHLETS, "P5_hops"),
        "T5": sampled_edges(NEW_GRAPHLETS, "T5"),
        "C5": sampled_edges(NEW_GRAPHLETS, "C5"),
    }
    exact_structures = {
        "triangle": triangle,
        "K4": k4,
        "inside": {edge_key(int(left), int(right)) for left, right in behavior_edges[within].tolist()},
        "cross": {edge_key(int(left), int(right)) for left, right in behavior_edges[~within].tolist()},
    }
    structures = {**new_structures, **exact_structures}
    old_structures.update(exact_structures)
    indices = {
        name: np.array([edge_to_index[edge] for edge in edges if edge in edge_to_index], dtype=np.int64)
        for name, edges in structures.items()
    }
    old_indices = {
        name: np.array([edge_to_index[edge] for edge in edges if edge in edge_to_index], dtype=np.int64)
        for name, edges in old_structures.items()
    }
    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    dmax = np.maximum(degree[behavior_edges[:, 0]], degree[behavior_edges[:, 1]])
    degree_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    structure_bins = {name: degree_bin[value] for name, value in indices.items()}

    # First pass: standard q graphs, to obtain the 44 collision<0.90 arms and the budget.
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
            skipped.append({"number": directory.name, "reason": "invalid SID tokens"})
            continue
        q = sum(codebooks[level][tokens[:, level]] for level in range(3)).astype(np.float32)
        ten = qknn_edges(q, k=10)
        recall = json.loads(recall_path.read_text()).get("test_recall@10")
        collision = analysis.collision_of(tokens)
        if recall is None:
            skipped.append({"number": directory.name, "reason": "no test_recall@10"})
            continue
        if collision >= COLLISION_MAX:
            skipped.append({"number": directory.name, "reason": f"collision {collision:.4f} >= {COLLISION_MAX}"})
            continue
        rows.append({
            "directory": directory,
            "tokens": tokens,
            "q": q,
            "recall": float(recall),
            "collision": float(collision),
            "ten_edges": len(ten),
        })
    target = int(np.median([row["ten_edges"] for row in rows]))
    print(f"kept arms {len(rows)}; common target edges {target}", flush=True)

    for row in rows:
        q = row["q"]
        candidate = qknn_edges(q, k=K_CANDIDATE)
        edges = trim_edges(q, candidate, target)
        edge_set = {(int(left), int(right)) for left, right in edges.tolist()}
        recovered = np.array([(int(left), int(right)) in edge_set for left, right in behavior_edges], dtype=bool)
        bin_rate = np.array([
            recovered[degree_bin == value].mean() if (degree_bin == value).any() else np.nan
            for value in range(5)
        ])
        row["q_edges"] = int(len(edges))
        row["overall"] = float(recovered.mean())
        for name, value in indices.items():
            observed = recovery(recovered, value)
            expected = float(np.nanmean(bin_rate[structure_bins[name]])) if len(value) else float("nan")
            row[f"{name}_rate"] = observed
            row[f"{name}_excess"] = observed - expected
        for name, value in old_indices.items():
            row[f"old_{name}_rate"] = recovery(recovered, value)
        print(f"{row['directory'].name} fixed_edges {len(edges)} overall {row['overall']:.5f} "
              f"P5 {row['P5_rate']:.5f} K4 {row['K4_rate']:.5f}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    collision = np.array([row["collision"] for row in rows])
    overall = np.array([row["overall"] for row in rows])
    inside = np.array([row["inside_rate"] for row in rows])
    cross = np.array([row["cross_rate"] for row in rows])

    stability = {}
    for name in SAMPLED:
        old_rate = np.array([row[f"old_{name}_rate"] for row in rows])
        new_rate = np.array([row[f"{name}_rate"] for row in rows])
        stability[name] = {
            "old_edges": int(len(old_indices[name])),
            "new_edges": int(len(indices[name])),
            "rate_spearman": analysis.spearman(old_rate, new_rate),
            "rate_pearson": analysis.pearson(old_rate, new_rate),
            "rate_mae": float(np.mean(np.abs(old_rate - new_rate))),
            "old_recall_spearman": analysis.spearman(old_rate, recall),
            "new_recall_spearman": analysis.spearman(new_rate, recall),
        }

    def base_controls(name: str) -> np.ndarray:
        if name in ("inside", "cross"):
            return np.column_stack([overall, collision, cross if name == "inside" else inside])
        return np.column_stack([overall, collision, inside, cross])

    cv = {}
    for name in STRUCTURES:
        control = base_controls(name)
        rate = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        base = cv_scores(control, recall)
        with_rate = cv_scores(np.column_stack([control, rate]), recall)
        with_excess = cv_scores(np.column_stack([control, excess]), recall)
        cv[name] = {
            "base": base,
            "rate": with_rate,
            "excess": with_excess,
            "delta_r2_rate": with_rate["r2_mean"] - base["r2_mean"],
            "delta_mae_rate": with_rate["mae_mean"] - base["mae_mean"],
            "delta_r2_excess": with_excess["r2_mean"] - base["r2_mean"],
            "delta_mae_excess": with_excess["mae_mean"] - base["mae_mean"],
        }

    graphlet_names = ("K1_4", "P5", "T5", "C5", "triangle", "K4")
    all_rates = np.column_stack([np.array([row[f"{name}_rate"] for row in rows]) for name in graphlet_names])
    all_excess = np.column_stack([np.array([row[f"{name}_excess"] for row in rows]) for name in graphlet_names])
    cv["all_graphlets"] = {
        "base": cv_scores(np.column_stack([overall, collision, inside, cross]), recall),
        "rate": cv_scores(np.column_stack([overall, collision, inside, cross, all_rates]), recall),
        "excess": cv_scores(np.column_stack([overall, collision, inside, cross, all_excess]), recall),
    }

    result = {
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "arms": len(rows),
        "collision_max": COLLISION_MAX,
        "target_edges": target,
        "candidate_k": K_CANDIDATE,
        "structures": {
            name: {
                "edges": int(len(indices[name])),
                "old_edges": int(len(old_indices[name])),
                "definition": "exact" if name in ("triangle", "K4", "inside", "cross") else "strict sampled",
            }
            for name in STRUCTURES
        },
        "sample_stability": stability,
        "cv": cv,
        "rows": [
            {key: value for key, value in row.items() if key in (
                "recall", "collision", "q_edges", "overall",
                *tuple(f"{name}_rate" for name in STRUCTURES),
                *tuple(f"{name}_excess" for name in STRUCTURES),
            )}
            for row in rows
        ],
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Robustness: fixed q-kNN budget, high-collision exclusion, larger graphlets, CV",
        "",
        f"Standard-VQ arms with L3 collision < **{COLLISION_MAX}**: **{len(rows)}**. "
        f"All q graphs are k={K_CANDIDATE} candidates trimmed to **{target}** edges, so q edge count is constant.",
        "",
        "## Sample-size stability (300 vs 3000 strict instances)",
        "",
        "| structure | old edges | new edges | rate Spearman | rate Pearson | rate MAE | old recall rho | new recall rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in stability.items():
        lines.append(
            f"| {name} | {value['old_edges']} | {value['new_edges']} | {value['rate_spearman']:+.4f} | "
            f"{value['rate_pearson']:+.4f} | {value['rate_mae']:.5f} | {value['old_recall_spearman']:+.4f} | "
            f"{value['new_recall_spearman']:+.4f} |"
        )
    lines += [
        "",
        "## Repeated 5-fold CV: incremental recall prediction",
        "",
        "Base controls: overall recovery, L3 collision, inside/cross recovery (for graphlet structures). "
        "Ridge(alpha=1) after StandardScaler; 20 repeats of 5-fold. ΔR²/ΔMAE are with minus base.",
        "",
        "| structure | base R² | +rate R² | ΔR² rate | ΔMAE rate | +excess R² | ΔR² excess | ΔMAE excess |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in STRUCTURES:
        value = cv[name]
        lines.append(
            f"| {name} | {value['base']['r2_mean']:.4f} | {value['rate']['r2_mean']:.4f} | "
            f"{value['delta_r2_rate']:+.4f} | {value['delta_mae_rate']:+.5f} | "
            f"{value['excess']['r2_mean']:.4f} | {value['delta_r2_excess']:+.4f} | {value['delta_mae_excess']:+.5f} |"
        )
    value = cv["all_graphlets"]
    lines.append(
        f"| all graphlets | {value['base']['r2_mean']:.4f} | {value['rate']['r2_mean']:.4f} | "
        f"{value['rate']['r2_mean'] - value['base']['r2_mean']:+.4f} | "
        f"{value['rate']['mae_mean'] - value['base']['mae_mean']:+.5f} | "
        f"{value['excess']['r2_mean']:.4f} | "
        f"{value['excess']['r2_mean'] - value['base']['r2_mean']:+.4f} | "
        f"{value['excess']['mae_mean'] - value['base']['mae_mean']:+.5f} |"
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
