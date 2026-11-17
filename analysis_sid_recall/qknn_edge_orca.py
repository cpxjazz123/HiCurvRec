"""Edge-ORCA structural roles vs q-kNN recovery and Stage3 recall.

No Louvain.  The original behaviour graph is processed by ORCA's EDGE orbit mode (5-node
graphlets, 68 edge orbits).  The 68-d edge-orbit profiles are summarised with MiniBatchKMeans into
edge structural archetypes; clusters may share orbit structure (the 68-d vector remains multi-label).
For each archived standard-VQ TIGER arm, the k=10 q-kNN graph is compared with each behaviour edge
class.  Degree + sentence-embedding similarity matching provides an excess over matched edges.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_edge_orca.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from qknn_community_correlation import checkpoint_vq, qknn_edges  # noqa: E402
from qknn_graphlet_cv import cv_scores  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
EDGE_ORCA = REPO / "results" / "orca_build" / "runs" / "behaviour_edges_cpu_5.out"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT_JSON = HERE / "results" / "qknn_edge_orca.json"
OUT_MD = HERE / "results" / "qknn_edge_orca.md"
K = 10
K_EDGE_ROLES = 10
N_ITEMS = 24587


def direct(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first, second, control = values[usable], recall[usable], controls[usable]
    if len(first) < 5:
        return {"n": int(len(first)), "spearman": None}
    rho = analysis.spearman(first, second)
    pear = analysis.pearson(first, second)
    design = np.column_stack([np.ones(len(first)), control])
    res_x = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    res_y = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    ranked_x = analysis.average_ranks(first)
    ranked_y = analysis.average_ranks(second)
    ranked_control = np.column_stack([analysis.average_ranks(control[:, i]) for i in range(control.shape[1])])
    ranked_design = np.column_stack([np.ones(len(first)), ranked_control])
    rank_res_x = ranked_x - ranked_design @ np.linalg.lstsq(ranked_design, ranked_x, rcond=None)[0]
    rank_res_y = ranked_y - ranked_design @ np.linalg.lstsq(ranked_design, ranked_y, rcond=None)[0]
    rng = np.random.default_rng(42)
    beats_s = beats_r = 0
    for _ in range(3000):
        shuffled = rng.permutation(second)
        beats_s += abs(analysis.spearman(first, shuffled)) >= abs(rho)
        beats_r += abs(analysis.pearson(first, shuffled)) >= abs(pear)
    return {
        "n": int(len(first)),
        "mean": float(first.mean()),
        "spearman": rho,
        "pearson": pear,
        "p_spearman": (beats_s + 1) / 3001,
        "p_pearson": (beats_r + 1) / 3001,
        "partial_spearman": analysis.pearson(rank_res_x, rank_res_y),
        "partial_pearson": analysis.pearson(res_x, res_y),
    }


def incremental(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    base = np.column_stack([np.ones(usable.sum()), controls[usable]])
    residue = recall[usable] - base @ np.linalg.lstsq(base, recall[usable], rcond=None)[0]
    total = np.sum((recall[usable] - recall[usable].mean()) ** 2)
    base_r2 = 1.0 - np.sum(residue ** 2) / total
    full = np.column_stack([base, values[usable]])
    residue_full = recall[usable] - full @ np.linalg.lstsq(full, recall[usable], rcond=None)[0]
    full_r2 = 1.0 - np.sum(residue_full ** 2) / total
    n = int(usable.sum())
    delta = full_r2 - base_r2
    p = (f_dist.sf((delta / 1.0) / ((1.0 - full_r2) / (n - controls.shape[1] - 2)),
                   1, n - controls.shape[1] - 2)
         if n > controls.shape[1] + 2 and delta > 0 else float("nan"))
    return {"n": n, "base_r2": float(base_r2), "with_r2": float(full_r2),
            "delta_r2": float(delta), "p_delta": p}


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    orbits = np.loadtxt(EDGE_ORCA, dtype=np.float64)
    if orbits.shape != (len(behavior_edges), 68):
        raise ValueError(f"Edge-ORCA shape {orbits.shape}, expected {(len(behavior_edges), 68)}")
    log_orbit = np.log1p(np.maximum(orbits, 0.0))
    profile = log_orbit / np.maximum(log_orbit.sum(axis=1, keepdims=True), 1.0)
    print("Edge-ORCA loaded", orbits.shape, flush=True)

    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    left, right = behavior_edges[:, 0], behavior_edges[:, 1]
    du, dv = degree[left], degree[right]
    dmax, dmin = np.maximum(du, dv), np.minimum(du, dv)
    common = np.array([len(adjacency[int(u)] & adjacency[int(v)]) for u, v in behavior_edges.tolist()], dtype=np.int64)
    embedding = np.load(EMBEDDING).astype(np.float32)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    semantic = np.einsum("ij,ij->i", embedding[left], embedding[right]).astype(np.float64)
    dmax_bin = np.asarray(pd.qcut(dmax, 5, labels=False, duplicates="drop"), dtype=np.int64)
    dmin_bin = np.asarray(pd.qcut(dmin, 5, labels=False, duplicates="drop"), dtype=np.int64)
    semantic_bin = np.asarray(pd.qcut(semantic, 5, labels=False, duplicates="drop"), dtype=np.int64)
    cell = dmax_bin * 25 + dmin_bin * 5 + semantic_bin

    model = MiniBatchKMeans(n_clusters=K_EDGE_ROLES, random_state=42, n_init=20, batch_size=4096)
    edge_role = model.fit_predict(profile).astype(np.int64)
    role_info = {}
    for role in range(K_EDGE_ROLES):
        mask = edge_role == role
        role_info[f"E{role}"] = {
            "edges": int(mask.sum()),
            "share": float(mask.mean()),
            "dmax_median": float(np.median(dmax[mask])),
            "dmin_median": float(np.median(dmin[mask])),
            "common_median": float(np.median(common[mask])),
            "semantic_median": float(np.median(semantic[mask])),
            "top_orbits": np.argsort(-profile[mask].mean(axis=0))[:5].tolist(),
        }
    print("edge roles", json.dumps(role_info, indent=2), flush=True)

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
        row = {"number": directory.name, "recall": float(recall), "q_edges": int(len(edges)),
               "collision": analysis.collision_of(tokens), "overall": float(recovered.mean())}
        for role in range(K_EDGE_ROLES):
            mask = edge_role == role
            observed = float(recovered[mask].mean())
            cell_rate = np.full(125, np.nan)
            for value in np.unique(cell):
                control = (~mask) & (cell == value)
                if control.any():
                    cell_rate[value] = recovered[control].mean()
            expected = float(np.nanmean(cell_rate[cell[mask]]))
            row[f"E{role}_rate"] = observed
            row[f"E{role}_matched"] = expected
            row[f"E{role}_excess"] = observed - expected
            row[f"E{role}_lift"] = observed / expected if expected > 0 else float("nan")
        rows.append(row)
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} "
              f"E0 {row['E0_rate']:.5f} E1 {row['E1_rate']:.5f}", flush=True)

    recall = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    overall = np.array([row["overall"] for row in rows], dtype=float)
    controls = np.column_stack([overall, q_edges, collision])
    results = {}
    for role in range(K_EDGE_ROLES):
        name = f"E{role}"
        rate = np.array([row[f"{name}_rate"] for row in rows])
        excess = np.array([row[f"{name}_excess"] for row in rows])
        lift = np.array([row[f"{name}_lift"] for row in rows])
        results[name] = {
            "rate": direct(rate, recall, controls),
            "excess": direct(excess, recall, controls),
            "lift": direct(lift, recall, controls),
            "incremental_rate": incremental(rate, recall, controls),
            "incremental_excess": incremental(excess, recall, controls),
            "cv_base": cv_scores(controls, recall),
            "cv_rate": cv_scores(np.column_stack([controls, rate]), recall),
            "cv_excess": cv_scores(np.column_stack([controls, excess]), recall),
        }

    output = {
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "knn_k": K,
        "edge_roles": K_EDGE_ROLES,
        "arms": len(rows),
        "role_info": role_info,
        "results": results,
        "rows": rows,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Edge-ORCA structural roles vs q-kNN recovery and recall",
        "",
        "No Louvain. ORCA edge-orbit mode (5-node graphlets, **68 edge orbits**) on the fixed "
        "TRAIN behaviour graph; MiniBatchKMeans(K=10) summaries of log-normalised edge-orbit profiles.",
        "Matching uses (dmax quintile, dmin quintile, sentence-embedding similarity quintile); "
        "excess = class recovery - matched non-class recovery.",
        "",
        f"Behaviour edges **{len(behavior_edges)}**; arms **{len(rows)}**.",
        "",
        "## Edge structural roles",
        "",
        "| role | edges | share | dmax median | dmin median | common median | semantic median | top edge orbits |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in role_info.items():
        lines.append(
            f"| {name} | {value['edges']} | {value['share']:.4f} | {value['dmax_median']:.1f} | "
            f"{value['dmin_median']:.1f} | {value['common_median']:.1f} | {value['semantic_median']:.3f} | "
            f"{value['top_orbits']} |"
        )
    lines += [
        "",
        "## Recall relations",
        "",
        "Partial controls overall recovery, q-kNN edge count and L3 collision; CV is repeated 5-fold Ridge.",
        "",
        "| role | mean rate | mean excess | direct rho | direct r | partial rho | ΔR² rate | p(rate) | CV +rate | CV +excess |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in range(K_EDGE_ROLES):
        name = f"E{role}"
        value = results[name]
        rate = value["rate"]
        excess = value["excess"]
        inc = value["incremental_rate"]
        base = value["cv_base"]["r2_mean"]
        lines.append(
            f"| {name} | {rate['mean']:.5f} | {excess['mean']:+.5f} | {rate['spearman']:+.4f} | "
            f"{rate['pearson']:+.4f} | {rate['partial_spearman']:+.4f} | {inc['delta_r2']:+.4f} | "
            f"{inc['p_delta']:.4f} | {value['cv_rate']['r2_mean'] - base:+.4f} | "
            f"{value['cv_excess']['r2_mean'] - base:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
