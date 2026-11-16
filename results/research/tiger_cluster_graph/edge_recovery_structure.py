"""Why are the recovered behaviour edges recovered?

TIGER only.  Build the same edge-budget-matched global kNN graphs as
`embedding_knn_compare.py`, then analyse the recovered behaviour edges by:

* endpoint ORCA roles and role-pair retention;
* local graphlet proxies (common neighbours, triangle, K4, chain/star-like);
* behaviour community within/cross;
* degree strata and logistic controls;
* z/q four-way fate: always / q-lost / q-added / never.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/edge_recovery_structure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from embedding_knn_compare import edged, trim_edges  # noqa: E402
from semantic_qknn_orca import (  # noqa: E402
    BEHAVIOUR_ORCA,
    ORBIT_GROUPS,
    qknn_edges,
    quantized_vectors,
    roles_from_table,
)
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

VECTORS = HERE / "tiger_vectors.npz"
OUT_JSON = HERE / "edge_recovery_structure.json"
OUT_MD = HERE / "edge_recovery_structure.md"
K = 10
ROLE_NAMES = list(json.loads(ORBIT_GROUPS.read_text()).keys())


def rate(mask: np.ndarray, recovered: np.ndarray, base: float) -> dict:
    total = int(mask.sum())
    hits = int((mask & recovered).sum())
    value = hits / total if total else float("nan")
    return {
        "total": total,
        "recovered": hits,
        "rate": value,
        "lift": value / base if base > 0 else float("nan"),
    }


def frame_from_edges(nodes: int, edges: np.ndarray, behavior_adj: list[set[int]],
                     role_dist: np.ndarray, top: np.ndarray, comm: np.ndarray, multi: np.ndarray) -> pd.DataFrame:
    left = edges[:, 0]
    right = edges[:, 1]
    degree = np.array([len(x) for x in behavior_adj], dtype=np.int64)
    du = degree[left]
    dv = degree[right]
    common = np.array([len(behavior_adj[u] & behavior_adj[v]) for u, v in zip(left, right)], dtype=np.int64)
    k4 = np.zeros(len(edges), dtype=bool)
    for index, (u, v) in enumerate(zip(left, right)):
        shared = list(behavior_adj[int(u)] & behavior_adj[int(v)])
        if len(shared) >= 2:
            for a in range(len(shared)):
                for b in range(a + 1, len(shared)):
                    if shared[b] in behavior_adj[shared[a]]:
                        k4[index] = True
                        break
                if k4[index]:
                    break
    role_cos = np.einsum("ij,ij->i", role_dist[left], role_dist[right]) / np.maximum(
        np.linalg.norm(role_dist[left], axis=1) * np.linalg.norm(role_dist[right], axis=1), 1e-12
    )
    low_role = np.minimum(top[left], top[right])
    high_role = np.maximum(top[left], top[right])
    role_pair = low_role * 9 + high_role
    return pd.DataFrame({
        "u": left,
        "v": right,
        "du": du,
        "dv": dv,
        "dmax": np.maximum(du, dv),
        "dmin": np.minimum(du, dv),
        "common": common,
        "triangle": common > 0,
        "k4": k4,
        "role_pair": role_pair,
        "role_cos": role_cos,
        "within_community": comm[left] == comm[right],
        "multi_u": multi[left],
        "multi_v": multi[right],
        "multi_edge": multi[left] | multi[right],
    })


def role_pair_table(frame: pd.DataFrame, recovered: np.ndarray, base: float, limit: int = 18) -> list[dict]:
    rows = []
    for pair, group in frame.groupby("role_pair"):
        count = len(group)
        if count < 100:
            continue
        hits = int(recovered[group.index].sum())
        value = hits / count
        left, right = divmod(int(pair), 9)
        rows.append({
            "left": ROLE_NAMES[left],
            "right": ROLE_NAMES[right],
            "edges": count,
            "recovered": hits,
            "rate": value,
            "lift": value / base if base > 0 else float("nan"),
        })
    rows.sort(key=lambda row: row["edges"], reverse=True)
    return rows[:limit]


def logistic(frame: pd.DataFrame, target: np.ndarray) -> dict:
    features = pd.DataFrame({
        "log_du": np.log1p(frame["du"].to_numpy()),
        "log_dv": np.log1p(frame["dv"].to_numpy()),
        "triangle": frame["triangle"].astype(float).to_numpy(),
        "k4": frame["k4"].astype(float).to_numpy(),
        "within_community": frame["within_community"].astype(float).to_numpy(),
    })
    x = np.column_stack([np.ones(len(features)), features.to_numpy()])
    y = target.astype(float)
    names = ["const", *features.columns]

    def negative_loglik(beta: np.ndarray) -> float:
        linear = x @ beta
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    fit = minimize(negative_loglik, np.zeros(x.shape[1]), method="BFGS")
    beta = fit.x
    probability = 1.0 / (1.0 + np.exp(-(x @ beta)))
    weight = probability * (1.0 - probability)
    covariance = np.linalg.pinv(x.T @ (weight[:, None] * x))
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
    z_value = beta / standard_error
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_value)))
    base = y.mean()
    null = -(y.sum() * np.log(max(base, 1e-12)) + (1.0 - y).sum() * np.log(max(1.0 - base, 1e-12)))
    return {
        "n": int(len(target)),
        "events": int(target.sum()),
        "pseudo_r2": float(1.0 - fit.fun / null) if null > 0 else float("nan"),
        "terms": [
            {
                "term": name,
                "coef": float(beta[index]),
                "odds_ratio": float(np.exp(beta[index])),
                "p_value": float(p_value[index]),
            }
            for index, name in enumerate(names)
        ],
    }


def main() -> int:
    nodes, behavior_edges, adjacency = load_behavior_graph()
    z, q = quantized_vectors(VECTORS)
    z_raw = qknn_edges(z, k=K)
    q_raw = qknn_edges(q, k=K)
    target = min(len(z_raw), len(q_raw))
    z_edges, z_trimmed = trim_edges(z, z_raw, target)
    q_edges, q_trimmed = trim_edges(q, q_raw, target)
    z_set = edged(z_edges)
    q_set = edged(q_edges)
    z_recovered = np.array([(int(u), int(v)) in z_set for u, v in behavior_edges], dtype=bool)
    q_recovered = np.array([(int(u), int(v)) in q_set for u, v in behavior_edges], dtype=bool)
    base_z = float(z_recovered.mean())
    base_q = float(q_recovered.mean())
    print(f"behaviour {len(behavior_edges)}, z recovered {z_recovered.sum()}, q recovered {q_recovered.sum()}, "
          f"z trimmed {z_trimmed}, q trimmed {q_trimmed}", flush=True)

    table = np.loadtxt(BEHAVIOUR_ORCA, dtype=np.float64)
    role_dist, top, multi, _ = roles_from_table(table)
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    comm = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        comm[list(members)] = index

    frame = frame_from_edges(nodes, behavior_edges, adjacency, role_dist, top, comm, multi)
    frame["z_recovered"] = z_recovered
    frame["q_recovered"] = q_recovered

    feature_groups = {
        "triangle": frame["triangle"].to_numpy(),
        "non_triangle": (~frame["triangle"]).to_numpy(),
        "k4": frame["k4"].to_numpy(),
        "non_k4": (~frame["k4"]).to_numpy(),
        "common_0": (frame["common"] == 0).to_numpy(),
        "common_1": (frame["common"] == 1).to_numpy(),
        "common_2": (frame["common"] == 2).to_numpy(),
        "common_3plus": (frame["common"] >= 3).to_numpy(),
        "within_community": frame["within_community"].to_numpy(),
        "cross_community": (~frame["within_community"]).to_numpy(),
        "multi_edge": frame["multi_edge"].to_numpy(),
        "single_edge": (~frame["multi_edge"]).to_numpy(),
        "star_like": ((frame["common"] == 0) & (frame["dmax"] >= 5) & (frame["dmin"] <= 2)).to_numpy(),
    }
    groups = {
        name: {"z": rate(mask, z_recovered, base_z), "q": rate(mask, q_recovered, base_q)}
        for name, mask in feature_groups.items()
    }

    # degree quintiles of dmax
    quintile = pd.qcut(frame["dmax"], 5, labels=False, duplicates="drop")
    degree_groups = {}
    for value in range(int(quintile.max()) + 1):
        mask = (quintile == value).to_numpy()
        degree_groups[f"Q{value + 1}"] = {
            "dmax_min": int(frame.loc[mask, "dmax"].min()),
            "dmax_max": int(frame.loc[mask, "dmax"].max()),
            "z": rate(mask, z_recovered, base_z),
            "q": rate(mask, q_recovered, base_q),
        }

    four = {
        "always": int((z_recovered & q_recovered).sum()),
        "q_lost": int((z_recovered & ~q_recovered).sum()),
        "q_added": int((~z_recovered & q_recovered).sum()),
        "never": int((~z_recovered & ~q_recovered).sum()),
    }

    role_pairs_z = role_pair_table(frame, z_recovered, base_z)
    role_pairs_q = role_pair_table(frame, q_recovered, base_q)

    # Q loss selectivity among z-recovered edges
    z_frame = frame[frame["z_recovered"]].copy()
    z_frame_target = z_frame["q_recovered"].to_numpy()
    loss_groups = {}
    for name, mask in {
        "triangle": z_frame["triangle"].to_numpy(),
        "non_triangle": (~z_frame["triangle"]).to_numpy(),
        "k4": z_frame["k4"].to_numpy(),
        "non_k4": (~z_frame["k4"]).to_numpy(),
        "within_community": z_frame["within_community"].to_numpy(),
        "cross_community": (~z_frame["within_community"]).to_numpy(),
        "multi_edge": z_frame["multi_edge"].to_numpy(),
        "single_edge": (~z_frame["multi_edge"]).to_numpy(),
    }.items():
        loss_groups[name] = {
            "total": int(mask.sum()),
            "retained": int((mask & z_frame_target).sum()),
            "lost": int((mask & ~z_frame_target).sum()),
            "retention_rate": float((mask & z_frame_target).sum() / max(mask.sum(), 1)),
        }

    result = {
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "k": K,
        "z_edges_raw": int(len(z_raw)),
        "q_edges_raw": int(len(q_raw)),
        "matched_edges": int(target),
        "z_trimmed": int(z_trimmed),
        "q_trimmed": int(q_trimmed),
        "z_recovered": int(z_recovered.sum()),
        "q_recovered": int(q_recovered.sum()),
        "base_rate_z": base_z,
        "base_rate_q": base_q,
        "groups": groups,
        "degree_quintiles": degree_groups,
        "four_fate": four,
        "role_pairs_z": role_pairs_z,
        "role_pairs_q": role_pairs_q,
        "q_loss_selectivity": loss_groups,
        "logistic_z": logistic(frame, z_recovered),
        "logistic_q_given_z": logistic(z_frame, z_frame_target),
        "q_fate_frame": {
            "triangle_rate": {
                "always": float(z_frame.loc[z_frame["q_recovered"], "triangle"].mean()),
                "q_lost": float(z_frame.loc[~z_frame["q_recovered"], "triangle"].mean()),
            },
            "within_community_rate": {
                "always": float(z_frame.loc[z_frame["q_recovered"], "within_community"].mean()),
                "q_lost": float(z_frame.loc[~z_frame["q_recovered"], "within_community"].mean()),
            },
            "median_dmax": {
                "always": float(z_frame.loc[z_frame["q_recovered"], "dmax"].median()),
                "q_lost": float(z_frame.loc[~z_frame["q_recovered"], "dmax"].median()),
            },
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Why are the recovered TIGER behaviour edges recovered?",
        "",
        "TIGER only. Edge-budget-matched global k=10 z/q graphs from `embedding_knn_compare.py`. "
        "Each row is a real TRAIN behaviour edge.",
        "",
        f"Behaviour edges **{len(behavior_edges)}**; z recovered **{z_recovered.sum()} ({base_z:.4%})**; "
        f"q recovered **{q_recovered.sum()} ({base_q:.4%})**.",
        f"Matched edge budget: z {len(z_raw)}→{len(z_edges)}, q {len(q_raw)}→{len(q_edges)}.",
        "",
        "## Retention by local structure and community",
        "",
        "| type | edges | z recovered | z rate | z lift | q recovered | q rate | q lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in groups.items():
        lines.append(
            f"| {name} | {value['z']['total']} | {value['z']['recovered']} | {value['z']['rate']:.5f} | "
            f"{value['z']['lift']:.3f} | {value['q']['recovered']} | {value['q']['rate']:.5f} | {value['q']['lift']:.3f} |"
        )
    lines += [
        "",
        "`non_triangle` is used as the chain-like/acyclic proxy; `star_like` = common neighbours 0, "
        "max degree ≥ 5, min degree ≤ 2. These are heuristics, not exact 5-node graphlet classes.",
        "",
        "## Degree control",
        "",
        "| dmax quintile | dmax range | edges | z rate | z lift | q rate | q lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in degree_groups.items():
        lines.append(
            f"| {name} | {value['dmax_min']}–{value['dmax_max']} | {value['z']['total']} | "
            f"{value['z']['rate']:.5f} | {value['z']['lift']:.3f} | {value['q']['rate']:.5f} | {value['q']['lift']:.3f} |"
        )
    lines += [
        "",
        "## Logistic controls (odds ratios)",
        "",
        "| model | term | coef | odds ratio | p |",
        "|---|---|---:|---:|---:|",
    ]
    for label, value in (("z recovery", result["logistic_z"]),
                         ("q retention given z", result["logistic_q_given_z"])):
        for term in value["terms"]:
            lines.append(
                f"| {label} | {term['term']} | {term['coef']:+.4f} | {term['odds_ratio']:.3f} | {term['p_value']:.4f} |"
            )
    lines += [
        "",
        "## Four fates of a behaviour edge",
        "",
        "| fate | edges | share |",
        "|---|---:|---:|",
    ]
    for name, value in four.items():
        lines.append(f"| {name} | {value} | {value / len(behavior_edges):.4f} |")
    lines += [
        "",
        "## ORCA role pairs: z recovery",
        "",
        "| left role | right role | edges | z recovered | z rate | z lift |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in role_pairs_z:
        lines.append(
            f"| {row['left']} | {row['right']} | {row['edges']} | {row['recovered']} | "
            f"{row['rate']:.5f} | {row['lift']:.3f} |"
        )
    lines += [
        "",
        "## Quantisation selectivity among z-recovered edges",
        "",
        "| group | total | retained by q | lost by q | q retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in loss_groups.items():
        lines.append(
            f"| {name} | {value['total']} | {value['retained']} | {value['lost']} | {value['retention_rate']:.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
