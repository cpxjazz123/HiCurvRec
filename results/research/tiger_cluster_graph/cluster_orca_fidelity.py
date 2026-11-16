"""Cluster-level ORCA structural fidelity for pure Euclidean TIGER and mixed-curvature F0.

For each arm independently:

    KMeans(k=256) on pre-quantisation z
    assign both z and q with the SAME centres
    aggregate the 73-d ORCA behaviour-GDV by cluster
    compare z-cluster and q-cluster role distributions (JSD / agreement)
    compare cluster-internal behaviour edges

The behaviour graph and its ORCA GDV are built once and are the answer key.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/cluster_orca_fidelity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BEHAVIOUR_ORCA = REPO / "results" / "orca_build" / "runs" / "behaviour_gpu_5.out"
GROUPS = HERE / "orbit_groups.json"
ARMS = {
    "pure_euclidean_TIGER": HERE / "tiger_vectors.npz",
    "mixed_F0": HERE / "mixed_vectors.npz",
}
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = HERE / "cluster_orca_fidelity.json"
MD = HERE / "cluster_orca_fidelity.md"
K = 256
SEED = 42
MIN_CLUSTER = 10


def behaviour_edges(nodes: int) -> np.ndarray:
    frame = pd.read_parquet(TRAIN, columns=["user", "seen_history", "target"])
    best = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)
    ):
        history = [] if history is None else [int(x) for x in history]
        user = int(user)
        if user not in best or len(history) > best[user][0]:
            best[user] = (len(history), [*history, int(target)])
    edges = set()
    for _, session in best.values():
        for left, right in zip(session, session[1:]):
            edges.add((left, right) if left < right else (right, left))
    return np.array(sorted(edges), dtype=np.int64)


def jsd(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.sum() <= 0 or second.sum() <= 0:
        return float("nan")
    first = first / first.sum()
    second = second / second.sum()
    middle = 0.5 * (first + second)
    value = 0.0
    for left, right in ((first, middle), (second, middle)):
        mask = left > 0
        value += 0.5 * float((left[mask] * np.log(left[mask] / right[mask])).sum())
    return value


def cluster_mean(matrix: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((K, matrix.shape[1]), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    np.add.at(sums, labels[valid], matrix[valid])
    np.add.at(counts, labels[valid], 1)
    means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    return means, counts


def within_share(labels: np.ndarray, edges: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    left, right = labels[edges[:, 0]], labels[edges[:, 1]]
    within = np.zeros(K, dtype=np.int64)
    incident = np.zeros(K, dtype=np.int64)
    np.add.at(incident, left, 1)
    np.add.at(incident, right, 1)
    both = np.flatnonzero(left == right)
    np.add.at(within, left[both], 1)
    # incident counts each within-cluster edge twice (both endpoints) and each boundary edge once;
    # the number of distinct edges touching c is therefore incident - within.
    denominator = incident - within
    rate = np.divide(within, denominator, out=np.full(K, np.nan), where=denominator > 0)
    total = int((incident - within).sum())
    global_share = float(within.sum() / max(total, 1))
    return global_share, rate, incident


def main() -> int:
    groups = json.loads(GROUPS.read_text())
    orbit = np.loadtxt(BEHAVIOUR_ORCA, dtype=np.float64)
    if orbit.ndim != 2 or orbit.shape[1] != 73:
        raise SystemExit(f"unexpected ORCA table shape: {orbit.shape}")
    nodes = orbit.shape[0]
    valid = orbit.sum(axis=1) > 0
    probability = np.divide(
        orbit, orbit.sum(axis=1, keepdims=True),
        out=np.zeros_like(orbit), where=orbit.sum(axis=1, keepdims=True) > 0,
    )
    role_scores = np.zeros((nodes, len(groups)), dtype=np.float64)
    for column, (name, indices) in enumerate(groups.items()):
        role_scores[:, column] = probability[:, indices].sum(axis=1)
    role_total = role_scores.sum(axis=1, keepdims=True)
    role_dist = np.divide(role_scores, role_total, out=np.zeros_like(role_scores), where=role_total > 0)
    multi_role = (role_dist >= 0.25).sum(axis=1) >= 2

    edges = behaviour_edges(nodes)
    print(f"behaviour edges {len(edges)}; valid ORCA rows {valid.sum()}; multi-role {multi_role.mean():.3f}", flush=True)

    result = {
        "orca": str(BEHAVIOUR_ORCA.relative_to(REPO)),
        "nodes": nodes,
        "behaviour_edges": int(len(edges)),
        "groups": list(groups),
        "multi_role_definition": "at least two ORCA orbit-group scores >= 0.25 after per-item normalisation",
        "arms": {},
    }

    for arm, vector_path in ARMS.items():
        payload = np.load(vector_path)
        z, q = payload["z"].astype(np.float32), payload["q"].astype(np.float32)
        model = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(z)
        labels_z = model.labels_.astype(np.int64)
        labels_q = model.predict(q).astype(np.int64)
        switch = labels_z != labels_q

        gdv_z, count_z = cluster_mean(probability, labels_z, valid)
        gdv_q, count_q = cluster_mean(probability, labels_q, valid)
        role_z, _ = cluster_mean(role_dist, labels_z, valid)
        role_q, _ = cluster_mean(role_dist, labels_q, valid)
        shared = np.flatnonzero((count_z >= MIN_CLUSTER) & (count_q >= MIN_CLUSTER))
        gdv_jsd = np.array([jsd(gdv_z[c], gdv_q[c]) for c in shared])
        role_jsd = np.array([jsd(role_z[c], role_q[c]) for c in shared])
        weight = np.minimum(count_z[shared], count_q[shared]).astype(np.float64)
        keep = np.isfinite(gdv_jsd) & np.isfinite(role_jsd)
        gdv_weighted = float(np.average(gdv_jsd[keep], weights=weight[keep]))
        role_weighted = float(np.average(role_jsd[keep], weights=weight[keep]))
        gdv_max = float(np.nanmax(gdv_jsd))
        role_max = float(np.nanmax(role_jsd))

        cocluster_z = float(np.mean(labels_z[edges[:, 0]] == labels_z[edges[:, 1]]))
        cocluster_q = float(np.mean(labels_q[edges[:, 0]] == labels_q[edges[:, 1]]))
        edge_cocluster_z = labels_z[edges[:, 0]] == labels_z[edges[:, 1]]
        edge_cocluster_q = labels_q[edges[:, 0]] == labels_q[edges[:, 1]]
        retention = float((edge_cocluster_z & edge_cocluster_q).sum() / max(edge_cocluster_z.sum(), 1))
        new_edges = float((~edge_cocluster_z & edge_cocluster_q).sum() / max((~edge_cocluster_z).sum(), 1))
        share_z, rate_z, incident_z = within_share(labels_z, edges)
        share_q, rate_q, incident_q = within_share(labels_q, edges)

        clusters = []
        for c in shared:
            members_z = labels_z == c
            members_q = labels_q == c
            clusters.append({
                "cluster": int(c),
                "size_z": int(members_z.sum()),
                "size_q": int(members_q.sum()),
                "overlap": float((members_z & members_q).sum() / max(members_z.sum(), 1)),
                "gdv_jsd": float(gdv_jsd[np.flatnonzero(shared == c)[0]]),
                "role_jsd": float(role_jsd[np.flatnonzero(shared == c)[0]]),
                "within_share_z": float(rate_z[c]),
                "within_share_q": float(rate_q[c]),
                "top_role": list(groups)[int(np.argmax(role_z[c]))],
            })
        clusters.sort(key=lambda row: row["role_jsd"], reverse=True)
        all_clusters = []
        for cluster in clusters:
            index = cluster["cluster"]
            all_clusters.append({
                **cluster,
                "role_z": role_z[index].tolist(),
                "role_q": role_q[index].tolist(),
            })

        multi_switch = float(switch[multi_role].mean()) if multi_role.any() else float("nan")
        single_switch = float(switch[~multi_role].mean()) if (~multi_role).any() else float("nan")
        result["arms"][arm] = {
            "vectors": str(vector_path.relative_to(REPO)),
            "k": K,
            "switch_rate": float(switch.mean()),
            "ari_z_q": float(adjusted_rand_score(labels_z, labels_q)),
            "nmi_z_q": float(normalized_mutual_info_score(labels_z, labels_q)),
            "clusters_kept": int(len(shared)),
            "gdv_jsd_weighted": gdv_weighted,
            "gdv_jsd_max": gdv_max,
            "gdv_agreement": 1.0 - gdv_weighted,
            "role_jsd_weighted": role_weighted,
            "role_jsd_max": role_max,
            "role_agreement": 1.0 - role_weighted,
            "behaviour_cocluster_rate_z": cocluster_z,
            "behaviour_cocluster_rate_q": cocluster_q,
            "behaviour_edge_retention": retention,
            "new_cocluster_rate": new_edges,
            "within_share_z": share_z,
            "within_share_q": share_q,
            "multi_role_fraction": float(multi_role.mean()),
            "multi_role_switch_rate": multi_switch,
            "single_role_switch_rate": single_switch,
            "top_clusters": clusters[:10],
            "all_clusters": all_clusters,
        }
        print(arm, json.dumps({k: v for k, v in result["arms"][arm].items() if k != "top_clusters"}, indent=1), flush=True)

    OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Cluster-level ORCA structural fidelity\n"]
    lines.append(
        "KMeans is fitted on pre-quantisation `z` (k=256); the same centres assign both `z` and `q`. "
        "ORCA is the exact 73-d node-orbit table of the TRAIN behaviour graph (GPU build).\n"
    )
    lines.append("| arm | switch | ARI | NMI | GDV JSD | GDV agreement | role JSD | role agreement | behaviour co-cluster z | q | edge retention | within-share z | q |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm, row in result["arms"].items():
        lines.append(
            f"| {arm} | {row['switch_rate']:.3f} | {row['ari_z_q']:.3f} | {row['nmi_z_q']:.3f} | "
            f"{row['gdv_jsd_weighted']:.4f} | {row['gdv_agreement']:.4f} | {row['role_jsd_weighted']:.4f} | "
            f"{row['role_agreement']:.4f} | {row['behaviour_cocluster_rate_z']:.4f} | "
            f"{row['behaviour_cocluster_rate_q']:.4f} | {row['behaviour_edge_retention']:.4f} | "
            f"{row['within_share_z']:.4f} | {row['within_share_q']:.4f} |\n"
        )
    lines.append("\n## Multi-role items\n\n")
    lines.append("| arm | multi-role fraction | multi-role switch | single-role switch |\n")
    lines.append("|---|---:|---:|---:|\n")
    for arm, row in result["arms"].items():
        lines.append(
            f"| {arm} | {row['multi_role_fraction']:.3f} | {row['multi_role_switch_rate']:.3f} | "
            f"{row['single_role_switch_rate']:.3f} |\n"
        )
    for arm, row in result["arms"].items():
        lines.append(f"\n## Top shifted clusters: {arm}\n\n")
        lines.append("| cluster | size z | size q | overlap | GDV JSD | role JSD | within z | within q | top role |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for cluster in row["top_clusters"]:
            lines.append(
                f"| {cluster['cluster']} | {cluster['size_z']} | {cluster['size_q']} | {cluster['overlap']:.3f} | "
                f"{cluster['gdv_jsd']:.4f} | {cluster['role_jsd']:.4f} | {cluster['within_share_z']:.4f} | "
                f"{cluster['within_share_q']:.4f} | {cluster['top_role']} |\n"
            )
    MD.write_text("".join(lines) + "\n")
    print(f"wrote {OUT} and {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
