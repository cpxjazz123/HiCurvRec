"""Issue #218 step 1-2: retention of the five induced graphlets in each archived SID.

For every arm and every prefix depth (L1, L1+L2, full) the script measures

    all_share   - instances whose nodes all share the prefix
    pair_share  - the share over all node pairs inside the instances, split by real edge /
                  non-edge and by same-orbit / different-orbit
    null        - the analytic expectation of each share under a random assignment of the
                  arm's own code frequencies (exact, no permutation noise)
    gain/lift   - observed minus / over that null

The instance pool is the frozen #217 pool (heldout split of the canonical TRAIN graph), used
unchanged for all arms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "analysis_sid_recall" / "r2"
INSTANCES = REPO / "results" / "research" / "graphlet_roles" / "instances.npz"
TYPES = ("K3", "P3", "C4", "P4", "K1_3")
DEPTHS = {"L1": 1, "L1L2": 2, "full": 3}
OUT = Path(__file__).resolve().parent
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
POOL = REPO / "results" / "research" / "g4_closure" / "pairs_heldout.npz"


def load_sid(number: str) -> np.ndarray:
    payload = json.loads((MIRROR / number / "item_sids.json").read_text())
    return np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)


def graph_arrays():
    payload = np.load(GRAPH, allow_pickle=False)
    rows, cols = payload["rows"].astype(np.int64), payload["cols"].astype(np.int64)
    adjacency = [set() for _ in range(int(payload["nodes"]))]
    for a, b in zip(rows.tolist(), cols.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    return rows, cols, adjacency


def analytic_null(frequencies: np.ndarray, nodes: int, size: int) -> tuple[float, float]:
    """P(all k share) and P(two share) under a random assignment with these frequencies."""
    total = frequencies.sum()
    denominator_all = np.prod([total - i for i in range(size)])
    denominator_pair = total * (total - 1)
    share_all = 0.0
    share_pair = 0.0
    for count in frequencies:
        if count >= size:
            share_all += np.prod([count - i for i in range(size)]) / denominator_all
        share_pair += count * (count - 1) / denominator_pair
    return float(share_all), float(share_pair)


def type_masks(nodes: np.ndarray, roles: np.ndarray, adjacency: list) -> dict:
    """The pair masks of one graphlet type: independent of the arm, so built once."""
    k = nodes.shape[1]
    pair = ~np.eye(k, dtype=bool)
    edge = np.zeros((len(nodes), k, k), dtype=bool)
    for index, row in enumerate(nodes.tolist()):
        for i in range(k):
            for j in range(k):
                if i != j and row[j] in adjacency[row[i]]:
                    edge[index, i, j] = True
    same = (roles[:, :, None] == roles[:, None, :])
    return {"pair": pair, "edge": edge & pair, "nonedge": (~edge) & pair,
            "same_role": same & pair, "diff_role": (~same) & pair, "k": k}


def measure(codes: np.ndarray, nodes: np.ndarray, masks: dict, depth: int) -> dict:
    sliced = codes[nodes, :depth]
    pair_eq = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    pair = masks["pair"]
    all_share = float(pair_eq[:, pair].all(axis=1).mean())
    pairs = float((pair_eq & pair).sum() / (len(nodes) * pair.sum()))
    def share(mask):
        count = mask.sum()
        return float((pair_eq & mask).sum() / count) if count else float("nan")
    return {"all_share": all_share, "pair_share": pairs,
            "edge_share": share(masks["edge"]), "nonedge_share": share(masks["nonedge"]),
            "same_role_share": share(masks["same_role"]), "diff_role_share": share(masks["diff_role"])}


def sample_triangles(adjacency: list, limit: int = 20000, seed: int = 218) -> np.ndarray:
    found = []
    for a in range(len(adjacency)):
        neighbours = sorted(adjacency[a])
        for index, b in enumerate(neighbours):
            for c in neighbours[index + 1:]:
                if c in adjacency[b]:
                    found.append((a, b, c))
        if len(found) >= limit * 4:
            break
    array = np.array(found, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return array[rng.choice(len(array), limit, replace=False)] if len(array) > limit else array


def global_metrics(codes: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                   triangles: np.ndarray) -> dict:
    triples = codes[:, :3]
    unique = len({tuple(row) for row in triples.tolist()})
    counts = np.bincount(codes[:, 0], minlength=int(codes[:, 0].max()) + 1)
    tri_codes = codes[triangles, 0]
    return {"triangle_L1": float((tri_codes[:, 0] == tri_codes[:, 1]).mean()
                                 * (tri_codes[:, 0] == tri_codes[:, 2]).mean()),
            "edge_L1": float((codes[rows, 0] == codes[cols, 0]).mean()),
            "edge_L1L2": float((codes[rows, :2] == codes[cols, :2]).all(axis=1).mean()),
            "collision": 1 - unique / len(codes),
            "L1_used": int((counts > 0).sum()),
            "L1_max_bucket": int(counts.max()),
            "L1_max_share": float(counts.max() / len(codes)),
            "L2_prefixes": int(len({tuple(row[:2]) for row in codes.tolist()}))}


def main() -> None:
    payload = np.load(INSTANCES, allow_pickle=False)
    rows, cols, adjacency = graph_arrays()
    triangles = sample_triangles(adjacency)
    pool = np.load(POOL)
    nonedge_pairs = pool["triples"][:, [1, 2]].astype(np.int64)
    manifest = {"pool": str(INSTANCES), "graph_sha256": json.loads(
        (REPO / "results" / "research" / "behavior_graph" / "metadata.json").read_text()
    )["outputs"]["graph.npz"]["sha256"], "types": {}}
    type_nodes, type_masks_cache = {}, {}
    for name in TYPES:
        keep = payload[f"{name}_split"] == 1
        type_nodes[name] = payload[f"{name}_nodes"][keep]
        roles = payload[f"{name}_roles"][keep]
        type_masks_cache[name] = type_masks(type_nodes[name], roles, adjacency)
        manifest["types"][name] = {"instances": int(keep.sum()), "roles": roles.tolist()[:0] or "recorded in #217"}
    (OUT / "graphlet_pool_manifest.json").write_text(json.dumps(manifest, indent=1))

    header = ["number", "arm", "type", "depth", "all_share", "pair_share", "edge_share",
              "nonedge_share", "same_role_share", "diff_role_share", "null_all", "null_pair",
              "gain_all", "gain_pair", "lift_pair"]
    lines = [",".join(header)]
    global_rows = [",".join(["number", "arm", "edge_L1", "edge_L1L2", "matched_nonedge_L1",
                             "triangle_L1", "collision", "L1_used", "L1_max_bucket",
                             "L1_max_share", "L2_prefixes"])]
    for number in sorted(p.name for p in MIRROR.iterdir() if p.is_dir()):
        sid = load_sid(number)
        arm = (REPO / "results" / "research" / "issue139" / "r2_ledger.json")
        ledger = json.loads(arm.read_text())
        name = (ledger.get(number) or {}).get("arm", "")
        metrics = global_metrics(sid, rows, cols, triangles)
        metrics["matched_nonedge_L1"] = float(
            (sid[nonedge_pairs[:, 0], 0] == sid[nonedge_pairs[:, 1], 0]).mean())
        ordered = ("edge_L1", "edge_L1L2", "matched_nonedge_L1", "triangle_L1", "collision",
                   "L1_used", "L1_max_bucket", "L1_max_share", "L2_prefixes")
        global_rows.append(",".join([number, f'"{name}"'] + [str(metrics[k]) for k in ordered]))
        for name_type in TYPES:
            nodes = type_nodes[name_type]
            for label, depth in DEPTHS.items():
                stats = measure(sid, nodes, type_masks_cache[name_type], depth)
                # the analytic null uses the marginal code frequencies at this depth
                if depth == 1:
                    codes = sid[:, 0]
                elif depth == 2:
                    codes = sid[:, 0] * 4096 + sid[:, 1]
                else:
                    codes = (sid[:, 0] * 4096 + sid[:, 1]) * 4096 + sid[:, 2]
                # np.unique, not bincount: the full-depth code is up to 256**3 and a dense
                # histogram of that range needs ~34 GB.
                _, frequency = np.unique(codes, return_counts=True)
                null_all, null_pair = analytic_null(frequency.astype(np.float64), len(codes), nodes.shape[1])
                rows_out = [number, name_type.replace(",", "_"), label]
                values = [stats["all_share"], stats["pair_share"], stats["edge_share"],
                          stats["nonedge_share"], stats["same_role_share"], stats["diff_role_share"],
                          null_all, null_pair, stats["all_share"] - null_all,
                          stats["pair_share"] - null_pair,
                          (stats["pair_share"] / null_pair) if null_pair else float("nan")]
                lines.append(",".join([number, f'"{name}"', name_type, label] +
                                      [f"{v:.8g}" if isinstance(v, float) else str(v) for v in values]))
    (OUT / "per_arm_retention.csv").write_text("\n".join(lines) + "\n")
    (OUT / "per_arm_global.csv").write_text("\n".join(global_rows) + "\n")
    print("wrote per_arm_retention.csv", len(lines) - 1, "rows; per_arm_global.csv", len(global_rows) - 1)


if __name__ == "__main__":
    main()
