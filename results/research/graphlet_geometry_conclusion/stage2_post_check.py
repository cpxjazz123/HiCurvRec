"""Post-Stage2 check for the two codeword arms.

Reads each exported SID, reports the layer statistics that decide whether the run
is usable, and compares it with F0: the control arm's table is constant, so its
SID should be F0's; if it is not, the pair needs its own Stage3 baseline.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/graphlet_geometry_conclusion/stage2_post_check.py
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGE2 = REPO / "results" / "stage2_RQ-VAE"
ARMS = ["TIGER_mixed_rq_F0", "TIGER_graphlet_codebook_rq",
        "TIGER_graphlet_codebook_global_rq"]


def load(arm: str) -> dict[int, tuple]:
    path = STAGE2 / arm / "item_sids.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): tuple(int(v) for v in row[:3]) for key, row in payload.items()}


def item_density(n_items: int) -> "np.ndarray":
    """The same TRAIN structural density the trainer allocates with."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    train = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    pairs = set()
    for history, target in zip(train["history"].to_numpy(), train["target"].to_numpy()):
        history = list(history)
        for left, right in zip(history, history[1:]):
            pairs.add((left, right) if left < right else (right, left))
        if history:
            tail = history[-1]
            pairs.add((tail, target) if tail < target else (target, tail))
    edges = np.array(sorted(pairs), dtype=np.int64)
    adjacency = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(n_items, n_items)).tocsr()
    adjacency = adjacency + adjacency.T
    triangles = np.rint(np.asarray((adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel())
    incident = np.zeros(n_items)
    count = np.zeros(n_items)
    np.add.at(incident, edges[:, 0], triangles)
    np.add.at(incident, edges[:, 1], triangles)
    np.add.at(count, edges[:, 0], 1.0)
    np.add.at(count, edges[:, 1], 1.0)
    return incident / np.maximum(count, 1.0)


def table_fidelity(arm: str, sid: dict) -> dict | None:
    """Does the saved codeword table rank the codes by their covered density?"""
    import numpy as np
    import torch
    path = STAGE2 / arm / "rqvae_best.pth"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    table = payload.get("code_alpha")
    if table is None:
        return None
    table = np.array(table, dtype=np.float64)
    density = item_density(max(sid) + 1)
    tokens = np.array([sid[item] for item in range(max(sid) + 1)])
    out = {"mode": payload.get("code_alpha_mode"), "levels": []}
    for level in range(min(3, table.shape[0])):
        codes = tokens[:, level]
        size = table.shape[1]
        sums = np.bincount(codes, weights=density, minlength=size)
        counts = np.bincount(codes, minlength=size)
        used = counts > 0
        mean = np.divide(sums, np.maximum(counts, 1))[used]
        alpha = table[level, :size][used]
        order_density = np.argsort(np.argsort(mean))
        order_alpha = np.argsort(np.argsort(alpha))
        rho = float(np.corrcoef(order_density, order_alpha)[0, 1]) if mean.size > 2 else None
        out["levels"].append({"level": level + 1, "codes_used": int(used.sum()),
                              "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
                              "spearman_density_alpha": rho})
    return out


def main() -> None:
    sids = {}
    rows = []
    for arm in ARMS:
        path = STAGE2 / arm / "item_sids.json"
        if not path.is_file():
            print(f"{arm}: missing SID")
            continue
        sid = load(arm)
        sids[arm] = sid
        triples = collections.Counter(sid.values())
        l1 = {value[0] for value in sid.values()}
        l2 = {value[:2] for value in sid.values()}
        rows.append({
            "arm": arm,
            "items": len(sid),
            "l1_distinct": len(l1),
            "l2_distinct": len(l2),
            "l3_distinct": len(triples),
            "l3_collision": 1.0 - len(triples) / len(sid),
            "sid_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    header = f"{'arm':38s} {'items':>6s} {'L1':>4s} {'L2':>6s} {'L3':>6s} {'collision':>10s}"
    print(header)
    for row in rows:
        print(f"{row['arm']:38s} {row['items']:6d} {row['l1_distinct']:4d} "
              f"{row['l2_distinct']:6d} {row['l3_distinct']:6d} {row['l3_collision']:10.4f}")
    for row in rows:
        print(f"{row['arm']}: {row['sid_sha256']}")

    base = sids.get("TIGER_mixed_rq_F0")
    if base is not None:
        for arm, sid in sids.items():
            if arm == "TIGER_mixed_rq_F0":
                continue
            shared = set(base) & set(sid)
            same_l1 = sum(1 for item in shared if base[item][0] == sid[item][0]) / max(len(shared), 1)
            same_l3 = sum(1 for item in shared if base[item] == sid[item]) / max(len(shared), 1)
            print(f"{arm} vs F0: same_L1={same_l1:.4f} same_L3={same_l3:.4f}")

    fidelity = {arm: table_fidelity(arm, sid) for arm, sid in sids.items()}
    for arm, entry in fidelity.items():
        if entry is None:
            continue
        print(f"{arm} codeword table ({entry['mode']}): " + " ".join(
            f"L{level['level']}:{level['alpha_min']:.3f}-{level['alpha_max']:.3f}"
            f"/{level['codes_used']}/rho={level['spearman_density_alpha']:.3f}"
            for level in entry["levels"]))
    gate = {row["arm"]: row["l3_collision"] < 0.40 for row in rows}
    print("stage3 collision gate (<0.40):", gate)
    (HERE / "stage2_post_check.json").write_text(
        json.dumps({"arms": rows, "stage3_gate": gate, "codeword_table": fidelity},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
