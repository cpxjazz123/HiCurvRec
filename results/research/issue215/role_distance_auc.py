"""Do behaviour neighbours land closer, role by role, before and after quantization?

For every ORCA role this asks the question the owner posed: in the model's own vector space, are the
items a real behaviour edge connects *closer* than items that are not connected?  A role where the
answer is strong is one whose behaviour relations the representation expresses well; a role where
it is weak (or inverted) is one where it does not.

Measured for both vectors the model produces:

    z    the encoder's vector before quantization
    z_q  the model's own quantized readout

and reported as an AUC, which is a rank statistic and therefore says nothing about the overall
scale: 0.5 means a behaviour pair is no closer than a random pair of the same role, 0.75 means the
behaviour pair is closer three times out of four.

Random pairs are drawn **inside the same role**, so a role cannot look good or bad merely because
its items sit at a different distance from the origin.

    python3 results/research/issue215/role_distance_auc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
ROLE_TABLE = OUT / "orca_roles_frozen.csv"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
JOURNAL = OUT / "role_distance_auc.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
ROLE_NAMES = {0: "dense hub", 1: "tight community", 2: "two-hop / fan", 3: "degree middle",
              4: "cycle / long path", 5: "star / chain"}
SEED = 42
RANDOM_PER_EDGE = 1


def auc(edge_distances: np.ndarray, random_distances: np.ndarray) -> float:
    """P(an edge pair is closer than a random pair), by rank, without materialising a matrix."""
    combined = np.concatenate([edge_distances, random_distances])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    n_edge = edge_distances.size
    rank_sum = ranks[:n_edge].sum()
    return float((rank_sum - n_edge * (n_edge + 1) / 2.0) / (n_edge * random_distances.size))


def main() -> int:
    roles: dict[int, int] = {}
    with ROLE_TABLE.open() as handle:
        next(handle)
        for line in handle:
            item, role = line.strip().split(",")
            roles[int(item)] = int(role)

    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        item, nxt = int(history[-1]), int(target)
        if item in roles and nxt in roles:
            pairs.add((item, nxt))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    edge_roles = np.asarray([roles[int(a)] for a in edges[:, 0]], dtype=np.int64)
    print(f"behaviour edges with roles on both ends: {len(edges)}", flush=True)

    rng = np.random.default_rng(SEED)
    by_role = {role: np.asarray(sorted(item for item, value in roles.items() if value == role), dtype=np.int64)
               for role in sorted(ROLE_NAMES)}
    baseline = {}
    for role, items in by_role.items():
        count = int((edge_roles == role).sum()) * RANDOM_PER_EDGE
        if count == 0 or items.size < 2:
            baseline[role] = np.zeros((0, 2), dtype=np.int64)
            continue
        left = rng.choice(items, size=count, replace=True)
        right = rng.choice(items, size=count, replace=True)
        keep = left != right
        baseline[role] = np.stack([left[keep], right[keep]], axis=1)

    # the same question asked in the identifier's own currency: does a behaviour pair share a
    # level-1 code more often than a random same-role pair, and by how much
    def share_rate(codes: np.ndarray, pair: np.ndarray) -> float:
        left, right = codes[pair[:, 0]], codes[pair[:, 1]]
        valid = (left >= 0) & (right >= 0)
        if not valid.any():
            return float("nan")
        return float((left[valid] == right[valid]).mean())

    report = {"edges": int(len(edges)), "role_table": str(ROLE_TABLE.relative_to(REPO)), "arms": {}}
    for arm in ARMS:
        z_path, q_path = VECTORS / f"{arm}_z.npy", VECTORS / f"{arm}_q.npy"
        if not z_path.is_file() or not q_path.is_file():
            print(f"{arm}: vectors missing", flush=True)
            continue
        z = np.load(z_path).astype(np.float64)
        q = np.load(q_path).astype(np.float64)
        block = {}
        for label, vectors in (("z", z), ("q", q)):
            per_role = {}
            for role in sorted(ROLE_NAMES):
                pick = edge_roles == role
                if not pick.any():
                    continue
                pair = edges[pick]
                edge_distance = np.linalg.norm(vectors[pair[:, 0]] - vectors[pair[:, 1]], axis=1)
                base = baseline[role]
                random_distance = np.linalg.norm(vectors[base[:, 0]] - vectors[base[:, 1]], axis=1)
                per_role[str(role)] = {
                    "edges": int(pick.sum()),
                    "edge_distance_mean": float(edge_distance.mean()),
                    "random_distance_mean": float(random_distance.mean()),
                    "relative_gap": float(1.0 - edge_distance.mean() / max(random_distance.mean(), 1e-12)),
                    "auc": auc(edge_distance, random_distance),
                }
            block[label] = per_role
        # the level-1 share, edge vs random, per role -- the statistic that stays meaningful when
        # the quantized distances collapse into ties
        tokens_path = VECTORS / f"{arm}_tokens.npy"
        if tokens_path.is_file():
            level1 = np.load(tokens_path)[:, 0]
            block["share1"] = {}
            for role in sorted(ROLE_NAMES):
                pick = edge_roles == role
                if not pick.any():
                    continue
                block["share1"][str(role)] = {
                    "edge_share": share_rate(level1, edges[pick]),
                    "random_share": share_rate(level1, baseline[role]),
                }
        report["arms"][arm] = block
        print(f"\n{arm}: AUC of {label} (P(behaviour pair closer than a random same-role pair))", flush=True)
        for role in sorted(ROLE_NAMES):
            if str(role) not in block["z"]:
                continue
            zr, qr = block["z"][str(role)], block["q"][str(role)]
            print(f"  {role} {ROLE_NAMES[role]:<20} z {zr['auc']:.4f} (gap {zr['relative_gap']:+.3f}) "
                  f"| q {qr['auc']:.4f} (gap {qr['relative_gap']:+.3f}) | delta {qr['auc'] - zr['auc']:+.4f}",
                  flush=True)

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))

    print("\nAUC by role and arm (z -> q):")
    header = " | ".join(f"{arm:>13}" for arm in ARMS)
    print(f"{'role':<22} | {header}")
    for role in sorted(ROLE_NAMES):
        cells = []
        for arm in ARMS:
            block = report["arms"].get(arm, {})
            if "z" not in block or str(role) not in block["z"]:
                cells.append(f"{'—':>13}")
                continue
            cells.append(f"{block['z'][str(role)]['auc']:.3f}->{block['q'][str(role)]['auc']:.3f}")
        print(f"{role} {ROLE_NAMES[role]:<20} | " + " | ".join(cells))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
