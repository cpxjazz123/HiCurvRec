"""Per-role retention of the arms, against the frozen ORCA roles.

    python3 results/research/issue215/retention_by_role.py

For every arm of issue #215 -- the frozen F0 baseline and the three routed arms -- this measures
how well the arm's SID keeps the behaviour edges of each structural role, using the same position
prefix definition everywhere:

    prefix@1   the two endpoints share the level-1 code
    prefix@2   they share levels 1 and 2
    prefix@3   they share all three codes

The roles are the frozen ORCA table (73 five-node orbits, NMF k=6, seed=42) and the edges are the
TRAIN behaviour edges only.  The centred column is the role's retention minus the arm's own overall
retention, which is what removes the arm's raw coarseness from the comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_role_geom_rq"))

from model.role_geometry import load_role_table, sha256_of  # noqa: E402

ROLE_TABLE = REPO / "results" / "research" / "issue215" / "orca_roles_frozen.csv"
ROLE_SHA = "c7a29e6559aed1385c5ce514e506be22d88bf6e85eb61120b6dbf69375ea251c"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = REPO / "results" / "research" / "issue215" / "retention_by_role.json"

ARMS = {
    "A F0": REPO / "analysis_sid_recall" / "r2" / "19" / "item_sids.json",
    "TIGER (hist.)": REPO / "analysis_sid_recall" / "r2" / "24" / "item_sids.json",
    "B global": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_global_rq" / "item_sids.json",
    "C role": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_rq" / "item_sids.json",
    "D shuffled": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_shuffled_rq" / "item_sids.json",
}
ROLE_NAMES = {0: "dense hub", 1: "tight community", 2: "two-hop / fan", 3: "degree middle",
              4: "cycle / long path", 5: "star / chain"}


def train_edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        pairs.add((int(history[-1]), int(target)))
    return np.asarray(sorted(pairs), dtype=np.int64)


def retention(tokens: np.ndarray, u: np.ndarray, v: np.ndarray, role: np.ndarray) -> dict:
    a1, a2, a3 = tokens[0], tokens[1], tokens[2]
    m1 = (a1[u] == a1[v]) & (a1[u] >= 0)
    m2 = m1 & (a2[u] == a2[v]) & (a2[u] >= 0)
    m3 = m2 & (a3[u] == a3[v]) & (a3[u] >= 0)
    out = {"edges": int(u.shape[0]),
           "prefix@1": float(m1.mean()), "prefix@2": float(m2.mean()), "prefix@3": float(m3.mean()),
           "per_role": {}}
    for r in sorted(ROLE_NAMES):
        mask = role == r
        count = int(mask.sum())
        out["per_role"][str(r)] = {
            "edges": count,
            "prefix@1": float(m1[mask].mean()) if count else None,
            "prefix@2": float(m2[mask].mean()) if count else None,
            "prefix@3": float(m3[mask].mean()) if count else None,
        }
    return out


def main() -> int:
    roles = load_role_table(ROLE_TABLE, ROLE_SHA)
    print(f"role table sha256 {sha256_of(ROLE_TABLE)[:12]} | roles {len(roles)}", flush=True)
    edges = train_edges()
    u, v = edges[:, 0], edges[:, 1]
    role = np.asarray([roles.get(int(item), -1) for item in u], dtype=np.int64)

    report = {"role_table_sha256": sha256_of(ROLE_TABLE), "edges": int(u.shape[0]), "arms": {}}
    for name, path in ARMS.items():
        if not Path(path).is_file():
            report["arms"][name] = {"status": "missing", "path": str(path)}
            print(f"{name}: missing ({path})", flush=True)
            continue
        payload = json.loads(Path(path).read_text())
        n_items = max(int(key) for key in payload) + 1
        tokens = np.full((3, n_items), -1, dtype=np.int64)
        for key, value in payload.items():
            index = int(key)
            for position in range(min(3, len(value))):
                tokens[position, index] = int(value[position])
        block = retention(tokens, u, v, role)
        block["status"] = "ok"
        block["path"] = str(path)
        block["n_items"] = len(payload)
        report["arms"][name] = block
        print(f"{name}: prefix@1 {block['prefix@1']:.4f} prefix@2 {block['prefix@2']:.4f} "
              f"prefix@3 {block['prefix@3']:.4f}", flush=True)

    OUT.write_text(json.dumps(report, indent=1, default=float))

    print("\nper-role prefix@1 (and the centred value against the arm's own overall):")
    header = " | ".join(f"{name:>14}" for name in ARMS)
    print(f"{'role':<22} | {header}")
    for r in sorted(ROLE_NAMES):
        cells = []
        for name in ARMS:
            block = report["arms"].get(name, {})
            if block.get("status") != "ok":
                cells.append(f"{'—':>14}")
                continue
            value = block["per_role"][str(r)]["prefix@1"]
            centred = value - block["prefix@1"]
            cells.append(f"{value:>7.4f}({centred:+.3f})")
        print(f"{r} {ROLE_NAMES[r]:<20} | " + " | ".join(cells))
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
