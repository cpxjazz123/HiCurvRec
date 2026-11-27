"""Issue #270: graph-distance distortion J on the frozen 20k/256 sample (common ruler)."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.issue253.evaluation_reference import (fixed_three_paired_numpy,
                                                fixed_three_squared_numpy)
OUT = REPO / "results/research/issue270"
GEOM = OUT / "assets/geometry"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
SCALE = 0.08433422446250916
SEED = 269
SAMPLE = 20000
REFS = 256
N = 24587


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> None:
    arm = sys.argv[1]
    cohort = json.loads((OUT / "COHORT.json").read_text())
    row = next(r for r in cohort["rows"] if r["arm"] == arm and r["usable_geometry"])
    sid = json.loads(Path(row["sid"]).read_text())
    codes = np.asarray([sid[str(i)] for i in range(N)], dtype=np.int64)[:, :3]
    payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    q = np.zeros((N, 32), dtype=np.float64)
    for level in range(3):
        key = (f"rq.vq_layers.{level}.embed.weight" if f"rq.vq_layers.{level}.embed.weight" in state
               else f"rq.levels.{level}.embed.weight")
        book = state[key].detach().cpu().numpy().astype(np.float64)
        q += book[codes[:, level]]
    from tools.graph_infra.assets import load_sparse
    g = load_sparse(GRAPH).tocsr().astype(np.int8)
    g = g.maximum(g.T).tocsr(); g.setdiag(0); g.eliminate_zeros()
    _, labels = connected_components(g, directed=False)
    rng = np.random.default_rng(SEED)
    blocks = []
    for c in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == c)
        if len(members) >= 2:
            blocks.append(np.sort(rng.choice(members, min(REFS, len(members)), replace=False)))
    refs = np.concatenate(blocks)
    gd = shortest_path(g, unweighted=True, directed=False, indices=refs).T
    use = np.isfinite(gd) & (gd > 0)
    gv = np.where(use, gd, 1.0).astype(np.float32)
    t = np.load(TARGETS, mmap_mode="r"); o = np.load(OFFSETS, mmap_mode="r")
    rows = np.sort(np.random.default_rng(SEED + 1).choice(int(o[-1]), SAMPLE, replace=False))
    rows = rows[np.isfinite(np.asarray(t["target"][rows]))]
    centre = np.searchsorted(o, rows, side="right") - 1
    b = np.asarray(t["b"][rows], dtype=np.int64); c = np.asarray(t["c"][rows], dtype=np.int64)
    total = np.zeros(len(rows)); denom = np.zeros(len(rows))
    for node_ids, sign in ((centre, 1.0), (b, 1.0), (c, 1.0)):
        d2 = fixed_three_squared_numpy(q[node_ids], q[refs], scale=SCALE)  # [rows, refs]
        ratio = np.sqrt(np.maximum(d2, 0.0)) / gv[node_ids]
        masked = np.where(use[node_ids], (ratio - 1.0) ** 2, 0.0)
        total += masked.sum(1)
        denom += use[node_ids].sum(1)
    lo = np.minimum(b, c) * N + np.maximum(b, c)
    coo = g.tocoo(); up = coo.row < coo.col
    keys = set((coo.row[up].astype(np.int64) * N + coo.col[up].astype(np.int64)).tolist())
    pair_g = np.fromiter((1.0 if int(k) in keys else 2.0 for k in lo), dtype=np.float32, count=len(lo))
    d2 = fixed_three_paired_numpy(q[b], q[c], scale=SCALE)
    total += ((np.sqrt(np.maximum(d2, 0.0)) / pair_g - 1.0) ** 2)
    denom += 1.0
    j = total / denom
    out = GEOM / (arm + ".distortion.json")
    out.write_text(json.dumps({
        "schema": "genrec.issue270.distortion/1", "arm": arm, "rows": int(len(rows)),
        "references": int(len(refs)), "scale": SCALE,
        "mean_j": float(j.mean()), "median_j": float(np.median(j)),
        "p90_j": float(np.quantile(j, 0.9)),
        "sid_sha256": row["sid_sha256"], "checkpoint_sha256": row["checkpoint_sha256"],
        "output": str(out)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm": arm, "mean_j": float(j.mean())}))


if __name__ == "__main__":
    main()
