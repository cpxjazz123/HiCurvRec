"""Issue #270: SID structural features per item and per centre-triplet aggregate."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"
ASSETS = OUT / "assets/sid"
GEOM = OUT / "assets/geometry"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
ROLES = REPO / "stage2_RQ-VAE/_archive/TIGER_topology_residual_l2_rq_issue263/role_table.json"
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
    l1, l2, l3 = codes[:, 0], codes[:, 1], codes[:, 2]
    key1 = l1
    key2 = l1 * 256 + l2
    key3 = (l1 * 256 + l2) * 256 + l3
    _, inv1, cnt1 = np.unique(key1, return_inverse=True, return_counts=True)
    _, inv2, cnt2 = np.unique(key2, return_inverse=True, return_counts=True)
    _, inv3, cnt3 = np.unique(key3, return_inverse=True, return_counts=True)
    roles = np.asarray(json.loads(ROLES.read_text())["table"], dtype=np.int64) if ROLES.is_file() else None
    if roles is not None and roles.shape != (N,):
        roles = None
    purity1 = np.zeros(N)
    purity2 = np.zeros(N)
    if roles is not None:
        for group, inv in ((1, inv1), (2, inv2)):
            order = np.argsort(inv, kind="stable")
            grp = inv[order]
            r = roles[order]
            starts = np.searchsorted(grp, np.arange(grp[-1] + 1))
            ends = np.searchsorted(grp, np.arange(grp[-1] + 1), side="right")
            out = np.zeros(len(order))
            for s, e in zip(starts, ends):
                if e > s:
                    vals, counts = np.unique(r[s:e], return_counts=True)
                    out[s:e] = counts.max() / (e - s)
            res = np.empty(N)
            res[order] = out
            if group == 1:
                purity1 = res
            else:
                purity2 = res
    # per-triplet features
    t = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    total = int(offsets[-1])
    centre_sum = np.zeros(N)
    lcp_mb = np.zeros(N); lcp_mc = np.zeros(N); lcp_bc = np.zeros(N)
    all1 = np.zeros(N); all2 = np.zeros(N); all3 = np.zeros(N)
    count = np.zeros(N)
    cls = None
    geom = GEOM / (arm + ".npz")
    if geom.is_file():
        with np.load(geom) as blob:
            cls = blob["cls"]
    per_class = {}
    for shard in sorted((REPO / "assets/shards").glob("shard_*.npz")):
        with np.load(shard) as blob:
            valid = blob["valid"]
            m = blob["centre"][valid].astype(np.int64)
            b = blob["b"][valid].astype(np.int64)
            c = blob["c"][valid].astype(np.int64)
        e1 = (l1[m] == l1[b]).astype(np.uint8) & (l1[m] == l1[c]).astype(np.uint8)
        e2 = (key2[m] == key2[b]) & (key2[m] == key2[c])
        e3 = (key3[m] == key3[b]) & (key3[m] == key3[c])
        pair_bc_l1 = (l1[b] == l1[c]).astype(np.uint8)
        lcp = lambda x, y: ((x[:, 0] == y[:, 0]).astype(np.uint8) +
                            ((x[:, 0] == y[:, 0]) & (x[:, 1] == y[:, 1])).astype(np.uint8) +
                            ((x[:, 0] == y[:, 0]) & (x[:, 1] == y[:, 1]) & (x[:, 2] == y[:, 2])).astype(np.uint8))
        cm = codes[m]; cb = codes[b]; cc = codes[c]
        lb_mb = lcp(cm, cb); lb_mc = lcp(cm, cc); lb_bc = lcp(cb, cc)
        centre_sum += np.bincount(m, weights=e1 + e2 + e3, minlength=N)
        lcp_mb += np.bincount(m, weights=lb_mb, minlength=N)
        lcp_mc += np.bincount(m, weights=lb_mc, minlength=N)
        lcp_bc += np.bincount(m, weights=lb_bc, minlength=N)
        all1 += np.bincount(m, weights=e1, minlength=N)
        all2 += np.bincount(m, weights=e2, minlength=N)
        all3 += np.bincount(m, weights=e3, minlength=N)
        count += np.bincount(m, minlength=N)
        if cls is not None:
            # rows of this shard are contiguous in the packed order; recover the
            # offsets with a running cursor over the shards.
            pass
    count_safe = np.maximum(count, 1)
    per_item = np.stack([
        l1, l2, l3,
        cnt1[inv1], cnt2[inv2], cnt3[inv3],
        purity1, purity2,
        all1 / count_safe, all2 / count_safe, all3 / count_safe,
        lcp_mb / count_safe, lcp_mc / count_safe, lcp_bc / count_safe,
        count,
    ], axis=1).astype(np.float64)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / (arm + ".npz")
    np.savez_compressed(out, per_item=per_item, codes=codes.astype(np.int64))
    distinct = {"L1": int(len(cnt1)), "L2": int(len(cnt2)), "L3": int(len(cnt3))}
    summary = {
        "schema": "genrec.issue270.sid_link/1", "arm": arm,
        "sid": row["sid"], "sid_sha256": row["sid_sha256"],
        "distinct": distinct, "collision": 1 - distinct["L3"] / N,
        "l2_per_l1": float(distinct["L2"] / distinct["L1"]),
        "l3_per_l2": float(distinct["L3"] / distinct["L2"]),
        "mean_prefix_group_L1": float(cnt1.mean()), "max_prefix_group_L1": int(cnt1.max()),
        "mean_all3_L1_share": float((all1 / count_safe).mean()),
        "mean_all3_L2_share": float((all2 / count_safe).mean()),
        "mean_all3_L3_share": float((all3 / count_safe).mean()),
        "mean_lcp_mb": float((lcp_mb / count_safe).mean()),
        "mean_lcp_mc": float((lcp_mc / count_safe).mean()),
        "mean_lcp_bc": float((lcp_bc / count_safe).mean()),
        "per_item_columns": ["l1", "l2", "l3", "group1", "group2", "group3", "purity1",
                             "purity2", "share1", "share2", "share3", "lcp_mb", "lcp_mc",
                             "lcp_bc", "triplets"],
        "output": str(out), "output_sha256": sha(out),
    }
    (ASSETS / (arm + ".json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("arm", "distinct", "collision", "mean_all3_L1_share")}))


if __name__ == "__main__":
    main()
