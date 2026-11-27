"""Issue #270: full-triplet A.1 geometry of one archived arm on the common ruler.

q is reconstructed exactly from the checkpoint codebooks and the exported SID; no
encoder forward is needed. The equal-weight three-expert metric and the A.1 moment
identity are the frozen evaluation (`tools/issue251.full_evaluation._moments`,
`tools/issue253.evaluation_reference`). Strata are computed from the directed
training graph and the undirected visible graph and aggregated per row/centre.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.issue251.full_evaluation import _moments
from tools.issue253.evaluation_reference import (fixed_three_paired_numpy,
                                                fixed_three_squared_numpy)

OUT = REPO / "results/research/issue271"
ASSETS = OUT / "assets/geometry"
SCALE = 0.08433422446250916  # #265 common tangent; see CONTRACT.md
N_ITEMS = 24587
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
SHARDS = REPO / "assets/shards"
NODE_MOMENTS = REPO / "assets/node_moments.npz"
STRATA = REPO / "results/research/issue270/assets/strata.npz"


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def code(q: np.ndarray, keys: np.ndarray, codes: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(keys, codes)
    pos = np.clip(pos, 0, len(keys) - 1)
    return np.where(keys[pos] == codes, q[pos], 0.0)


def main() -> None:
    arm = sys.argv[1]
    cohort = json.loads((OUT / "geometry_cohort.json").read_text())
    row = next(r for r in cohort["rows"] if r["arm"] == arm and r["usable_geometry"])
    started = time.time()
    sid = json.loads(Path(row["sid"]).read_text())
    codes = np.asarray([sid[str(i)] for i in range(N_ITEMS)], dtype=np.int64)
    if codes.shape != (N_ITEMS, 4):
        raise ValueError("SID shape mismatch")
    payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    def book_key(level):
        for candidate in (f"rq.vq_layers.{level}.embed.weight", f"rq.levels.{level}.embed.weight"):
            if candidate in state:
                return candidate
        raise KeyError("no codebook key for level %d" % level)
    books = [state[book_key(level)].detach().cpu().numpy().astype(np.float64)
             for level in range(3)]
    if any(b.shape != (256, 32) for b in books):
        raise ValueError("codebook shape mismatch")
    q = np.zeros((N_ITEMS, 32), dtype=np.float64)
    for level, book in enumerate(books):
        q += book[codes[:, level]]
    if not np.isfinite(q).all():
        raise ValueError("nonfinite q")
    labels = np.load(NODE_MOMENTS)["component"]
    if labels.shape != (N_ITEMS,):
        raise ValueError("component shape mismatch")
    moments = _moments(q, labels, SCALE, fixed_three_squared_numpy, 512)
    if not np.isfinite(moments).all():
        raise ValueError("nonfinite moments")

    # strata inputs
    t = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    total = int(offsets[-1])
    cache = np.load(STRATA)
    dkeys, ukeys, deg_decile = cache["dkeys"], cache["ukeys"], cache["deg_decile"]

    err = np.empty(total, dtype=np.float16)
    cls = np.empty(total, dtype=np.uint8)
    per_centre = np.zeros((N_ITEMS, 5), dtype=np.float64)  # sum, abssum, count, signok, absgt
    per_class = {}
    per_closure = np.zeros((2, 5), dtype=np.float64)
    per_deg = np.zeros((10, 5), dtype=np.float64)
    per_class_arr = np.zeros((128, 5), dtype=np.float64)
    per_closure_arr = np.zeros((2, 5), dtype=np.float64)
    per_deg_arr = np.zeros((10, 5), dtype=np.float64)
    for shard in sorted(SHARDS.glob("shard_*.npz")):
        with np.load(shard) as blob:
            valid = blob["valid"]
            centre = blob["centre"][valid].astype(np.int64)
            b = blob["b"][valid].astype(np.int64)
            c = blob["c"][valid].astype(np.int64)
            target = blob["curvature"][valid].astype(np.float64)
        pair = fixed_three_paired_numpy(q[b], q[c], scale=SCALE)
        predicted = moments[centre] - 0.5 * (moments[b] + moments[c]) + 0.25 * pair
        if not hasattr(main, "_sample"):
            main._sample = []
        if len(main._sample) < 13:
            stride = max(1, len(predicted) // 2000)
            main._sample.append(np.stack((predicted[::stride], target[::stride]), axis=1))
        e = predicted - target
        abs_e = np.abs(e)
        sign_ok = (np.sign(predicted) == np.sign(target)).astype(np.float64)
        abs_gt = (abs_e < np.abs(target)).astype(np.float64)
        idx = centre
        per_centre[:, 0] += np.bincount(idx, weights=e, minlength=N_ITEMS)
        per_centre[:, 1] += np.bincount(idx, weights=abs_e, minlength=N_ITEMS)
        per_centre[:, 2] += np.bincount(idx, minlength=N_ITEMS)
        per_centre[:, 3] += np.bincount(idx, weights=sign_ok, minlength=N_ITEMS)
        per_centre[:, 4] += np.bincount(idx, weights=abs_gt, minlength=N_ITEMS)
        bits = np.zeros(len(idx), dtype=np.uint8)
        for u, v, bit in ((centre, b, 0), (b, centre, 1), (centre, c, 2), (c, centre, 3),
                          (b, c, 4), (c, b, 5)):
            key = u * N_ITEMS + v
            pos = np.searchsorted(dkeys, key)
            pos = np.clip(pos, 0, len(dkeys) - 1)
            bits |= (dkeys[pos] == key).astype(np.uint8) << bit
        lo = np.minimum(b, c) * N_ITEMS + np.maximum(b, c)
        pos = np.searchsorted(ukeys, lo)
        pos = np.clip(pos, 0, len(ukeys) - 1)
        closure = (ukeys[pos] == lo).astype(np.uint8)
        cls_chunk = bits | (closure << 6)
        weights = {"e": e, "a": abs_e, "s": sign_ok, "g": abs_gt}
        for column, key in enumerate(("e", "a", "s", "g")):
            per_class_arr[:, column] += np.bincount(cls_chunk, weights=weights[key], minlength=128)
        per_class_arr[:, 2] += np.bincount(cls_chunk, minlength=128)
        per_closure_arr[:, 0] += np.bincount(closure, weights=e, minlength=2)
        per_closure_arr[:, 1] += np.bincount(closure, weights=abs_e, minlength=2)
        per_closure_arr[:, 2] += np.bincount(closure, minlength=2)
        per_closure_arr[:, 3] += np.bincount(closure, weights=sign_ok, minlength=2)
        dec = deg_decile[centre]
        per_deg_arr[:, 0] += np.bincount(dec, weights=e, minlength=10)
        per_deg_arr[:, 1] += np.bincount(dec, weights=abs_e, minlength=10)
        per_deg_arr[:, 2] += np.bincount(dec, minlength=10)
        per_deg_arr[:, 3] += np.bincount(dec, weights=sign_ok, minlength=10)
        if not hasattr(main, "_cursor"):
            main._cursor = 0
        cursor = main._cursor
        n = len(idx)
        err[cursor:cursor + n] = e.astype(np.float16)
        cls[cursor:cursor + n] = cls_chunk
        main._cursor = cursor + n
    if main._cursor != total:
        raise ValueError("row coverage mismatch %d/%d" % (main._cursor, total))
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / (arm + ".npz")
    np.savez(out, q=q.astype(np.float32), error=err, cls=cls,
                        centre_sum=per_centre[:, 0], centre_abs=per_centre[:, 1],
                        centre_count=per_centre[:, 2], centre_signok=per_centre[:, 3],
                        centre_absgt=per_centre[:, 4], scale=np.array(SCALE))
    summary = {
        "schema": "genrec.issue271.local_geometry/1", "arm": arm,
        "sid": row["sid"], "sid_sha256": row["sid_sha256"],
        "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
        "binding": row["binding"], "record_verified": row["record_verified"],
        "scale": SCALE, "rows": total, "runtime_s": time.time() - started,
        "mean_error": float(np.mean(err.astype(np.float64))),
        "mean_abs_error": float(np.mean(np.abs(err.astype(np.float64)))),
        "spearman": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
            np.concatenate(main._sample), axis=0).statistic),
        "sign_accuracy": float(per_centre[:, 3].sum() / total),
        "abs_better_than_zero": float(per_centre[:, 4].sum() / total),
        "per_class": {str(k): {"sum": float(v[0]), "abs": float(v[1]), "n": int(v[2]),
                               "sign": float(v[3]), "absgt": float(v[4])}
                      for k, v in enumerate(per_class_arr)},
        "closure": {str(k): {"sum": float(v[0]), "abs": float(v[1]), "n": int(v[2]),
                             "sign": float(v[3])} for k, v in enumerate(per_closure_arr)},
        "degree_decile": {str(k): {"sum": float(v[0]), "abs": float(v[1]), "n": int(v[2]),
                                   "sign": float(v[3])} for k, v in enumerate(per_deg_arr)},
        "output": str(out), "output_sha256": sha(out),
    }
    (ASSETS / (arm + ".json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("arm", "mean_abs_error", "sign_accuracy", "runtime_s")}))


if __name__ == "__main__":
    main()
