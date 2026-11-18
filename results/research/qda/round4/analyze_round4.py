"""Round 4: does the training-time assignment survive the export, and is the prefix
lift drop a granularity artefact?

Two diagnostics, both from observation snapshots that already exist:

A. **The export gap.** L3 is assigned by Sinkhorn during training and by argmin at
   export. For every held-out behaviour edge we ask whether the two rules keep the
   pair inside one code, and cross-tabulate kept / broken / created / neither,
   overall and per frozen edge stratum. A "broken" pair is a pair the training run
   treated as one cell and the exported SID splits -- that is the structural
   information the export throws away.

B. **Lift at matched capacity.** A prefix lift computed over 256 live codes is not
   comparable with one computed over 53: sharing is mechanically rarer. Each arm's
   codes are therefore coarsened, in usage rank order, to K buckets for a grid of K,
   and the lift is recomputed. If B3 matches or beats F0 at the same K, its low lift
   is granularity; if it stays below at every K, the relation itself was lost.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QDA = HERE.parent
ROUND2 = QDA / "round2"
SAMPLE = QDA / "sample" / "sample.npz"
STRATA = {
    "acyclic": lambda tri, inside: tri == 0,
    "triangle": lambda tri, inside: tri == 1,
    "dense": lambda tri, inside: tri >= 2,
    "intra_community": lambda tri, inside: inside == 1,
    "bridge": lambda tri, inside: inside == 0,
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load("qda_analyze", QDA / "analyze.py")


def snapshot(arm: str, epoch: int) -> Path:
    return ROUND2 / "runs" / arm / "observations" / f"epoch_{epoch:04d}.npz"


def _same(codes: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return codes[left] == codes[right]


def _train_assignment(record, level: int):
    """The assignment the training rule makes at `level`, or None if not recorded."""
    key = f"train_code_{level}"
    if key in record:
        return record[key].astype(np.int64)
    if level == 3 and "codes_sinkhorn" in record:
        # round-2/3 observers only stored the last level's balanced assignment
        return record["codes_sinkhorn"].astype(np.int64)
    return None


def assignment_gap(arm: str, epoch: int, sample, level: int = 3) -> dict:
    record = np.load(snapshot(arm, epoch))
    train = _train_assignment(record, level)
    if train is None:
        return None
    export = record["codes"][:, level - 1].astype(np.int64)
    split = sample["split"] == 1
    left = sample["edges"][:, 0].astype(np.int64)
    right = sample["edges"][:, 1].astype(np.int64)
    same_train = _same(train, left, right)
    same_export = _same(export, left, right)
    triangles, inside = sample["triangles"], sample["inside"]

    def cells(mask):
        t = same_train[mask]
        e = same_export[mask]
        return {
            "edges": int(mask.sum()),
            "kept": float((t & e).mean()),
            "broken": float((t & ~e).mean()),
            "created": float((~t & e).mean()),
            "neither": float((~t & ~e).mean()),
            "item_disagreement": float((train != export).mean()),
        }

    out = {"level": level, "overall": cells(split), "strata": {}}
    for name, rule in STRATA.items():
        mask = split & rule(triangles, inside)
        out["strata"][name] = cells(mask)
    return out


def coarsen(codes: np.ndarray, buckets: int) -> np.ndarray:
    """Map codes to `buckets` groups in usage-rank order (frequency-preserving)."""
    present, counts = np.unique(codes, return_counts=True)
    order = present[np.argsort(-counts, kind="stable")]
    rank = np.empty(codes.max() + 1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    scale = np.arange(len(order)) * buckets // max(len(order), 1)
    table = np.full(codes.max() + 1, -1, dtype=np.int64)
    table[order] = scale
    return table[codes]


def coarsen_centroid(readout: np.ndarray, codes: np.ndarray, buckets: int) -> np.ndarray:
    """Merge codes by where their items sit, not by how many they hold.

    Usage-rank coarsening is arbitrary; this variant places each code at the centroid
    of the readouts that selected it and runs KMeans over those centroids, which is
    what a smaller codebook would plausibly look like.
    """
    from sklearn.cluster import KMeans

    present = np.unique(codes)
    centroids = np.stack([readout[codes == code].mean(0) for code in present])
    labels = KMeans(n_clusters=min(buckets, len(present)), n_init=10,
                    random_state=0).fit_predict(centroids)
    table = np.empty(codes.max() + 1, dtype=np.int64)
    table[present] = labels
    return table[codes]


def lift_at(codes: np.ndarray, sample, split, buckets: int) -> float:
    """Prefix lift on one token, optionally coarsened to `buckets` groups."""
    grouped = coarsen(codes, buckets) if buckets else codes
    return float(analyze.prefix_lift(grouped[:, None], sample, split, 1)["lift"])


def capacity_curve(arm: str, epoch: int, sample, grid) -> dict:
    record = np.load(snapshot(arm, epoch))
    split = sample["split"] == 1
    out = {}
    for level in (1, 2, 3):
        codes = record["codes"][:, level - 1].astype(np.int64)
        row = {"live": int(len(np.unique(codes)))}
        for buckets in grid:
            if buckets >= row["live"]:
                continue
            row[f"lift@{buckets}"] = lift_at(codes, sample, split, buckets)
        row["lift@all"] = lift_at(codes, sample, split, 0)
        # robustness: the same capacity reached by merging code centroids
        readout = record[f"readout_{level}"].astype(np.float32)
        for buckets in grid:
            if buckets >= row["live"]:
                continue
            grouped = coarsen_centroid(readout, codes, buckets)
            row[f"cmap@{buckets}"] = float(
                analyze.prefix_lift(grouped[:, None], sample, split, 1)["lift"])
        out[f"l{level}"] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=[
        "A_fine:3000", "B3_long:3000", "B0_replay:300", "B3_l1_balanced:300"])
    parser.add_argument("--grid", nargs="*", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--out", default="report_round4.json")
    args = parser.parse_args()

    sample = np.load(SAMPLE)
    report = {"assignment": {}, "capacity": {}}
    for spec in args.pairs:
        arm, epoch = spec.split(":")
        if not snapshot(arm, int(epoch)).exists():
            print(f"{spec}: no snapshot")
            continue
        per_level = {}
        for level in (1, 2, 3):
            gap = assignment_gap(arm, int(epoch), sample, level)
            if gap is not None:
                per_level[f"l{level}"] = gap
        report["assignment"][spec] = per_level
        report["capacity"][spec] = capacity_curve(arm, int(epoch), sample, args.grid)

        for level, gap in report["assignment"][spec].items():
            overall = gap["overall"]
            print(f"{spec:22s} {level} item disagreement "
                  f"{overall['item_disagreement']:.3f} | kept {overall['kept']:.4f} "
                  f"broken {overall['broken']:.4f} created {overall['created']:.4f}")
            for name in ("bridge", "intra_community"):
                row = gap["strata"][name]
                share = row["broken"] / max(row["kept"] + row["broken"], 1e-12)
                print(f"    {name:16s} broken_share {share:.3f}")
        row = report["capacity"][spec]["l1"]
        print("    L1 lift: " + "  ".join(
            f"{key}={value:.4f}" for key, value in row.items() if key != "live"))
    (HERE / args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
