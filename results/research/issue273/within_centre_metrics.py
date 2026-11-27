"""Issue #273: within-centre difference fidelity on the frozen #270/#271 ruler.

The frozen evaluator (``results/research/issue271/local_geometry.py``) is reused as-is for the
per-row geometry: ``q`` is reconstructed from the checkpoint codebooks and the exported SID, the
A.1 moments come from ``tools.issue251.full_evaluation._moments`` and the per-row prediction is
the same section formula with the frozen scale ``0.08433422446250916``.

This script adds the issue-273 statistics (all computed from the same per-row quantities, no new
assets and no new ruler):

* ``variance_ratio`` = var(predicted) / var(target)                     (C4)
* ``slope``         = cov(predicted, target) / var(target)              (C5)
* ``difference_rmse`` = sqrt( mean over centres of 2 * var_c(error) )   (C6)
  With ``e = p - y`` the mean squared same-centre difference is exactly ``2 Var_c(e)``
  (see CONTRACT.md section 3), so the centre-balanced difference RMSE is exact and needs no
  pairwise enumeration.  The same value is reported per target-sign stratum.
* ``ordering_agreement`` = centre-balanced fraction of same-centre pairs whose *order* is
  preserved; estimated from a bounded, seeded pair sample per centre (the estimator is identical
  for every arm, so the comparisons are consistent).

The script also recomputes the frozen MAE/Spearman/sign and compares them with the frozen
evaluator's stored summary (self-check) so the new statistics can only be trusted when the old
ones reproduce.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.issue251.full_evaluation import _moments
from tools.issue253.evaluation_reference import fixed_three_paired_numpy, fixed_three_squared_numpy

SCALE = 0.08433422446250916
N_ITEMS = 24587
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
SHARDS = REPO / "assets/shards"
NODE_MOMENTS = REPO / "assets/node_moments.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_q(sid_path: Path, checkpoint_path: Path) -> np.ndarray:
    sid = json.loads(sid_path.read_text())
    codes = np.asarray([sid[str(i)] for i in range(N_ITEMS)], dtype=np.int64)
    if codes.shape != (N_ITEMS, 4):
        raise ValueError("SID shape mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]

    def book_key(level: int) -> str:
        for candidate in (f"rq.vq_layers.{level}.embed.weight", f"rq.levels.{level}.embed.weight"):
            if candidate in state:
                return candidate
        raise KeyError(f"no codebook key for level {level}")

    books = [state[book_key(level)].detach().cpu().numpy().astype(np.float64) for level in range(3)]
    if any(book.shape != (256, 32) for book in books):
        raise ValueError("codebook shape mismatch")
    q = np.zeros((N_ITEMS, 32), dtype=np.float64)
    for level, book in enumerate(books):
        q += book[codes[:, level]]
    if not np.isfinite(q).all():
        raise ValueError("nonfinite q")
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--pairs-per-centre", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args()

    started = time.time()
    q = reconstruct_q(Path(args.sid), Path(args.checkpoint))
    labels = np.load(NODE_MOMENTS)["component"]
    if labels.shape != (N_ITEMS,):
        raise ValueError("component shape mismatch")
    moments = _moments(q, labels, SCALE, fixed_three_squared_numpy, 512)
    if not np.isfinite(moments).all():
        raise ValueError("nonfinite moments")

    total = int(np.load(OFFSETS, mmap_mode="r")[-1])
    count = np.zeros(N_ITEMS, dtype=np.int64)
    e_sum = np.zeros(N_ITEMS)
    e_sq = np.zeros(N_ITEMS)
    y_sum = np.zeros(N_ITEMS)
    y_sq = np.zeros(N_ITEMS)
    ye_sum = np.zeros(N_ITEMS)
    sign_ok = np.zeros(N_ITEMS)
    stratum_count = [np.zeros(N_ITEMS, dtype=np.int64) for _ in range(2)]
    stratum_e = [np.zeros(N_ITEMS) for _ in range(2)]
    stratum_e2 = [np.zeros(N_ITEMS) for _ in range(2)]
    row_centre = np.empty(total, dtype=np.int32)
    row_pred = np.empty(total, dtype=np.float32)
    row_target = np.empty(total, dtype=np.float32)
    spearman_sample = []
    cursor = 0
    for shard in sorted(SHARDS.glob("shard_*.npz")):
        with np.load(shard) as blob:
            valid = blob["valid"]
            centre = blob["centre"][valid].astype(np.int64)
            b = blob["b"][valid].astype(np.int64)
            c = blob["c"][valid].astype(np.int64)
            target = blob["curvature"][valid].astype(np.float64)
        pair = fixed_three_paired_numpy(q[b], q[c], scale=SCALE)
        predicted = moments[centre] - 0.5 * (moments[b] + moments[c]) + 0.25 * pair
        error = predicted - target
        count += np.bincount(centre, minlength=N_ITEMS)
        e_sum += np.bincount(centre, weights=error, minlength=N_ITEMS)
        e_sq += np.bincount(centre, weights=error * error, minlength=N_ITEMS)
        y_sum += np.bincount(centre, weights=target, minlength=N_ITEMS)
        y_sq += np.bincount(centre, weights=target * target, minlength=N_ITEMS)
        ye_sum += np.bincount(centre, weights=target * error, minlength=N_ITEMS)
        sign_ok += np.bincount(centre, weights=(np.sign(predicted) == np.sign(target)).astype(np.float64),
                               minlength=N_ITEMS)
        positive = target > 0
        for index, mask in enumerate((positive, ~positive)):
            idx = centre[mask]
            stratum_count[index] += np.bincount(idx, minlength=N_ITEMS)
            stratum_e[index] += np.bincount(idx, weights=error[mask], minlength=N_ITEMS)
            stratum_e2[index] += np.bincount(idx, weights=error[mask] ** 2, minlength=N_ITEMS)
        if len(spearman_sample) < 13:
            stride = max(1, len(predicted) // 2000)
            spearman_sample.append(np.stack((predicted[::stride], target[::stride]), axis=1))
        size = len(centre)
        row_centre[cursor:cursor + size] = centre
        row_pred[cursor:cursor + size] = predicted
        row_target[cursor:cursor + size] = target
        cursor += size
    if cursor != total:
        raise ValueError(f"row coverage mismatch {cursor}/{total}")

    def within_centre_variance(n, s1, s2):
        usable = n >= 2
        if not np.any(usable):
            return float("nan"), 0
        variance = (s2[usable] - s1[usable] ** 2 / n[usable]) / (n[usable] - 1)
        return float(np.mean(variance)), int(usable.sum())

    overall_variance, centres_used = within_centre_variance(count, e_sum, e_sq)
    per_stratum = {}
    for index, name in enumerate(("positive", "negative")):
        variance, used = within_centre_variance(stratum_count[index], stratum_e[index], stratum_e2[index])
        per_stratum[name] = {"difference_rmse": float(np.sqrt(2.0 * variance)) if used else None,
                             "centres": used}
    difference_rmse = float(np.sqrt(2.0 * overall_variance))

    pred = row_pred.astype(np.float64)
    target = row_target.astype(np.float64)
    var_pred = float(np.var(pred))
    var_target = float(np.var(target))
    cov = float(np.mean((pred - pred.mean()) * (target - target.mean())))
    variance_ratio = float(var_pred / var_target)
    slope = float(cov / var_target)

    order = np.argsort(row_centre, kind="stable")
    sorted_centre = row_centre[order]
    sorted_pred = pred[order]
    sorted_target = target[order]
    bounds = np.searchsorted(sorted_centre, np.arange(N_ITEMS + 1))
    agreements = []
    for centre in range(N_ITEMS):
        start, stop = int(bounds[centre]), int(bounds[centre + 1])
        size = stop - start
        if size < 2:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, centre]))
        left = rng.integers(0, size, size=args.pairs_per_centre)
        right = rng.integers(0, size, size=args.pairs_per_centre)
        same = (sorted_pred[start + left] - sorted_pred[start + right]) * (
            sorted_target[start + left] - sorted_target[start + right]) >= 0
        agreements.append(float(same.mean()))
    ordering_agreement = float(np.mean(agreements)) if agreements else float("nan")

    # The issue-272 statistic (results/research/issue272/local_diagnostics.py): the within-centre
    # Spearman over a seeded row sample and a seeded centre sample.  Reproduced here so the
    # candidate can be compared with the frozen issue-272 numbers for the same arms.
    rng = np.random.default_rng(272)
    sample = np.sort(rng.choice(len(pred), 400_000, replace=False))
    sampled_centre = row_centre[sample]
    sampled_pred = pred[sample]
    sampled_target = target[sample]
    unique_centres = np.unique(row_centre)
    rng2 = np.random.default_rng(273)
    within_rho = []
    for centre in rng2.choice(unique_centres, min(400, len(unique_centres)), replace=False):
        mask = sampled_centre == centre
        if int(mask.sum()) >= 8:
            rho = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
                sampled_pred[mask], sampled_target[mask]).statistic
            if np.isfinite(rho):
                within_rho.append(float(rho))
    within_centre_spearman = float(np.mean(within_rho)) if within_rho else None
    within_centre_spearman_n = len(within_rho)

    mean_abs_error = float(np.mean(np.abs(pred - target)))
    sign_accuracy = float(sign_ok.sum() / total)
    spearman = float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
        np.concatenate(spearman_sample), axis=0).statistic)
    summary = {
        "schema": "genrec.issue273.within_centre/1",
        "arm": args.arm,
        "sid": str(args.sid), "sid_sha256": sha256(Path(args.sid)),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "scale": SCALE, "rows": total, "pairs_per_centre": args.pairs_per_centre,
        "seed": args.seed, "runtime_s": time.time() - started,
        "mean_abs_error": mean_abs_error, "spearman": spearman, "sign_accuracy": sign_accuracy,
        "variance_ratio": variance_ratio, "slope": slope,
        "difference_rmse": difference_rmse,
        "difference_rmse_by_sign": per_stratum,
        "ordering_agreement": ordering_agreement,
        "within_centre_spearman": within_centre_spearman,
        "within_centre_spearman_n": within_centre_spearman_n,
        "centres_with_pairs": centres_used,
    }
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        summary["self_check"] = {
            "reference": str(args.reference),
            "mean_abs_error_delta": mean_abs_error - float(reference["mean_abs_error"]),
            "spearman_delta": spearman - float(reference["spearman"]),
            "sign_accuracy_delta": sign_accuracy - float(reference["sign_accuracy"]),
        }
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("arm", "mean_abs_error", "spearman", "sign_accuracy", "variance_ratio",
                       "slope", "difference_rmse", "ordering_agreement", "within_centre_spearman",
                       "self_check")}, indent=2))


if __name__ == "__main__":
    main()
