"""Hard discrete SID L1 ER gate for the R6 arms (issues #74-#76).

One metric, one threshold, fail closed:

    pos_share = P[sid_L1(anchor) == sid_L1(observed_target)]
    neg_share = P[sid_L1(anchor) == sid_L1(matched_frozen_negative)]
    Hard_L1_ER = pos_share / neg_share
    stage3_allowed = ( ... and finite(ER) and ER > 16.38953942232631 )

Everything that is not the metric is a precondition, and every precondition must be verified
against the bytes on disk: the frozen heldout artifact by hash, the run by an explicit
train_end(epoch=3000), the checkpoint link by its metadata, and the exported SIDs by their hash.
A missing, unreadable or defaulted value is a FAIL, never a pass -- in particular a `train_end`
row without an explicit epoch is refused rather than assumed to be 3000.

The gate writes stage2_metrics.json (the metric) and gate.json (the whole verdict).

Usage:  <project python> tools/stage2_hard_l1_er_gate.py <arm> [--tree PATH] [--spec-sha SHA]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PAIRS = REPO / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
PAIRS_SHA256 = "04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0"
BASELINE_ER = 16.38953942232631
BASELINE_SID_SHA256 = "b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10"
EXPECTED_POSITIVES = 52855
EXPECTED_NEGATIVES = 10
EPOCHS = 3000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(tree: Path, arm: str, spec_sha: str | None) -> dict:
    result: dict = {"arm": arm, "tree": str(tree), "spec_sha256": spec_sha,
                    "preconditions": {}, "metric": {}, "stage3_allowed": False}
    failures: list[str] = []

    def require(name: str, ok: bool, detail: str) -> bool:
        result["preconditions"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(f"{name}: {detail}")
        return ok

    result_dir = REPO / "results/stage2_RQ-VAE" / arm
    metrics_path = tree / "logs/resample/training_metrics.jsonl"
    sids_path = result_dir / "sids_for_hgrec.npy"
    items_path = result_dir / "item_sids.json"
    checkpoint_path = result_dir / "rqvae_best.pth"

    require("frozen_pairs_present", PAIRS.is_file(), str(PAIRS))
    if PAIRS.is_file():
        require("frozen_pairs_sha256", sha256(PAIRS) == PAIRS_SHA256,
                f"{sha256(PAIRS)} != {PAIRS_SHA256}")

    rows: list[dict] = []
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    require("metrics_present", bool(rows), f"{len(rows)} rows in {metrics_path}")

    train_ends = [row for row in rows if row.get("event") == "train_end"]
    checkpoints = [row for row in rows if row.get("event") == "ckpt_saved"]
    # An explicit epoch is required. `row.get("epoch", EPOCHS)` would silently accept a run that
    # stopped early, which is exactly the audit hole this check exists to close.
    require("train_end_present", bool(train_ends), f"{len(train_ends)} train_end rows")
    if train_ends:
        epochs_seen = [row.get("epoch") for row in train_ends]
        require("train_end_epoch_explicit", all(value is not None for value in epochs_seen),
                f"epochs seen: {epochs_seen}")
        require("train_end_epoch_is_3000", all(value == EPOCHS for value in epochs_seen),
                f"declared {epochs_seen}, required {EPOCHS}")
    require("ckpt_saved_present", bool(checkpoints), f"{len(checkpoints)} ckpt_saved rows")
    if checkpoints:
        require("ckpt_saved_epoch_3000",
                all(row.get("epoch") == EPOCHS for row in checkpoints),
                f"epochs seen: {[row.get('epoch') for row in checkpoints]}")

    checkpoint_epoch = None
    if checkpoint_path.is_file():
        try:
            import torch

            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_epoch = payload.get("epoch") if isinstance(payload, dict) else None
        except Exception as exc:  # noqa: BLE001
            failures.append(f"checkpoint_unreadable: {exc}")
        require("checkpoint_readable", checkpoint_epoch is not None,
                f"epoch metadata = {checkpoint_epoch}")
        require("checkpoint_epoch_is_3000", checkpoint_epoch == EPOCHS,
                f"{checkpoint_epoch} != {EPOCHS}")
    else:
        require("checkpoint_present", False, f"missing {checkpoint_path}")

    require("sids_present", sids_path.is_file(), str(sids_path))
    require("item_map_present", items_path.is_file(), str(items_path))
    if not sids_path.is_file():
        result["failures"] = failures
        return result

    sids = np.load(sids_path)
    require("sids_shape", sids.ndim == 2 and sids.shape[1] >= 3,
            f"shape {sids.shape}; column 0 is the native hard SID")
    hard = sids[:, 0].astype(np.int64)

    pairs = np.load(PAIRS)
    anchors = pairs["anchors"].astype(np.int64)
    positives = pairs["positives"].astype(np.int64)
    negatives = pairs["negatives"].astype(np.int64)
    require("pair_count", len(anchors) == EXPECTED_POSITIVES,
            f"{len(anchors)} pairs, required {EXPECTED_POSITIVES}")
    require("negative_count", negatives.shape[1] == EXPECTED_NEGATIVES,
            f"{negatives.shape} negatives, required {EXPECTED_NEGATIVES}")
    require("indices_in_range",
            bool(anchors.max() < len(hard) and positives.max() < len(hard)
                 and negatives.max() < len(hard)),
            f"max index {max(int(anchors.max()), int(positives.max()), int(negatives.max()))} "
            f"vs {len(hard)} items")

    pos_share = float(np.mean(hard[anchors] == hard[positives]))
    neg_share = float(np.mean(hard[anchors[:, None]] == hard[negatives]))
    er = pos_share / neg_share if neg_share > 0 else float("nan")
    sid_sha = sha256(sids_path)

    result["metric"] = {
        "pos_share": pos_share,
        "neg_share": neg_share,
        "hard_l1_er": er,
        "baseline_hard_l1_er": BASELINE_ER,
        "delta_vs_baseline": (er - BASELINE_ER) if math.isfinite(er) else None,
        "sid_sha256": sid_sha,
        "frozen_pairs_sha256": PAIRS_SHA256,
        "baseline_reference_sid_sha256": BASELINE_SID_SHA256,
    }

    require("pos_share_finite", math.isfinite(pos_share), str(pos_share))
    require("neg_share_finite", math.isfinite(neg_share), str(neg_share))
    require("neg_share_positive", neg_share > 0.0, str(neg_share))
    require("er_finite", math.isfinite(er), str(er))
    require("er_above_baseline", math.isfinite(er) and er > BASELINE_ER,
            f"{er} > {BASELINE_ER} required (equal is a FAIL)")

    result["failures"] = failures
    result["stage3_allowed"] = not failures
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--tree", default=None)
    parser.add_argument("--spec-sha", default=None)
    arguments = parser.parse_args()
    tree = Path(arguments.tree) if arguments.tree else \
        REPO / "stage2_RQ-VAE" / f"TIGER_{arguments.arm}"
    result = evaluate(tree, arguments.arm, arguments.spec_sha)
    result_dir = REPO / "results/stage2_RQ-VAE" / arguments.arm
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "stage2_metrics.json").write_text(
        json.dumps(result["metric"], indent=2, default=str) + "\n", encoding="utf-8")
    (result_dir / "gate.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"\nstage3_allowed = {result['stage3_allowed']}")
    return 0 if result["stage3_allowed"] else 1


if __name__ == "__main__":
    sys.exit(main())
