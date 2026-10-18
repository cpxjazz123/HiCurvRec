"""R4 shared gate evaluation: frozen held-out Hard L1 ER and the fail-closed Stage3 gate.

One metric, one gate, used by every R4-NN arm (issues #51-#60). Nothing here reads
train/valid/test parquet: the only inputs are the frozen held-out pair file and the
arm's own exported final SID column 0.

Gate (verbatim from the issue contract):

    stage3_allowed = tests_passed && train_end_3000 && hashes_valid
                     && pos_neg_support_valid && isfinite(ER_L1)
                     && neg_share > 0 && (ER_L1 > 16.38953942232631)

Any missing or non-finite ingredient defaults to False. There is no rounding and
no second threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PAIRS_PATH = REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
CANONICAL_SID_PATH = (
    REPO_ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
)
CANONICAL_SID_SHA256 = "b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10"
PAIRS_SHA256 = "04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0"
BASELINE_ER_L1 = 16.38953942232631
EXPECTED_POSITIVES = 52855
EXPECTED_NEGATIVES = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(path: Path = PAIRS_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["anchors"], data["positives"], data["negatives"]


def load_sid_matrix(item_sids_path: Path, n_items: int | None = None) -> np.ndarray:
    """Return the (n_items, 4) SID matrix; row index is the item id."""
    payload = json.loads(Path(item_sids_path).read_text(encoding="utf-8"))
    ids = sorted(int(key) for key in payload)
    if ids != list(range(len(ids))):
        raise ValueError("item_sids.json keys are not a dense 0..N-1 range")
    if n_items is not None and len(ids) != n_items:
        raise ValueError(f"SID catalogue has {len(ids)} rows, expected {n_items}")
    return np.asarray([payload[str(i)] for i in ids], dtype=np.int64)


def hard_l1_er(sid_matrix: np.ndarray) -> dict[str, Any]:
    """Hard L1 ER on the frozen held-out pairs: pos_share / neg_share, column 0 only."""
    anchors, positives, negatives = load_pairs()
    if positives.shape[0] != EXPECTED_POSITIVES or negatives.shape[1] != EXPECTED_NEGATIVES:
        raise ValueError(
            f"held-out file is not the frozen one: {positives.shape} / {negatives.shape}"
        )
    col0 = sid_matrix[:, 0]
    pos_share = float(np.mean(col0[anchors.astype(np.int64)] == col0[positives.astype(np.int64)]))
    neg_share = float(
        np.mean(col0[anchors.astype(np.int64)[:, None]] == col0[negatives.astype(np.int64)])
    )
    er_l1 = pos_share / neg_share if neg_share > 0 else float("nan")
    return {
        "n_positive_rows": int(positives.shape[0]),
        "n_negatives_per_row": int(negatives.shape[1]),
        "pos_share": pos_share,
        "neg_share": neg_share,
        "ER_L1": er_l1,
    }


def train_end_epoch(metrics_path: Path) -> int | None:
    """Epoch of the last ``train_end`` row, or None when the run never finished."""
    if not metrics_path.is_file():
        return None
    epoch: int | None = None
    with metrics_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "train_end":
                epoch = int(row.get("epoch", 3000))
    return epoch


def evaluate(
    *,
    arm: str,
    issue: int,
    item_sids_path: Path,
    metrics_path: Path,
    tests_passed: bool,
    canonical_sid_sha: str | None = None,
    metrics_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (stage2_metrics, gate). Fails closed on every missing ingredient."""
    item_sids_path = Path(item_sids_path)
    metrics_path = Path(metrics_path)

    sid_sha = sha256_file(item_sids_path) if item_sids_path.is_file() else None
    ckpt_path = item_sids_path.parent / "rqvae_best.pth"
    ckpt_sha = sha256_file(ckpt_path) if ckpt_path.is_file() else None
    er = {"pos_share": float("nan"), "neg_share": float("nan"), "ER_L1": float("nan")}

    sid_error = None
    if sid_sha is None:
        sid_error = "missing item_sids.json"
    else:
        try:
            er = hard_l1_er(load_sid_matrix(item_sids_path))
            if er["n_positive_rows"] != EXPECTED_POSITIVES or er["n_negatives_per_row"] != EXPECTED_NEGATIVES:
                sid_error = "held-out support is not the frozen 52,855 x 10 set"
        except Exception as exc:  # noqa: BLE001 - any failure must fail the gate closed
            sid_error = f"{type(exc).__name__}: {exc}"

    epoch = train_end_epoch(metrics_path)
    expected_canonical_sha = canonical_sid_sha or CANONICAL_SID_SHA256
    actual_canonical_sha = sha256_file(CANONICAL_SID_PATH) if CANONICAL_SID_PATH.is_file() else None
    actual_pairs_sha = sha256_file(PAIRS_PATH) if PAIRS_PATH.is_file() else None
    hashes_valid = (
        bool(sid_sha)
        and bool(ckpt_sha)
        and (actual_canonical_sha == expected_canonical_sha)
        and (actual_pairs_sha == PAIRS_SHA256)
    )
    pos_neg_support_valid = (
        er["n_positive_rows"] if "n_positive_rows" in er else False
    ) == EXPECTED_POSITIVES and er.get("n_negatives_per_row", 0) == EXPECTED_NEGATIVES
    finite_er = bool(math.isfinite(er["ER_L1"]))
    positive_delta = finite_er and er["ER_L1"] > BASELINE_ER_L1

    stage2_metrics = {
        "arm": arm,
        "issue": issue,
        "metric_name": "hard_l1_enrichment_ratio",
        "metric_definition": "pos_share / neg_share on SID column 0",
        "held_out_pairs": "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz",
        "held_out_pairs_sha256": actual_pairs_sha,
        "canonical_sid_path": "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
        "canonical_sid_sha256": actual_canonical_sha,
        "baseline_er_l1": BASELINE_ER_L1,
        **er,
        "exceeds_baseline_strictly": positive_delta,
        "train_end_epoch": epoch,
        "item_sids_sha256": sid_sha,
        "checkpoint_sha256": ckpt_sha,
        "sid_error": sid_error,
        **(metrics_extra or {}),
    }

    gate = {
        "arm": arm,
        "issue": issue,
        "tests_passed": bool(tests_passed),
        "train_end_3000": epoch == 3000,
        "hashes_valid": hashes_valid,
        "pos_neg_support_valid": bool(pos_neg_support_valid),
        "isfinite_ER_L1": finite_er,
        "neg_share_positive": bool(er["neg_share"] > 0),
        "ER_L1_above_baseline": positive_delta,
        "baseline_er_l1": BASELINE_ER_L1,
        "measured_er_l1": er["ER_L1"],
        "comparison": "strict >, no rounding",
        "sid_error": sid_error,
    }
    gate["stage3_allowed"] = bool(
        gate["tests_passed"]
        and gate["train_end_3000"]
        and gate["hashes_valid"]
        and gate["pos_neg_support_valid"]
        and gate["isfinite_ER_L1"]
        and gate["neg_share_positive"]
        and gate["ER_L1_above_baseline"]
    )
    return stage2_metrics, gate


def maybe_launch_stage3(gate: dict[str, Any], launcher: Any) -> bool:
    """Invoke ``launcher()`` iff ``gate['stage3_allowed']`` is strictly True."""
    if not isinstance(gate, dict) or gate.get("stage3_allowed") is not True:
        return False
    launcher()
    return True

