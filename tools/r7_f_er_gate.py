"""The ONE frozen Stage2 Hard-L1 ER gate for issue #89.

Reads the immutable held-out pair artifact EXACTLY ONCE, after the final Stage2 checkpoint and its
SID export exist. Nothing here trains, tunes or selects: it measures and compares against the
frozen reference ``16.38953942232631`` with a strict ``>`` and no rounding.

    pos_share = mean(SID_L1[source] == SID_L1[positive])
    neg_share = mean(SID_L1[source] == SID_L1[negative])
    Hard_L1_ER = pos_share / neg_share

The gate is fail-closed: a missing artifact, a checksum mismatch, an item-map mismatch, a
non-finite share or ``neg_share <= 0`` refuses to produce a number.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")

HELDOUT = REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
HELDOUT_SHA256 = "04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0"
FROZEN_BASELINE = 16.38953942232631
HISTORICAL_R4_08 = 16.402627511591962


class GateRefused(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sid_l1(item_sids_path: Path) -> tuple[np.ndarray, str]:
    """The exported L1 code per item, plus a hash of the item->SID mapping."""
    if not item_sids_path.is_file():
        raise GateRefused("missing_artifact", f"{item_sids_path} does not exist")
    payload = json.loads(item_sids_path.read_text())
    mapping = payload.get("item_to_sid") or payload.get("sids") or payload
    if not isinstance(mapping, dict):
        raise GateRefused("invalid_artifact",
                          "item_sids.json is not a mapping from item id to SID")
    size = max(int(key) for key in mapping) + 1
    l1 = np.full(size, -1, dtype=np.int64)
    for key, value in mapping.items():
        codes = value if isinstance(value, (list, tuple)) else [value]
        if not codes:
            raise GateRefused("invalid_artifact", f"item {key} has an empty SID")
        l1[int(key)] = int(codes[0])
    if (l1 < 0).any():
        raise GateRefused("invalid_artifact",
                          f"{int((l1 < 0).sum())} items have no L1 code")
    ordered = sorted((int(key), int(l1[int(key)])) for key in mapping)
    digest = hashlib.sha256(np.array(ordered, dtype=np.int64).tobytes()).hexdigest()
    return l1, digest


def evaluate(item_sids_path: Path, *, artifact: Path = HELDOUT,
             expected_sha256: str = HELDOUT_SHA256) -> dict[str, Any]:
    """Measure the frozen Hard-L1 ER once. Refuses rather than guesses."""
    if not artifact.is_file():
        raise GateRefused("missing_artifact", f"{artifact} does not exist")
    actual_sha = sha256_file(artifact)
    if expected_sha256 and actual_sha != expected_sha256:
        raise GateRefused("checksum_mismatch",
                          f"the held-out artifact hash is {actual_sha}, expected "
                          f"{expected_sha256}")
    l1, mapping_sha = load_sid_l1(item_sids_path)
    pairs = np.load(artifact)
    anchors = pairs["anchors"].astype(np.int64)
    positives = pairs["positives"].astype(np.int64)
    negatives = pairs["negatives"].astype(np.int64)
    highest = int(max(anchors.max(), positives.max(), negatives.max()))
    if highest >= l1.shape[0]:
        raise GateRefused("item_map_mismatch",
                          f"the SID export covers {l1.shape[0]} items but the pairs reach "
                          f"index {highest}")
    pos_hits = int((l1[anchors] == l1[positives]).sum())
    neg_hits = int((l1[anchors][:, None] == l1[negatives]).sum())
    pos_share = pos_hits / float(anchors.shape[0])
    neg_share = neg_hits / float(anchors.shape[0] * negatives.shape[1])
    if not np.isfinite(pos_share) or not np.isfinite(neg_share):
        raise GateRefused("non_finite", f"pos_share={pos_share} neg_share={neg_share}")
    if neg_share <= 0.0:
        raise GateRefused("zero_neg_share",
                          "neg_share is 0, so Hard-L1 ER is undefined and cannot pass")
    er = pos_share / neg_share
    return {
        "artifact": str(artifact),
        "artifact_sha256": actual_sha,
        "item_sids": str(item_sids_path),
        "item_map_sha256": mapping_sha,
        "positives": int(anchors.shape[0]),
        "negatives_per_positive": int(negatives.shape[1]),
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "pos_share": pos_share,
        "neg_share": neg_share,
        "hard_l1_er": er,
        "frozen_baseline": FROZEN_BASELINE,
        "comparison": "strict >, no rounding, equality fails",
        "passed": bool(er > FROZEN_BASELINE),
        "historical_r4_08_reference": HISTORICAL_R4_08,
        "stronger_than_historical_r4_08": bool(er > HISTORICAL_R4_08),
        "note": "The frozen baseline is the ONLY Stage3 gate. Exceeding the historical R4-08 "
                "value is a separate research comparison, not the gate.",
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: r7_f_er_gate.py <item_sids.json> [out.json]")
        return 2
    item_sids = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    try:
        result = evaluate(item_sids)
    except GateRefused as exc:
        print(json.dumps({"refused": True, "status": exc.status, "reason": exc.reason},
                         indent=2))
        return 1
    text = json.dumps(result, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
