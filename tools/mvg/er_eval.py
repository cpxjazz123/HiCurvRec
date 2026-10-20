"""The frozen Hard-L1 ER evaluation, run once per admitted arm.

The evaluation set is fixed -- the held-out valid transitions, seed 42 -- and the
threshold is frozen: the historical Stage-2 hard-L1 ER is
``16.38953942232631`` and the R4-08 arm measured ``16.402627511591962``.

This is a **measurement, not a knob**. It is run once, after the mechanism is admitted,
and nothing is ever retuned against it: a mechanism that misses the threshold is
reported as it is. The threshold is not an oracle either -- a mechanism can pass it and
still be mathematically wrong, which is what the independent judges are for.

Run with the project's own environment (it needs numpy):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/er_eval.py --sid results/stage2_RQ-VAE/<arm>/<arm>/sids_for_hgrec.npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_PAIRS = REPO_ROOT / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"

#: The frozen Stage-2 hard-L1 ER threshold and the historical R4-08 value.
FROZEN_L1_ER = 16.38953942232631
FROZEN_R4_08_ER = 16.402627511591962


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sid_matrix(path: Path) -> np.ndarray:
    """The SID matrix of an arm, in the form the frozen evaluation reads it.

    The frozen evaluation reads ``item_sids.json``: a dict of item id -> the four
    columns ``[token1, token2, token3, collision_token]``. A ``.npy`` export is read
    as the matrix it already is.
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    raw = path.read_bytes()
    sids = json.loads(raw.decode("utf-8"))
    max_id = max(int(k) for k in sids.keys())
    matrix = np.zeros((max_id + 1, 4), dtype=np.int64)
    for key, value in sids.items():
        matrix[int(key)] = value
    return matrix


def evaluate(sid_path: Path, *, bootstrap: bool = False) -> Dict[str, Any]:
    """The L1 prefix enrichment ratio of one arm's exported SIDs."""
    sid_path = Path(sid_path)
    if not sid_path.is_file():
        return {"ok": False, "error": f"no SID export at {sid_path}"}
    if not VALID_PAIRS.is_file():
        return {"ok": False, "error": f"the frozen evaluation set is missing: {VALID_PAIRS}"}
    sid = load_sid_matrix(sid_path)
    if sid.ndim != 2 or sid.shape[1] < 1:
        return {"ok": False, "error": f"the SID export has the wrong shape: {sid.shape}"}
    data = np.load(VALID_PAIRS)
    anchors = data["anchors"].astype(np.int64)
    positives = data["positives"].astype(np.int64)
    negatives = data["negatives"].astype(np.int64)

    anchor_codes = sid[anchors, 0]
    positive_codes = sid[positives, 0]
    negative_codes = sid[negatives, 0]
    positive_match = anchor_codes == positive_codes
    negative_match = anchor_codes[:, None] == negative_codes

    pos_share = float(positive_match.mean())
    neg_share = float(negative_match.mean())
    if pos_share == 1.0 and neg_share == 1.0:
        er: Any = 1.0
        status = "trivial_collapse_er_1"
    elif neg_share == 0.0:
        er = "undefined"
        status = "undefined_zero_neg_share"
    else:
        er = float(pos_share / neg_share)
        status = "defined"

    report: Dict[str, Any] = {
        "ok": True,
        "sid_path": str(sid_path),
        "sid_sha256": sha256_file(sid_path),
        "n_items": int(sid.shape[0]),
        "n_pairs": int(positive_match.size),
        "n_negatives_per_pair": int(negative_match.shape[1]),
        "pos_share": pos_share,
        "neg_share": neg_share,
        "gap": pos_share - neg_share,
        "l1_er": er,
        "l1_er_status": status,
        "frozen_threshold": FROZEN_L1_ER,
        "r4_08_reference": FROZEN_R4_08_ER,
        "evaluation_set": str(VALID_PAIRS.relative_to(REPO_ROOT)),
        "evaluation_set_sha256": sha256_file(VALID_PAIRS),
    }
    if isinstance(er, float):
        report["above_frozen_threshold"] = bool(er > FROZEN_L1_ER)
    if bootstrap:
        report["bootstrap"] = _bootstrap(positive_match, negative_match)
    return report


def _bootstrap(positive_match: np.ndarray, negative_match: np.ndarray,
               replicates: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """The paired bootstrap of the enrichment ratio, as the historical report gives it."""
    rng = np.random.default_rng(seed)
    rows = positive_match.shape[0]
    draws = negative_match.shape[1]
    ers: List[float] = []
    for _ in range(replicates):
        index = rng.integers(0, rows, rows)
        p = float(positive_match[index].mean())
        n = float(negative_match[index].mean())
        if p == 1.0 and n == 1.0:
            ers.append(1.0)
        elif n == 0.0:
            continue
        else:
            ers.append(p / n)
    if not ers:
        return {"replicates": replicates, "er_ci95": ["undefined", "undefined"]}
    low, high = np.quantile(ers, [0.025, 0.975])
    return {"replicates": replicates, "er_ci95": [float(low), float(high)]}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.mvg.er_eval", description=__doc__)
    parser.add_argument("--sid", required=True, help="the arm's exported sids_for_hgrec.npy")
    parser.add_argument("--bootstrap", action="store_true", help="also compute the bootstrap CI")
    parser.add_argument("--out", default="", help="write the report here as JSON")
    args = parser.parse_args(argv)
    sid_path = Path(args.sid)
    if not sid_path.is_absolute():
        sid_path = REPO_ROOT / sid_path
    report = evaluate(sid_path, bootstrap=args.bootstrap)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
