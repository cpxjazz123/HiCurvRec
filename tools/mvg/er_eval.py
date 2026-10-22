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


def _evaluation_set_label() -> str:
    """The evaluation set's path, relative to the repository when it lives inside it."""
    try:
        return str(VALID_PAIRS.relative_to(REPO_ROOT))
    except ValueError:
        return str(VALID_PAIRS)


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
        "evaluation_set": _evaluation_set_label(),
        "evaluation_set_sha256": sha256_file(VALID_PAIRS),
    }
    if isinstance(er, float):
        report["above_frozen_threshold"] = bool(er > FROZEN_L1_ER)
    report["d2"] = conditional_l2_discrimination(
        sid, anchors, positives, negatives
    )
    if bootstrap:
        report["bootstrap"] = _bootstrap(positive_match, negative_match)
    return report


#: A conditional set with fewer pairs than this is reported as low support.
LOW_SUPPORT_PAIRS = 1000

#: The replicates of the anchor-cluster bootstrap of the D2.
D2_BOOTSTRAP_REPLICATES = 1000
D2_BOOTSTRAP_SEED = 42

#: The replicates are drawn in chunks, so the resampling matrix stays small.
D2_BOOTSTRAP_CHUNK = 50


def conditional_l2_discrimination(
    sid: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    replicates: int = D2_BOOTSTRAP_REPLICATES,
    seed: int = D2_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """The conditional level-2 discrimination, from the SAME frozen read.

    ``D2 = P(shared L2 | shared L1, positive pair) - P(shared L2 | shared L1, negative pair)``

    It is computed here, from the pairs this evaluation has already loaded, because the
    frozen pair read may happen only once per arm: asking for D2 must never cost a second
    read of the held-out pairs.

    The uncertainty is a bootstrap over ANCHORS, because the pairs that share an anchor are
    not independent, and the resampling unit has to be the anchor.
    """
    if sid.shape[1] < 3:
        return {"ok": False, "reason": "the export carries no level-2 column"}

    anchor_l1 = sid[anchors, 1]
    positive_l1 = sid[positives, 1]
    anchor_l2 = sid[anchors, 2]
    positive_l2 = sid[positives, 2]
    negative_l1 = sid[negatives, 1]
    negative_l2 = sid[negatives, 2]

    positive_shared_l1 = anchor_l1 == positive_l1
    positive_shared_l2 = anchor_l2 == positive_l2
    negative_shared_l1 = anchor_l1[:, None] == negative_l1
    negative_shared_l2 = anchor_l2[:, None] == negative_l2

    # the per-anchor sufficient statistics, which are what the bootstrap resamples
    positive_denominator = positive_shared_l1.astype(np.int64)
    positive_numerator = (positive_shared_l1 & positive_shared_l2).astype(np.int64)
    negative_denominator = negative_shared_l1.sum(axis=1).astype(np.int64)
    negative_numerator = (negative_shared_l1 & negative_shared_l2).sum(axis=1).astype(np.int64)

    denominator_positive = int(positive_denominator.sum())
    numerator_positive = int(positive_numerator.sum())
    denominator_negative = int(negative_denominator.sum())
    numerator_negative = int(negative_numerator.sum())

    flags: List[str] = []
    if denominator_positive == 0 or denominator_negative == 0:
        flags.append("undefined_zero_denominator")
    if (
        denominator_positive < LOW_SUPPORT_PAIRS
        or denominator_negative < LOW_SUPPORT_PAIRS
    ):
        flags.append("low_support")

    if denominator_positive == 0 or denominator_negative == 0:
        conditional_positive: Any = None
        conditional_negative: Any = None
        difference: Any = None
    else:
        conditional_positive = float(numerator_positive / denominator_positive)
        conditional_negative = float(numerator_negative / denominator_negative)
        difference = float(conditional_positive - conditional_negative)

    interval = _anchor_cluster_interval(
        positive_numerator,
        positive_denominator,
        negative_numerator,
        negative_denominator,
        replicates=replicates,
        seed=seed,
    )

    return {
        "ok": True,
        "definition": (
            "P(shared L2 | shared L1, positive pair) - P(shared L2 | shared L1, negative pair)"
        ),
        "conditional_positive": conditional_positive,
        "conditional_negative": conditional_negative,
        "d2": difference,
        "numerator_positive": numerator_positive,
        "denominator_positive": denominator_positive,
        "numerator_negative": numerator_negative,
        "denominator_negative": denominator_negative,
        "anchors": int(anchors.shape[0]),
        "anchors_with_a_shared_l1_positive": int((positive_denominator > 0).sum()),
        "anchors_with_a_shared_l1_negative": int((negative_denominator > 0).sum()),
        "flags": flags,
        "undefined": "undefined_zero_denominator" in flags,
        "low_support": "low_support" in flags,
        "anchor_cluster_ci95": interval,
        "computed_from": "the same frozen pair read as the enrichment ratio",
    }


def _anchor_cluster_interval(
    positive_numerator: np.ndarray,
    positive_denominator: np.ndarray,
    negative_numerator: np.ndarray,
    negative_denominator: np.ndarray,
    *,
    replicates: int = D2_BOOTSTRAP_REPLICATES,
    seed: int = D2_BOOTSTRAP_SEED,
    chunk: int = D2_BOOTSTRAP_CHUNK,
) -> Dict[str, Any]:
    """The bootstrap confidence interval of the D2, resampling the anchors."""
    rows = int(positive_denominator.shape[0])
    if rows == 0:
        return {"low": None, "high": None, "replicates": replicates, "reason": "no anchors"}
    rng = np.random.default_rng(seed)
    differences: List[np.ndarray] = []
    remaining = replicates
    while remaining > 0:
        size = min(chunk, remaining)
        draws = rng.integers(0, rows, size=(size, rows))
        positive_num = positive_numerator[draws].sum(axis=1)
        positive_den = positive_denominator[draws].sum(axis=1)
        negative_num = negative_numerator[draws].sum(axis=1)
        negative_den = negative_denominator[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(
                (positive_den > 0) & (negative_den > 0),
                positive_num / positive_den - negative_num / negative_den,
                np.nan,
            )
        differences.append(values)
        remaining -= size
    finite = np.concatenate(differences)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"low": None, "high": None, "replicates": replicates, "reason": "no defined draw"}
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "replicates": int(replicates),
        "unit": "anchor",
        "seed": int(seed),
    }


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
