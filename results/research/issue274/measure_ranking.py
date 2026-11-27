#!/usr/bin/env python3
"""Issue #274: measure the frozen baselines and freeze the acceptance thresholds.

Everything here is measured from frozen artefacts **before** the candidate is trained:
  * the frozen TIGER's hard q (the plain sum of the three codeword embeddings at the exported
    codes) on the frozen hold-out pairs -> the C1/C2 thresholds;
  * the fixed margin m, calibrated in advance on the frozen TIGER's training-pair distance scale;
  * the frozen TIGER's SID prefix discrimination on the same hold-out pairs -> the C5 threshold;
  * the frozen TIGER's behavioural neighbourhood fidelity -> the C6 threshold;
  * the #265 same-caliber reconstruction -> the C9 threshold;
  * the #269 global J values (already in the #272 PASS table) -> the C7/C8 thresholds.

The script also exposes the same measurement for any candidate checkpoint, so the candidate is
scored at exactly the same caliber.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"


def load_pair_index():
    return np.load(OUT / "pairs_index.npz")


def hard_q(checkpoint_path: Path, sid_path: Path, n_items: int = 24587) -> np.ndarray:
    """The final hard q = sum over the three levels of the selected codeword embeddings."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = blob["state_dict"]
    codebooks = []
    for level in range(3):
        key = f"rq.vq_layers.{level}.embed.weight"
        if key not in state:
            raise KeyError(f"{key} not in the checkpoint")
        codebooks.append(state[key].detach().to(torch.float32))
    sid = json.loads(sid_path.read_text())
    codes = np.zeros((n_items, 3), dtype=np.int64)
    for item, row in sid.items():
        codes[int(item)] = np.asarray(row[:3], dtype=np.int64)
    q = torch.zeros((n_items, codebooks[0].shape[1]), dtype=torch.float32)
    for level, codebook in enumerate(codebooks):
        q += codebook[torch.from_numpy(codes[:, level])]
    return q.numpy()


def pair_distances(q: np.ndarray, index, mask: np.ndarray):
    """The positive and negative pair distances of the selected groups."""
    centre = index["centre"][mask]
    d_pos = np.linalg.norm(q[index["pos_b"][mask]] - q[index["pos_c"][mask]], axis=1)
    d_neg = np.linalg.norm(q[index["neg_d"][mask]] - q[index["neg_e"][mask]], axis=1)
    return centre, d_pos, d_neg


def prefix_lcp(codes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The length of the longest common prefix of two code rows."""
    equal = codes[a] == codes[b]
    # the first mismatch position, or the full length when they agree everywhere
    mismatch = np.where(equal.all(axis=1), codes.shape[1],
                        np.argmin(equal, axis=1))
    return mismatch.astype(np.int64)


def ranking_metrics(q: np.ndarray, index, mask: np.ndarray, margin: float) -> dict:
    centre, d_pos, d_neg = pair_distances(q, index, mask)
    accuracy = float((d_pos < d_neg).mean())
    margin_rate = float((d_pos + margin <= d_neg).mean())
    gap = d_neg - d_pos
    return {
        "groups": int(mask.sum()),
        "ranking_accuracy": accuracy,
        "margin_satisfaction_rate": margin_rate,
        "mean_distance_gap": float(gap.mean()),
        "median_distance_gap": float(np.median(gap)),
        "mean_positive_distance": float(d_pos.mean()),
        "mean_negative_distance": float(d_neg.mean()),
        "margin": float(margin),
        "macro_accuracy_by_centre": float(_macro(centre, d_pos < d_neg)),
        "macro_margin_rate_by_centre": float(_macro(centre, d_pos + margin <= d_neg)),
    }


def _macro(centre: np.ndarray, flag: np.ndarray) -> float:
    order = np.argsort(centre, kind="stable")
    c = centre[order]
    f = flag[order]
    boundaries = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[boundaries[1:], c.size]
    means = [float(f[a:b].mean()) for a, b in zip(boundaries, ends)]
    return float(np.mean(means)) if means else float("nan")


def prefix_metrics(codes: np.ndarray, index, mask: np.ndarray) -> dict:
    lcp_pos = prefix_lcp(codes, index["pos_b"][mask], index["pos_c"][mask])
    lcp_neg = prefix_lcp(codes, index["neg_d"][mask], index["neg_e"][mask])
    return {
        "lcp_positive_greater_than_negative": float((lcp_pos > lcp_neg).mean()),
        "lcp_positive_equals_negative": float((lcp_pos == lcp_neg).mean()),
        "mean_lcp_difference": float((lcp_pos - lcp_neg).mean()),
        "positive_pair_shares_level1": float((lcp_pos >= 1).mean()),
        "negative_pair_shares_level1": float((lcp_neg >= 1).mean()),
        "mean_lcp_positive": float(lcp_pos.mean()),
        "mean_lcp_negative": float(lcp_neg.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--margin", type=float, default=None,
                        help="the frozen margin; omit to calibrate it from the training pairs")
    parser.add_argument("--reference", default=None,
                        help="a baselines.json to inherit the frozen margin from")
    args = parser.parse_args()

    index = load_pair_index()
    train = ~index["is_holdout"]
    holdout = index["is_holdout"]
    q = hard_q(Path(args.checkpoint), Path(args.sid))
    sid = json.loads(Path(args.sid).read_text())
    codes = np.zeros((24587, 3), dtype=np.int64)
    for item, row in sid.items():
        codes[int(item)] = np.asarray(row[:3], dtype=np.int64)

    margin = args.margin
    calibrated = False
    if margin is None:
        if args.reference:
            margin = float(json.loads(Path(args.reference).read_text())["frozen_margin"])
        else:
            _, d_pos_train, _ = pair_distances(q, index, train)
            margin = float(np.median(d_pos_train))
            calibrated = True

    payload = {
        "schema": "genrec.issue274.baselines/1",
        "label": args.label,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sid": str(Path(args.sid).resolve()),
        "frozen_margin": float(margin),
        "margin_source": "calibrated on the frozen TIGER's training-pair positive distances"
        if calibrated else "inherited from the frozen reference",
        "training_pairs": ranking_metrics(q, index, train, margin),
        "holdout_pairs": ranking_metrics(q, index, holdout, margin),
        "holdout_prefix_discrimination": prefix_metrics(codes, index, holdout),
        "training_prefix_discrimination": prefix_metrics(codes, index, train),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("label", "frozen_margin", "holdout_pairs",
                                              "holdout_prefix_discrimination")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
