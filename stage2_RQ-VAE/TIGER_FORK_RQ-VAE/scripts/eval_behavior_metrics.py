"""Behaviour and SID metrics for one model, on the canonical seed-42 pairs.

Reusable by every step of the programme: it takes a SID path and the frozen pair
file, so the baseline and every mechanism are measured on identical anchors,
identical positives and identical negatives. Nothing is redrawn here.

Reported per level (L1, L1+L2, L1+L2+L3): positive sharing rate, negative sharing
rate, gap, enrichment ratio. Plus the three-token collision rate, per-level
codebook utilisation, and the L1 positive-coverage figures that step 9 needs to
tell a genuine selectivity gain from a mechanism that simply stopped sharing.

Enrichment with a zero negative rate is reported as undefined, never as infinity
and never with an epsilon: a model that shares no prefix with anything would
otherwise look perfectly selective.

The fourth stored SID token is the collision extension token and is never counted
as a level.

Recall@10 is NOT computed here. It comes from the Stage 3 native test_final.json
of the run that produced the SID, and is read from disk so no number is retyped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402

ROOT = AP.ROOT
OUT_DIR = MM.OUT_DIR
PAIRS = OUT_DIR / "behavior_pairs_s42.npz"
MIN_CELL = AP.MIN_CELL
CODEBOOK_SIZE = 256


def load_pairs(path: Path) -> dict:
    z = np.load(path)
    return {
        "anchors": z["anchors"].astype(np.int64),
        "positives": z["positives"].astype(np.int64),
        "negatives": z["negatives"].astype(np.int64),
    }


def codebook_utilisation(levels: np.ndarray) -> dict:
    """Distinct codes used per level, over the codebook capacity."""
    out = {}
    for k in range(levels.shape[1]):
        used = int(len(np.unique(levels[:, k])))
        out[f"L{k + 1}"] = {
            "distinct_codes": used,
            "capacity": CODEBOOK_SIZE,
            "utilisation": used / CODEBOOK_SIZE,
        }
    triples = int(len(np.unique(levels[:, :3], axis=0)))
    out["full_sid"] = {
        "distinct_triples": triples,
        "n_items": int(len(levels)),
        "collision_rate": float(1.0 - triples / len(levels)),
    }
    return out


def _macro_mean(values: np.ndarray, groups: np.ndarray) -> float:
    """Mean over groups of the within-group mean, in O(n) not O(groups * n).

    The previous form rescanned the whole array once per anchor, which is
    quadratic and dominated the runtime at 24k anchors and 3.2M negatives.
    """
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(np.float64))
    counts = np.bincount(inv)
    keep = counts > 0
    return float((sums[keep] / counts[keep]).mean())


def metrics_for(levels: np.ndarray, pairs: dict) -> dict:
    anchors, positives, negatives = (
        pairs["anchors"], pairs["positives"], pairs["negatives"]
    )
    k = negatives.shape[1]
    anchor_rep = np.repeat(anchors, k)
    neg_flat = negatives.reshape(-1)
    out = {"k_negatives": k, "n_positive_pairs": int(len(anchors)),
           "n_negative_pairs": int(len(neg_flat)),
           "n_anchors": int(len(np.unique(anchors)))}

    for lvl in (1, 2, 3):
        pre = AP.prefix_ids(levels, lvl)
        ps = pre[positives] == pre[anchors]
        ns = pre[neg_flat] == pre[anchor_rep]
        p_pos, p_neg = float(ps.mean()), float(ns.mean())
        entry = {
            "p_share_pos": p_pos,
            "p_share_neg": p_neg,
            "gap": p_pos - p_neg,
            "er": (p_pos / p_neg) if p_neg > 0 else None,
            "er_undefined_reason": None if p_neg > 0 else "negative sharing rate is zero",
            "macro_p_share_pos": _macro_mean(ps, anchors),
            "macro_p_share_neg": _macro_mean(ns, anchor_rep),
            "sampled_precision_given_share": (
                int(ps.sum()) / (int(ps.sum()) + int(ns.sum()))
                if (int(ps.sum()) + int(ns.sum())) > 0 else None
            ),
        }
        entry["macro_gap"] = entry["macro_p_share_pos"] - entry["macro_p_share_neg"]
        out[{1: "L1", 2: "L1_L2", 3: "L1_L2_L3"}[lvl]] = entry

    # Conditional retention, the figure that separates a coarse L1 from a genuine
    # L2/L3 refinement effect.
    pre = {k2: AP.prefix_ids(levels, k2) for k2 in (1, 2, 3)}
    for k2 in (2, 3):
        prev, cur = pre[k2 - 1], pre[k2]
        entry = {}
        for side, others, anch in (
            ("pos", positives, anchors), ("neg", neg_flat, anchor_rep)
        ):
            cond = prev[anch] == prev[others]
            entry[f"n_effective_{side}"] = int(cond.sum())
            entry[f"p_retain_{side}"] = (
                float((cur[anch][cond] == cur[others][cond]).mean())
                if cond.sum() >= MIN_CELL else None
            )
        out[f"L{k2}_given_L{k2 - 1}"] = entry

    # L1 positive coverage: what fraction of behaviour positives still land in a
    # shared L1 at all, and how large those shared groups are. A mechanism that
    # raises ER by pushing positives into singleton L1 groups is visible here.
    pre1 = AP.prefix_ids(levels, 1)
    sizes = AP.group_sizes(pre1)[pre1]
    out["L1_coverage"] = {
        "frac_positives_sharing": float((pre1[positives] == pre1[anchors]).mean()),
        "mean_L1_group_size_of_positives": float(sizes[positives].mean()),
        "mean_L1_group_size_of_negatives": float(sizes[neg_flat].mean()),
        "frac_positives_in_singleton_group": float((sizes[positives] == 1).mean()),
        "L1_max_group": int(sizes.max()),
    }
    out["codebook"] = codebook_utilisation(levels)
    return out


def recall_for(sid_path: str) -> dict | None:
    """Native Stage 3 result for the run that produced this SID, read from disk."""
    inv_path = OUT_DIR / "model_inventory.json"
    if not inv_path.exists():
        return None
    inv = json.loads(inv_path.read_text())
    q = Path(sid_path)
    if not q.is_absolute():
        q = ROOT / q
    rel = str(q.relative_to(ROOT))
    for m in inv["models"]:
        if m["sid_path"] == rel:
            return {
                "model": m["model"], "recall@10": m["recall@10"],
                "ndcg@10": m["ndcg@10"], "n_eval": m["n_eval"],
                "stage3_runs": m["stage3_runs"],
            }
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="path to item_sids.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = load_pairs(PAIRS)
    levels, info = AP.load_levels(Path(args.sid))
    if len(levels) != MM.N_ITEMS:
        raise ValueError(f"{args.sid}: {len(levels)} items, expected {MM.N_ITEMS}")

    report = {
        "label": args.label,
        "sid_path": args.sid,
        "sid_token_info": info,
        "pairs_file": str(PAIRS.relative_to(ROOT)),
        "seed": 42,
        "metrics": metrics_for(levels, pairs),
        "stage3": recall_for(args.sid),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    m = report["metrics"]
    print(f"{args.label}")
    for lvl in ("L1", "L1_L2", "L1_L2_L3"):
        e = m[lvl]
        er = f"{e['er']:.3f}" if e["er"] is not None else "undefined"
        print(f"  {lvl:10s} pos={e['p_share_pos']:.5f} neg={e['p_share_neg']:.5f} "
              f"gap={e['gap']:+.5f} ER={er}")
    print(f"  collision       {m['codebook']['full_sid']['collision_rate']:.4f}")
    print(f"  codebook used   L1={m['codebook']['L1']['distinct_codes']} "
          f"L2={m['codebook']['L2']['distinct_codes']} "
          f"L3={m['codebook']['L3']['distinct_codes']}")
    if report["stage3"]:
        print(f"  Recall@10       {report['stage3']['recall@10']:.6f}  "
              f"NDCG@10 {report['stage3']['ndcg@10']:.6f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()