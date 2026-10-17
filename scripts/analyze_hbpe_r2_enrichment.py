"""Held-out evaluation for the HBPE R2 rounds (issues #34, #35, #36).

Everything is measured on the fixed independent validation set
``results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz``: 52,855
transitions whose scored relations were never observed in ``train.parquet``, 10
popularity/frequency-matched negatives each, seed 42.

Two kinds of interval are reported and they are NOT interchangeable:

  per-arm CI
      the bootstrap percentile interval of that arm's own enrichment ratio.

  paired delta-ER CI
      the interval of ``ER(reference) - ER(candidate)`` where BOTH ratios are
      recomputed on the SAME resampled rows in the same replicate. Whether two
      separate per-arm intervals happen to overlap is not a test of their
      difference -- two arms can have overlapping marginal intervals and still
      differ consistently -- so every comparison in the report is stated as a
      paired delta, and the marginal overlap is only shown for context.

Each arm additionally reports the L1 codebook's distinct-code count and its
effective usage ``exp(H)`` over all items, plus the L1+L2 distinct-prefix count and
the full three-token collision rate.

Stage 2 only: no Stage 3 job is started, queued or referenced, and no Recall or
NDCG number is produced or estimated.

Usage:
  python3.9 scripts/analyze_hbpe_r2_enrichment.py --round a1
  python3.9 scripts/analyze_hbpe_r2_enrichment.py --round a2
  python3.9 scripts/analyze_hbpe_r2_enrichment.py --round a3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_PAIRS_PATH = (
    REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
)
RESULT2 = REPO_ROOT / "results/stage2_RQ-VAE"

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95
DEPTHS = (1, 2)

BASELINE_L1_ER = 16.3895
BASELINE_L1L2_ER = 58.3871


def _arm(key: str, label: str, tree: str, sub: str) -> tuple[str, str, Path]:
    return key, label, RESULT2 / tree / sub / "item_sids.json"


RESAMPLE = _arm("resample", "Exp1a Resample (canonical baseline)",
                "TIGER_BEHAVIOR_SAMPLE_RQ-VAE", "resample")

ROUNDS: dict[str, dict[str, Any]] = {
    "a1": {
        "issue": 34,
        "output": RESULT2 / "hbpe_a1_r2",
        "reference_keys": ["resample", "hbpe_a1"],
        "models": [
            RESAMPLE,
            _arm("hbpe_a1", "HBPE-A1 (#31) lambda_ER1 = 0.020, no occupancy guard",
                 "TIGER_HBPE_A1", "hbpe_a1"),
            _arm("a1r2_w005", "A1-R2 lambda_ER1 = 0.005",
                 "TIGER_HBPE_A1_R2_W005", "a1r2_w005"),
            _arm("a1r2_w010", "A1-R2 lambda_ER1 = 0.010",
                 "TIGER_HBPE_A1_R2_W010", "a1r2_w010"),
            _arm("a1r2_w010_occ", "A1-R2 lambda_ER1 = 0.010 + occupancy guard 0.01",
                 "TIGER_HBPE_A1_R2_W010_OCC", "a1r2_w010_occ"),
        ],
    },
    "a2": {
        "issue": 35,
        "output": RESULT2 / "hbpe_a2_r2",
        "reference_keys": ["resample", "hbpe_a1", "hbpe_a2"],
        "models": [
            RESAMPLE,
            _arm("hbpe_a1", "HBPE-A1 (#31) L1 only, no L2 term",
                 "TIGER_HBPE_A1", "hbpe_a1"),
            _arm("hbpe_a2", "HBPE-A2 (#32) lambda_ER12 = 0.020",
                 "TIGER_HBPE_A2", "hbpe_a2"),
            _arm("a2r2_l2w002", "A2-R2 lambda_ER12 = 0.002",
                 "TIGER_HBPE_A2_R2_L2W002", "a2r2_l2w002"),
            _arm("a2r2_l2w005", "A2-R2 lambda_ER12 = 0.005",
                 "TIGER_HBPE_A2_R2_L2W005", "a2r2_l2w005"),
            _arm("a2r2_l2w005_sgL1",
                 "A2-R2 lambda_ER12 = 0.005 with s12 = stopgrad(s1) * s2",
                 "TIGER_HBPE_A2_R2_L2W005_SGL1", "a2r2_l2w005_sgL1"),
        ],
    },
    "a3": {
        "issue": 36,
        "output": RESULT2 / "hbpe_a3_r2",
        "reference_keys": ["resample", "hbpe_a2", "hbpe_a3"],
        "models": [
            RESAMPLE,
            _arm("hbpe_a2", "HBPE-A2 (#32) no negative cap",
                 "TIGER_HBPE_A2", "hbpe_a2"),
            _arm("hbpe_a3", "HBPE-A3 (#33) inactive coverage floors",
                 "TIGER_HBPE_A3", "hbpe_a3"),
            _arm("a3r2_negcap", "A3-R2 + negative-sharing cap 0.01",
                 "TIGER_HBPE_A3_R2_NEGCAP", "a3r2_negcap"),
            _arm("a3r2_negcap_occ", "A3-R2 + negative cap 0.01 + occupancy guard 0.01",
                 "TIGER_HBPE_A3_R2_NEGCAP_OCC", "a3r2_negcap_occ"),
        ],
    },
}


def load_sid_matrix(path: Path) -> tuple[np.ndarray, str, int]:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    sids = json.loads(raw.decode("utf-8"))
    max_id = max(int(k) for k in sids)
    matrix = np.zeros((max_id + 1, 4), dtype=np.int64)
    for key, value in sids.items():
        matrix[int(key)] = value
    return matrix, sha, len(sids)


def match_arrays(
    sid_matrix: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row positive match and per-negative match arrays for one prefix level."""
    if depth == 1:
        a, p, n = sid_matrix[anchors, 0], sid_matrix[positives, 0], sid_matrix[negatives, 0]
        return a == p, a[:, None] == n
    a = sid_matrix[anchors, :2]
    p = sid_matrix[positives, :2]
    n = sid_matrix[negatives, :2]
    return np.all(a == p, axis=1), np.all(a[:, None, :] == n, axis=2)


def _ratio(p: float, n: float) -> float:
    if p == 1.0 and n == 1.0:
        return 1.0
    return float("nan") if n == 0.0 else p / n


def _quantile(values: list[float], q: float) -> float | str:
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < len(values) * 0.5:
        return "undefined"
    return float(np.quantile(finite, q))


def bootstrap_indices(n_rows: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=(replicates, n_rows))


def per_arm_bootstrap(
    pos_match: np.ndarray,
    neg_match: np.ndarray,
    draws: np.ndarray,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    neg_row_mean = neg_match.mean(axis=1)
    p_samples = pos_match[draws].mean(axis=1)
    n_samples = neg_row_mean[draws].mean(axis=1)
    gaps = p_samples - n_samples
    ers = [_ratio(float(p), float(n)) for p, n in zip(p_samples, n_samples)]
    alpha = 1.0 - confidence
    return {
        "gap_ci95": [
            float(np.quantile(gaps, alpha / 2.0)),
            float(np.quantile(gaps, 1.0 - alpha / 2.0)),
        ],
        "er_ci95": [_quantile(ers, alpha / 2.0), _quantile(ers, 1.0 - alpha / 2.0)],
        "_ers": ers,
    }


def paired_delta(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """delta = ER(reference) - ER(candidate) on paired replicates.

    Both arms' replicate ratios come from ``per_arm_bootstrap`` called with the SAME
    ``draws`` matrix, so replicate i compares the two arms on identical rows. That
    shared ``draws`` is the whole point: the marginal intervals of the two arms are
    not what is being differenced.
    """
    deltas: list[float] = []
    for ref_er, cand_er in zip(reference["_boot_ers"], candidate["_boot_ers"]):
        if np.isfinite(ref_er) and np.isfinite(cand_er):
            deltas.append(ref_er - cand_er)
    if len(deltas) < len(reference["_boot_ers"]) * 0.5:
        return {"delta_er": "undefined", "delta_er_ci95": ["undefined", "undefined"],
                "replicates_used": len(deltas)}
    alpha = 1.0 - confidence
    ref_point = reference["er"]
    cand_point = candidate["er"]
    point = (ref_point - cand_point
             if isinstance(ref_point, float) and isinstance(cand_point, float)
             else "undefined")
    return {
        "delta_er": point,
        "delta_er_ci95": [float(np.quantile(deltas, alpha / 2.0)),
                          float(np.quantile(deltas, 1.0 - alpha / 2.0))],
        "replicates_used": len(deltas),
        "significant_at_95": bool(
            (np.quantile(deltas, alpha / 2.0) > 0.0)
            or (np.quantile(deltas, 1.0 - alpha / 2.0) < 0.0)
        ) if deltas else False,
    }


def codebook_diagnostics(sid_matrix: np.ndarray, n_items: int) -> dict[str, Any]:
    tokens_l1 = sid_matrix[:, 0]
    counts = np.bincount(tokens_l1, minlength=int(tokens_l1.max()) + 1).astype(np.float64)
    share = counts / counts.sum()
    nonzero = share[share > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    l123_unique = int(len(np.unique(sid_matrix[:, :3], axis=0)))
    return {
        "l1_distinct_codes": int((counts > 0).sum()),
        "l1_effective_usage_exp_H": float(np.exp(entropy)),
        "l1_entropy_nats": entropy,
        "l1_max_possible_entropy": float(np.log(len(counts))),
        "l12_distinct_prefixes": int(len(np.unique(sid_matrix[:, :2], axis=0))),
        "l123_unique_tokens": l123_unique,
        "full_3token_collision": float(1.0 - l123_unique / n_items),
        "n_items": n_items,
    }


def evaluate(round_key: str) -> dict[str, Any]:
    spec = ROUNDS[round_key]
    data = np.load(VALID_PAIRS_PATH)
    anchors = data["anchors"].astype(np.int64)
    positives = data["positives"].astype(np.int64)
    negatives = data["negatives"].astype(np.int64)
    draws = bootstrap_indices(len(anchors), BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)

    missing = [k for k, _, p in spec["models"] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"round {round_key}: missing SID exports for {missing}")

    results: list[dict[str, Any]] = []
    for key, label, sid_path in spec["models"]:
        sid_matrix, sha, n_items = load_sid_matrix(sid_path)
        entry: dict[str, Any] = {
            "arm": key,
            "label": label,
            "sid_sha256": sha,
            "sid_path": str(sid_path.relative_to(REPO_ROOT)),
            **codebook_diagnostics(sid_matrix, n_items),
            "levels": {},
        }
        for depth in DEPTHS:
            pos_match, neg_match = match_arrays(
                sid_matrix, anchors, positives, negatives, depth
            )
            boot = per_arm_bootstrap(pos_match, neg_match, draws)
            entry["levels"][depth] = {
                "prefix_level": "L1" if depth == 1 else "L1+L2",
                "pos_count": int(pos_match.sum()),
                "neg_count": int(neg_match.sum()),
                "pos_pairs": int(pos_match.size),
                "total_neg_draws": int(neg_match.size),
                "pos_share": float(pos_match.mean()),
                "neg_share": float(neg_match.mean()),
                "gap": float(pos_match.mean() - neg_match.mean()),
                "gap_ci95": boot["gap_ci95"],
                "er": _ratio(float(pos_match.mean()), float(neg_match.mean())),
                "er_ci95": boot["er_ci95"],
                "_boot_ers": boot["_ers"],
            }
        results.append(entry)

    by_key = {r["arm"]: r for r in results}
    for entry in results:
        entry["paired_delta_vs"] = {}
        for ref_key in spec["reference_keys"]:
            if ref_key == entry["arm"] or ref_key not in by_key:
                continue
            reference = by_key[ref_key]
            entry["paired_delta_vs"][ref_key] = {
                depth: paired_delta(reference["levels"][depth], entry["levels"][depth])
                for depth in DEPTHS
            }

    baseline = by_key["resample"]
    meta = {
        "issue": spec["issue"],
        "stage2_only": True,
        "no_stage3_job": (
            "this round starts, queues and references no Stage 3 job and reports no "
            "Recall@10 or NDCG@10"
        ),
        "valid_pairs_path": str(VALID_PAIRS_PATH.relative_to(REPO_ROOT)),
        "n_held_out_transitions": int(len(anchors)),
        "negatives_per_transition": int(negatives.shape[1]),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "paired_bootstrap_note": (
            "delta intervals resample the SAME rows for both arms in each replicate; "
            "marginal CI overlap is reported for context only and is not a difference test"
        ),
        "baseline_check": {
            "L1_ER_expected": BASELINE_L1_ER,
            "L1_ER_actual": baseline["levels"][1]["er"],
            "L1_ER_match": abs(float(baseline["levels"][1]["er"]) - BASELINE_L1_ER) < 5e-4,
            "L1L2_ER_expected": BASELINE_L1L2_ER,
            "L1L2_ER_actual": baseline["levels"][2]["er"],
            "L1L2_ER_match": abs(float(baseline["levels"][2]["er"]) - BASELINE_L1L2_ER) < 5e-4,
        },
    }
    return {"metadata": meta, "results": results}


def save(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in report["results"]:
        for level in entry["levels"].values():
            level.pop("_boot_ers", None)
    (output_dir / "hbpe_r2_enrichment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    rows: list[list[Any]] = []
    for entry in report["results"]:
        for depth in DEPTHS:
            level = entry["levels"][depth]
            row = [
                entry["arm"], entry["sid_sha256"][:10], level["prefix_level"],
                f"{level['pos_share']:.6f}", f"{level['neg_share']:.6f}",
                f"{level['gap']:.6f}",
                f"{level['er']:.6f}" if isinstance(level["er"], float) else level["er"],
                str(level["er_ci95"]),
                entry["l1_distinct_codes"], f"{entry['l1_effective_usage_exp_H']:.4f}",
                entry["l12_distinct_prefixes"], f"{entry['full_3token_collision']:.6f}",
            ]
            for ref_key, deltas in entry["paired_delta_vs"].items():
                d = deltas[depth]
                row += [
                    ref_key,
                    f"{d['delta_er']:.6f}" if isinstance(d["delta_er"], float) else "undefined",
                    str(d["delta_er_ci95"]),
                    d["significant_at_95"],
                ]
            rows.append(row)
    header = [
        "arm", "sid_sha", "prefix_level", "pos_share", "neg_share", "gap", "er",
        "er_ci95", "l1_codes", "l1_effective_usage", "l12_prefixes", "collision",
    ]
    for ref_key in sorted({k for e in report["results"] for k in e["paired_delta_vs"]}):
        header += [f"delta_er_vs_{ref_key}", f"delta_er_ci95_vs_{ref_key}",
                   f"delta_sig95_vs_{ref_key}"]
    with (output_dir / "hbpe_r2_enrichment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, choices=sorted(ROUNDS))
    args = parser.parse_args()
    report = evaluate(args.round)
    report["metadata"]["round"] = args.round
    save(report, ROUNDS[args.round]["output"])
    checks = report["metadata"]["baseline_check"]
    print(f"round {args.round} -> {ROUNDS[args.round]['output']}")
    print(f"baseline check: L1 {checks['L1_ER_actual']:.4f} "
          f"({'PASS' if checks['L1_ER_match'] else 'FAIL'}), "
          f"L1+L2 {checks['L1L2_ER_actual']:.4f} "
          f"({'PASS' if checks['L1L2_ER_match'] else 'FAIL'})")
    for entry in report["results"]:
        l1, l12 = entry["levels"][1], entry["levels"][2]
        print(f"  {entry['arm']:20s} L1 ER {l1['er']:8.4f}  L1+L2 ER {l12['er']:8.4f}  "
              f"L1 codes {entry['l1_distinct_codes']:3d}  eff {entry['l1_effective_usage_exp_H']:7.2f}  "
              f"coll {entry['full_3token_collision']:.4f}")


if __name__ == "__main__":
    main()
