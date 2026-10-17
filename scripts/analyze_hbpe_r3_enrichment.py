"""Evaluation and Stage 3 Gate automation for HBPE-R3 arms (#37-#46).

Evaluates on the fixed independent validation set:
  results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz
  (52,855 held-out valid transitions, 10 matched negatives per anchor, seed 42)

Enforces the mandatory Stage 3 Gate:
  stage3_allowed = (finite(L1_ER) && L1_ER > 16.38953942232631 && finite(L1L2_ER) && L1L2_ER > 58.38709677419355)
Both must be strictly greater than canonical Resample constants.
If false, writes gate.json, reports Stage 2 results, and closes issue.
If true, leaves issue open for Stage 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_PAIRS_PATH = REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
BASE_RESAMPLE_SID = REPO_ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
BASE_A1R2_W010_SID = REPO_ROOT / "results/stage2_RQ-VAE/TIGER_HBPE_A1_R2_W010/a1r2_w010/item_sids.json"

BASELINE_L1_ER = 16.38953942232631
BASELINE_L1L2_ER = 58.38709677419355
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95

R3_EXPERIMENTS = {
    "01": {"issue": 37, "name": "HBPE-R3-01", "desc": "Hard-ST prefix ranking"},
    "02": {"issue": 38, "name": "HBPE-R3-02", "desc": "Top-2 sparse boundary assignment"},
    "03": {"issue": 39, "name": "HBPE-R3-03", "desc": "Multi-successor supervised prefix contrast"},
    "04": {"issue": 40, "name": "HBPE-R3-04", "desc": "Target-popularity matched negative sampling"},
    "05": {"issue": 41, "name": "HBPE-R3-05", "desc": "Same-prefix semi-hard negatives"},
    "06": {"issue": 42, "name": "HBPE-R3-06", "desc": "Encoder gradient budget"},
    "07": {"issue": 43, "name": "HBPE-R3-07", "desc": "Text-semantic neighborhood preservation"},
    "08": {"issue": 44, "name": "HBPE-R3-08", "desc": "L1-conditional L2 enrichment"},
    "09": {"issue": 45, "name": "HBPE-R3-09", "desc": "Within-run L1 assignment trust region"},
    "10": {"issue": 46, "name": "HBPE-R3-10", "desc": "Degree-corrected graph modularity"},
}


def load_sid_matrix(path: Path) -> tuple[np.ndarray, str, int]:
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    sids_dict = json.loads(raw_bytes.decode("utf-8"))
    n_items = len(sids_dict)
    max_id = max(int(k) for k in sids_dict.keys())
    matrix = np.zeros((max_id + 1, 4), dtype=np.int64)
    for k, v in sids_dict.items():
        matrix[int(k)] = v
    return matrix, sha, n_items


def evaluate_sid(
    sid_mat: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    draws: np.ndarray,
) -> dict[str, Any]:
    levels = {}
    for depth in (1, 2):
        if depth == 1:
            a = sid_mat[anchors, 0]
            p = sid_mat[positives, 0]
            n = sid_mat[negatives, 0]
            pos_m = (a == p)
            neg_m = (a[:, None] == n)
        else:
            a = sid_mat[anchors, :2]
            p = sid_mat[positives, :2]
            n = sid_mat[negatives, :2]
            pos_m = np.all(a == p, axis=1)
            neg_m = np.all(a[:, None, :] == n, axis=2)

        pos_share = float(pos_m.mean())
        neg_share = float(neg_m.mean())
        gap = pos_share - neg_share
        er = float(pos_share / neg_share) if neg_share > 0 else float("nan")

        # Bootstrap
        n_rows = len(pos_m)
        neg_row_mean = neg_m.mean(axis=1)
        p_samples = pos_m[draws].mean(axis=1)
        n_samples = neg_row_mean[draws].mean(axis=1)
        boot_ers = np.where(n_samples > 0, p_samples / n_samples, np.nan)
        finite_ers = boot_ers[np.isfinite(boot_ers)]
        alpha = 1.0 - CONFIDENCE
        ci = [float(np.quantile(finite_ers, alpha / 2.0)), float(np.quantile(finite_ers, 1.0 - alpha / 2.0))]

        levels[f"L{depth}" if depth == 1 else "L1L2"] = {
            "pos_share": pos_share,
            "neg_share": neg_share,
            "gap": gap,
            "er": er,
            "ci95": ci,
            "_boot_ers": boot_ers,
        }

    # Codebook diagnostics
    tokens_l1 = sid_mat[:, 0]
    counts = np.bincount(tokens_l1, minlength=int(tokens_l1.max()) + 1).astype(np.float64)
    share = counts / counts.sum()
    nonzero = share[share > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    l123_unique = int(len(np.unique(sid_mat[:, :3], axis=0)))

    return {
        "l1_distinct_codes": int((counts > 0).sum()),
        "l1_effective_usage_exp_H": float(np.exp(entropy)),
        "l1_entropy_nats": entropy,
        "l12_distinct_prefixes": int(len(np.unique(sid_mat[:, :2], axis=0))),
        "full_3token_collision": float(1.0 - l123_unique / len(sid_mat)),
        "levels": levels,
    }


def compute_paired_delta(cand_ers: np.ndarray, ref_ers: np.ndarray) -> dict[str, Any]:
    deltas = cand_ers - ref_ers
    finite = deltas[np.isfinite(deltas)]
    alpha = 1.0 - CONFIDENCE
    ci = [float(np.quantile(finite, alpha / 2.0)), float(np.quantile(finite, 1.0 - alpha / 2.0))]
    point = float(finite.mean())
    sig = bool((ci[0] > 0.0) or (ci[1] < 0.0))
    return {"delta": point, "ci95": ci, "significant": sig}


def process_r3_arm(nn: str, auto_close: bool = True) -> dict[str, Any]:
    info = R3_EXPERIMENTS[nn]
    issue_num = info["issue"]
    arm_name = f"hbpe_r3_{nn}"
    tree_dir = REPO_ROOT / f"stage2_RQ-VAE/TIGER_HBPE_R3_{nn}"
    result_arm_dir = REPO_ROOT / f"results/stage2_RQ-VAE/TIGER_HBPE_R3_{nn}/{arm_name}"
    delivery_dir = REPO_ROOT / f"results/stage2_RQ-VAE/hbpe_r3_{nn}"
    delivery_dir.mkdir(parents=True, exist_ok=True)

    sid_path = result_arm_dir / "item_sids.json"
    if not sid_path.is_file():
        raise FileNotFoundError(f"SID file missing for {arm_name}: {sid_path}")

    data = np.load(VALID_PAIRS_PATH)
    anchors = data["anchors"].astype(np.int64)
    positives = data["positives"].astype(np.int64)
    negatives = data["negatives"].astype(np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(anchors), size=(BOOTSTRAP_REPLICATES, len(anchors)))

    # Evaluate candidate
    cand_mat, cand_sha, n_items = load_sid_matrix(sid_path)
    cand_metrics = evaluate_sid(cand_mat, anchors, positives, negatives, draws)

    # Evaluate baselines
    ref_mat, _, _ = load_sid_matrix(BASE_RESAMPLE_SID)
    ref_metrics = evaluate_sid(ref_mat, anchors, positives, negatives, draws)

    a1_mat, _, _ = load_sid_matrix(BASE_A1R2_W010_SID)
    a1_metrics = evaluate_sid(a1_mat, anchors, positives, negatives, draws)

    # Paired deltas
    delta_vs_resample = {
        "L1": compute_paired_delta(cand_metrics["levels"]["L1"]["_boot_ers"], ref_metrics["levels"]["L1"]["_boot_ers"]),
        "L1L2": compute_paired_delta(cand_metrics["levels"]["L1L2"]["_boot_ers"], ref_metrics["levels"]["L1L2"]["_boot_ers"]),
    }
    delta_vs_a1w010 = {
        "L1": compute_paired_delta(cand_metrics["levels"]["L1"]["_boot_ers"], a1_metrics["levels"]["L1"]["_boot_ers"]),
        "L1L2": compute_paired_delta(cand_metrics["levels"]["L1L2"]["_boot_ers"], a1_metrics["levels"]["L1L2"]["_boot_ers"]),
    }

    l1_er = cand_metrics["levels"]["L1"]["er"]
    l1l2_er = cand_metrics["levels"]["L1L2"]["er"]

    # GATE CONDITION: BOTH strictly greater than canonical Resample constants
    stage3_allowed = bool(
        np.isfinite(l1_er) and l1_er > BASELINE_L1_ER
        and np.isfinite(l1l2_er) and l1l2_er > BASELINE_L1L2_ER
    )

    reason = (
        f"Stage 2 Hard L1 ER={l1_er:.6f} (baseline={BASELINE_L1_ER:.6f}), "
        f"Hard L1+L2 ER={l1l2_er:.6f} (baseline={BASELINE_L1L2_ER:.6f}). "
        + ("BOTH exceed baseline strictly -> Stage 3 allowed." if stage3_allowed else "Failed dual threshold gate -> Stage 3 refused.")
    )

    gate_payload = {
        "issue": issue_num,
        "arm": arm_name,
        "mechanism": info["desc"],
        "stage3_allowed": stage3_allowed,
        "hard_l1_er": l1_er,
        "hard_l1l2_er": l1l2_er,
        "baseline_l1_er": BASELINE_L1_ER,
        "baseline_l1l2_er": BASELINE_L1L2_ER,
        "l1_exceeds_baseline": bool(l1_er > BASELINE_L1_ER),
        "l1l2_exceeds_baseline": bool(l1l2_er > BASELINE_L1L2_ER),
        "reason": reason,
    }
    (delivery_dir / "gate.json").write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")

    # Clean boot arrays for json
    for lvl in ("L1", "L1L2"):
        cand_metrics["levels"][lvl].pop("_boot_ers", None)
        ref_metrics["levels"][lvl].pop("_boot_ers", None)
        a1_metrics["levels"][lvl].pop("_boot_ers", None)

    stage2_metrics_payload = {
        "issue": issue_num,
        "arm": arm_name,
        "mechanism": info["desc"],
        "sid_sha256": cand_sha,
        "metrics": cand_metrics,
        "delta_vs_resample": delta_vs_resample,
        "delta_vs_a1w010": delta_vs_a1w010,
    }
    (delivery_dir / "stage2_metrics.json").write_text(json.dumps(stage2_metrics_payload, indent=2), encoding="utf-8")

    # Copy artifacts to delivery directory
    for f in ("item_sids.json", "sids_for_hgrec.npy", "rqvae_best.pth"):
        if (result_arm_dir / f).is_file():
            shutil.copy2(result_arm_dir / f, delivery_dir / f)

    # Markdown report
    md_lines = [
        f"# {info['name']} (#{issue_num}) — {info['desc']}",
        "",
        "## Stage 2 Results and Conditional Stage 3 Gate",
        "",
        f"- **Arm**: `{arm_name}`",
        f"- **SID SHA256**: `{cand_sha}`",
        f"- **L1 distinct codes**: {cand_metrics['l1_distinct_codes']} (effective `exp(H)`: {cand_metrics['l1_effective_usage_exp_H']:.2f})",
        f"- **L1+L2 distinct prefixes**: {cand_metrics['l12_distinct_prefixes']}",
        f"- **Full 3-token collision**: {cand_metrics['full_3token_collision']:.4f}",
        "",
        "### Held-Out Behavioral Prefix Enrichment",
        "",
        "| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |",
        "|:---:|---:|---:|---:|---:|:---:|---:|:---:|",
        f"| L1 | {cand_metrics['levels']['L1']['pos_share']:.5f} | {cand_metrics['levels']['L1']['neg_share']:.5f} | {cand_metrics['levels']['L1']['gap']:+.5f} | **{l1_er:.4f}** | [{cand_metrics['levels']['L1']['ci95'][0]:.2f}, {cand_metrics['levels']['L1']['ci95'][1]:.2f}] | {BASELINE_L1_ER:.4f} | {delta_vs_resample['L1']['delta']:+.4f} [{delta_vs_resample['L1']['ci95'][0]:+.2f}, {delta_vs_resample['L1']['ci95'][1]:+.2f}] |",
        f"| L1+L2 | {cand_metrics['levels']['L1L2']['pos_share']:.5f} | {cand_metrics['levels']['L1L2']['neg_share']:.5f} | {cand_metrics['levels']['L1L2']['gap']:+.5f} | **{l1l2_er:.4f}** | [{cand_metrics['levels']['L1L2']['ci95'][0]:.2f}, {cand_metrics['levels']['L1L2']['ci95'][1]:.2f}] | {BASELINE_L1L2_ER:.4f} | {delta_vs_resample['L1L2']['delta']:+.4f} [{delta_vs_resample['L1L2']['ci95'][0]:+.2f}, {delta_vs_resample['L1L2']['ci95'][1]:+.2f}] |",
        "",
        "### Stage 3 Gate Verdict",
        "",
        f"- **Verdict**: `stage3_allowed = {str(stage3_allowed).lower()}`",
        f"- **Reason**: {reason}",
        "",
    ]
    if not stage3_allowed:
        md_lines.extend([
            "**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. "
            "Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.",
        ])
    else:
        md_lines.extend([
            "**Gate passed**: Both levels strictly exceeded the baseline. Stage 3 may proceed.",
        ])

    (delivery_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Post to issue and close if failed
    if auto_close and not stage3_allowed:
        try:
            cmd = ["gh", "issue", "comment", str(issue_num), "--repo", "cpxjazz123/HiCurvRec", "--body-file", str(delivery_dir / "README.md")]
            subprocess.run(cmd, check=True)
            cmd_close = ["gh", "issue", "close", str(issue_num), "--repo", "cpxjazz123/HiCurvRec", "--reason", "completed"]
            subprocess.run(cmd_close, check=True)
            print(f"Posted report and closed Issue #{issue_num}")
        except Exception as e:
            print(f"Warning: gh issue action failed for #{issue_num}: {e}")

    return gate_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, help="Arm number: 01 to 10")
    args = parser.parse_args()
    res = process_r3_arm(args.arm)
    print(json.dumps(res, indent=2))
