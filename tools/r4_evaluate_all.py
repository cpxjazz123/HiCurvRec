"""Post-training evaluation for every R4 arm (#51-#60).

For each arm that has finished its 3000-epoch Stage 2 run, compute the frozen
held-out Hard L1 ER on column 0 of its exported ``item_sids.json``, evaluate the
fail-closed gate, and write ``stage2_metrics.json``, ``gate.json`` and ``report.md``.

    python3 tools/r4_evaluate_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "results/stage2_RQ-VAE/r4_common"))
import r4_gate as G  # noqa: E402

ARMS = {
    "r4_01": (51, "Candidate-popularity logit correction in baseline L1 InfoNCE",
              "logits'_ij = logits_ij - 0.5 * log q_train(j)"),
    "r4_02": (52, "Symmetric row-column L1 InfoNCE on frozen Resample pairs",
              "L1_new = 0.5 * CE_rows(S) + 0.5 * CE_rows(S.T)"),
    "r4_03": (53, "Inverse-anchor-frequency weighting of baseline L1 contrastive rows",
              "w(a) = (1 + f(a))**-0.5; L1_new = sum_i w(a_i) CE_i / sum_i w(a_i)"),
    "r4_04": (54, "Unique-transition-uniform sampling for baseline Resample behavior batches",
              "IID uniform draws with replacement from sorted unique train (history[-1], target) edges"),
    "r4_05": (55, "One-skip-history anchor in unchanged three-level Resample contrastive supervision",
              "source = history[-2] if len(history) >= 2 else history[-1]; target unchanged"),
    "r4_06": (56, "Unit-normalized cosine distance only for L1 contrastive logits",
              "S'_ij = -(1 - cos(z_i, z_j)) / 0.07 with eps=1e-8"),
    "r4_07": (57, "Huberized robust distance only for L1 behavior comparison",
              "D_ij = sqrt(2 * sum_k smooth_l1(a_i[k] - b_j[k], beta=1.0) + 1e-12)"),
    "r4_08": (58, "Encoder-only conflict projection for baseline L1 behavioral gradient",
              "PCGrad projection of g_L1 against g_base on encoder parameters only when <g_L1, g_base> < 0"),
    "r4_09": (59, "Target-branch stop-gradient in original L1 InfoNCE",
              "candidates = residuals[0, batch_size:].detach() at L1 only"),
    "r4_10": (60, "Train-target-frequency-weighted L1 KMeans initialization only",
              "L1 KMeans fit with sample_weight w_i = sqrt(1 + count_train_target(i)), mean-normalised"),
}


def green_passed(arm_dir: Path) -> bool:
    tdd = arm_dir / "tdd_red_green.md"
    if not tdd.is_file():
        return False
    text = tdd.read_text(encoding="utf-8")
    return "## GREEN" in text and "Exit code: 0" in text and "ALL TESTS PASSED" in text


def write_report(
    arm_dir: Path,
    arm: str,
    issue: int,
    title: str,
    formula: str,
    metrics: dict,
    gate: dict,
) -> None:
    cfg = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    verdict = "PASS — Stage 3 eligible" if gate["stage3_allowed"] else "FAIL — Stage 3 refused"
    lines = [
        f"# {arm.upper()} (Issue #{issue}) — {title}",
        "",
        "## 1. Sole mechanism and formula",
        "",
        f"- **Parent**: canonical Exp1a Resample (`stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE`, `EXPERIMENT_ARM = \"resample\"`).",
        f"- **Mechanism flag**: `{cfg['mechanism_flag']}`.",
        f"- **Formula**: `{formula}`.",
        f"- **Changed files vs parent**: `{', '.join(cfg['changed_files'])}`.",
        f"- **Trainer SHA256**: `{cfg['train_rqvae_sha256']}`.",
        "",
        "## 2. TDD RED → GREEN",
        "",
        f"Full verbatim RED (against the unchanged canonical Resample parent) and GREEN (against this arm's tree) logs, plus the single-delta diff, are recorded in [`tdd_red_green.md`](tdd_red_green.md). Runnable suite: `results/stage2_RQ-VAE/{arm}/test_{arm}.py` (`tests_passed = {gate['tests_passed']}`).",
        "",
        "## 3. Stage 2 training completion",
        "",
        f"- **Epochs completed**: `{metrics['train_end_epoch']}` / 3000 (`train_end_3000 = {gate['train_end_3000']}`).",
        f"- **Final checkpoint**: `results/stage2_RQ-VAE/{arm}/rqvae_best.pth` (`sha256 = {metrics['checkpoint_sha256']}`).",
        f"- **Final SIDs**: `results/stage2_RQ-VAE/{arm}/item_sids.json` (`sha256 = {metrics['item_sids_sha256']}`), `sids_for_hgrec.npy`.",
        f"- **Canonical reference SID SHA256**: `{cfg['baseline_sid_sha256']}`.",
        "",
        "## 4. Frozen held-out Hard L1 ER and Stage 3 gate",
        "",
        "Evaluated on `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` (52,855 positive transitions, 10 frozen matched negatives per row, seed 42), column 0 of the exported hard discrete SID only:",
        "",
        "| Quantity | Value |",
        "|:---|---:|",
        f"| `pos_share` | `{metrics['pos_share']:.10f}` |",
        f"| `neg_share` | `{metrics['neg_share']:.10f}` |",
        f"| **`ER_L1 = pos_share / neg_share`** | **`{metrics['ER_L1']:.14f}`** |",
        f"| Canonical Resample `ER_L1` | `{metrics['baseline_er_l1']:.14f}` |",
        f"| `Δ ER_L1` vs canonical | `{metrics['ER_L1'] - metrics['baseline_er_l1']:+.14f}` |",
        f"| Strict `ER_L1 > 16.38953942232631` | `{gate['ER_L1_above_baseline']}` |",
        f"| **`stage3_allowed`** | **`{gate['stage3_allowed']}`** ({verdict}) |",
        "",
        "## 5. Stage 3 disposition",
        "",
    ]
    if gate["stage3_allowed"]:
        lines.append(
            "Gate passed strictly. Stage 3 must be run under the frozen protocol (150 epochs, seed 42, beam 20, `NO_EVAL=True`, `SKIP_TEST=False`) before this issue can be closed."
        )
    else:
        lines.append(
            "Gate failed (`ER_L1` did not strictly exceed the canonical Resample baseline `16.38953942232631`). Per the issue contract, Stage 3 was **not** launched, queued, or prepared."
        )
    (arm_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    canonical_metrics, _ = G.evaluate(
        arm="resample",
        issue=0,
        item_sids_path=G.CANONICAL_SID_PATH,
        metrics_path=Path("/nonexistent"),
        tests_passed=True,
    )
    assert abs(canonical_metrics["ER_L1"] - G.BASELINE_ER_L1) < 1e-12, canonical_metrics

    summary = []
    for arm, (issue, title, formula) in ARMS.items():
        arm_dir = REPO_ROOT / "results/stage2_RQ-VAE" / arm
        sid_path = arm_dir / "item_sids.json"
        metrics_path = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "logs" / arm / "training_metrics.jsonl"
        passed = green_passed(arm_dir)
        metrics, gate = G.evaluate(
            arm=arm,
            issue=issue,
            item_sids_path=sid_path,
            metrics_path=metrics_path,
            tests_passed=passed,
        )
        (arm_dir / "stage2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (arm_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        write_report(arm_dir, arm, issue, title, formula, metrics, gate)
        summary.append((arm, issue, metrics["train_end_epoch"], metrics["ER_L1"], gate["stage3_allowed"]))
        print(
            f"{arm} (#{issue}): epoch={metrics['train_end_epoch']} "
            f"ER_L1={metrics['ER_L1']:.6f} vs {G.BASELINE_ER_L1:.6f} "
            f"stage3_allowed={gate['stage3_allowed']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
