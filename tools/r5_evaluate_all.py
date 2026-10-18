"""Post-training Hard L1 ER gate evaluation for R5 arms (#61-#70)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "results/stage2_RQ-VAE/r5_common"))
import r5_gate as G  # noqa: E402

ARMS = {
    "r5_01": (61, "Reverse PCGrad: protect L1 behavior against base gradient",
              "project g_base against g_L1 when conflict"),
    "r5_02": (62, "Reconstruction-only reference gradient for original L1 PCGrad",
              "project g_L1 against g_recon"),
    "r5_03": (63, "Quantization-only reference gradient for original L1 PCGrad",
              "project g_L1 against g_quant"),
    "r5_04": (64, "Per-encoder-layer blockwise PCGrad",
              "PCGrad per Linear weight+bias block"),
    "r5_05": (65, "Half-strength conflict projection alpha=0.5",
              "g'_L1 = g_L1 - 0.5 * proj_component"),
    "r5_06": (66, "Norm-preserving post-PCGrad L1 behavior gradient",
              "rescale projected g_L1 to original norm"),
    "r5_07": (67, "Strong-conflict-only cosine trigger",
              "project only if cos < -0.2"),
    "r5_08": (68, "Late-epoch-only R4-08 conflict projection",
              "activate projection only for epoch >= 2250"),
    "r5_09": (69, "EMA-smoothed base gradient as PCGrad reference",
              "m_t=0.9 m + 0.1 g_base as reference"),
    "r5_10": (70, "Inter-level PCGrad against L2+L3",
              "project g_L1 against g_other"),
}


def green_passed(arm_dir: Path) -> bool:
    tdd = arm_dir / "tdd_red_green.md"
    if not tdd.is_file():
        return False
    text = tdd.read_text(encoding="utf-8")
    return "## GREEN" in text and "Exit code: 0" in text and "ALL TESTS PASSED" in text


def write_report(arm_dir, arm, issue, title, formula, metrics, gate):
    cfg = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    verdict = "PASS — Stage 3 eligible" if gate["stage3_allowed"] else "FAIL — Stage 3 refused"
    lines = [
        f"# {arm.upper()} (Issue #{issue}) — {title}", "",
        "## 1. Sole mechanism and formula", "",
        f"- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).",
        f"- **Mechanism flag**: `{cfg['mechanism_flag']}`.",
        f"- **Formula**: `{formula}`.",
        f"- **Changed files vs parent**: `{', '.join(cfg['changed_files'])}`.",
        f"- **Trainer SHA256**: `{cfg.get('train_rqvae_sha256')}`.", "",
        "## 2. TDD RED → GREEN", "",
        f"See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = {gate['tests_passed']}`.", "",
        "## 3. Stage 2 training completion", "",
        f"- **Epochs completed**: `{metrics['train_end_epoch']}` / 3000.",
        f"- **SID SHA256**: `{metrics['item_sids_sha256']}`.",
        f"- **Canonical Resample SID SHA256**: `{cfg['baseline_sid_sha256']}`.", "",
        "## 4. Frozen held-out Hard L1 ER and Stage 3 gate", "",
        "| Quantity | Value |", "|:---|---:|",
        f"| `pos_share` | `{metrics['pos_share']:.10f}` |",
        f"| `neg_share` | `{metrics['neg_share']:.10f}` |",
        f"| **`ER_L1`** | **`{metrics['ER_L1']:.14f}`** |",
        f"| Canonical Resample `ER_L1` | `{metrics['baseline_er_l1']:.14f}` |",
        f"| `Δ ER_L1` | `{metrics['ER_L1'] - metrics['baseline_er_l1']:+.14f}` |",
        f"| **`stage3_allowed`** | **`{gate['stage3_allowed']}`** ({verdict}) |", "",
        "## 5. Stage 3 disposition", "",
    ]
    if gate["stage3_allowed"]:
        lines.append("Gate passed. Stage 3 must run under the frozen protocol before closing.")
    else:
        lines.append("Gate failed. Stage 3 was not launched.")
    (arm_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for arm, (issue, title, formula) in ARMS.items():
        arm_dir = REPO_ROOT / "results/stage2_RQ-VAE" / arm
        metrics, gate = G.evaluate(
            arm=arm,
            issue=issue,
            item_sids_path=arm_dir / "item_sids.json",
            metrics_path=REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "logs" / arm / "training_metrics.jsonl",
            tests_passed=green_passed(arm_dir),
        )
        (arm_dir / "stage2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (arm_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        write_report(arm_dir, arm, issue, title, formula, metrics, gate)
        print(
            f"{arm} (#{issue}): epoch={metrics['train_end_epoch']} "
            f"ER_L1={metrics['ER_L1']:.6f} stage3_allowed={gate['stage3_allowed']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
