"""Analyse the L2 codebook capacity ablation.

Reports, per arm, the four things a smaller second level can change: how many
L1+L2 prefixes survive, whether the sharing is semantic rather than indiscriminate
(same-category versus different-category sharing), whether L3 can still separate
the items that now share a prefix, and what reconstruction and codebook usage
cost.

Sharing is reported as absolute rates. The within/cross ratio is only meaningful
once the rates leave the 1e-6 range, and it is recorded alongside them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))

from category_structure import load_category_supervision, structure_metrics  # noqa: E402
from analyze_l2_assignment import prefix_ids, sharing  # noqa: E402

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/l2_capacity_arms"
SEED = 42
ARMS = [
    ("L2_256_baseline", 256),
    ("L2_128", 128),
    ("L2_64", 64),
    ("L2_32", 32),
]
L3_CODES = 256


def prefix_capacity(tokens: np.ndarray) -> dict:
    """How hard the third level must work to undo prefix sharing.

    L3 has one code per item inside a prefix, so a prefix holding more items than
    L3 has codes forces collisions whatever the third level learns. This separates
    "not enough capacity" from "the third level never bothered to spread".
    """
    l12 = prefix_ids(tokens, 2)
    sizes = np.unique(l12, return_counts=True)[1].astype(np.int64)
    return {
        "max_items_in_one_prefix": int(sizes.max()),
        "mean_items_per_prefix": float(sizes.mean()),
        "prefixes_over_l3_codes": int((sizes > L3_CODES).sum()),
        "items_in_prefixes_over_l3_codes": int(sizes[sizes > L3_CODES].sum()),
        "l3_codes": L3_CODES,
        "collision_forced": bool(sizes.max() > L3_CODES),
    }


def main() -> None:
    coarse, fine, _ = load_category_supervision()
    summary: dict = {"arms": {}}
    for name, l2 in ARMS:
        arm_dir = RESULTS / f"{name}_s{SEED}"
        tokens = np.load(arm_dir / "out/rqvae/instruments/sids_raw.npy")
        report = structure_metrics(tokens, coarse, fine)
        l12 = prefix_ids(tokens, 2)
        coarse_rows, fine_rows = coarse >= 0, fine >= 0
        coarse_sharing = sharing(
            tokens[coarse_rows], l12[coarse_rows], coarse[coarse_rows]
        )
        fine_sharing = sharing(
            tokens[fine_rows], l12[fine_rows], fine[fine_rows]
        )
        rows = [
            json.loads(line)
            for line in (arm_dir / "logs/training_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        final_train = [row for row in rows if row.get("event") == "train"][-1]
        usage = {
            level: {
                "used_codewords": block["used_codewords"],
                "gini": block["gini"],
                "normalized_entropy": block["normalized_entropy"],
            }
            for level, block in report["codeword_usage"].items()
        }
        summary["arms"][name] = {
            "l2_codes": l2,
            "structure": {
                "l1l2_prefix_clusters": report["l1l2_prefix_clusters"],
                "l1l2_prefix_singleton_fraction": report[
                    "l1l2_prefix_singleton_fraction"
                ],
                "full_sid_unique": report["full_sid_unique"],
                "full_sid_collision_rate": report["full_sid_collision_rate"],
                "coarse_vs_L1_ami": report["coarse_vs_L1"]["ami"],
                "coarse_vs_L1_ami_control": report["coarse_vs_L1"]["ami_control"],
                "fine_vs_L1L2_ami": report["fine_vs_L1L2"]["ami"],
            },
            "sharing_coarse": coarse_sharing,
            "sharing_fine": fine_sharing,
            "prefix_capacity": prefix_capacity(tokens),
            "codeword_usage": usage,
            "recon": final_train["recon"],
            "loss": final_train["loss"],
            "global_step": final_train["global_step"],
        }
        print(
            f"[cap] {name}: prefixes={report['l1l2_prefix_clusters']} "
            f"max_prefix={summary['arms'][name]['prefix_capacity']['max_items_in_one_prefix']} "
            f"singleton={report['l1l2_prefix_singleton_fraction']:.3f} "
            f"collision={report['full_sid_collision_rate']:.4f} | "
            f"within_coarse={coarse_sharing['within_share']:.6f} "
            f"cross_coarse={coarse_sharing['cross_share']:.6f} "
            f"ratio={coarse_sharing['within_over_cross']:.1f} | "
            f"fine_ami={report['fine_vs_L1L2']['ami']:.4f} "
            f"coarse_ami={report['coarse_vs_L1']['ami']:.4f} | "
            f"recon={final_train['recon']:.7f} "
            f"used={[b['used_codewords'] for b in usage.values()]} "
            f"gini={[round(b['gini'], 4) for b in usage.values()]}",
            flush=True,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\n[cap] wrote {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
