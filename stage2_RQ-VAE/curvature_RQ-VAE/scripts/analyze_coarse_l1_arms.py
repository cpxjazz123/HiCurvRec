"""Read the coarse-on-level-1 round from its native outputs.

The mechanism moves the coarse cone onto level 1 and leaves level 2 to the
reconstruction objective. The reading that decides whether it did anything is
therefore per level, not per prefix: level 1 is the level that already carries
the category (fine-category AMI 0.3210 in the accepted model) and level 2 is the
level that carries item identity, which is what keeps the L1L2 prefix unique.
The accepted model's level 2 scores 0.0076, so an aligned level 2 would mean the
mechanism went to the wrong level.

The three numbers below are, in order: generalised containment on held-out
items, which is the only thing the mechanism is judged on; whether level 1's
category AMI rose without the prefix losing uniqueness, which is the transfer;
and Stage3, which is the decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG))

from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402
from category_structure import (  # noqa: E402
    codebook_sizes_from_checkpoint,
    structure_metrics,
)
from model.category_cone import load_labels  # noqa: E402

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE"
ARMS = (
    ("accepted bar (euclid L2=256)", "euclid_l2_256_72k_s42"),
    ("hyperbolic no cone", "."),
    ("euclid coarse-on-L1", "coarse_l1_arms/A_euclid_coarse_l1_s42"),
    ("hyperbolic coarse-on-L1", "coarse_l1_arms/B_poincare_coarse_l1_s42"),
)


def main() -> None:
    coarse, fine, _, _ = load_labels()
    print(f"{'arm':<30} {'L1 AMI(fine)':>12} {'L2 AMI(fine)':>12} "
          f"{'prefix AMI':>10} {'prefix uniq':>11} {'items':>7}")
    for label, relative in ARMS:
        directory = RESULTS / relative
        raw = directory / "out/rqvae/instruments/sids_raw.npy"
        if not raw.is_file():
            print(f"{label:<30} {'(no raw sids yet)':>12}")
            continue
        tokens = np.load(raw)
        sizes = codebook_sizes_from_checkpoint(directory)
        report = structure_metrics(tokens, coarse, fine, sizes)
        print(
            f"{label:<30} {adjusted_mutual_info_score(fine, tokens[:, 0]):>12.4f} "
            f"{adjusted_mutual_info_score(fine, tokens[:, 1]):>12.4f} "
            f"{report['fine_vs_L1L2']['ami']:>10.4f} "
            f"{len(np.unique(tokens[:, :2], axis=0)):>11} {len(tokens):>7}"
        )

    for label, relative in ARMS:
        metrics = RESULTS / relative / "logs/training_metrics.jsonl"
        if not metrics.is_file():
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        containment = [r for r in rows if r.get("event") == "cone_containment"]
        if not containment:
            continue
        print(f"\n{label}")
        print("   step | HELDOUT coarse_q2 pos/neg | HELDOUT fine_q3 pos/neg")
        for row in containment:
            held = row["heldout"]
            print(
                f"   {row['global_step']:6d} | "
                f"{held['coarse_q2_containment']:.4f} {held['coarse_q2_negative_containment']:.4f} | "
                f"{held['fine_q3_containment']:.4f} {held['fine_q3_negative_containment']:.4f}"
            )


if __name__ == "__main__":
    main()