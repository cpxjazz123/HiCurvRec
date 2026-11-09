"""Read the level-aligned round from its native outputs.

At 256 codes per level the L1L2 prefix is nearly unique, so the sharing gap that
served as the transfer reading at 64 codes has nothing left to measure. What
matters at this compression is whether the prefix carries the fine category at
all: the accepted model's prefix scores a fine-category AMI of 0.0001, which is
the headroom the mechanism is meant to fill, and the compressed configuration
reaches 0.0901 at the cost of the recall Stage3 measures.

So the readings here are, in order: generalised containment on held-out items,
positive and negative, which is the only thing the mechanism is judged on; then
whether the prefix became fine-semantic, which is the transfer; then Stage3,
which is the decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG))

from category_structure import (  # noqa: E402
    codebook_sizes_from_checkpoint,
    structure_metrics,
)
from model.category_cone import load_labels  # noqa: E402

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE"
ARMS = (
    ("euclid no cone (bar)", "euclid_l2_256_72k_s42"),
    ("hyperbolic no cone", None),
    ("euclid aligned", "aligned_arms/A_euclid_aligned_s42"),
    ("hyperbolic aligned", "aligned_arms/B_poincare_aligned_s42"),
)


def _raw_sids(relative: str) -> np.ndarray:
    """The three raw levels, the same array the historical AMI numbers used.

    ``item_sids.json`` carries a fourth column, the collision-extension id, so
    reading it here would score a different object than the 0.0001 and 0.0901
    the comparison is against.
    """
    return np.load(RESULTS / relative / "out/rqvae/instruments/sids_raw.npy")


def main() -> None:
    coarse, fine, _, _ = load_labels()
    print(f"{'arm':<24} {'fine AMI of L1L2':>17} {'full collision':>15} {'items':>7}")
    for label, relative in ARMS:
        if relative is None:
            continue
        path = RESULTS / relative / "item_sids.json"
        if not path.is_file():
            print(f"{label:<24} {'(no SIDs yet)':>17}")
            continue
        tokens = _raw_sids(relative)
        report = structure_metrics(
            tokens, coarse, fine, codebook_sizes_from_checkpoint(RESULTS / relative)
        )
        ami = report["fine_vs_L1L2"]["ami"]
        collision = report["full_sid_collision_rate"]
        print(f"{label:<24} {ami:>17.4f} {collision:>15.4f} {len(tokens):>7}")

    for label, relative in ARMS:
        if relative is None:
            continue
        metrics = RESULTS / relative / "logs/training_metrics.jsonl"
        if not metrics.is_file():
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        containment = [r for r in rows if r.get("event") == "cone_containment"]
        if not containment:
            print(f"\n{label}: no cone in this arm")
            continue
        print(f"\n{label}")
        print("   step | HELDOUT pos q2(fine)/q3 | HELDOUT neg q2(fine)/q3")
        for row in containment:
            held = row["heldout"]
            print(
                f"   {row['global_step']:6d} | "
                f"{held.get('fine_q2_containment', float('nan')):.4f} "
                f"{held.get('fine_q3_containment', float('nan')):.4f} | "
                f"{held.get('fine_q2_negative_containment', float('nan')):.4f} "
                f"{held.get('fine_q3_negative_containment', float('nan')):.4f}"
            )


if __name__ == "__main__":
    main()