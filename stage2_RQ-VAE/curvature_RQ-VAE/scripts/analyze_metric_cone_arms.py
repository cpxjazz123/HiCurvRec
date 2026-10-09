"""Read the metric-margin cone round from its native outputs.

Three readings, in the order the round's claim needs them.

1. Generalised containment, on held-out items, positive and negative, which is
   the only thing the mechanism is allowed to be judged on. The training
   diagnostics measure the supervised batch, so they cannot answer this.
2. Whether the geometric difference reached the discrete code assignment: the
   assigned level-2 code is what the cone objective acts on, so a containment
   advantage that never changes which code an item receives has not transferred.
3. The sharing structure of those codes inside a coarse category, which is the
   link between the assignment and Stage3: raising same-fine sharing helps
   recommendation while raising different-fine sharing merges categories.

Sharing is reported as a diagnostic, never as a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))

from prefix_sharing_detail import fine_sharing_within_l1  # noqa: E402
sys.path.insert(0, str(PKG))
from model.category_cone import load_labels  # noqa: E402

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE"
ARMS = (
    ("euclid", "metric_cone_arms/A_euclid_metriccone_s42"),
    ("poincare", "metric_cone_arms/B_poincare_metriccone_s42"),
)


def _metrics(arm_dir: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (arm_dir / "logs/training_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    containment = [r for r in rows if r.get("event") == "cone_containment"]
    calibration = [r for r in rows if r.get("event") == "cone_calibration"]
    return {
        "containment": containment,
        "calibration": calibration[0] if calibration else {},
        "train_end": [r for r in rows if r.get("event") == "train_end"],
    }


def _supervision() -> tuple[np.ndarray, np.ndarray]:
    coarse, fine, _, _ = load_labels()
    return coarse, fine


def main() -> None:
    coarse, fine = _supervision()
    print(f"labelled items: {(fine >= 0).sum()} of {len(fine)}")
    for name, relative in ARMS:
        arm_dir = RESULTS / relative
        if not (arm_dir / "item_sids.json").is_file():
            print(f"\n== {name}: no SIDs yet")
            continue
        report = _metrics(arm_dir)
        print(f"\n== {name}  ({relative})")
        cal = report["calibration"]
        print(
            f"   calibration: k_q2={cal.get('k_q2')} k_q3={cal.get('k_q3')} "
            f"aperture_proto={cal.get('aperture_deg_proto')}deg "
            f"aperture_fine={cal.get('aperture_deg_fine_proto')}deg "
            f"heldout={cal.get('heldout_items')}"
        )
        print(
            "   step | HELDOUT pos q2/q3 | HELDOUT neg q2/q3 | sup pos q2/q3"
        )
        for row in report["containment"]:
            held = row.get("heldout", {})
            sup = row.get("supervised", {})
            print(
                f"   {row['global_step']:6d} | "
                f"{held.get('coarse_q2_containment', float('nan')):.4f} "
                f"{held.get('fine_q3_containment', float('nan')):.4f} | "
                f"{held.get('coarse_q2_negative_containment', float('nan')):.4f} "
                f"{held.get('fine_q3_negative_containment', float('nan')):.4f} | "
                f"{sup.get('coarse_q2_containment', float('nan')):.4f} "
                f"{sup.get('fine_q3_containment', float('nan')):.4f}"
            )
        tokens = np.array(
            [
                row
                for _, row in sorted(
                    json.loads((arm_dir / "item_sids.json").read_text()).items(),
                    key=lambda pair: int(pair[0]),
                )
            ],
            dtype=np.int64,
        )
        sharing = fine_sharing_within_l1(tokens, coarse, fine)
        print(
            "   L2 sharing inside one coarse category: "
            f"same-fine {sharing['within_l1_same_fine_share']:.4f} | "
            f"different-fine {sharing['within_l1_different_fine_share']:.4f} | "
            f"gap {sharing['within_l1_same_fine_share'] - sharing['within_l1_different_fine_share']:+.4f}"
        )


if __name__ == "__main__":
    main()