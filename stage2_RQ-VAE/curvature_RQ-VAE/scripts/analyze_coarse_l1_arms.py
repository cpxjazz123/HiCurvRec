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


def _level_containment(relative: str, level: int, geometry: str) -> dict:
    """Containment of one level's assigned code vectors in their category's cone.

    ``CategoryCone.evaluate`` reports the coarse cone against the level-2 code
    and the fine cone against the level-3 code, which is the historical pairing.
    This round moves the coarse cone to level 1, so the reading that matters -
    does the assigned level-1 code sit inside its coarse category's cone - has
    to be reconstructed from the checkpoint and the exported SIDs.
    """
    import torch
    from model.cones import containment_rate, to_point
    from model.category_cone import load_labels as _load

    directory = RESULTS / relative
    checkpoint = torch.load(
        directory / "out/rqvae/instruments/rqvae_best.pth", map_location="cpu"
    )
    state = checkpoint.get("state_dict", checkpoint)
    codes = state[f"rq.vq_layers.{level}.embed.weight"].float()
    prototypes = state["category_cone.prototypes.coarse"].float()
    prototypes_fine = state["category_cone.prototypes.fine"].float()
    k_proto = float(state["category_cone.k_proto"])
    k_q3 = float(state["category_cone.k_q3"])
    tokens = np.load(directory / "out/rqvae/instruments/sids_raw.npy")
    coarse_labels, fine_labels, _, _ = _load()
    curvature = torch.tensor(1.0)
    assigned = codes[torch.as_tensor(tokens[:, level], dtype=torch.long)]
    coarse = torch.as_tensor(coarse_labels, dtype=torch.long)
    fine = torch.as_tensor(fine_labels, dtype=torch.long)
    known = fine >= 0
    apex_coarse = prototypes[coarse[known]]
    apex_fine = prototypes_fine[fine[known]]
    points = assigned[known]
    out = {
        "coarse_cone_contains_assigned": float(
            containment_rate(geometry, curvature, apex_coarse, points, k_proto).float().mean()
        ),
        "fine_cone_contains_assigned": float(
            containment_rate(geometry, curvature, apex_fine, points, k_q3).float().mean()
        ),
    }
    # rejection: each item against another category's cone, deterministic shift
    order = torch.argsort(fine[known])
    counts = torch.unique_consecutive(fine[known][order], return_counts=True)[1]
    shift = int(counts.max().item()) if len(counts) else 0
    if shift:
        partner = order[torch.roll(torch.arange(int(known.sum())), -shift)]
        out["coarse_cone_negative"] = float(
            containment_rate(
                geometry, curvature, prototypes[coarse[known][partner]], points, k_proto
            ).float().mean()
        )
    return out


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

    print()
    print("level-1 code containment in its own coarse cone (this round's core metric)")
    for label, relative in ARMS:
        raw = RESULTS / relative / "out/rqvae/instruments/sids_raw.npy"
        if not raw.is_file():
            continue
        geometry = "euclid" if "euclid" in relative else "poincare"
        try:
            report = _level_containment(relative, 0, geometry)
        except (KeyError, FileNotFoundError) as error:
            print(f"{label:<30} unavailable: {type(error).__name__}")
            continue
        print(
            f"{label:<30} coarse={report['coarse_cone_contains_assigned']:.4f} "
            f"fine={report['fine_cone_contains_assigned']:.4f} "
            f"negative={report.get('coarse_cone_negative', float('nan')):.4f}"
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