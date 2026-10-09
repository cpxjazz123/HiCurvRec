"""Read the smoothness round, including a containment reading it does not train.

The mechanism is a term on the quantiser, not a cone, so nothing in the training
loop emits a containment number and the arm's own metrics cannot supply one.
Containment is still the reading the program is judged on, so it is measured
here on the trained codebooks: each category's prototype is fitted as the mean
of its items' code vectors at the level that category's cone would supervise,
the aperture is calibrated the same way the trained arms calibrate theirs, and
the held-out positive and negative rates are then read off.

That is a fitted cone, not a trained one, and the two are not the same object -
the fitted version asks how cone-compatible the codebook turned out to be rather
than what a cone pushed it to. The distinction is stated with the numbers.

Per-level category AMI and prefix uniqueness come along because this round's
transfer question is whether making the quantiser smooth also made it
category-aligned, which is the thing that has cost every previous round.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG))

from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402
from category_structure import (  # noqa: E402
    codebook_sizes_from_checkpoint,
    structure_metrics,
)
from model.category_cone import load_labels  # noqa: E402
from model.cones import aperture, containment_rate, k_for_aperture, to_point  # noqa: E402

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE"
ARMS = (
    ("accepted bar (euclid no cone)", "euclid_l2_256_72k_s42", "euclid"),
    ("hyperbolic no cone", ".", "poincare"),
    ("euclid smooth", "smoothness_arms/A_euclid_smooth_s42", "euclid"),
    ("hyperbolic smooth", "smoothness_arms/B_poincare_smooth_s42", "poincare"),
)
APERTURE_DEGREES = 20.0


def _heldout_split(fine: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    held = np.zeros(len(fine), dtype=bool)
    for node in np.unique(fine):
        if node < 0:
            continue
        members = np.flatnonzero(fine == node)
        if len(members) < 2:
            continue
        held[rng.choice(members, size=max(1, int(round(len(members) * fraction))), replace=False)] = True
    return held


def _fitted_containment(directory: Path, geometry: str) -> dict:
    state = torch.load(
        directory / "out/rqvae/instruments/rqvae_best.pth", map_location="cpu"
    )
    state = state.get("state_dict", state)
    codes = [
        state[f"rq.vq_layers.{level}.embed.weight"].float() for level in range(3)
    ]
    tokens = np.load(directory / "out/rqvae/instruments/sids_raw.npy")
    coarse_labels, fine_labels, n_coarse, n_fine = load_labels()
    curvature = torch.tensor(1.0)
    held = _heldout_split(fine_labels, 0.10, 2026)

    def mean_code(labels: np.ndarray, level: int, count: int) -> torch.Tensor:
        assigned = codes[level][torch.as_tensor(tokens[:, level], dtype=torch.long)]
        out = torch.zeros(count, assigned.shape[-1])
        for node in range(count):
            members = np.flatnonzero(labels == node)
            if len(members):
                out[node] = assigned[torch.as_tensor(members, dtype=torch.long)].mean(0)
        return out

    # the pairing the trained arms use: coarse cone on level 2, fine cone on level 3
    coarse_proto = mean_code(coarse_labels, 1, n_coarse)
    fine_proto = mean_code(fine_labels, 2, n_fine)
    coarse_points = to_point(geometry, coarse_proto, curvature)
    fine_points = to_point(geometry, fine_proto, curvature)
    k_coarse = k_for_aperture(geometry, coarse_points, APERTURE_DEGREES)
    k_fine = k_for_aperture(geometry, fine_points, APERTURE_DEGREES)

    index = torch.as_tensor(np.flatnonzero(held & (fine_labels >= 0)), dtype=torch.long)
    order = np.argsort(fine_labels[held & (fine_labels >= 0)])
    pool_fine = fine_labels[held & (fine_labels >= 0)][order]
    counts = np.unique(pool_fine, return_counts=True)[1]
    shift = int(counts.max()) if len(counts) else 0
    partner = torch.as_tensor(order[np.roll(np.arange(len(index)), -shift)], dtype=torch.long) if shift else torch.arange(len(index))
    item_coarse = torch.as_tensor(coarse_labels, dtype=torch.long)[index]
    item_fine = torch.as_tensor(fine_labels, dtype=torch.long)[index]
    level2 = codes[1][torch.as_tensor(tokens[:, 1], dtype=torch.long)][index]
    level3 = codes[2][torch.as_tensor(tokens[:, 2], dtype=torch.long)][index]
    other_coarse = torch.as_tensor(coarse_labels, dtype=torch.long)[index[partner]]
    other_fine = torch.as_tensor(fine_labels, dtype=torch.long)[index[partner]]
    return {
        "aperture_deg": (APERTURE_DEGREES, APERTURE_DEGREES),
        "coarse_positive": float(containment_rate(geometry, curvature, coarse_proto[item_coarse], level2, k_coarse).float().mean()),
        "coarse_negative": float(containment_rate(geometry, curvature, coarse_proto[other_coarse], level2, k_coarse).float().mean()),
        "fine_positive": float(containment_rate(geometry, curvature, fine_proto[item_fine], level3, k_fine).float().mean()),
        "fine_negative": float(containment_rate(geometry, curvature, fine_proto[other_fine], level3, k_fine).float().mean()),
    }


def main() -> None:
    coarse, fine, _, _ = load_labels()
    print(f"{'arm':<30} {'L1 AMI':>8} {'L2 AMI':>8} {'L3 AMI':>8} {'prefix uniq':>11} "
          f"{'coarse pos/neg':>16} {'fine pos/neg':>14}")
    for label, relative, geometry in ARMS:
        directory = RESULTS / relative
        raw = directory / "out/rqvae/instruments/sids_raw.npy"
        if not raw.is_file():
            print(f"{label:<30} {'(no sids yet)':>8}")
            continue
        tokens = np.load(raw)
        report = structure_metrics(tokens, coarse, fine, codebook_sizes_from_checkpoint(directory))
        try:
            containment = _fitted_containment(directory, geometry)
            cc = f"{containment['coarse_positive']:.4f}/{containment['coarse_negative']:.4f}"
            fc = f"{containment['fine_positive']:.4f}/{containment['fine_negative']:.4f}"
        except (KeyError, FileNotFoundError):
            cc = fc = "n/a"
        print(
            f"{label:<30} {adjusted_mutual_info_score(fine, tokens[:, 0]):>8.4f} "
            f"{adjusted_mutual_info_score(fine, tokens[:, 1]):>8.4f} "
            f"{adjusted_mutual_info_score(fine, tokens[:, 2]):>8.4f} "
            f"{len(np.unique(tokens[:, :2], axis=0)):>11} {cc:>16} {fc:>14}"
        )
    print("\ncontainment here is read off a cone fitted to the trained codebooks; the")
    print("trained arms' cones were optimised instead, so the two are not the same object.")


if __name__ == "__main__":
    main()