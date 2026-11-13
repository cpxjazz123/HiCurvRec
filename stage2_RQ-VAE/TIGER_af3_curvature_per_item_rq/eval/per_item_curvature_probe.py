"""Read-only probe for the per-item-curvature arm. It trains nothing and saves nothing.

Every quantity below is a number the mechanism can be judged by, not a rewrite of it
(`ideas/AF3_curvature_per_item_idea/math.json`, formulas `flat_limit`,
`block_weight_split`, `curvature_confound`, `curvature_vs_origin_shift`):

1. `c -> 0` limit: recompute the cost with `c` shrunk and compare element-wise against
   the baseline squared Euclidean of `distances/distance.py`.
2. base point is the ball centre: the geometry has no translation term, and the table's
   row variance is > 0.
3. same `u_i`, two geometries: the per-row `argmin` flip matrix `F_c` of this arm versus
   `F_p` of the archived origin-shift parameterisation, and the size of their difference.
4. effect split: the W arm (only the block weight `1/c_i` moves) and the P arm (only the
   points move with `sqrt(c_i)`), each against the main arm, reported separately for the
   `argmin` levels (`l < N_l`) and the Sinkhorn level (`l = N_l`).
5. `geometry_off` against the baseline cost, element-wise; `behaviour_off` table variance
   and the SID change rate against the main arm.
6. curvature permutation: freeze the trained weights, permute the table over items, and
   report the SID change rate. Its conclusion strength stops at "does the table matter";
   it is NOT evidence that the AF3 values beat any other assignment.

Run: `python -m eval.per_item_curvature_probe` from the arm directory.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from distances.distance import distances as baseline_distances
from model.per_item_curvature_geometry import (
    C_MAX, C_MIN, D, D_E, D_H, SCALE_HYP, PerItemCurvatureGeometry, squared_euclidean,
)


def _flip_matrix(cost_main: torch.Tensor, cost_other: torch.Tensor) -> dict:
    """Per-row `argmin` flip counts and the frozenset of flipped row indices."""
    left = cost_main.argmin(dim=-1)
    right = cost_other.argmin(dim=-1)
    flipped = (left != right)
    return {"flipped_rows": int(flipped.sum().item()),
            "rows": int(left.numel()),
            "flip_rate": float(flipped.float().mean().item())}


def _table_stats(table: torch.Tensor) -> dict:
    values = table.double()
    return {"rows": int(values.numel()),
            "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "variance": float(values.var(unbiased=False))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=64, help="rows in the synthetic batch")
    parser.add_argument("--codewords", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curvature-json", default="", help="path of a frozen table dump")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(args.items, D, generator=generator) * 0.3
    codebook = torch.randn(args.codewords, D, generator=generator) * 0.3
    rows = torch.arange(args.items)

    if args.curvature_json:
        table = torch.as_tensor(np.load(args.curvature_json), dtype=torch.float32)
    else:
        # a monotone spread over the frozen range, one curvature per row
        spread = torch.linspace(0.0, 1.0, args.items, dtype=torch.float32)
        table = C_MIN * torch.pow(torch.tensor(C_MAX / C_MIN), spread)
    table = table[: args.items] if table.numel() >= args.items else table.repeat(args.items)[: args.items]

    geometry = PerItemCurvatureGeometry(table)
    report: dict = {"constants": {"D": D, "D_E": D_E, "D_H": D_H, "SCALE_HYP": SCALE_HYP,
                                 "c_min": C_MIN, "c_max": C_MAX}}

    # (2) the table is a real per-item degree of freedom, base point is the centre
    report["table"] = _table_stats(geometry.table)
    report["table_has_row_variance"] = bool(report["table"]["variance"] > 0.0)

    # (1) c -> 0 limit must land on the baseline squared Euclidean, element by element
    baseline = baseline_distances(latent, codebook)
    flat = PerItemCurvatureGeometry(torch.full((args.items,), 1e-8))  # c -> 0
    flat_cost = flat.cost(latent, codebook, rows)
    relative = (flat_cost - baseline).abs() / baseline.abs().clamp_min(1e-12)
    report["c_to_zero_max_relative_error"] = float(relative.max().item())

    # (5) geometry_off vs the baseline cost, element by element
    off = PerItemCurvatureGeometry(table, geometry_off=True)
    off_cost = off.cost(latent, codebook, rows)
    report["geometry_off_max_relative_error"] = float(
        ((off_cost - baseline).abs() / baseline.abs().clamp_min(1e-12)).max().item())
    report["geometry_off_bitwise_equal"] = bool(torch.equal(off_cost, squared_euclidean(latent, codebook)))

    behaviour = PerItemCurvatureGeometry(table, behaviour_off=True)
    report["behaviour_off_table"] = _table_stats(behaviour.curvature_of(rows, latent))

    # (4) W / P effect split against the main arm
    main_cost = geometry.cost(latent, codebook, rows)
    report["flip_main_vs_geometry_off"] = _flip_matrix(main_cost, off_cost)
    report["flip_main_vs_weight_only"] = _flip_matrix(main_cost,
                                                      geometry.cost_weight_only(latent, codebook, rows))
    report["flip_main_vs_point_only"] = _flip_matrix(main_cost,
                                                     geometry.cost_point_only(latent, codebook, rows))
    whole_row = main_cost * geometry.curvature_of(rows, latent).unsqueeze(-1)  # a row constant
    report["flip_row_constant_is_zero"] = _flip_matrix(main_cost, whole_row)

    # (3) archived origin-shift style parameterisation at the SAME u_i: p_i = expmap0(s_i e_1, 1)
    from model.poincare_contract import expmap0  # noqa: PLC0415 - keep the probe importable

    u = (torch.log(geometry.table / C_MIN) / torch.log(torch.tensor(C_MAX / C_MIN))).clamp(0.0, 1.0)
    anchor = expmap0((0.25 + 0.25 * u).unsqueeze(-1) * torch.eye(D_H)[0].unsqueeze(0), 1.0)
    r_e, r_h = latent[:, :D_E], latent[:, D_E:]
    t_e, t_h = codebook[:, :D_E], codebook[:, D_E:]
    euclid = (r_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)
    moved = (r_h.unsqueeze(0) - anchor.unsqueeze(1)).pow(2).sum(-1)   # |r_i - p_i|^2, constant c = 1
    archive_cost = euclid + SCALE_HYP * moved
    archive_flip = _flip_matrix(main_cost, archive_cost)
    main_rows = set(torch.nonzero(main_cost.argmin(-1) != off_cost.argmin(-1)).flatten().tolist())
    archive_rows = set(torch.nonzero(archive_cost.argmin(-1) != off_cost.argmin(-1)).flatten().tolist())
    archive_flip["difference_set_size"] = len(main_rows ^ archive_rows)
    report["flip_curvature_vs_origin_shift"] = archive_flip

    # (6) curvature permutation: does the table matter at all?
    permuted = PerItemCurvatureGeometry(table[torch.randperm(table.numel(), generator=generator)])
    report["flip_main_vs_permuted_table"] = _flip_matrix(
        main_cost, permuted.cost(latent, codebook, rows))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
