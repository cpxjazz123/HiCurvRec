"""Category cone supervision on the residual-quantized prefixes.

Assembles the quantized cone objective

    L_cone = E(P_coarse, P_fine) + E(P_coarse, Q2) + E(P_fine, Q3)
             + radial_weight * L_radial

for either arm. The prototypes are learnable tangent-space parameters shared by
both arms (30 coarse and 59 fine of dimension 32); the two arms differ only in
the metric the energies are measured in.

Negatives are drawn inside the batch, so a negative Q2/Q3 is a representation the
model actually computed this step, and they are a deterministic function of the
step index and the batch's item ids - the two arms therefore train on exactly the
same negative partners. ``E(P_fine, Q3)`` prefers a partner from the same coarse
category but a different fine category, which is the comparison that tests
fine-grained discrimination, and falls back to another coarse category when the
batch holds no sibling fine category.

Aperture constants are fitted per arm on the supervised items only; the held-out
items never take part in the fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .cones import (
    CategoryPrototypes,
    aperture,
    k_for_aperture,
    aperture_factor,
    containment_loss,
    containment_rate,
    energy,
    from_point,
    radial_order_loss,
    to_point,
)

SUPERVISION_PATH = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE"
    "/dataset/Instruments/category_supervision.npz"
)


def load_labels() -> tuple[np.ndarray, np.ndarray, int, int]:
    """item -> (coarse, fine) node ids, with -1 where the label is unknown."""
    if not SUPERVISION_PATH.is_file():
        raise FileNotFoundError(
            "Category supervision cache missing. Build it with "
            "python stage2_RQ-VAE/curvature_RQ-VAE/scripts/category_structure.py"
        )
    with np.load(SUPERVISION_PATH, allow_pickle=True) as saved:
        coarse = saved["coarse"].astype(np.int64)
        fine = saved["fine"].astype(np.int64)
    return coarse, fine, int(coarse.max()) + 1, int(fine.max()) + 1


def stratified_holdout(
    fine: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split items into supervised / held-out, stratified by fine category.

    Stratifying guarantees a small category still contributes held-out items
    instead of vanishing from the split. The held-out items keep training the
    RQ-VAE itself; they are only excluded from the cone supervision, so the
    reading this produces is the generalisation of the category supervision,
    not of an unseen catalogue.
    """
    rng = np.random.default_rng(seed)
    held_out = np.zeros(len(fine), dtype=bool)
    for node in np.unique(fine):
        if node < 0:
            continue
        members = np.flatnonzero(fine == node)
        if len(members) < 2:
            continue
        count = max(1, int(round(len(members) * fraction)))
        held_out[rng.choice(members, size=count, replace=False)] = True
    return ~held_out, held_out


class CategoryCone(nn.Module):
    """Prototypes, negative partners and the cone objective."""

    def __init__(
        self,
        geometry: str,
        codebook_dim: int,
        holdout_fraction: float,
        data_seed: int,
        margin: float,
        radial_weight: float,
        radial_margin: float,
        device: torch.device,
        radius_band_low: float = 0.6,
        radius_band_high: float = 1.4,
    ) -> None:
        super().__init__()
        self.radius_band_low = float(radius_band_low)
        self.radius_band_high = float(radius_band_high)
        self.geometry = geometry
        self.margin = float(margin)
        self.radial_weight = float(radial_weight)
        self.radial_margin = float(radial_margin)
        self.data_seed = int(data_seed)

        coarse, fine, n_coarse, n_fine = load_labels()
        self.register_buffer("coarse_of_item", torch.as_tensor(coarse, device=device))
        self.register_buffer("fine_of_item", torch.as_tensor(fine, device=device))
        self.supervised, self.held_out = stratified_holdout(
            fine, holdout_fraction, data_seed
        )
        self.supervised_index = np.flatnonzero(self.supervised)
        self.heldout_index = np.flatnonzero(self.held_out)
        self.labelled_index = np.flatnonzero(fine >= 0)

        # parent coarse of every fine node, for the prototype-level negative
        parent = np.full(n_fine, -1, dtype=np.int64)
        for node in range(n_fine):
            members = np.flatnonzero(fine == node)
            if len(members):
                parent[node] = coarse[members[0]]
        self.register_buffer(
            "coarse_of_fine", torch.as_tensor(parent, device=device)
        )
        self.fine_children = [
            (node, np.flatnonzero(parent == node))
            for node in range(n_coarse)
            if int((parent == node).sum())
        ]

        generator = torch.Generator(device=device).manual_seed(data_seed)
        self.prototypes = CategoryPrototypes(
            torch.arange(n_coarse, device=device),
            torch.arange(n_fine, device=device),
            codebook_dim,
            reference_radius=1.0,
            coarse_radius_ratio=0.5,
            generator=generator,
            device=device,
        )
        # Radius band, expressed in the arm's own point space and relative to
        # each prototype's initial radius so the rule is identical in both arms.
        # Without it the cone has a degenerate attractor: the cheapest way to
        # pull positives into the cone is to move the apex toward the origin,
        # where g = (1-r^2)/r diverges, so the aperture silently saturates at
        # pi/2 and starts swallowing the negatives too.
        curvature_unit = torch.ones((), device=device)
        with torch.no_grad():
            coarse_radius = to_point(
                geometry, self.prototypes.coarse.detach(), curvature_unit
            ).norm(dim=-1)
            fine_radius = to_point(
                geometry, self.prototypes.fine.detach(), curvature_unit
            ).norm(dim=-1)
        self.register_buffer("init_coarse_radius", coarse_radius)
        self.register_buffer("init_fine_radius", fine_radius)

        self.register_buffer("k_proto", torch.zeros((), device=device))
        self.register_buffer("k_q2", torch.zeros((), device=device))
        self.register_buffer("k_q3", torch.zeros((), device=device))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool, device=device))

    @torch.no_grad()
    def project_prototypes(self, curvature: torch.Tensor) -> None:
        """Hold each prototype inside its relative radius band."""
        for parameter, initial in (
            (self.prototypes.coarse, self.init_coarse_radius),
            (self.prototypes.fine, self.init_fine_radius),
        ):
            points = to_point(self.geometry, parameter, curvature)
            norm = points.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            lower = self.radius_band_low * initial.unsqueeze(-1)
            upper = self.radius_band_high * initial.unsqueeze(-1)
            scaled = (norm.clamp(min=lower, max=upper) / norm)
            parameter.copy_(
                from_point(self.geometry, points * scaled, curvature)
            )

    # ---------------------------------------------------------------- sampling
    def _partners(
        self,
        coarse: np.ndarray,
        fine: np.ndarray,
        seed: int,
        mode: str,
    ) -> np.ndarray:
        """Pick a negative partner per row, inside the batch, deterministically.

        ``q2`` rejects only a shared coarse category. ``q3`` rejects a shared
        coarse *and* fine category, so a sibling fine category is preferred and
        another coarse category is the fallback.

        The implementation groups the rows by the key they must differ on and
        draws each group's partners from the complementary rows. That is linear
        in the batch. Building a (rows x rows) "is this pair invalid" matrix
        costs ~17M boolean comparisons per step at batch 4096 and made the cone
        arms run about four times slower than the no-cone arms.
        """
        rng = np.random.default_rng(seed)
        size = len(coarse)
        if size < 2:
            raise RuntimeError("Need at least two rows to draw a negative partner")
        if mode == "q2":
            key = coarse
        else:
            key = coarse.astype(np.int64) * (int(fine.max()) + 1) + fine
        chosen = np.empty(size, dtype=np.int64)
        for value in np.unique(key):
            rows = np.flatnonzero(key == value)
            if len(rows) == size:
                raise RuntimeError(
                    "The batch holds a single category, so no out-of-category "
                    "partner exists for it"
                )
            complement = np.setdiff1d(np.arange(size), rows, assume_unique=False)
            picks = rng.integers(0, len(complement), size=len(rows))
            chosen[rows] = complement[picks]
        if mode == "q2":
            blocked = coarse[chosen] == coarse
        else:
            blocked = (coarse[chosen] == coarse) & (fine[chosen] == fine)
        if bool(blocked.any()):
            raise RuntimeError("Partner sampling produced an invalid partner")
        return chosen

    def _prototype_pairs(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """One (coarse, one of its fine children) pair per coarse category.

        Drawn from the tree rather than from the item batch: the prototype term
        constrains the prototypes themselves, and a batch can easily contain all
        30 coarse categories, which would leave no valid out-of-category fine
        prototype to reject.
        """
        rng = np.random.default_rng(seed)
        coarse = np.asarray([node for node, _ in self.fine_children], dtype=np.int64)
        pick = rng.random(len(self.fine_children))
        fine = np.asarray(
            [
                children[int(np.floor(pick[index] * len(children)))]
                for index, (_, children) in enumerate(self.fine_children)
            ],
            dtype=np.int64,
        )
        return coarse, fine

    def _fine_negatives(self, coarse: np.ndarray, seed: int) -> np.ndarray:
        """Fine prototypes whose coarse parent differs from the row's coarse."""
        rng = np.random.default_rng(seed)
        parent = self.coarse_of_fine.detach().cpu().numpy()
        size = len(parent)
        if bool((parent[:, None] == np.asarray(coarse)[None, :]).all()):
            raise RuntimeError("Every fine prototype shares a coarse category")
        chosen = rng.integers(0, size, size=len(coarse))
        blocked = parent[chosen] == coarse
        for _ in range(8):
            if not bool(blocked.any()):
                return chosen
            rows = np.flatnonzero(blocked)
            chosen[rows] = rng.integers(0, size, size=len(rows))
            blocked = parent[chosen] == coarse
        for row in np.flatnonzero(blocked):
            chosen[row] = int(np.flatnonzero(parent != coarse[row])[0])
        return chosen

    # ------------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(
        self,
        prefixes: list[torch.Tensor],
        curvature: torch.Tensor,
        aperture_degrees: float,
        sample_size: int = 8_192,
    ) -> dict:
        """Fit one aperture constant per loss term on the supervised items."""
        rng = np.random.default_rng(self.data_seed)
        pool = self.supervised_index[self.supervised_index < len(prefixes[0])]
        pool = pool[np.isin(pool, self.labelled_index)]
        if len(pool) > sample_size:
            pool = rng.choice(pool, size=sample_size, replace=False)
        index = torch.as_tensor(pool, dtype=torch.long, device=prefixes[0].device)
        coarse = self.coarse_of_item[index]
        fine = self.fine_of_item[index]
        apex_proto = self.prototypes.coarse[coarse]
        point_proto = self.prototypes.fine[fine]
        point_coarse = to_point(self.geometry, apex_proto, curvature)
        point_fine = to_point(self.geometry, self.prototypes.fine[fine], curvature)
        k_proto = k_for_aperture(
            self.geometry, point_coarse, aperture_degrees
        )
        k_q2 = k_for_aperture(
            self.geometry, point_coarse, aperture_degrees
        )
        k_q3 = k_for_aperture(
            self.geometry, point_fine, aperture_degrees
        )
        self.k_proto.fill_(k_proto)
        self.k_q2.fill_(k_q2)
        self.k_q3.fill_(k_q3)
        self.calibrated.fill_(True)
        return self.aperture_report()

    @torch.no_grad()
    def aperture_report(self) -> dict:
        report = {
            "k_proto": float(self.k_proto),
            "k_q2": float(self.k_q2),
            "k_q3": float(self.k_q3),
        }
        curvature = torch.ones((), device=self.k_proto.device)
        coarse = self.prototypes.coarse
        fine = self.prototypes.fine
        for name, apex in (
            ("proto", coarse),
            ("fine_proto", fine),
        ):
            point = to_point(self.geometry, apex, curvature)
            factors = aperture_factor(self.geometry, point)
            report[f"max_kg_{name}"] = float(
                factors.max() * float(self.k_proto)
            )
            psi = aperture(self.geometry, point, float(self.k_proto))
            degrees = torch.rad2deg(psi)
            report[f"aperture_deg_{name}"] = {
                "min": float(degrees.min()),
                "median": float(degrees.median()),
                "max": float(degrees.max()),
            }
        return report

    # ------------------------------------------------------------------- loss
    def loss(
        self,
        items: torch.Tensor,
        prefixes: list[torch.Tensor],
        curvature: torch.Tensor,
        global_step: int,
    ) -> tuple[torch.Tensor, dict]:
        """Cone objective plus the per-term diagnostics it is built from."""
        coarse_np = self.coarse_of_item[items].detach().cpu().numpy()
        fine_np = self.fine_of_item[items].detach().cpu().numpy()
        supervised = self.supervised[items.detach().cpu().numpy()]
        rows = np.flatnonzero(supervised & (coarse_np >= 0) & (fine_np >= 0))
        if len(rows) > 1:
            # A batch can hold a single coarse category - "Instrument
            # Accessories" alone is 48% of the catalogue - and then no
            # out-of-category partner exists for the coarse/Q2 term.
            #
            # Both availability questions are answered without a pair matrix: a
            # row has a Q2 partner exactly when the batch holds more than one
            # coarse category, and it lacks a Q3 partner only when every row
            # shares its coarse *and* fine category, i.e. the batch is a single
            # category pair. A (rows x rows) mask was the earlier formulation and
            # cost ~17M CPU comparisons per step at batch 4096, which made the
            # cone arms about three times slower than the no-cone arms.
            coarse_here, fine_here = coarse_np[rows], fine_np[rows]
            if np.unique(coarse_here).size < 2 or np.unique(
                np.stack([coarse_here, fine_here]), axis=1
            ).shape[0] < 2:
                rows = rows[:0]
        if len(rows) == 0:
            zero = prefixes[2].sum() * 0.0
            return zero, {
                "cone_total": 0.0,
                "cone_prototype_pair": 0.0,
                "cone_coarse_to_q2": 0.0,
                "cone_fine_to_q3": 0.0,
                "cone_radial": 0.0,
                "cone_rows": 0,
                "nonzero_negative_ratio": 0.0,
                "negative_containment": 0.0,
                "radial_order_rate": 0.0,
            }

        index = torch.as_tensor(rows, dtype=torch.long, device=items.device)
        coarse = coarse_np[rows]
        fine = fine_np[rows]
        seed = self.data_seed * 1_000_003 + int(global_step)
        # The partners are positions inside the already-selected rows: q2 and q3
        # below are the row subset, not the batch, so batch row ids would index
        # out of bounds.
        partner_q2 = self._partners(coarse, fine, seed, "q2")
        partner_q3 = self._partners(coarse, fine, seed + 1, "q3")
        prototype_coarse, prototype_fine = self._prototype_pairs(seed + 2)
        negative_fine = self._fine_negatives(prototype_coarse, seed + 3)
        partner_q2_index = torch.as_tensor(partner_q2, dtype=torch.long, device=items.device)
        partner_q3_index = torch.as_tensor(partner_q3, dtype=torch.long, device=items.device)
        negative_fine_index = torch.as_tensor(
            negative_fine, dtype=torch.long, device=items.device
        )

        apex_coarse = self.prototypes.coarse[
            torch.as_tensor(coarse, dtype=torch.long, device=items.device)
        ]
        apex_fine = self.prototypes.fine[
            torch.as_tensor(fine, dtype=torch.long, device=items.device)
        ]
        q2, q3 = prefixes[1][index], prefixes[2][index]

        prototype_coarse_index = torch.as_tensor(
            prototype_coarse, dtype=torch.long, device=items.device
        )
        prototype_fine_index = torch.as_tensor(
            prototype_fine, dtype=torch.long, device=items.device
        )
        loss_pf, positive_pf, negative_pf = containment_loss(
            self.geometry, curvature,
            self.prototypes.coarse[prototype_coarse_index],
            self.prototypes.fine[prototype_fine_index],
            self.prototypes.fine[negative_fine_index],
            float(self.k_proto), self.margin,
        )
        loss_cq2, positive_cq2, negative_cq2 = containment_loss(
            self.geometry, curvature, apex_coarse, q2, q2[partner_q2_index],
            float(self.k_q2), self.margin,
        )
        loss_fq3, positive_fq3, negative_fq3 = containment_loss(
            self.geometry, curvature, apex_fine, q3, q3[partner_q3_index],
            float(self.k_q3), self.margin,
        )
        # The radial term spans every prototype, not only the ones in this
        # batch, so each prototype parameter receives a gradient on every step.
        # Distributed training runs with find_unused_parameters=False and would
        # abort on any parameter left without one.
        radial = radial_order_loss(
            self.geometry,
            curvature,
            self.prototypes.coarse,
            self.prototypes.fine,
            self.radial_margin,
        )
        total = loss_pf + loss_cq2 + loss_fq3 + self.radial_weight * radial

        with torch.no_grad():
            negative_containment = torch.cat(
                [
                    containment_rate(
                        self.geometry, curvature,
                        self.prototypes.coarse[prototype_coarse_index],
                        self.prototypes.fine[negative_fine_index],
                        float(self.k_proto),
                    ).reshape(-1),
                    containment_rate(
                        self.geometry, curvature, apex_coarse,
                        q2[partner_q2_index], float(self.k_q2),
                    ).reshape(-1),
                    containment_rate(
                        self.geometry, curvature, apex_fine,
                        q3[partner_q3_index], float(self.k_q3),
                    ).reshape(-1),
                ]
            )
            positive_containment = torch.cat(
                [
                    containment_rate(
                        self.geometry, curvature,
                        self.prototypes.coarse[prototype_coarse_index],
                        self.prototypes.fine[prototype_fine_index],
                        float(self.k_proto),
                    ).reshape(-1),
                    containment_rate(
                        self.geometry, curvature, apex_coarse, q2, float(self.k_q2)
                    ).reshape(-1),
                    containment_rate(
                        self.geometry, curvature, apex_fine, q3, float(self.k_q3)
                    ).reshape(-1),
                ]
            )
            # Per-pair negative energies, not the batch means: the diagnostic
            # is the share of individual negative partners that still sit inside
            # the cone and therefore still produce gradient.
            negative_margin_gap = torch.cat(
                [
                    self.margin
                    - energy(
                        self.geometry,
                        self.prototypes.coarse[prototype_coarse_index],
                        self.prototypes.fine[negative_fine_index],
                        float(self.k_proto),
                    ).reshape(-1),
                    self.margin
                    - energy(
                        self.geometry, apex_coarse, q2[partner_q2_index],
                        float(self.k_q2),
                    ).reshape(-1),
                    self.margin
                    - energy(
                        self.geometry, apex_fine, q3[partner_q3_index],
                        float(self.k_q3),
                    ).reshape(-1),
                ]
            )
            diagnostics = {
                "cone_total": float(total.detach()),
                "cone_prototype_pair": float(loss_pf.detach()),
                "cone_coarse_to_q2": float(loss_cq2.detach()),
                "cone_fine_to_q3": float(loss_fq3.detach()),
                "cone_radial": float(radial.detach()),
                "cone_positive_proto_q2": float(positive_cq2.detach()),
                "cone_positive_fine_q3": float(positive_fq3.detach()),
                "cone_rows": int(len(rows)),
                "negative_containment": float(negative_containment.float().mean()),
                "positive_containment": float(positive_containment.float().mean()),
                "nonzero_negative_ratio": float(
                    (negative_margin_gap > 0).float().mean()
                ),
                "radial_order_rate": float(
                    (radial.detach() <= 0).float().mean()
                ),
            }
        return total, diagnostics

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(
        self,
        prefixes: list[torch.Tensor],
        curvature: torch.Tensor,
    ) -> dict:
        """Containment on the supervised and held-out items, split by prefix."""
        report: dict[str, dict] = {}
        for name, pool in (
            ("supervised", self.supervised_index),
            ("heldout", self.heldout_index),
        ):
            pool = pool[np.isin(pool, self.labelled_index)]
            if len(pool) == 0:
                report[name] = {"n_items": 0}
                continue
            index = torch.as_tensor(pool, dtype=torch.long, device=prefixes[0].device)
            coarse = self.coarse_of_item[index]
            fine = self.fine_of_item[index]
            apex_coarse = self.prototypes.coarse[coarse]
            apex_fine = self.prototypes.fine[fine]
            report[name] = {
                "n_items": int(len(pool)),
                "coarse_q2_containment": float(
                    containment_rate(
                        self.geometry, curvature, apex_coarse,
                        prefixes[1][index], float(self.k_q2),
                    ).float().mean()
                ),
                "fine_q3_containment": float(
                    containment_rate(
                        self.geometry, curvature, apex_fine,
                        prefixes[2][index], float(self.k_q3),
                    ).float().mean()
                ),
                "prototype_pair_containment": float(
                    containment_rate(
                        self.geometry, curvature, apex_coarse, apex_fine,
                        float(self.k_proto),
                    ).float().mean()
                ),
            }
        return report

    def state_report(self) -> dict:
        return {
            "geometry": self.geometry,
            "n_coarse_prototypes": int(self.prototypes.coarse.shape[0]),
            "n_fine_prototypes": int(self.prototypes.fine.shape[0]),
            "prototype_dim": int(self.prototypes.fine.shape[1]),
            "n_prototype_parameters": int(
                self.prototypes.coarse.numel() + self.prototypes.fine.numel()
            ),
            "supervised_items": int(self.supervised_index.size),
            "heldout_items": int(self.heldout_index.size),
            "labelled_items": int(self.labelled_index.size),
            "margin": self.margin,
            "radius_band": [self.radius_band_low, self.radius_band_high],
            "radial_weight": self.radial_weight,
            "radial_margin": self.radial_margin,
            "calibrated": bool(self.calibrated),
            **self.aperture_report(),
        }


def cone_report(cone: CategoryCone) -> str:
    report = cone.state_report()
    return json.dumps(report, sort_keys=True)