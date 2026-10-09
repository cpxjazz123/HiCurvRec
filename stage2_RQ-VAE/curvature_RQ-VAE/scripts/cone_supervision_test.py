"""Invariants of the production cone supervision.

Runs without pytest: ``python scripts/cone_supervision_test.py`` exits non-zero
on the first broken invariant. The cone energy is the quantity the mechanism
optimises, so its orientation, its per-geometry aperture and the calibration
rule are checked here rather than discovered through a training run.

The pair cases mirror the entailment directions: a descendant further out along
the apex's own spoke must land inside the cone, the apex's own ancestor
direction must land outside, and a sibling must sit strictly between them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from model.cones import (  # noqa: E402
    CategoryPrototypes,
    aperture,
    aperture_factor,
    apex_angle,
    containment_loss,
    containment_rate,
    energy,
    fit_k,
    radial_order_loss,
    to_point,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
K = 0.1
MARGIN = 0.01
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)
    if not condition:
        FAILURES.append(name)


def spoke_case(radius: float = 0.3, step: float = 0.05):
    along = torch.tensor([1.0, 0.0], device=DEVICE)
    across = torch.tensor([0.0, 1.0], device=DEVICE)
    return {
        "apex": (radius * along).double(),
        "child": ((radius + step) * along).double(),
        "ancestor": ((radius - step) * along).double(),
        "sibling": (radius * across).double(),
    }


def main() -> None:
    for geometry in ("euclid", "poincare"):
        case = spoke_case()
        apex = to_point(geometry, case["apex"], 1.0)
        child = to_point(geometry, case["child"], 1.0)
        ancestor = to_point(geometry, case["ancestor"], 1.0)
        sibling = to_point(geometry, case["sibling"], 1.0)
        psi = float(aperture(geometry, apex, K))

        xi_child = float(apex_angle(geometry, apex, child))
        xi_ancestor = float(apex_angle(geometry, apex, ancestor))
        xi_sibling = float(apex_angle(geometry, apex, sibling))
        check(
            f"{geometry}: descendant on the apex spoke is inside the cone",
            xi_child < 1e-3 and xi_child <= psi,
            f"xi={xi_child:.6f} psi={psi:.6f}",
        )
        check(
            f"{geometry}: ancestor direction is outside the cone",
            xi_ancestor > math.pi - 1e-3 and xi_ancestor > psi,
            f"xi={xi_ancestor:.6f}",
        )
        check(
            f"{geometry}: sibling lies between the two and outside the cone",
            xi_child < xi_sibling < xi_ancestor and xi_sibling > psi,
            f"xi={xi_sibling:.6f}",
        )
        check(
            f"{geometry}: containment agrees with the energy sign",
            bool(containment_rate(geometry, 1.0, case["apex"], case["child"], K).reshape(-1)[0])
            == bool(float(energy(geometry, apex, child, K)) == 0.0),
            "rate and energy consistent",
        )
        check(
            f"{geometry}: reversed pair flips the verdict",
            not bool(
                containment_rate(geometry, 1.0, case["child"], case["apex"], K).reshape(-1)[0]
            ),
            "apex/child swapped is outside",
        )
        loss, positive, negative = containment_loss(
            geometry,
            1.0,
            case["apex"].unsqueeze(0),
            case["child"].unsqueeze(0),
            case["ancestor"].unsqueeze(0),
            K,
            MARGIN,
        )
        violating, violating_positive, _ = containment_loss(
            geometry,
            1.0,
            case["apex"].unsqueeze(0),
            case["sibling"].unsqueeze(0),
            case["child"].unsqueeze(0),
            K,
            MARGIN,
        )
        check(
            f"{geometry}: loss is zero on a satisfied pair and positive on a violation",
            bool(torch.isfinite(loss.detach())) and float(positive.detach()) < 1e-6
            and float(loss) == 0.0
            and float(violating_positive) > 0.0
            and float(violating) > 0.0,
            f"satisfied={float(loss.detach()):.6f} "
            f"violated={float(violating.detach()):.6f} "
            f"negative_energy={float(negative.detach()):.6f}",
        )

    euclid = spoke_case()
    point = to_point("euclid", euclid["apex"], 1.0)
    sibling = to_point("euclid", euclid["sibling"], 1.0)
    euclid_sibling = float(apex_angle("euclid", point, sibling))
    poincarre_sibling = float(
        apex_angle(
            "poincare",
            to_point("poincare", euclid["apex"], 1.0),
            to_point("poincare", euclid["sibling"], 1.0),
        )
    )
    check(
        "poincare widens the apex angle relative to euclid",
        poincarre_sibling > euclid_sibling,
        f"poincare={poincarre_sibling:.6f} euclid={euclid_sibling:.6f}",
    )
    euclid_psi = float(aperture("euclid", point, K))
    poincarre_psi = float(
        aperture("poincare", to_point("poincare", euclid["apex"], 1.0), K)
    )
    check(
        "poincare aperture is narrower at equal tangent radius",
        0.0 < poincarre_psi < euclid_psi,
        f"poincare={poincarre_psi:.6f} euclid={euclid_psi:.6f}",
    )

    for geometry in ("euclid", "poincare"):
        fit_rng = torch.Generator(device=DEVICE).manual_seed(7)
        direction = torch.randn(
            256, 8, dtype=torch.float64, device=DEVICE, generator=fit_rng
        )
        direction = direction / direction.norm(dim=-1, keepdim=True)
        # Real parent/child pairs sit near each other's spoke, so the fit uses
        # an outward offset plus a small perpendicular wobble. A cone can never
        # contain a point more than 90 degrees from its apex, so an adversarial
        # fit set would be unreachable for any K and would only test the clamp.
        wobble = direction - (direction * direction).sum(-1, keepdim=True) * direction
        fit_apex = to_point(geometry, direction * 0.45, 1.0)
        fit_point = to_point(
            geometry, direction * 0.47 + 0.01 * wobble, 1.0
        )
        k = fit_k(geometry, fit_apex, fit_point, 0.9)
        apex_factors = aperture_factor(geometry, fit_apex)
        saturated = float((k * apex_factors).max())
        inside = apex_angle(geometry, fit_apex, fit_point) <= aperture(
            geometry, fit_apex, k
        )
        check(
            f"{geometry}: fit_k hits the requested coverage without saturating",
            k > 0.0 and saturated < 1.0 and bool(inside.float().mean() >= 0.85),
            f"k={k:.6f} max(K*g)={saturated:.4f} coverage={float(inside.float().mean()):.3f}",
        )
        far = to_point(
            geometry, direction[:1] * -0.9, 1.0
        )
        check(
            f"{geometry}: the calibrated aperture rejects an opposite point",
            not bool(
                (apex_angle(geometry, fit_apex[:1], far)
                 <= aperture(geometry, fit_apex[:1], k)).reshape(-1)[0]
            ),
            "opposite point outside",
        )

    # Build the supervision module exactly as the trainer builds it. A wrong
    # keyword here once survived two rounds because the cone was off in both,
    # so the construction path itself is now covered.
    from model.category_cone import CategoryCone
    from curvature_config import (
        CATEGORY_CONE_DATA_SEED,
        CATEGORY_CONE_HOLDOUT_FRACTION,
        CATEGORY_CONE_MARGIN,
        CATEGORY_CONE_RADIAL_MARGIN,
        CATEGORY_CONE_RADIAL_WEIGHT,
    )

    supervised = CategoryCone(
        geometry="poincare",
        codebook_dim=32,
        holdout_fraction=CATEGORY_CONE_HOLDOUT_FRACTION,
        data_seed=CATEGORY_CONE_DATA_SEED,
        margin=CATEGORY_CONE_MARGIN,
        radial_weight=CATEGORY_CONE_RADIAL_WEIGHT,
        radial_margin=CATEGORY_CONE_RADIAL_MARGIN,
        device=torch.device(DEVICE),
    )
    check(
        "cone module builds with the trainer's keywords",
        bool(supervised.prototypes.coarse.shape[0] >= 1)
        and bool(supervised.heldout_index.size > 0),
        f"supervised={supervised.supervised_index.size} "
        f"heldout={supervised.heldout_index.size}",
    )

    generator = torch.Generator(device=DEVICE).manual_seed(0)
    coarse_ids = torch.arange(30, device=DEVICE)
    fine_ids = torch.arange(59, device=DEVICE)
    prototypes = CategoryPrototypes(
        coarse_ids, fine_ids, 32, reference_radius=1.0,
        coarse_radius_ratio=0.5, generator=generator, device=DEVICE,
    ).to(DEVICE)
    check(
        "prototypes: both arms would hold the same parameter count",
        prototypes.coarse.numel() + prototypes.fine.numel() == (30 + 59) * 32,
        f"params={prototypes.coarse.numel() + prototypes.fine.numel()}",
    )
    coarse_norm = prototypes.coarse.detach().norm(dim=-1)
    fine_norm = prototypes.fine.detach().norm(dim=-1)
    check(
        "prototypes: coarse start interior to fine",
        bool(coarse_norm.max() < fine_norm.min()),
        f"coarse<={float(coarse_norm.max()):.4f} fine>={float(fine_norm.min()):.4f}",
    )
    for geometry in ("euclid", "poincare"):
        inside = radial_order_loss(
            geometry, 1.0,
            prototypes.coarse[:8], prototypes.fine[:8], 0.02,
        )
        check(
            f"{geometry}: radial order holds for the initial prototypes",
            float(inside.detach()) == 0.0,
            f"loss={float(inside.detach()):.6f}",
        )

    # The negative rejection term must stay active for a negative that is
    # already inside the cone. Building it from the one-sided energy would give
    # a positive loss with a zero gradient there, so a mis-assigned item could
    # never be pushed back out. These checks back-propagate the negative term on
    # its own, and confirm the positive term is untouched.
    for geometry in ("euclid", "poincare"):
        curvature, k, margin = 1.0, 0.20, 0.05
        apex = torch.tensor([[0.35, 0.0]], dtype=torch.float64, device=DEVICE)
        child = torch.tensor([[0.37, 0.0]], dtype=torch.float64, device=DEVICE)
        inside = torch.tensor([[0.36, 0.0]], dtype=torch.float64, device=DEVICE)
        opposite = torch.tensor([[-1.2, 0.0]], dtype=torch.float64, device=DEVICE)
        psi = float(aperture(geometry, to_point(geometry, apex, curvature), k))
        near_boundary = torch.tensor(
            [[0.35 * (1.0) + psi * 0.6, 0.0]], dtype=torch.float64, device=DEVICE
        )

        def rejection(negative_point):
            """Loss and gradient of the rejection term alone."""
            apex_t = apex.clone().requires_grad_(True)
            _, positive_term, negative_term = containment_loss(
                geometry, curvature, apex_t, child, negative_point, k, margin
            )
            negative_term.sum().backward()
            grad = apex_t.grad
            return (
                float(negative_term.detach()),
                float(positive_term.detach()),
                0.0 if grad is None else float(grad.norm()),
            )

        inside_loss, positive_inside, inside_grad = rejection(inside)
        boundary_loss, _, boundary_grad = rejection(near_boundary)
        far_loss, positive_far, far_grad = rejection(opposite)
        check(
            f"{geometry}: in-cone negative keeps a rejection gradient",
            inside_loss > 0.0 and inside_grad > 0.0,
            f"loss={inside_loss:.4f} grad={inside_grad:.3e}",
        )
        check(
            f"{geometry}: near-boundary negative keeps a rejection gradient",
            boundary_loss > 0.0 and boundary_grad > 0.0,
            f"loss={boundary_loss:.4f} grad={boundary_grad:.3e}",
        )
        check(
            f"{geometry}: a comfortably outside negative costs nothing",
            far_loss == 0.0 and far_grad == 0.0,
            f"loss={far_loss:.4f} grad={far_grad:.3e}",
        )
        check(
            f"{geometry}: the positive term is unchanged by the negative one",
            positive_inside == 0.0 and positive_far == 0.0,
            f"positive inside={positive_inside:.4f} far={positive_far:.4f}",
        )
        check(
            f"{geometry}: the in-cone negative costs more than the far one",
            inside_loss > far_loss,
            f"inside={inside_loss:.4f} far={far_loss:.4f}",
        )

    if FAILURES:
        raise SystemExit(f"cone supervision invariants failed: {FAILURES}")
    print("all cone supervision invariants hold", flush=True)


if __name__ == "__main__":
    main()