"""Geometric invariants for the entailment-cone angle and aperture.

Runs without pytest: ``python bench_hier_behavior/test_cone_geometry.py`` exits
non-zero on the first broken invariant.

The invariants are the ones the cone loss depends on. Apex ``x``, child ``y``:

* a child further out along the apex's own spoke is *inside* the cone, so
  ``xi(x, y)`` must be near zero;
* a point inward along that spoke, i.e. the apex's own ancestor direction, must
  be near pi, the opposite of the child case;
* a sibling at the same radius on an orthogonal spoke must sit strictly
  between those two, and outside a small aperture;
* the same configuration gives a strictly wider angle in the Poincare ball than
  in Euclidean space, because the models agree only in the flat limit;
* at a common radius the Poincare aperture is strictly narrower than the
  Euclidean one, since ``(1-||x||^2)/||x|| < 1/||x||`` inside the ball;
* the loss the training code actually calls must reward the child and punish
  the ancestor direction.
"""

from __future__ import annotations

import math

import torch

from .cones import FACTOR, entailment_margin_loss, euclid_xi_pairs, poincare_xi_pairs
from .geometry import make_geometry

CURVATURE = 1.0
K = 0.1
TOLERANCE = 1e-3


def _angles(geometry: str, apex: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    if geometry == "euclid":
        return euclid_xi_pairs(apex, point)
    return poincare_xi_pairs(apex, point, CURVATURE)


def _case(radius: float = 0.3, step: float = 0.05) -> dict[str, torch.Tensor]:
    spoke = torch.tensor([1.0, 0.0], dtype=torch.float64)
    orthogonal = torch.tensor([0.0, 1.0], dtype=torch.float64)
    return {
        "apex": radius * spoke,
        "child": (radius + step) * spoke,
        "ancestor": (radius - step) * spoke,
        "sibling": radius * orthogonal,
    }


def check(name: str, condition: bool, detail: str) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)
    return condition


def main() -> None:
    failures: list[str] = []
    for geometry in ("euclid", "poincare"):
        case = _case()
        apex, child = case["apex"], case["child"]
        ancestor, sibling = case["ancestor"], case["sibling"]
        xi_child = float(_angles(geometry, apex, child))
        xi_ancestor = float(_angles(geometry, apex, ancestor))
        xi_sibling = float(_angles(geometry, apex, sibling))
        aperture = float(
            torch.asin((K * FACTOR[geometry](apex)).clamp(0.0, 1.0))
        )

        if not check(
            f"{geometry}: child on the apex spoke is inside the cone",
            xi_child < TOLERANCE and xi_child <= aperture,
            f"xi(child)={xi_child:.6f} aperture={aperture:.6f}",
        ):
            failures.append(f"{geometry}/child_inside")

        if not check(
            f"{geometry}: ancestor direction is outside the cone",
            xi_ancestor > math.pi - TOLERANCE and xi_ancestor > aperture,
            f"xi(ancestor)={xi_ancestor:.6f}",
        ):
            failures.append(f"{geometry}/ancestor_outside")

        if not check(
            f"{geometry}: sibling lies between child and ancestor, outside a small cone",
            xi_child < xi_sibling < xi_ancestor and xi_sibling > aperture,
            f"xi(sibling)={xi_sibling:.6f}",
        ):
            failures.append(f"{geometry}/sibling_ordering")

        xi_reversed = float(_angles(geometry, child, apex))
        if not check(
            f"{geometry}: the two apex angles are supplementary",
            abs(xi_child + xi_reversed - math.pi) < TOLERANCE,
            f"xi(child,apex)={xi_reversed:.6f}",
        ):
            failures.append(f"{geometry}/antisymmetry")

    euclid_sibling = float(_angles("euclid", _case()["apex"], _case()["sibling"]))
    poincare_sibling = float(
        _angles("poincare", _case()["apex"], _case()["sibling"])
    )
    if not check(
        "poincare widens the apex angle relative to euclid",
        poincare_sibling > euclid_sibling,
        f"poincare={poincare_sibling:.6f} euclid={euclid_sibling:.6f}",
    ):
        failures.append("poincare/wider_angle")

    euclid_aperture = float(torch.asin((K * FACTOR["euclid"](_case()["apex"])).clamp(0, 1)))
    poincare_aperture = float(
        torch.asin((K * FACTOR["poincare"](_case()["apex"])).clamp(0, 1))
    )
    if not check(
        "poincare aperture is narrower than euclid at equal radius",
        0.0 < poincare_aperture < euclid_aperture,
        f"poincare={poincare_aperture:.6f} euclid={euclid_aperture:.6f}",
    ):
        failures.append("aperture/narrower")

    for geometry in ("euclid", "poincare"):
        model_geometry = make_geometry(geometry)
        case = _case()
        for label, point in (("child", case["child"]), ("ancestor", case["ancestor"])):
            loss, positive, _, _ = entailment_margin_loss(
                geometry,
                model_geometry,
                case["apex"].unsqueeze(0),
                point.unsqueeze(0),
                None,
                k=K,
            )
            del loss
            rewarded = float(positive) < 1e-6
            if not check(
                f"{geometry}: loss rewards the {label} direction"
                if label == "child"
                else f"{geometry}: loss punishes the {label} direction",
                rewarded if label == "child" else not rewarded,
                f"positive term={float(positive):.6f}",
            ):
                failures.append(f"{geometry}/loss_{label}")

    if failures:
        raise SystemExit(f"cone geometry invariants failed: {failures}")
    print("all cone geometry invariants hold", flush=True)


if __name__ == "__main__":
    main()
