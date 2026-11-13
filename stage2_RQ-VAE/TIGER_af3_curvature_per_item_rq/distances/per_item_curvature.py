"""`per_item_curvature`: the cost matrix read in each item's own curvature ball.

Registered as a FACTORY (see `distances/__init__.py`): the instance needs the frozen
curvature table the trainer built before training started, which the registry cannot
know. The object is what `VQLayer` calls, with the metric contract of its sibling
`af3_offset`:

    needs_rows = True
    __call__(latent, codebook, rows) -> (B, K)

`rows=None` raises: there is no fallback, because any default (e.g. `arange`) would
apply item A's curvature to item B silently. A latent that is not `(B, 32)` raises too:
the factor split `D_E=16 + D_H=16` is part of the mechanism, not a runtime choice.

The geometry itself lives in `model/per_item_curvature_geometry.py`; this module only
declares the registry face and re-checks its arguments.
"""

from __future__ import annotations

import torch

NAME = "per_item_curvature"
DESCRIPTION = ("squared Euclidean on the R^16 factor + SCALE_HYP * squared geodesic on the "
               "H^16(c_i) factor, c_i frozen per item row")

#: constants of the mechanism (math.json `constants`), fixed and NOT scannable
D = 32
D_E = 16
D_H = 16
SCALE_HYP = 0.25


class PerItemCurvature:
    """The registry face of `PerItemCurvatureGeometry`.

    Attributes the quantiser probes for (the same convention as `af3_offset`):

    * `needs_rows = True` -- the batch's item rows must be handed to the metric;
    * `quant_parts(...)` -- the two quantisation errors must be read in the same
      geometry, so the layer asks the metric instead of building an MSE itself.
    """

    needs_rows = True
    #: the geometry is not per-level: one curvature table for the whole residual stack
    has_levels = False

    def __init__(self, curvature_table=None, *, behaviour_off: bool = False,
                 geometry_off: bool = False) -> None:
        from model.per_item_curvature_geometry import PerItemCurvatureGeometry

        if curvature_table is None:
            raise ValueError(
                "per_item_curvature needs the frozen curvature table; build it with "
                "af3_per_item_curvature.build(...) and pass curvature_table=<(N,) array>"
            )
        self.geometry = PerItemCurvatureGeometry(
            curvature_table,
            behaviour_off=behaviour_off,
            geometry_off=geometry_off,
        )

    # ------------------------------------------------------------------ faces
    @property
    def table(self) -> torch.Tensor:
        return self.geometry.table

    @property
    def behaviour_off(self) -> bool:
        return self.geometry.behaviour_off

    @property
    def geometry_off(self) -> bool:
        return self.geometry.geometry_off

    def curvature_of(self, rows, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.geometry.curvature_of(rows, like)

    def distances(self, latent: torch.Tensor, codebook: torch.Tensor, rows) -> torch.Tensor:
        """`(B, D)` against `(K, D)` at the rows' own curvatures -> `(B, K)`."""
        if rows is None:
            raise ValueError(
                "per_item_curvature requires `rows` (the batch's item rows); "
                "the two-argument registry contract has no meaning here"
            )
        if latent.dim() != 2 or int(latent.shape[1]) != D:
            raise ValueError(
                "per_item_curvature expects a (B, %d) tangent vector, got %s"
                % (D, tuple(latent.shape))
            )
        return self.geometry.cost(latent, codebook, rows)

    # `VQLayer` calls the object as a function
    __call__ = distances

    def quant_parts(self, latent: torch.Tensor, assigned_codebook: torch.Tensor, rows):
        """The `(commitment, codebook)` pair, read in the same row-ball geometry."""
        if rows is None:
            raise ValueError("per_item_curvature.quant_parts requires `rows`")
        return self.geometry.quant_parts(latent, assigned_codebook, rows)

    # ------------------------------------------------- read-only decompositions
    def cost_weight_only(self, latent: torch.Tensor, codebook: torch.Tensor, rows) -> torch.Tensor:
        if rows is None:
            raise ValueError("per_item_curvature.cost_weight_only requires `rows`")
        return self.geometry.cost_weight_only(latent, codebook, rows)

    def cost_point_only(self, latent: torch.Tensor, codebook: torch.Tensor, rows) -> torch.Tensor:
        if rows is None:
            raise ValueError("per_item_curvature.cost_point_only requires `rows`")
        return self.geometry.cost_point_only(latent, codebook, rows)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "PerItemCurvature(%r)" % (self.geometry,)
