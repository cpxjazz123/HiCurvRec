"""Geometry plug-ins for the benchmark quantizer.

The Poincare implementation is imported verbatim from GeneRec's production
RQ-VAE (``stage2_RQ-VAE/curvature_RQ-VAE/model/layers.py``) so the hyperbolic
arm runs the same exp/log maps, Mobius addition and pairwise distance that
Stage 2 trains with. The Euclidean arm is the same code with the geometry
replaced by its flat limit: identity maps, vector subtraction, and the Euclidean
pairwise distance. Nothing else differs between the two arms.

Encoder outputs and codebook vectors are stored as origin-tangent vectors in
both arms, which is what makes the shared quantizer below valid: a tangent
vector is a legitimate point in a Euclidean vector space and maps into the
Poincare ball through ``exp_0``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from . import bench_config as cfg

_LAYERS_PATH = (
    cfg.REPO_ROOT / "stage2_RQ-VAE" / "curvature_RQ-VAE" / "model" / "layers.py"
)


def _load_production_layers():
    if not _LAYERS_PATH.is_file():
        raise FileNotFoundError(f"GeneRec layers module not found: {_LAYERS_PATH}")
    spec = importlib.util.spec_from_file_location(
        "generec_rqvae_layers", _LAYERS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prod = _load_production_layers()
expmap0 = _prod._expmap0_tangent
logmap0 = _prod._logmap0_point
mobius_add = _prod._mobius_add
poincare_pair_distance = _prod._pairwise_poincare_distance_tangents
poincare_tangent_pair_distance = _prod._poincare_distance_tangent_pairs


class Geometry:
    """Flat-space baseline."""

    name = "euclid"

    def to_point(self, tangent: torch.Tensor) -> torch.Tensor:
        return tangent

    def from_point(self, point: torch.Tensor) -> torch.Tensor:
        return point

    def residual(self, residual: torch.Tensor,
                 code: torch.Tensor) -> torch.Tensor:
        return residual - code

    def pairwise_distance(self, left: torch.Tensor,
                          right: torch.Tensor) -> torch.Tensor:
        return torch.cdist(left, right)

    def pair_distance(self, left: torch.Tensor,
                      right: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(left - right, dim=-1)

    def projection_norm(self, point: torch.Tensor) -> torch.Tensor:
        del point
        return torch.zeros((), device=point.device, dtype=point.dtype)


class PoincareGeometry(Geometry):
    """Constant negative curvature, curvature ``-c``."""

    name = "poincare"

    def __init__(self, curvature: float = cfg.CURVATURE):
        self.curvature = float(curvature)

    def to_point(self, tangent: torch.Tensor) -> torch.Tensor:
        return expmap0(tangent, self.curvature)

    def from_point(self, point: torch.Tensor) -> torch.Tensor:
        return logmap0(point, self.curvature)

    def residual(self, residual: torch.Tensor,
                 code: torch.Tensor) -> torch.Tensor:
        residual_point = self.to_point(residual)
        code_point = self.to_point(code)
        return logmap0(mobius_add(-code_point, residual_point, self.curvature),
                       self.curvature)

    def pairwise_distance(self, left: torch.Tensor,
                          right: torch.Tensor) -> torch.Tensor:
        return poincare_pair_distance(left, right, self.curvature)

    def pair_distance(self, left: torch.Tensor,
                      right: torch.Tensor) -> torch.Tensor:
        return poincare_tangent_pair_distance(left, right, self.curvature)

    def projection_norm(self, point: torch.Tensor) -> torch.Tensor:
        del point
        # log(sqrt(c)) -> 0 is the tangent-space radius that exp_0 maps onto the
        # ideal boundary; cones are undefined out there.
        return torch.log(torch.tensor(self.curvature, device=point.device,
                                      dtype=point.dtype))


def make_geometry(name: str) -> Geometry:
    if name == "euclid":
        return Geometry()
    if name == "poincare":
        return PoincareGeometry()
    raise ValueError(f"unknown geometry: {name}")
