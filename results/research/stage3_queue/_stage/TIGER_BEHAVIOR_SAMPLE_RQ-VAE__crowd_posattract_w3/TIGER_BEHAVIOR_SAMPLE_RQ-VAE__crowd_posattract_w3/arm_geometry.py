"""Frozen, one-factor configuration for the E0/H1/E1 geometry comparison."""

from geom_losses import RADIAL_SCALE

ARM_CONFIGS = {
    "E0": {
        "distance": "squared_euclidean",
        "mapping": "identity",
        "radial_scale": 1.0,
        "loss_form": "softplus(d(anchor,positive)-d(anchor,negative))",
        "weight": 0.2,
        "parameter_count": 0,
        "curvature": None,
    },
    "H1": {
        "distance": "poincare_geodesic",
        "mapping": "expmap0",
        "radial_scale": RADIAL_SCALE,
        "loss_form": "softplus(d(anchor,positive)-d(anchor,negative))",
        "weight": 0.2,
        "parameter_count": 0,
        "curvature": 1.0,
    },
    "E1": {
        "distance": "mapped_squared_euclidean",
        "mapping": "expmap0",
        "radial_scale": RADIAL_SCALE,
        "loss_form": "softplus(d(anchor,positive)-d(anchor,negative))",
        "weight": 0.2,
        "parameter_count": 0,
        "curvature": 1.0,
    },
}
