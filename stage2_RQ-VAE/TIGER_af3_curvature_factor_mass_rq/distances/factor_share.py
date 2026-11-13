"""The product-manifold distance, as a registry entry: `factor_share`.

The registry (`distances/__init__.py`) stores `NAME`, `DESCRIPTION` and a callable
`distances(latent, codebook) -> (B, K)`. This module holds NO numbers and no formula:
it forwards to `factor_share_geometry.product_cost`, which is the single implementation
of formula `product_cost` in this tree.

`p` is the per-item share `p_i = u_i` (see `factor_share_geometry.share`). It is
REQUIRED: a missing or mis-shaped `p` is a `ValueError`, never a silent fallback to the
Euclidean metric -- a fallback would make a broken wiring look like a working arm.

`latent` is the 32-vector `(r_i^{E,l} | h1_i | h2_i)`; `codebook` is `(K, 32)`, exactly
the shapes the baseline quantiser uses, so the layer can hand over whatever the metric
object is.

Package direction: this module imports the tree's TOP-LEVEL `factor_share_geometry`
only. It never imports `model.*`, so `model/__init__ -> model/layers -> distances`
stays acyclic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

NAME = "factor_share"
DESCRIPTION = ("product-manifold cost |r^E - tau^E|^2 + SCALE_HYP * "
               "(d_{c_hi}^2 + d_{c_lo}^2) with the codeword side split per item by p")

_GEOMETRY = None


def geometry():
    """The tree's top-level `factor_share_geometry` (cached, fail loud)."""
    global _GEOMETRY
    if _GEOMETRY is not None:
        return _GEOMETRY
    try:
        import factor_share_geometry as module
    except ImportError as error:
        # The tree directory is normally on sys.path (the trainer runs from it); when a
        # test imports this package from elsewhere, load the sibling module by walking up
        # to the FILE MARKER -- no parent counting, exactly like `repository_root()`.
        path = None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "factor_share_geometry.py"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            raise RuntimeError("cannot import factor_share_geometry (%s); no tree above %s "
                               "contains the module"
                               % (error, Path(__file__).resolve())) from error
        spec = importlib.util.spec_from_file_location("factor_share_geometry", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("no import spec could be built for " + str(path)) from error
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    _GEOMETRY = module
    return module


def distances(latent, codebook: torch.Tensor,
              p=None, mode: str = "both_on") -> torch.Tensor:
    """`(B, 32)` against `(K, 32)` -> `(B, K)` on the product manifold.

    `latent` is either the tangent 32-vector `(r_i^{E,l} | h1_i | h2_i)`, in which case
    the chart head is applied by the geometry module, or the already charted triple
    `(r_i^{E,l}, x_i^hi, x_i^lo)` that `model/model.py` produces. Both forms reach the
    ONE formula in `factor_share_geometry`; this module still holds no numbers.

    `p` is `(B,)` (per item, frozen). `mode` selects the wiring check:
    'both_on' (main arm), 'behaviour_off' (`p_i = SHARE_REF`) or 'geometry_off'
    (the baseline squared Euclidean expansion, used to prove the two are elementwise
    identical); `geometry_off` is only defined on the tangent vector.
    """
    if p is None:
        raise ValueError("factor_share needs the per-item share p (u_i); it is never "
                         "defaulted to a constant")
    if not torch.is_tensor(p) and not hasattr(p, "__len__"):
        raise ValueError("p must be a per-item sequence or tensor, got %r" % (type(p),))
    geometry_module = geometry()
    if isinstance(latent, (tuple, list)):
        if len(latent) != 3:
            raise ValueError("the charted product point must be (rE, x_hi, x_lo), got %d "
                             "blocks" % (len(latent),))
        rE, x_hi, x_lo = latent
        return geometry_module.product_cost_blocks(rE, x_hi, x_lo, codebook, p, mode=mode)
    return geometry_module.product_cost(latent, codebook, p, mode=mode)


__all__ = ["NAME", "DESCRIPTION", "distances", "geometry"]
