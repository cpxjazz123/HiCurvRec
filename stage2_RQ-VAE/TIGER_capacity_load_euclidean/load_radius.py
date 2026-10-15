"""Radius-sensitive load capacity: one operator, two geometries.

Why this operator
-----------------
The earlier arm added ``lambda * log(1 + usage_i)`` to the *score*, which is a
load penalty expressed in whatever metric the level already uses. The crowded
codeword's reach was never changed -- only how far away it looked -- so the term
carried no geometry of its own and attributing an effect to a hyperbolic mechanism
was not supported.

Here the load signal acts on the codeword's **position in the quantiser's space**:

    usage_i is high  ->  r_i = 1 + lambda * log(1 + usage_i)  >  1
    the codeword is placed at radius r_i along its own direction

A crowded codeword therefore sits further out, and in the Poincare ball the
geodesic distance to a point grows *faster* with radius than the Euclidean one --
the ball's geometry is what makes over-used codes expensive to reach. Nothing else
changes: no extra loss term, no extra codewords, same three levels, same capacity
mask, same Sinkhorn.

The two arms that use this module differ in one value, ``GEOMETRY``:

* ``geodesic`` -- encode both sides into the ball of curvature ``CURVATURE`` and
  measure with the geodesic distance;
* ``euclidean`` -- the identical radial rule applied to the raw vectors, measured
  with the squared Euclidean distance.

That is what makes the pair informative: the load rule is held fixed and only the
metric moves, so a difference between the arms is attributable to the metric.

The counter records the indices the level actually assigned, Sinkhorn included,
and is a persistent buffer so a checkpoint carries the load history it was
exported with.
"""
from __future__ import annotations

import contextlib
from types import MethodType

import torch

from hyperbolic import pair_poincare_distance
from scripts.hyp_losses import encode_points, shared_tangent_scale

USAGE_DECAY = 0.9
LOAD_STRENGTH = 0.5
CURVATURE = 1.0
GEOMETRY = "euclidean"      # pinned for this arm


def load_radii(usage: torch.Tensor, strength: float = LOAD_STRENGTH) -> torch.Tensor:
    """Radius multiplier per codeword: crowded codes move outward."""
    return 1.0 + float(strength) * torch.log1p(usage)


def install_load_radius(layer, decay: float = USAGE_DECAY,
                        strength: float = LOAD_STRENGTH,
                        geometry: str = GEOMETRY,
                        c: float = CURVATURE) -> None:
    """Install the load-aware assignment and keep its usage EMA current.

    Only ``_distances`` is replaced. ``_indices`` is wrapped to count what the
    level actually assigned and then delegated to, so the balanced-assignment
    control (Sinkhorn on the last level) still decides.
    """
    if getattr(layer, "_load_radius_active", False):
        return
    if geometry not in ("geodesic", "euclidean"):
        raise ValueError(f"geometry must be 'geodesic' or 'euclidean', got {geometry!r}")
    original = layer._distances
    original_indices = layer._indices
    codes_base = layer.get_code_embs()
    n_embed = int(layer.n_embed)
    if not hasattr(layer, "_usage_ema"):
        layer.register_buffer(
            "_usage_ema",
            torch.zeros(n_embed, dtype=torch.float32, device=codes_base.device),
        )
    usage = layer._usage_ema

    def distances(self, latent):
        # Directions come from the *live* codebook on every call. A snapshot taken
        # at installation would keep assigning codes at positions the VQ objective
        # has since moved away from, so the learned codewords and the SID selection
        # would disagree; and it lived only in a closure, so it was absent from
        # state_dict() and a reloaded model would rebuild it from whatever weights it
        # happened to hold.
        directions = torch.nn.functional.normalize(self.get_code_embs(), dim=-1)
        radii = load_radii(usage, strength).unsqueeze(-1)
        codes = directions * radii
        if geometry == "euclidean":
            return (latent.pow(2).sum(1, keepdim=True)
                    - 2 * latent @ codes.t()
                    + codes.pow(2).sum(1).t())
        scale = shared_tangent_scale(self.get_code_embs(), c)
        points = encode_points(latent, c, scale)
        moved = encode_points(codes, c, scale)
        return pair_poincare_distance(points, moved, c)

    def indices(self, distance_matrix, infer_use_sk):
        chosen = original_indices(distance_matrix, infer_use_sk)
        if self.training and not _SUPPRESSED[0]:
            with torch.no_grad():
                counts = torch.bincount(
                    chosen.reshape(-1).to(torch.long), minlength=n_embed
                ).to(usage.dtype)
                usage.mul_(float(decay)).add_(counts, alpha=1.0 - float(decay))
                _record_load_update()
        return chosen

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)
    layer._load_radius_active = True
    layer._load_radius_geometry = str(geometry)
    layer._load_radius_strength = float(strength)
    layer._load_radius_decay = float(decay)
    layer.usage_ema = usage


# Load-EMA update boundary.
#
# The EMA is defined as **one update per optimizer step, from the main training
# batch**. Auxiliary quantiser calls -- the behaviour residual trace, mechanism
# traces, diagnostics -- walk the same layers in training mode, so a wrapper that
# updated on every call would decay the state once per quantiser forward rather
# than once per step, and would mix the pair-batch distribution into the measured
# load. Those call sites now run inside ``suppress_load_updates()``.
_SUPPRESSED = [False]
_UPDATE_COUNT = [0]


@contextlib.contextmanager
def suppress_load_updates():
    """Run a quantiser call without advancing the load EMA.

    Use around any auxiliary forward: behaviour traces, mechanism traces,
    diagnostics. Assignment, Sinkhorn and gradients are unaffected -- only the
    occupancy observation is skipped.
    """
    previous = _SUPPRESSED[0]
    _SUPPRESSED[0] = True
    try:
        yield
    finally:
        _SUPPRESSED[0] = previous


def load_update_count() -> int:
    """How many EMA updates have happened. One per optimizer step, by definition."""
    return _UPDATE_COUNT[0]


def reset_load_update_count() -> None:
    _UPDATE_COUNT[0] = 0


def _record_load_update() -> None:
    _UPDATE_COUNT[0] += 1
