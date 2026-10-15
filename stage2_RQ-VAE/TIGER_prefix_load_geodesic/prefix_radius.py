"""Prefix-keyed radius capacity: one load rule, two metrics.

The arm under review scored a candidate with ``d(x, i) + lambda * log(1 + counts[prefix, i])``
on the control's Euclidean distance. That does use the SID prefix, but only as a
*lookup key* for a scalar penalty: the prefix never changes where a codeword sits
in the quantiser's space, so nothing hyperbolic entered and the run is not evidence
for a hyperbolic prefix-aware quantiser.

Here the prefix keys the same radius rule the capacity arm uses:

    counts[prefix, i] high  ->  r_{prefix,i} = 1 + lambda * log(1 + counts[prefix, i])
    the codeword is placed at radius r_{prefix,i} along its own direction

so a codeword that is over-used *within its prefix* is pushed outward for the
items that share that prefix, and the effect is geometric: measured geodesically,
the ball makes an outward codeword more expensive to reach than a Euclidean metric
does. Nothing else changes -- no extra loss, no extra codewords, same levels, same
capacity mask, same Sinkhorn, same sampling and training protocol.

Two arms use this module and differ in one value, ``GEOMETRY``, so any difference
between them is attributable to the metric rather than to the load rule.

Prefix keys keep the packing the trainer already uses, so the counter addresses the
same cells as before:

    level 2: key = l1
    level 3: key = l1 * base + l2            (base = 256)

The counter records the indices the level actually assigned, Sinkhorn included,
decays every slot each step, and is a persistent buffer.
"""
from __future__ import annotations

import contextlib
from types import MethodType

import torch

# poincare_distance is used directly: pair_poincare_distance is a two-batch
# API that inserts the pair dimensions itself, and the candidate positions here
# already carry the row dimension.
from hyperbolic import poincare_distance
from scripts.hyp_losses import encode_points, shared_tangent_scale

PREFIX_DECAY = 0.9
LOAD_STRENGTH = 0.5
CURVATURE = 1.0
GEOMETRY = "geodesic"      # pinned for this arm


def prefix_radii(counts: torch.Tensor, strength: float = LOAD_STRENGTH) -> torch.Tensor:
    """Radius multiplier per (prefix, codeword) cell."""
    return 1.0 + float(strength) * torch.log1p(counts)


def install_prefix_radius(layer, level: int, decay: float = PREFIX_DECAY,
                          strength: float = LOAD_STRENGTH,
                          base: int = 256,
                          geometry: str = GEOMETRY,
                          c: float = CURVATURE) -> None:
    """Prefix-aware assignment whose crowded cells are placed further out.

    ``_prefix_key`` is read from the layer, exactly as the additive version did, so
    the trainer's trace helper needs no change. ``_indices`` is wrapped to count and
    delegated to, so Sinkhorn still decides.
    """
    if getattr(layer, "_prefix_radius_active", False):
        return
    if geometry not in ("geodesic", "euclidean"):
        raise ValueError(f"geometry must be 'geodesic' or 'euclidean', got {geometry!r}")
    original = layer._distances
    original_indices = layer._indices
    codes_base = layer.get_code_embs()
    n_embed = int(layer.n_embed)
    n_prefix = int(base) ** max(int(level) - 1, 1)
    if not hasattr(layer, "_prefix_counts"):
        layer.register_buffer(
            "_prefix_counts",
            torch.zeros(n_prefix, n_embed, dtype=torch.float32, device=codes_base.device),
        )
    counts = layer._prefix_counts
    flat = counts.view(-1)

    def distances(self, latent):
        # Live codebook directions on every call: a snapshot taken at installation
        # would pin the candidate geometry to positions the VQ objective has since
        # moved, and it was never in state_dict(), so a reload could silently
        # change distances and exported SIDs for identical weights and counters.
        directions = torch.nn.functional.normalize(self.get_code_embs(), dim=-1)
        # Each row has its own candidate positions: the radius is per (prefix,
        # codeword) cell, so a codeword's position depends on which prefix the row
        # came from. That makes this (N, K, d), not a single shared (K, d) matrix --
        # broadcasting the shared directions against a (N, K) radius table silently
        # produces a 3-D tensor instead.
        keys = getattr(self, "_prefix_key", None)
        radii = directions.unsqueeze(0)                      # (1, K, d)
        if keys is not None:
            keys = keys.to(torch.long)
            radii = directions.unsqueeze(0) * prefix_radii(
                counts[keys], strength).unsqueeze(-1)         # (N, K, d)
        if geometry == "euclidean":
            cross = torch.matmul(latent.unsqueeze(1), radii.transpose(-1, -2)).squeeze(1)
            return (latent.pow(2).sum(1, keepdim=True)
                    - 2.0 * cross
                    + radii.pow(2).sum(-1))
        scale = shared_tangent_scale(self.get_code_embs(), c)
        points = encode_points(latent, c, scale)
        moved = encode_points(radii, c, scale)
        return poincare_distance(points.unsqueeze(1), moved, c)

    def indices(self, distance_matrix, infer_use_sk):
        chosen = original_indices(distance_matrix, infer_use_sk)
        keys = getattr(self, "_prefix_key", None)
        if self.training and keys is not None and not _SUPPRESSED[0]:
            with torch.no_grad():
                keys = keys.to(torch.long)
                slot = keys.reshape(-1) * n_embed + chosen.reshape(-1).to(torch.long)
                unique, inverse = torch.unique(slot, return_inverse=True)
                batch_counts = torch.bincount(
                    inverse, minlength=unique.numel()
                ).to(flat.dtype)
                # Every slot decays each step, so a prefix the batch never visits
                # cannot keep steering assignments with a stale load.
                flat.mul_(float(decay))
                flat[unique] += (1.0 - float(decay)) * batch_counts
                _record_load_update()
        return chosen

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)
    layer._prefix_radius_active = True
    layer._prefix_radius_geometry = str(geometry)
    layer._prefix_radius_level = int(level)
    layer._prefix_radius_strength = float(strength)
    layer._prefix_radius_decay = float(decay)
    layer.prefix_counts = counts


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
