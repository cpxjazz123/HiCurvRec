"""The baseline squared-Euclidean cost plus the codeword-potential offset.

`assignment_cost` of `ideas/AF3_curvature_drive_idea/math.json`:

    C_ik^(l) = d(z_i^(l), c_k) + lambda_l * H_delta(s_i - g_k^(l))

`d` is the baseline metric, unchanged to the last bit: this module calls
`distances.distance.distances` for it instead of re-deriving the expansion. The
offset comes from `model.codeword_potential.CodewordPotential`, which owns `s`, `g`
and `lambda_l`; the metric owns neither.

The registry hands one callable to the whole residual stack and the quantiser calls
it as `distance(latent, codebook)`. That signature cannot say which level is being
quantised, while `C` depends on `l` (each level has its own `g` and `lambda`). So this
module exposes `at_level(level)`, and `model/layers.py` binds one cheap view per level
instead of sharing a mutable "current level" pointer between levels.

Item ids: `C` is indexed by item (`s_i`), so a metric that needs the row's item id
declares `needs_rows = True`, and the quantiser passes the ids it was given. A metric
that does not need them (the baseline `euclidean`) is called exactly as before.
"""

from __future__ import annotations

import torch

from distances.distance import distances as euclidean

NAME = "af3_offset"
DESCRIPTION = (
    "squared Euclidean |x|^2 - 2 x.c + |c|^2 plus lambda_l * H_delta(s_i - g_k^(l)), "
    "the AF3 curvature term on every assignment cost"
)

__all__ = ["Af3Offset", "Af3OffsetLevel", "NAME", "DESCRIPTION", "build"]


class Af3OffsetLevel:
    """One level's view: `d(latent, codebook)` with that level's structural offset."""

    #: the quantiser must hand this metric the batch's item ids
    needs_rows = True

    def __init__(self, base, potential, level: int) -> None:
        self.base = base
        self.potential = potential
        self.level = int(level)

    def __call__(self, latent: torch.Tensor, codebook: torch.Tensor, rows=None) -> torch.Tensor:
        """`(B, D)` against `(K, D)` -> `(B, K)` cost matrix for this level."""
        base = self.base(latent, codebook)
        if rows is None:
            # Evaluation over the whole item space: row i is item i, which is how
            # train_rqvae.py builds `all_embeddings`.
            rows = torch.arange(latent.shape[0], device=latent.device)
        offset = self.potential.bias(self.level, rows).to(base.dtype)
        if offset.shape != base.shape:
            raise ValueError(
                "af3_offset: offset %r does not match the cost %r"
                % (tuple(offset.shape), tuple(base.shape))
            )
        return base + offset

    def base_cost(self, latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """`d(z, c)` alone, the quantity `lambda_calibration` divides."""
        return self.base(latent, codebook)

    def initialise_level(self, latent: torch.Tensor, codebook: torch.Tensor, rows, tokens) -> dict:
        """Calibrate this level right after KMeans, from the base metric (no offset)."""
        if rows is None:
            rows = torch.arange(latent.shape[0], device=latent.device)
        cost = self.base(latent, codebook)
        index = torch.as_tensor(tokens, dtype=torch.long, device=cost.device).reshape(-1, 1)
        assigned = cost.gather(1, index).squeeze(1)
        return self.potential.initialise_level(self.level, rows, tokens, assigned)


class Af3Offset:
    """The factory: one base metric, one codeword potential, one view per level."""

    #: the registry hands this object to RQLayer, which binds the levels
    needs_rows = True
    has_levels = True

    def __init__(self, base=None, potential=None) -> None:
        if potential is None:
            raise ValueError("af3_offset needs the CodewordPotential that owns s, g and lambda")
        self.base = base if base is not None else euclidean
        self.potential = potential
        self.levels = potential.levels

    def at_level(self, level: int) -> Af3OffsetLevel:
        if not 0 <= int(level) < self.levels:
            raise ValueError("af3_offset: level %r is outside [0, %d)" % (level, self.levels))
        return Af3OffsetLevel(self.base, self.potential, int(level))

    def __call__(self, latent: torch.Tensor, codebook: torch.Tensor, rows=None) -> torch.Tensor:
        """Level 1's cost; the quantiser always binds a level, this is the fallback."""
        return self.at_level(0)(latent, codebook, rows)


def build(base=None, potential=None) -> Af3Offset:
    """Build the metric from the pieces the trainer already holds."""
    return Af3Offset(base=base, potential=potential)
