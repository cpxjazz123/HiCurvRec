"""The codeword potentials `g_k^(l)` and the per-level scale `lambda_l`.

Everything here is one of the formulas of `ideas/AF3_curvature_drive_idea/math.json`:

    H_delta(x) = 0.5 x^2 if |x| <= delta else delta (|x| - 0.5 delta)   (`huber`, delta = 1)
    g_k^(l) <- mean{ s_i : token_l(i) = k }                            (`codeword_potential_init`)
    den_l = median_i |s_i - g_{token_l(i)}^(l)|                        (`lambda_calibration`)
    lambda_l = median_i d(z_i^(l), c_{token_l(i)}) / den_l             (`lambda_calibration`)
    C_ik^(l) = d(z_i^(l), c_k) + lambda_l * H_delta(s_i - g_k^(l))     (`assignment_cost`)

The module owns the trainable scalars `g` (one vector per residual level, so DDP
synchronises them like any other parameter), the frozen standardised potential `s`
(a buffer: `af3_potential.py` computes it once from the training split and nothing
updates it), and the two calibration numbers per level, `den_l` and `lambda_l`
(buffers, never trained -- math.json fixes them at initialisation).

`lambda_l` is deliberately NOT a parameter: it is calibrated once, from the ratio of
two medians, and written into the metrics stream. Making it trainable would move the
one coefficient the mechanism declares fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: constants of the mechanism, copied from math.json constants
DELTA = 1.0
EPS = 1e-8

__all__ = ["CodewordPotential", "DELTA", "EPS", "huber"]


def huber(x: torch.Tensor, delta: float = DELTA) -> torch.Tensor:
    """`H_delta(x)` (formula `huber`): quadratic inside, linear outside."""
    magnitude = x.abs()
    quadratic = 0.5 * x * x
    linear = delta * (magnitude - 0.5 * delta)
    return torch.where(magnitude <= delta, quadratic, linear)


class CodewordPotential(nn.Module):
    """Per-level codeword potentials and the frozen structure they are fitted to."""

    def __init__(
        self,
        codebook_sizes,
        num_items: int,
        delta: float = DELTA,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        sizes = tuple(int(size) for size in codebook_sizes)
        if not sizes:
            raise ValueError("the residual stack needs at least one level")
        if any(size <= 0 for size in sizes):
            raise ValueError("every codebook size must be positive")
        if int(num_items) <= 0:
            raise ValueError("the potential table needs at least one item")
        self.codebook_sizes = sizes
        self.levels = len(sizes)
        self.num_items = int(num_items)
        self.delta = float(delta)
        self.eps = float(eps)

        #: `g_k^(l)`, one trainable scalar per codeword, per level. Zero is the
        #: init the mechanism needs: with `s = 0` the whole term is a stationary
        #: point at zero (switch `behaviour_off`).
        self.g = nn.ParameterList([nn.Parameter(torch.zeros(size)) for size in sizes])

        #: the frozen standardised potential `s_i`, indexed by item id.
        self.register_buffer("s", torch.zeros(self.num_items))
        #: `den_l` and `lambda_l` per level; both recorded in metrics.
        self.register_buffer("dens", torch.zeros(self.levels))
        self.register_buffer("lambdas", torch.zeros(self.levels))
        self.register_buffer("initialised", torch.zeros(self.levels, dtype=torch.bool))

    # ------------------------------------------------------------------ frozen s
    def set_potentials(self, potential) -> None:
        """Copy the frozen standardised potential into the buffer (no grad)."""
        value = torch.as_tensor(potential, dtype=self.s.dtype, device=self.s.device)
        if value.dim() != 1 or value.shape[0] != self.num_items:
            raise ValueError(
                "%s: expected a (%d,) potential, got %r"
                % (self.__class__.__name__, self.num_items, tuple(value.shape))
            )
        if not torch.isfinite(value).all():
            raise ValueError("the structural potential must be finite")
        self.s.copy_(value.detach())

    def potentials_for(self, item_ids) -> torch.Tensor:
        """`s_i` for the rows of a batch; the ids are the index into the table."""
        if item_ids is None:
            raise ValueError(
                "the AF3 term indexes the potential by item id: the batch must carry "
                "its item ids, otherwise the cost would mix up two items"
            )
        ids = torch.as_tensor(item_ids, dtype=torch.long, device=self.s.device)
        ids = ids.reshape(-1)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.num_items):
            raise ValueError(
                "item id out of the frozen potential table: got [%d, %d], table has %d rows"
                % (int(ids.min()), int(ids.max()), self.num_items)
            )
        return self.s.index_select(0, ids)

    # ------------------------------------------------------------------- the cost
    def bias(self, level: int, rows) -> torch.Tensor:
        """`lambda_l * H_delta(s_i - g_k^(l))` as a `(B, K)` offset on the cost.

        The value is detached: it only decides argmin / Sinkhorn, so it must not
        pretend to be a differentiable term. `losses/structural.py` is the term that
        carries the gradient into `g`.
        """
        s_i = self.potentials_for(rows)
        difference = s_i.unsqueeze(1) - self.g[level].unsqueeze(0)
        source = self.s if difference.device == self.s.device else difference
        del source
        deviation = huber(difference, self.delta).detach()
        return self.lambdas[level].to(deviation.dtype) * deviation

    def gather(self, level: int, tokens) -> torch.Tensor:
        """`g_{token_l(i)}^(l)` for every row, keeping the gradient into `g`."""
        index = torch.as_tensor(tokens, dtype=torch.long, device=self.g[level].device)
        return self.g[level].index_select(0, index.reshape(-1))

    # ------------------------------------------------------------- calibration
    @torch.no_grad()
    def initialise_level(self, level: int, rows, tokens, assigned_distances) -> dict:
        """`codeword_potential_init` then `lambda_calibration`, once, on this level.

        `tokens` is the KMeans-time assignment `token_l(i)` and
        `assigned_distances` is `d(z_i^(l), c_{token_l(i)})` under the BASE metric
        (`euclidean`), so the calibration measures the quantity math.json names.
        """
        s_i = self.potentials_for(rows)
        index = torch.as_tensor(tokens, dtype=torch.long, device=s_i.device).reshape(-1)
        distances = torch.as_tensor(assigned_distances, dtype=s_i.dtype, device=s_i.device)
        distances = distances.reshape(-1)
        if index.numel() != s_i.numel() or distances.numel() != s_i.numel():
            raise ValueError(
                "codeword_potential_init: %d tokens / %d distances for %d items"
                % (index.numel(), distances.numel(), s_i.numel())
            )
        if index.numel() == 0:
            raise ValueError("codeword_potential_init: the batch is empty")

        size = self.codebook_sizes[level]
        counts = torch.zeros(size, dtype=s_i.dtype, device=s_i.device)
        counts.index_add_(0, index, torch.ones_like(s_i))
        totals = torch.zeros(size, dtype=s_i.dtype, device=s_i.device)
        totals.index_add_(0, index, s_i)
        means = torch.zeros(size, dtype=s_i.dtype, device=s_i.device)
        nonempty = counts > 0
        means[nonempty] = totals[nonempty] / counts[nonempty]  # empty code stays 0
        self.g[level].copy_(means)

        residual = (s_i - means.index_select(0, index)).abs()
        denominator = float(residual.median())
        if denominator > self.eps:
            scale = float(distances.median()) / denominator
        else:
            scale = 0.0
        self.dens[level] = denominator
        self.lambdas[level] = scale
        self.initialised[level] = True
        return {"level": int(level), "den": denominator, "lambda": scale}

    def metrics(self) -> dict:
        """`den_l` and `lambda_l` per level, for the trainer's metrics stream."""
        record = {
            "structural_delta": self.delta,
            "structural_eps": self.eps,
            "structural_table_items": self.num_items,
            "structural_potential_absmean": float(self.s.abs().mean()),
        }
        for level in range(self.levels):
            record["den_level%d" % (level + 1)] = float(self.dens[level])
            record["lambda_level%d" % (level + 1)] = float(self.lambdas[level])
        record["lambda_max"] = float(self.lambdas.max())
        return record

    def geometry_off(self) -> bool:
        """Switch `geometry_off`: every `lambda_l` is zero, so `C_ik = d`."""
        return bool((self.lambdas == 0).all())
