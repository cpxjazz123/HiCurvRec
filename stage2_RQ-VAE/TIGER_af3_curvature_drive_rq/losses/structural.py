"""The structural term: the codeword potentials are fitted to the frozen potential.

`structural_loss` of `ideas/AF3_curvature_drive_idea/math.json`:

    L_struct = mean_{i, l} H_delta(s_i - g_{token_l(i)}^(l))          (`structural_loss`)

`H_delta` is the `huber` formula with `delta = 1`. The weight `structural_weight = 1.0`
is applied by `losses/__init__.py:TotalLoss`, in the same place the baseline applies
`beta`, so the trainer still adds no weight of its own.

Gradient routing (`gradient_routing`): the mean runs over `g` gathered at the frozen
tokens, and `s` is a buffer, so `dL_struct/dg != 0` while `dL_struct/dz = 0` -- the
encoder gets its gradient from the quantisation term alone, through the
straight-through estimator, exactly as before.

Switches, as math.json states them:

* `behaviour_off`: `s = 0` and `g = 0` -> every deviation is 0 -> `L_struct = 0`.
* `geometry_off`: `lambda_l = 0` -> the cost is the plain distance again; this term
  still trains `g`, which is why it is a separate switch.
* `structural_weight = 0` alone is NOT `behaviour_off`: it removes this term only.
"""

from __future__ import annotations

import torch

NAME = "structural"
#: What a test may assume about this formula (see the tree's tests/).
SPEC = {
    "form": "huber_potential",
    "delta": 1.0,
    "zero_when_equal": True,
    "symmetric": True,
    "shift_invariant": False,   # the paired sides are a potential and a codeword value
    "scale_power": 1,           # Huber is linear outside |x| <= delta
    "detached": "encoder",      # the gradient reaches g only, never z
    "weight": 1.0,
}
DESCRIPTION = (
    "mean over items and levels of H_delta(s_i - g_{token_l(i)}^(l)); trains the "
    "codeword potentials only"
)

#: `structural_weight` and `delta`, copied from math.json constants.
STRUCTURAL_WEIGHT = 1.0
DELTA = 1.0

__all__ = ["NAME", "SPEC", "DESCRIPTION", "STRUCTURAL_WEIGHT", "DELTA",
           "huber", "structural_loss"]


def huber(x: torch.Tensor, delta: float = DELTA) -> torch.Tensor:
    """`H_delta(x)` (formula `huber`)."""
    magnitude = x.abs()
    quadratic = 0.5 * x * x
    linear = delta * (magnitude - 0.5 * delta)
    return torch.where(magnitude <= delta, quadratic, linear)


def structural_loss(potentials, tokens, codeword, delta: float = DELTA) -> torch.Tensor:
    """`mean_{i,l} H_delta(s_i - g_{token_l(i)}^(l))`.

    `potentials` is `(B,)` (`s_i`, already detached -- it is a frozen buffer),
    `tokens` is `(B, L)` (`token_l(i)`) and `codeword` is the
    `model.codeword_potential.CodewordPotential` that owns `g`.
    """
    value = torch.as_tensor(potentials).reshape(-1)
    if value.numel() == 0:
        raise ValueError("%s: the batch is empty, so the mean is undefined" % NAME)
    if not value.is_floating_point():
        raise TypeError("%s: the potential must be floating point, got %r"
                        % (NAME, value.dtype))
    index = torch.as_tensor(tokens)
    if index.dim() != 2:
        raise ValueError("%s: tokens must be (B, levels), got %r"
                         % (NAME, tuple(index.shape)))
    if index.shape[0] != value.numel():
        raise ValueError(
            "%s: %d tokens for %d items; the two sides must be the same batch"
            % (NAME, index.shape[0], value.numel())
        )
    if not value.requires_grad:
        value = value.detach()

    terms = []
    for level in range(index.shape[1]):
        deviation = value - codeword.gather(level, index[:, level])
        terms.append(huber(deviation, delta))
    return torch.cat(terms).mean()
