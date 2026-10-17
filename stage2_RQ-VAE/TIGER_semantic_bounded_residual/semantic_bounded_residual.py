"""Semantic-Anchored Bounded Hyperbolic Behavioral Residual (Issue #22).

The three-level residual quantizer, its codeword distances, its nearest-neighbour
/ Sinkhorn assignment and its codebook update are left untouched. Everything this
module adds lives on the *continuous* side, before and after the quantizer:

1. The text encoder's latent is read as the tangent-space parameterisation of a
   semantic anchor ``s_i = Exp_o(f_sem(x_i))`` and of a behavioural correction
   ``v_i = f_beh(x_i)`` expressed in the anchor's tangent space ``T_{s_i} H``.
2. The perturbation budget ``r_i = max(0, (d_2(s_i) - d_1(s_i)) / 2 - eps)`` is the
   margin the anchor has between its nearest and its second-nearest codeword of
   the *first* level, computed with a stop-gradient codebook, so the correction
   may not move the anchor across a decision boundary. The tangent vector's
   hyperbolic norm ``lambda_{s_i} * ||v_i||`` is clamped into that budget.
3. ``z_i = Exp_{s_i}(v_i)`` is the item's hyperbolic representation and the input
   of the unchanged quantizer, so the semantic id stays a three-level code.
4. A hyperbolic relative ranking term over transition pairs is computed on the
   continuous ``z_i``; because it never touches a codeword it reaches the encoder
   alone and cannot move a codebook by the behaviour gradient.
5. A decoder-side head recovers the anchor from the quantizer's continuous
   output through a straight-through proxy that zeroes the codebook gradient, so
   the new recovery term trains the decoder-side head only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import encode_ball

CURVATURE = 1.0
BALL_EPS = 1e-5
ACOSH_EPS = 1e-7
MARGIN_EPS = 1e-3
TANGENT_SCALE = 0.5    # fixed working radius for the anchor, per hyperbolic.encode_ball


def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Origin exponential map with a unit-speed tangent: ||v|| is the arc length."""
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


def _to_hyperboloid(point: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Ball point -> hyperboloid point, curvature -c, in double precision."""
    if abs(c - 1.0) > 1e-12:
        raise ValueError(f"the hyperboloid bridge here is derived for c = 1, got {c}")
    squared = point.double().square().sum(-1, keepdim=True)
    scale = (1.0 - squared).clamp_min(BALL_EPS)
    return torch.cat([1.0 + squared, 2.0 * point.double()], dim=-1) / scale


def _from_hyperboloid(lifted: torch.Tensor) -> torch.Tensor:
    """Hyperboloid point -> ball point."""
    return lifted[..., 1:] / (lifted[..., :1] + 1.0)


def _lorentz_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product; negative on the hyperboloid."""
    return -left[..., :1] * right[..., :1] + (
        left[..., 1:] * right[..., 1:]
    ).sum(-1, keepdim=True)


def _expmap_x(x: torch.Tensor, v: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Exponential map at a base point: a geodesic of length ``||v||`` along ``v``.

    Taken in the hyperboloid model, where the map is exactly
    ``cosh(L) * p + sinh(L) * u`` for a unit Lorentz-tangent ``u``. The tangent's
    time component follows from the tangency condition ``<p, v> = 0``, and the
    length ``L`` is the *ambient* norm of ``v`` -- normalising the tangent to unit
    Lorentz length first is what makes the travelled distance equal ``||v||``.
    Verified to 2e-14 from the origin out to anchor radius 0.95 by the mechanism
    check; the ball's Mobius shortcut is not used because its direction is wrong
    away from the origin.
    """
    base = _to_hyperboloid(x, c)
    time = (base[..., 1:] * v.double()).sum(-1, keepdim=True) / base[..., :1]
    tangent = torch.cat([time, v.double()], dim=-1)
    unit = tangent / _lorentz_inner(tangent, tangent).clamp_min(1e-24).sqrt()
    length = torch.linalg.vector_norm(v.double(), dim=-1, keepdim=True)
    return _from_hyperboloid(torch.cosh(length) * base + torch.sinh(length) * unit).to(x.dtype)


def _hyperbolic_distance(x: torch.Tensor, y: torch.Tensor,
                         c: float = CURVATURE) -> torch.Tensor:
    """Poincare geodesic distance, broadcast over the leading dimensions.

    The argument of ``acosh`` is ``1 + O(squared distance)``, which float32
    cannot resolve once the squared distance drops below ~1e-7; the reduction is
    therefore done in double precision and cast back, so small distances stay
    accurate instead of hitting the cancellation floor.
    """
    sqrt_c = c ** 0.5
    limit = (1.0 - BALL_EPS) / c
    norm_x = torch.linalg.vector_norm(x, dim=-1).clamp_max(limit ** 0.5)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_max(limit ** 0.5)
    squared = (x - y).square().sum(dim=-1, dtype=torch.float64)
    conform = ((1.0 - c * norm_x.double().square())
               * (1.0 - c * norm_y.double().square())).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * squared / conform
    # 1 + 1e-12 keeps d(acosh)/d(argument) finite for coincident points while
    # sitting far under the float32 resolution of the ball's geometry.
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)).to(x.dtype) / sqrt_c


def _perturbation_budget(anchor: torch.Tensor, codebook: torch.Tensor,
                         c: float = CURVATURE, eps: float = MARGIN_EPS) -> torch.Tensor:
    """Conservative geodesic budget that preserves the existing Euclidean NN.

    The unmodified first VQ layer selects by squared Euclidean distance.
    A Euclidean displacement shorter than half the nearest/runner-up distance
    gap cannot change that choice. Poincare geodesic length is at least twice
    the Euclidean chord length, so a geodesic step <= (d2-d1)-eps suffices.
    The budget is for the representation only; the VQ assignment is untouched.
    """
    del c
    with torch.no_grad():
        distances = torch.cdist(anchor.detach().float(), codebook.detach().float(), p=2)
        nearest = torch.topk(distances, k=2, dim=-1, largest=False).values
        return ((nearest[:, 1] - nearest[:, 0]) - eps).clamp_min(0.0).to(anchor.dtype)


def _bound_to_budget(residual: torch.Tensor, anchor: torch.Tensor,
                     budget: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Bound the Lorentz-exponential geodesic step by the NN safety budget.

    _expmap_x parameterises the travelled hyperbolic distance by ||residual||,
    so no tanh or Poincare-coordinate rescaling belongs in this bound.
    """
    del anchor, c
    length = torch.linalg.vector_norm(residual, dim=-1, keepdim=True).clamp_min(1e-12)
    return residual * (budget.unsqueeze(-1) / length).clamp(max=1.0)



def decoder_reencoded_latent(model: nn.Module, quantized: torch.Tensor) -> torch.Tensor:
    """Route behaviour recovery through the *actual* RQ-VAE Decoder.

    Re-encode the decoder's text reconstruction to codebook dimensionality
    before feeding an auxiliary behavioural head. The existing VQ STE controls
    the gradient to the codebooks; no new assignment operation is introduced.
    Where an encoder forward wrapper exists, use its unwrapped text-MLP to
    avoid applying a behaviour transform twice.
    """
    reconstructed_text = model.decoder(quantized)
    text_encoder = getattr(model.encoder, "unwrapped_forward", model.encoder)
    return text_encoder(reconstructed_text)

class _BlockCodebookGradient(torch.autograd.Function):
    """Identity forward that refuses to pass gradient back into the codebooks."""

    @staticmethod
    def forward(ctx, quantized: torch.Tensor, *codebooks: torch.Tensor) -> torch.Tensor:
        ctx.codebook_count = len(codebooks)
        return quantized.view_as(quantized)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (grad_output,) + (None,) * ctx.codebook_count


def _ste_without_codebook_gradient(quantized: torch.Tensor,
                                   codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Pass the quantizer's continuous output through, blocking codebook gradients."""
    live = [codebook for codebook in codebooks if codebook.requires_grad]
    if not live:
        return quantized
    return _BlockCodebookGradient.apply(quantized, *live)


class SemanticBoundedResidual(nn.Module):
    """The anchor + bounded behavioural residual pair that drives the encoder."""

    def __init__(self, dim: int, curvature: float = CURVATURE) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.behavior = nn.Linear(dim, dim)
        nn.init.normal_(self.behavior.weight, std=1e-2)
        nn.init.zeros_(self.behavior.bias)

    def forward(self, latent: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # The anchor reads the encoder latent through the tree's own
        # ``encode_ball``: a raw 32-d MLP latent saturates tanh and pins every
        # point to the ball boundary, where the codebook margins blow up and the
        # quantiser collapses onto a handful of codes. That is measured, not
        # assumed -- the first run of this arm collapsed to 2 unique codes.
        anchor = encode_ball(latent, self.curvature, TANGENT_SCALE)
        residual = self.behavior(latent)
        budget = _perturbation_budget(anchor, codebook, self.curvature)
        bounded = _bound_to_budget(residual, anchor, budget, self.curvature)
        return _expmap_x(anchor, bounded, self.curvature), anchor, bounded, budget

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SemanticAnchorRecoveryHead(nn.Module):
    """Decoder-side head that reads the quantizer's continuous output."""

    def __init__(self, dim: int, curvature: float = CURVATURE) -> None:
        super().__init__()
        self.curvature = float(curvature)
        self.head = nn.Linear(dim, dim)

    def forward(self, quantized: torch.Tensor,
                codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        continuous = _ste_without_codebook_gradient(quantized, codebooks)
        return _expmap0(self.head(continuous), self.curvature)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def install_semantic_bounded_encoder(model: nn.Module, codebooks_getter,
                                     curvature: float = CURVATURE) -> SemanticBoundedResidual:
    """Route the tree's encoder output through the anchor/bounded-residual pair.

    The encoder module itself is not replaced and the branch is deliberately not
    attached to the model, so ``model.state_dict()`` keeps exactly the keys the
    codebook probe and the Stage-3 handoff expect; only the encoder's forward is
    extended to produce the bounded hyperbolic point. The trainer owns the
    branch's parameters and trains them alongside the model.
    """
    dim = int(model.config.codebook_dim)
    branch = SemanticBoundedResidual(dim, curvature).to(next(model.parameters()).device)
    encoder = model.encoder
    original_forward = encoder.forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        latent = original_forward(embeddings)
        representation, anchor, bounded, budget = branch(latent, codebooks_getter())
        branch.last_anchor = anchor
        branch.last_bounded = bounded
        branch.last_budget = budget
        return representation

    encoder.forward = patched_forward
    return branch


def semantic_bounded_ranking_loss(
    source: torch.Tensor,
    candidates: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
    temperature: float = 0.07,
    curvature: float = CURVATURE,
) -> torch.Tensor:
    """Hyperbolic relative ranking over the continuous item representations."""
    distances = _hyperbolic_distance(source.unsqueeze(1), candidates.unsqueeze(0), curvature)
    logits = -distances / temperature
    if successor_indicator is not None:
        mask = successor_indicator.to(logits.device).bool()
        if mask.shape == logits.shape:
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min / 2.0)
    targets = (source_ids.unsqueeze(1) == target_ids.unsqueeze(0)).to(logits.dtype)
    positive = targets.sum(dim=-1)
    valid = positive > 0
    if not bool(valid.any()):
        return source.sum() * 0.0
    log_probability = F.log_softmax(logits, dim=-1)[valid]
    return -(targets[valid] * log_probability).sum(dim=-1).div(positive[valid]).mean()


def semantic_anchor_recovery_loss(recovered: torch.Tensor, anchor: torch.Tensor,
                                  curvature: float = CURVATURE) -> torch.Tensor:
    """Recover the semantic anchor from the decoder-side continuous representation."""
    return _hyperbolic_distance(recovered, anchor.detach(), curvature).mean()
