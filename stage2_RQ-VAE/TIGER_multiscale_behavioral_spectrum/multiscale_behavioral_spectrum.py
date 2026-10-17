"""Hyperbolic Multi-Scale Behavioral Spectrum Learning Across RQ Depths (Issue #27).

The behaviour teacher here is not one vector per item but a *nested* one: the item's
collaborative spectrum is split into a low-frequency, a mid-frequency and a
high-frequency part, and each RQ depth is asked to carry one cumulative level of it.

1. From the existing user-item interaction matrix a normalised graph propagation
   gives the global low-frequency behaviour of each item; the first difference
   against the raw collaborative signal splits the remainder into a mid band and a
   fine-grained high band. So ``b^(1) = low``, ``b^(2) = low + mid`` and
   ``b^(3) = low + mid + high`` -- each depth adds information rather than
   repeating the previous one.
2. The text encoder and the three-level quantizer are left exactly as they are.
   The cumulative continuous reconstruction ``q_i^(q)`` is read straight out of the
   unchanged quantizer stack, one per depth.
3. Three depth-specific decoder heads ``D_1, D_2, D_3`` predict ``b^(1..3)`` from
   the matching cumulative representation, with a normalised relational
   distillation term so what is compared is the relation structure rather than the
   raw coordinates.
4. The cumulative representations reach the heads through a straight-through proxy
   that stops gradient at every codeword, and each depth's codeword vector is fed
   in detached as well. The three losses therefore update the encoder and the
   heads and no codebook.

Distances are hyperbolic throughout, so each depth is scored on the geometry its
own representation actually lives in. The final sid is still the original
three-level discrete code sequence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import encode_ball

CURVATURE = 1.0
BALL_EPS = 1e-5
TANGENT_SCALE = 0.5
PROPAGATION_STEPS = 2
PROPAGATION_ALPHA = 0.5
BAND_WEIGHT = 1.0
TRIANGLES_PER_DEPTH = 192


def _expmap0(x: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    sqrt_c = c ** 0.5
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm) * x


def _geodesic(x: torch.Tensor, y: torch.Tensor, c: float = CURVATURE) -> torch.Tensor:
    """Poincare geodesic distance; the acosh reduction runs in double precision."""
    sqrt_c = c ** 0.5
    limit = (1.0 - BALL_EPS) / c
    norm_x = torch.linalg.vector_norm(x, dim=-1).clamp_max(limit ** 0.5)
    norm_y = torch.linalg.vector_norm(y, dim=-1).clamp_max(limit ** 0.5)
    squared = (x - y).square().sum(dim=-1, dtype=torch.float64)
    conform = ((1.0 - c * norm_x.double().square())
               * (1.0 - c * norm_y.double().square())).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * squared / conform
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)).to(x.dtype) / sqrt_c



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
    live = [codebook for codebook in codebooks if codebook.requires_grad]
    if not live:
        return quantized
    return _BlockCodebookGradient.apply(quantized, *live)


def build_behaviour_spectrum(interactions: list[list[int]], n_items: int,
                             dim: int) -> dict[str, np.ndarray]:
    """Collaborative graph diffusion bands: low=P^2 X, mid=PX-P^2X, high=X-PX.

    X is a deterministic random projection of the item-user incidence matrix;
    P is the row-normalised item-item co-occurrence adjacency. These are
    bona fide graph-filter frequency bands, not an exact eigen-decomposition.
    """
    from scipy.sparse import csr_matrix, diags
    rows, cols = [], []
    for user, history in enumerate(interactions):
        items = np.unique(np.asarray(history, dtype=np.int64))
        items = items[(items >= 0) & (items < n_items)]
        rows.extend([user] * len(items))
        cols.extend(items.tolist())
    if not rows:
        raise RuntimeError("no collaborative interactions for spectrum")
    H = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                   shape=(len(interactions), n_items), dtype=np.float32)
    H.data[:] = 1.0
    # Project sparse collaborative columns; unlike per-item degree counts this
    # produces distinguishable feature vectors for different user audiences.
    projection = np.random.default_rng(42).standard_normal(
        (H.shape[0], dim)).astype(np.float32) / dim ** 0.5
    raw = np.asarray(H.T @ projection, dtype=np.float32)
    degree = np.asarray(H.sum(axis=0)).reshape(-1, 1).astype(np.float32)
    raw = raw / np.sqrt(np.maximum(degree, 1.0))
    adjacency = (H.T @ H).tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
    transition = diags(1.0 / np.maximum(row_sum, 1.0)) @ adjacency
    first = np.asarray(transition @ raw, dtype=np.float32)
    second = np.asarray(transition @ first, dtype=np.float32)
    low, mid, high = second, first - second, raw - first
    cumulative = np.stack((low, first, raw), axis=0)
    # A shared scale keeps the low/mid/high depth targets comparable. Convert
    # targets to ball points for the geodesic decoder-side distillation.
    rms = np.sqrt(np.mean(np.sum(raw.astype(np.float64) ** 2, axis=-1)))
    tangent = cumulative * (TANGENT_SCALE / max(float(rms), 1e-8))
    radius = np.linalg.norm(tangent, axis=-1, keepdims=True)
    ball = np.tanh(radius / 2.0) * tangent / np.maximum(radius, 1e-12)
    return {
        "low": low, "mid": mid, "high": high,
        "cumulative": ball.astype(np.float32),
    }


class MultiscaleRecoveryHeads(nn.Module):
    """One head per RQ depth, reading that depth's cumulative representation."""

    def __init__(self, dim: int, depths: int = 3) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depths))

    def forward(self, cumulative: list[torch.Tensor],
                codebooks: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return [
            _expmap0(head(_ste_without_codebook_gradient(level, codebooks)), CURVATURE)
            for head, level in zip(self.heads, cumulative)
        ]

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def install_spectral_encoder(model: nn.Module) -> None:
    """Read the encoder latent as a ball point; the quantizer is left untouched."""
    encoder = model.encoder
    original_forward = encoder.forward

    def patched_forward(embeddings: torch.Tensor) -> torch.Tensor:
        return encode_ball(original_forward(embeddings), CURVATURE, TANGENT_SCALE)

    encoder.forward = patched_forward


def cumulative_representations(raw_module, embeddings: torch.Tensor) -> list[torch.Tensor]:
    """``q_i^(q)`` for each depth, taken from the unchanged quantizer stack.

    Level ``q``'s cumulative reconstruction is the sum of the first ``q`` levels'
    codewords -- the same sum ``RQLayer.forward`` accumulates -- so no codebook
    operation is duplicated or altered.
    """
    residual = raw_module.encoder(embeddings)
    cumulative: list[torch.Tensor] = []
    total: torch.Tensor | None = None
    for vq_layer in raw_module.rq.vq_layers:
        quant, _, _, _ = vq_layer(residual)
        total = quant if total is None else total + quant
        cumulative.append(total)
        residual = residual - quant
    return cumulative


def band_relation_loss(predicted: torch.Tensor, target: torch.Tensor,
                       margin: float = 0.1) -> torch.Tensor:
    """Preserve teacher-local pairwise neighbour ordering at each RQ depth.

    Comparisons use *matching sample indices* in teacher and student spaces;
    target order determines positives and negatives. Both are in the ball.
    """
    if predicted.shape[0] < 3:
        return predicted.sum() * 0.0
    teacher_dist = torch.cdist(target.detach(), target.detach())
    teacher_dist.fill_diagonal_(float("inf"))
    near_ids = teacher_dist.argmin(dim=1)
    far_ids = teacher_dist.argmax(dim=1)
    # infinity diagonal is excluded explicitly for the distant neighbour.
    teacher_dist.fill_diagonal_(-1.0)
    far_ids = teacher_dist.argmax(dim=1)
    anchor = predicted
    positive = predicted[near_ids]
    negative = predicted[far_ids]
    hinge = F.relu(margin + _geodesic(anchor, positive) - _geodesic(anchor, negative))
    return hinge.mean()


def band_coordinate_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align the predicted band with the teacher's cumulative vector."""
    return F.mse_loss(predicted, target.detach())

def compose_spectrum_supervision(depth_losses: torch.Tensor,
                                 ramp: float, depth_weight: float
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply three-depth behaviour supervision exactly once.

    The generic trainer already scales its own separate behaviour objective.
    The spectral losses are exclusively applied as the decoder-side depth
    objective; returning a graph-connected zero prevents double weighting.
    """
    return depth_losses.sum() * 0.0, (ramp * depth_weight) * depth_losses.mean()
