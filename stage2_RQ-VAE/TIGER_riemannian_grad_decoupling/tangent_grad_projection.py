"""M7: Hyperbolic Gradient Decoupling.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
Every gradient the encoder receives through the quantizer is split into a
component along the L1 codeword direction the row was assigned to and the
component orthogonal to it. Only the orthogonal component is allowed to reach the
encoder; the radial component is re-routed to that codeword's own gradient. L1
structure therefore stops being overwritten by the behaviour objective while the
removed signal is conserved in the codebook update rather than discarded.

Operator, item by item (all three parts are the mechanism; nothing else is added)
--------------------------------------------------------------------------------
1. assignment read-back: at every encoder call the row's L1 codeword is
   ``argmin`` over ``l1_layer._distances(latent.detach())``, i.e. the *installed*
   distance, so the shared capacity mask applies and no disabled codeword can ever
   be the projection target;
2. projection: with ``u = normalise(code[idx])`` detached,
   ``g <- g - <g, u> u`` on the encoder-output gradient;
3. conservation: the removed ``<g, u> u`` is accumulated into the gradient of the
   corresponding L1 codeword row (``index_add_`` on the codebook's own gradient),
   so the radial signal is transplanted, not deleted.

No second training objective is introduced: the forward values are untouched (the
operator is the identity in the forward direction) and no loss term is added.

Composition with the shared layer
---------------------------------
Read-only w.r.t. the shared layer: it consumes the installed L1 distance function
(hence the capacity mask) and never changes the feasibility set.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask and diagnostics, assignment rules of every level, all losses,
pair sampling, codebook sizes and parameters, optimiser, schedule, seed, epochs,
evaluation interval.

Fixed parameters (no sweep): none beyond the operator itself (``c`` is not used;
the direction is the codeword's Euclidean unit vector, exactly as specified).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.hyp_losses import encode_points

NORM_EPS = 1e-12


class TangentProjection(torch.autograd.Function):
    """Identity forward; gradient projected off ``u`` and re-routed to the codebook.

    ``codes`` is the live L1 codebook parameter, so the returned code gradient is
    accumulated into the codebook's own gradient by autograd -- that is the
    conservation part of the operator. ``index`` is a detached long tensor.
    """

    @staticmethod
    def forward(ctx, latent: torch.Tensor, codes: torch.Tensor, index: torch.Tensor):
        ctx.save_for_backward(codes, index)
        return latent

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        codes, index = ctx.saved_tensors
        unit = F.normalize(codes[index], dim=-1, eps=NORM_EPS)
        radial = (grad_output * unit).sum(dim=-1, keepdim=True) * unit
        projected = grad_output - radial
        # ``index`` may carry leading level dimensions, as it does when the
        # operator is applied to a per-level trace: flatten them so every level's
        # radial component lands on the codeword it was assigned, and summing the
        # levels is correct because they share one codebook per level only when the
        # caller passes the matching codebook -- otherwise the caller is expected to
        # apply the operator per level.
        flat_index = index.reshape(-1)
        code_grad = torch.zeros_like(codes)
        code_grad.index_add_(0, flat_index, radial.reshape(flat_index.numel(), -1))
        return projected, code_grad, None


def projection_scope_note() -> str:
    """The scope of the installed projection, for the docstring and the metrics.

    The hook sits on the encoder's *output*, so it rewrites every gradient that
    reaches that tensor -- the reconstruction term's included, not only the
    deep-level supervision the arm is meant to protect L1 from. Stated here so
    the claim can be checked rather than read: ``covers_all_losses`` is what the
    implementation does, ``intended`` is what the mechanism name promises.
    """
    return (
        "installed on the encoder output: every gradient through that tensor is "
        "projected, including the reconstruction term, so L1's own objective is "
        "not left intact; the radial part is re-routed to the codebook, not "
        "dropped"
    )


def install_tangent_projection(encoder, l1_layer) -> None:
    """Install the projection on the encoder's output, in every training call.

    A forward hook is used instead of editing ``model/model.py``: the module tree and
    therefore the checkpoint key layout stay exactly as in the control, while every
    path that runs the encoder (main loop, residual trace, mechanism forwards) is
    covered. Inference calls whose output carries no gradient are passed through.

    What the projection is, precisely: an **Euclidean** orthogonal projection.
    The removed direction is the radial component along the assigned L1 codeword
    in ordinary inner-product geometry. "Tangent" here means the tangent
    complement of that one direction in the Euclidean sense; no Poincare metric,
    no ball exponential/logarithmic map and no Riemannian gradient is involved. A
    genuinely hyperbolic projection would be a different mechanism, not a fix.

    Known scope gap, deliberately left visible rather than silently accepted: the
    hook is on the encoder output, so the projection covers the reconstruction
    term's gradient as well as the deep-level supervision. The arm is therefore
    *not* yet isolated to deep supervision, and the numbers it produced should be
    read with that in mind. ``projection_scope_note`` states this in one line for
    the metrics, and the tree's checks pin the behaviour that produces it.
    """
    if getattr(encoder, "_tangent_projection_active", False):
        return

    def hook(module, inputs, output):
        if not torch.is_tensor(output) or not output.requires_grad:
            return output
        with torch.no_grad():
            index = l1_layer._distances(output.detach()).argmin(dim=-1)
        return TangentProjection.apply(output, l1_layer.get_code_embs(), index)

    encoder.register_forward_hook(hook)
    encoder._tangent_projection_active = True
    encoder._tangent_projection_eps = NORM_EPS


def project_deep_supervision(rq_module, residuals: torch.Tensor) -> torch.Tensor:
    """Route only the deep supervision's gradient through the projection.

    ``residuals`` is the per-level trace the behaviour term consumes. The
    projection is applied to *that tensor only*: the forward values pass through
    unchanged, so the loss is identical to the control's, while the gradient that
    reaches the encoder through this path has its radial component removed and
    handed to the assigned L1 codeword.

    Nothing else is touched. The reconstruction and L1 optimisation paths never see
    this operator, which is what makes the arm a decoupling mechanism rather than a
    global gradient rewrite -- and that is exactly what the encoder-output hook in
    the previous arm could not do.
    """
    l1 = rq_module.rq.vq_layers[0]
    with torch.no_grad():
        index = l1._indices(l1._distances(residuals[0]), False)
    codes = l1.get_code_embs()
    expanded = index.unsqueeze(0).expand(residuals.shape[0], -1)
    return TangentProjection.apply(residuals, codes, expanded)


def decoupled_scope_note() -> str:
    """One line for the metrics: what this arm does and does not touch."""
    return (
        "projection applied only to the deep supervision trace; reconstruction and "
        "L1 gradients are untouched; radial component routed to the L1 codeword"
    )


# ---------------------------------------------------------------------------
# Riemannian variant -- NEGATIVE RESULT, kept deliberately.
#
# The obvious way to make the radial removal "hyperbolic" is to take the component
# along the base point in the hyperbolic inner product:
#
#     radial = <g, x>_x / <x, x>_x * x
#
# With A = 1 - c|x|^2 and the Poincare inner product
# <u, v>_x = (u.v)/A + c (u.x)(v.x)/A^2, the conformal factor cancels exactly:
#
#     <g, x>_x = (g.x)/A^2        (since A + c|x|^2 = 1)
#     <x, x>_x = |x|^2 / A^2
#     alpha    = (g.x) / |x|^2
#
# which is the Euclidean coefficient. So that operator is *algebraically identical*
# to the Euclidean one for every curvature, and the arm's own check measures the
# difference at c = 1 to be 0.0000. Claiming it as a distinct hyperbolic mechanism
# would repeat exactly the misattribution these issues are about.
#
# What a genuinely distinct hyperbolic projector needs is the metric-correct
# tangent-space projection (the operator whose removal coefficient is not the
# Euclidean radial one), derived rather than recalled, with the c -> 0 limit stated
# for that operator. That derivation is not in this tree, so the tree documents the
# negative result instead of shipping an operator that does not differ.
# ---------------------------------------------------------------------------
RIEMANN_CURVATURE = 1.0


def hyperbolic_inner(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor,
                     c: float) -> torch.Tensor:
    """<u, v>_x in the ball of curvature c, with all three shaped (..., d)."""
    norm_sq = (x * x).sum(dim=-1, keepdim=True)
    denom = (1.0 - float(c) * norm_sq).clamp_min(1e-6)
    return (u * v).sum(dim=-1, keepdim=True) / denom \
        + float(c) * (u * x).sum(dim=-1, keepdim=True) \
        * (v * x).sum(dim=-1, keepdim=True) / (denom * denom)


def riemannian_radial(g: torch.Tensor, x: torch.Tensor,
                      c: float = RIEMANN_CURVATURE) -> torch.Tensor:
    """The hyperbolic-inner-product radial removal -- provably the Euclidean one.

    Kept with its derivation because the cancellation is the useful part: it is
    what rules out this construction for a genuinely hyperbolic operator.
    """
    return hyperbolic_inner(g, x, x, c) / hyperbolic_inner(x, x, x, c).clamp_min(1e-6) * x
