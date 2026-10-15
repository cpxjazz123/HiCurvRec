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

Classification of this operator
------------------------------
This arm is a **Euclidean control**, not a hyperbolic mechanism. The projection
removes the component of the gradient along the assigned L1 codeword in the ordinary
inner product; no Poincare metric, ball map or Riemannian gradient is involved.
"Tangent" is used in its Euclidean sense -- the tangent complement of one direction
in a normed space -- and the arm is named for that, not for ball geometry.

The hyperbolic reading was tested and does not hold: taking the same component in the
Poincare inner product cancels the conformal factor and yields the identical
operator (see TIGER_riemannian_grad_decoupling, which records that derivation and
measurement). An operator that is provably the Euclidean one must not be reported as
hyperbolic, so it is filed as the control it is.

"""
from __future__ import annotations

import torch
import torch.nn.functional as F

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
        code_grad = torch.zeros_like(codes)
        code_grad.index_add_(0, index, radial)
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
