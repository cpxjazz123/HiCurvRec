"""Exact adjoints for paired, static full-ego row-degree means.

Incoming and outgoing CSR supports are reverses of the same directed edges.
Their already-sorted integer supports can therefore be reused for the opposite
operator's transpose; only the row-degree values change. No support is sorted,
transposed, or converted during backward, and no activation is cached globally.
"""
import torch


def _transpose_pair(incoming, outgoing):
    """Build exact CSR transposes using opposite support, in linear work."""
    in_reciprocal = incoming.crow_indices().diff().to(incoming.dtype).clamp_min(1).reciprocal()
    out_reciprocal = outgoing.crow_indices().diff().to(outgoing.dtype).clamp_min(1).reciprocal()
    incoming_transpose = torch.sparse_csr_tensor(
        outgoing.crow_indices(), outgoing.col_indices(),
        in_reciprocal[outgoing.col_indices()],
        size=(incoming.shape[1], incoming.shape[0]), device=incoming.device)
    outgoing_transpose = torch.sparse_csr_tensor(
        incoming.crow_indices(), incoming.col_indices(),
        out_reciprocal[incoming.col_indices()],
        size=(outgoing.shape[1], outgoing.shape[0]), device=outgoing.device)
    return incoming_transpose, outgoing_transpose


class _StaticSparseMM(torch.autograd.Function):
    """Keep higher-order dense gradients on the same known CSR support pair."""
    @staticmethod
    def forward(ctx, x, operator, adjoint):
        ctx.save_for_backward(operator, adjoint)
        return torch.sparse.mm(operator, x)

    @staticmethod
    def backward(ctx, gradient):
        operator, adjoint = ctx.saved_tensors
        return _StaticSparseMM.apply(gradient, adjoint, operator), None, None


def _adjoint_mm(gradient, adjoint, operator):
    # Ordinary training needs no extra autograd node; create_graph retains the
    # exact reverse pair so double backward cannot trigger native CSR conversion.
    if torch.is_grad_enabled():
        return _StaticSparseMM.apply(gradient, adjoint, operator)
    return torch.sparse.mm(adjoint, gradient)


class _DirectionalMeans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, incoming, outgoing):
        incoming_transpose, outgoing_transpose = _transpose_pair(incoming, outgoing)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(incoming, outgoing, incoming_transpose, outgoing_transpose)
        return torch.sparse.mm(incoming, x), torch.sparse.mm(outgoing, x)

    @staticmethod
    def backward(ctx, gradient_in, gradient_out):
        incoming, outgoing, incoming_transpose, outgoing_transpose = ctx.saved_tensors
        gradient = None
        if gradient_in is not None:
            gradient = _adjoint_mm(gradient_in, incoming_transpose, incoming)
        if gradient_out is not None:
            outgoing_gradient = _adjoint_mm(gradient_out, outgoing_transpose, outgoing)
            gradient = outgoing_gradient if gradient is None else gradient + outgoing_gradient
        return gradient, None, None


def normalized_directional_means(x, incoming, outgoing):
    """Return incoming/outgoing means with exact, conversion-free dense gradients.

    The two square CSR operators must have paired reverse support and static
    reciprocal-row-degree values, in ``x``'s dtype and on ``x``'s device. Isolated
    rows stay zero. Only ``x`` is differentiable; operators have no learned state.
    Transposes live only as long as this autograd graph, not in a module cache.
    """
    if not torch.is_grad_enabled() or not x.requires_grad:
        return torch.sparse.mm(incoming, x), torch.sparse.mm(outgoing, x)
    return _DirectionalMeans.apply(x, incoming, outgoing)
