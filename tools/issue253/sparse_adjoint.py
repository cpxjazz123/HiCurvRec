"""Exact transposes for paired full-ego directional degree means.

Opposite directed support supplies the transpose topology; each gathered column
keeps its ORIGINAL direction's reciprocal degree, computed in the live dtype.
"""
import torch

from tools.issue253 import ego_kernel


def _transpose_pair(incoming, outgoing):
    """Paired uniform-per-occurrence CSR, retaining repeated occurrences."""
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
    @staticmethod
    def forward(ctx, x, operator, adjoint):
        ctx.save_for_backward(operator, adjoint)
        return torch.sparse.mm(operator, x)

    @staticmethod
    def backward(ctx, gradient):
        operator, adjoint = ctx.saved_tensors
        return _StaticSparseMM.apply(gradient, adjoint, operator), None, None


def _adjoint_mm(gradient, adjoint, operator):
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
    """CPU/reference CSR path, with its existing exact higher-order adjoints.

    Supports must be opposite directions of the same occurrences, with uniform
    per-occurrence reciprocal full-row degrees in x's dtype/device. Coalescing
    repeated edges into nonuniform weights is outside this operator contract.
    """
    if not torch.is_grad_enabled() or not x.requires_grad:
        return torch.sparse.mm(incoming, x), torch.sparse.mm(outgoing, x)
    return _DirectionalMeans.apply(x, incoming, outgoing)


class _FusedDirectionalMeans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layout, geometry):
        ctx.set_materialize_grads(False)
        ctx.layout = layout
        ctx.geometry = geometry
        return (ego_kernel.row_means(layout.in_ptr, layout.in_col, x, geometry),
                ego_kernel.row_means(layout.out_ptr, layout.out_col, x, geometry))

    @staticmethod
    def backward(ctx, gradient_in, gradient_out):
        layout, geometry = ctx.layout, ctx.geometry
        gradient = None
        if gradient_in is not None:
            scale_in = ego_kernel.reciprocal_degrees(
                layout, geometry, 'in', gradient_in.device, dtype=gradient_in.dtype)
            gradient = ego_kernel.row_means(
                layout.out_ptr, layout.out_col, gradient_in, geometry, scale=scale_in)
        if gradient_out is not None:
            scale_out = ego_kernel.reciprocal_degrees(
                layout, geometry, 'out', gradient_out.device, dtype=gradient_out.dtype)
            outgoing_gradient = ego_kernel.row_means(
                layout.in_ptr, layout.in_col, gradient_out, geometry, scale=scale_out)
            gradient = outgoing_gradient if gradient is None else gradient + outgoing_gradient
        return gradient, None, None


def fused_directional_means(x, layout, geometry):
    """Read each full row directly from immutable packed ego support."""
    if not torch.is_grad_enabled() or not x.requires_grad:
        return (ego_kernel.row_means(layout.in_ptr, layout.in_col, x, geometry),
                ego_kernel.row_means(layout.out_ptr, layout.out_col, x, geometry))
    return _FusedDirectionalMeans.apply(x, layout, geometry)


def directional_means(x, layout, geometry, operators=None):
    if x.is_cuda:
        return fused_directional_means(x, layout, geometry)
    if operators is None:
        raise ValueError('the CPU reference path needs the CSR operators')
    return normalized_directional_means(x, *operators)
