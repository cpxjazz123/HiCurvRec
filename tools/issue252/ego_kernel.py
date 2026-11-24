"""Fused per-ego directional means over the frozen ego layout.

The two-hop ego of an item is a complete induced subgraph stored in the frozen
layout: every ego owns a contiguous node range, a row pointer and the local
column indices of its directed edges. A batch of items is therefore a list of
egos, and the runtime directional mean of a batch is a per-row segmented sum of
the batch-local node vectors.

This module evaluates that segmented sum directly on the frozen layout. It does
not build the batch's block-diagonal CSR, and it never reads a device value to
decide a shape: the row count, the ego bases and the layout starts are host
facts. The arithmetic is the same row-degree-normalised mean the sparse CSR
operator computes; only the summation order inside a row may differ.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from tools.issue252.ego_operators import ragged_indices
from tools.issue252.host_transfer import host_int64


@triton.jit
def _row_mean_kernel(row_ptr, col_idx, x_ptr, out_ptr, row_ego_ptr, ego_base_ptr,
                     ego_start_ptr, scale_ptr, n_rows, DIM: tl.constexpr,
                     HAS_SCALE: tl.constexpr, IS_DOUBLE: tl.constexpr):
    """One program per batch-local row: sum the row's columns and normalise."""
    row = tl.program_id(0)
    if row >= n_rows:
        return
    ego = tl.load(row_ego_ptr + row)
    base = tl.load(ego_base_ptr + ego)
    layout_start = tl.load(ego_start_ptr + ego)
    layout_row = layout_start + (row - base)
    begin = tl.load(row_ptr + layout_row)
    stop = tl.load(row_ptr + layout_row + 1)
    dims = tl.arange(0, DIM)
    if IS_DOUBLE:
        total = tl.zeros([DIM], dtype=tl.float64)
    else:
        total = tl.zeros([DIM], dtype=tl.float32)
    for column in range(begin, stop):
        local = tl.load(col_idx + column).to(tl.int64)
        values = tl.load(x_ptr + (base + local) * DIM + dims)
        if HAS_SCALE:
            total += values * tl.load(scale_ptr + base + local)
        else:
            total += values
    if HAS_SCALE:
        result = total
    else:
        degree = (stop - begin).to(tl.float32)
        result = total / tl.maximum(degree, 1.0)
    tl.store(out_ptr + row * DIM + dims, result)


def row_means(row_ptr, col_idx, x, geometry, out=None, scale=None):
    """Row-degree-normalised means of `x` over the frozen layout's rows.

    `geometry` carries the batch mapping: the ego of every batch-local row, the
    batch-local node base of every ego and the layout's first node of every ego.
    With `scale` the kernel accumulates the scaled columns instead, which is the
    transposed operator the adjoint needs.
    """
    rows = int(x.shape[0])
    dim = int(x.shape[1])
    if out is None:
        out = torch.empty_like(x)
    _row_mean_kernel[(rows,)](
        row_ptr, col_idx, x, out, geometry['row_ego'], geometry['ego_base'],
        geometry['ego_start'], scale if scale is not None else x, rows,
        DIM=dim, HAS_SCALE=scale is not None, IS_DOUBLE=x.dtype == torch.float64,
        num_warps=4)
    return out


def batch_geometry(layout, items, device):
    """Host-known batch geometry for the frozen layout.

    `items` is the host array of the batch's item ids. The ego of every
    batch-local row, the batch-local node base of every ego, the layout's first
    node of every ego and the layout's rows of the batch are all built from
    host-known counts; no device value decides a shape.
    """
    items = np.asarray(items, dtype=np.int64)
    node_counts = layout.node_counts[items]
    starts = layout.offsets_np[items]
    sizes = host_int64(node_counts, device)
    total = int(node_counts.sum())
    ego_base = torch.zeros(len(items) + 1, dtype=torch.int64, device=device)
    torch.cumsum(sizes, 0, out=ego_base[1:])
    row_ego = torch.repeat_interleave(
        torch.arange(len(items), device=device, dtype=torch.int64), sizes,
        output_size=total)
    ego_start = host_int64(starts, device)
    layout_rows = ragged_indices(ego_start, sizes.to(torch.int64), total)
    return {'row_ego': row_ego, 'ego_base': ego_base, 'ego_start': ego_start,
            'layout_rows': layout_rows, 'nodes': total}


def reciprocal_degrees(layout, geometry, direction, device):
    """Per batch-local node the reciprocal degree of one direction.

    The layout stores every ego's rows contiguously, so the batch's rows are
    exactly the batch's nodes in order and the degrees follow directly.
    """
    ptr = getattr(layout, direction + '_ptr')
    rows = geometry['layout_rows']
    degrees = (ptr[rows + 1] - ptr[rows]).to(torch.float32)
    return degrees.reciprocal().clamp(max=1.0)
