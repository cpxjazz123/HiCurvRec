"""Full directed per-ego row means with fixed exact-precision edge reductions.

Each program owns a complete row. Short rows use 8/32-edge vector reductions;
long rows reduce every 128-edge chunk and divide once by the FULL row degree.
The scaled mode is an unnormalised weighted sum for the exact transpose.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from tools.issue252.ego_operators import ragged_indices
from tools.issue252.host_transfer import host_int64


@triton.jit
def _edge_sum(col_idx, x_ptr, scale_ptr, base, begin, stop,
              N_FEATURES: tl.constexpr, BLOCK_FEATURES: tl.constexpr,
              BLOCK_EDGES: tl.constexpr, X_ROW: tl.constexpr,
              X_FEATURE: tl.constexpr, SCALE_STRIDE: tl.constexpr,
              HAS_SCALE: tl.constexpr, IS_DOUBLE: tl.constexpr):
    edges = begin + tl.arange(0, BLOCK_EDGES)
    dims = tl.arange(0, BLOCK_FEATURES)
    valid = edges < stop
    local = tl.load(col_idx + edges, valid, other=0).to(tl.int64)
    values = tl.load(x_ptr + (base + local[:, None]) * X_ROW
                     + dims[None, :] * X_FEATURE,
                     valid[:, None] & (dims[None, :] < N_FEATURES), other=0)
    if IS_DOUBLE:
        values = values.to(tl.float64)
    else:
        values = values.to(tl.float32)
    if HAS_SCALE:
        scale = tl.load(scale_ptr + (base + local) * SCALE_STRIDE,
                        valid, other=0)
        values = values * scale[:, None]
    return tl.sum(values, axis=0)


@triton.jit
def _row_mean_kernel(row_ptr, col_idx, x_ptr, out_ptr, row_ego_ptr,
                     ego_base_ptr, ego_start_ptr, scale_ptr,
                     N_FEATURES: tl.constexpr, BLOCK_FEATURES: tl.constexpr,
                     X_ROW: tl.constexpr, X_FEATURE: tl.constexpr,
                     OUT_ROW: tl.constexpr, OUT_FEATURE: tl.constexpr,
                     SCALE_STRIDE: tl.constexpr, HAS_SCALE: tl.constexpr,
                     IS_DOUBLE: tl.constexpr):
    row = tl.program_id(0)
    ego = tl.load(row_ego_ptr + row)
    base = tl.load(ego_base_ptr + ego)
    layout_row = tl.load(ego_start_ptr + ego) + row - base
    begin = tl.load(row_ptr + layout_row)
    stop = tl.load(row_ptr + layout_row + 1)
    degree = stop - begin
    if IS_DOUBLE:
        total = tl.zeros((BLOCK_FEATURES,), tl.float64)
    else:
        total = tl.zeros((BLOCK_FEATURES,), tl.float32)
    if degree > 0:
        if degree <= 8:
            total = _edge_sum(col_idx, x_ptr, scale_ptr, base, begin, stop,
                              N_FEATURES, BLOCK_FEATURES, 8, X_ROW, X_FEATURE,
                              SCALE_STRIDE, HAS_SCALE, IS_DOUBLE)
        elif degree <= 32:
            total = _edge_sum(col_idx, x_ptr, scale_ptr, base, begin, stop,
                              N_FEATURES, BLOCK_FEATURES, 32, X_ROW, X_FEATURE,
                              SCALE_STRIDE, HAS_SCALE, IS_DOUBLE)
        else:
            for start in range(begin, stop, 128):
                total += _edge_sum(col_idx, x_ptr, scale_ptr, base, start, stop,
                                   N_FEATURES, BLOCK_FEATURES, 128, X_ROW, X_FEATURE,
                                   SCALE_STRIDE, HAS_SCALE, IS_DOUBLE)
    if not HAS_SCALE:
        denominator = tl.maximum(degree, 1)
        if IS_DOUBLE:
            total = total / denominator.to(tl.float64)
        else:
            total = tl.div_rn(total, denominator.to(tl.float32))
    dims = tl.arange(0, BLOCK_FEATURES)
    tl.store(out_ptr + row * OUT_ROW + dims * OUT_FEATURE,
             total, dims < N_FEATURES)


def row_means(row_ptr, col_idx, x, geometry, out=None, scale=None):
    """Return complete degree-normalised means, or column-scaled transpose sums.

    Supports noncontiguous inputs/output buffers without copying. All launch
    sizes and strides are tensor metadata or host-known geometry, never device
    values. Only float32/float64 are part of the frozen numerical contract.
    """
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError('ego row means require float32 or float64')
    rows, features = x.shape
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError('ego output must match input shape, dtype and device')
    if scale is not None and (scale.shape != (rows,) or scale.dtype != x.dtype
                              or scale.device != x.device):
        raise ValueError('ego transpose scale must match input rows, dtype and device')
    if rows == 0 or features == 0:
        return out
    _row_mean_kernel[(rows,)](
        row_ptr, col_idx, x, out, geometry['row_ego'], geometry['ego_base'],
        geometry['ego_start'], scale if scale is not None else x,
        N_FEATURES=features, BLOCK_FEATURES=triton.next_power_of_2(features),
        X_ROW=x.stride(0), X_FEATURE=x.stride(1),
        OUT_ROW=out.stride(0), OUT_FEATURE=out.stride(1),
        SCALE_STRIDE=scale.stride(0) if scale is not None else 0,
        HAS_SCALE=scale is not None, IS_DOUBLE=x.dtype == torch.float64,
        num_warps=4, enable_fp_fusion=False)
    return out


def batch_geometry(layout, items, device):
    """Map full static ego rows to batch rows using host-known exact sizes."""
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
    layout_rows = ragged_indices(ego_start, sizes, total)
    return {'row_ego': row_ego, 'ego_base': ego_base, 'ego_start': ego_start,
            'layout_rows': layout_rows, 'nodes': total}


def reciprocal_degrees(layout, geometry, direction, device, *, dtype):
    """Original-direction degrees in the live input dtype, including float64."""
    ptr = getattr(layout, direction + '_ptr')
    rows = geometry['layout_rows']
    degrees = (ptr[rows + 1] - ptr[rows]).clamp_min(1).to(dtype=dtype, device=device)
    return degrees.reciprocal()
