"""M5: Multi-Interest Hyperbolic Branching.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
Inside one crowded L1 bucket the items belong to several unrelated interests. The
level-2 assignment is made interest-aware: every L1 bucket gets K axes estimated
from the training transition graph of the items in that bucket, the anchor is
matched to the axis it is closest to, and child codewords are penalised for
pointing away from that axis. Items of different interests therefore take
different children of the same parent instead of competing for one.

Math
----
Per L1 bucket ``b`` let ``A`` be the bucket's transition co-occurrence matrix
(``A[i, j]`` = observed ``anchor -> successor`` count, deflated power iteration on
``M = A + A^T``) and let ``r_i`` be the residual entering level 2 for item ``i``.
The axes are

    v_1, ..., v_K  = top-K eigenvectors of M (power iteration, 20 steps, deflation)
    d_{b,j}        = normalise( sum_i v_j[i] * r_i )        axis in residual space

For a row whose parent (level-1) codeword is ``p`` and whose residual is ``x``:

    u_x     = (x - p) / ||x - p||
    j*(x)   = argmax_j <u_x, d_{b(x), j}>        the row's own interest
    u_c     = (c - p) / ||c - p||                candidate direction inside the parent
    d(x, c) = d_P(x, c) + strength * (1 - <u_c, d_{b, j*(x)}>)

The penalty depends on the candidate through its direction inside the parent, which
is what makes it change the argmin: a row-constant penalty (the naive
``1 - cos(u_x, d)`` with no candidate term) would cancel out of the comparison and
leave the assignment untouched. Closed form, no ``(N, K, d)`` tensor:

    <c - p, d>   = <c, d> - <p, d>
    ||c - p||    = sqrt(||c||^2 - 2<p, c> + ||p||^2)

Composition with the shared layer
---------------------------------
Installed on level 2 only, before ``scripts/capacity_mask.py`` wraps L1, so level 1
keeps the crowded 128-codeword Euclidean+sentinel rule and level 3 keeps the
control's rule. Buckets are measured from an assignment that already respects that
128-codeword capacity, so the axes describe the crowded regime.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask, L1 and L3 assignment rules, all losses (reconstruction,
commitment, behaviour contrastive), pair sampling, codebook sizes and parameters,
optimiser, schedule, seed, epochs, evaluation interval.

Fixed parameters (no sweep): ``c = 1.0``, ``strength = 0.5``, ``K = 2``,
power-iteration steps 20.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import torch

from hyperbolic import pair_poincare_distance
from scripts.hyp_losses import encode_points, shared_tangent_scale

BRANCH_CURVATURE = 1.0
BRANCH_STRENGTH = 0.5
BRANCH_K = 2
POWER_ITERS = 20


def _power_iteration(matrix: np.ndarray, iters: int) -> np.ndarray:
    """Dominant eigenvector of a non-negative symmetric matrix."""
    vector = matrix.sum(axis=1)
    if not np.any(vector > 0):
        vector = np.ones(matrix.shape[0])
    vector = vector / np.linalg.norm(vector)
    for _ in range(int(iters)):
        updated = matrix @ vector
        norm = np.linalg.norm(updated)
        if norm <= 0:
            break
        vector = updated / norm
    return vector


def interest_directions(module, embeddings: torch.Tensor, transitions: np.ndarray,
                        k: int = BRANCH_K, iters: int = POWER_ITERS,
                        allowed: int | None = None, c: float = BRANCH_CURVATURE,
                        n_embed: int = 256) -> torch.Tensor:
    """``(n_embed, k, dim)`` interest axes, one row per L1 bucket (zeros if unused)."""
    with torch.no_grad():
        latent = module.encoder(embeddings)
        matrix = module.rq.vq_layers[0]._distances(latent)
        if allowed is not None:
            keep = torch.arange(int(allowed), device=matrix.device)
            disabled = torch.ones(matrix.shape[-1], dtype=torch.bool, device=matrix.device)
            disabled[keep] = False
            matrix = matrix.masked_fill(disabled, float("inf"))
        codes = matrix.argmin(dim=-1)
        residual = latent - module.rq.vq_layers[0].embed_code(codes)

    code_np = codes.cpu().numpy()
    residual_np = residual.cpu().numpy().astype(np.float64)
    transitions = np.asarray(transitions, dtype=np.int64)
    anchor, successor = transitions[:, 0], transitions[:, 1]
    bucket_of_anchor = code_np[anchor]
    bucket_of_successor = code_np[successor]
    dim = residual_np.shape[1]
    directions = np.zeros((n_embed, k, dim), dtype=np.float64)
    local_index = np.full(code_np.shape[0], -1, dtype=np.int64)

    for bucket in range(n_embed):
        members = np.flatnonzero(code_np == bucket)
        if members.size < 2:
            continue
        local_index[:] = -1
        local_index[members] = np.arange(members.size)
        selected = (bucket_of_anchor == bucket) & (bucket_of_successor == bucket)
        if not bool(selected.any()):
            continue
        rows = local_index[anchor[selected]]
        cols = local_index[successor[selected]]
        cooccurrence = np.zeros((members.size, members.size), dtype=np.float64)
        np.add.at(cooccurrence, (rows, cols), 1.0)
        graph = cooccurrence + cooccurrence.T
        if graph.sum() <= 0:
            continue
        for axis in range(k):
            vector = _power_iteration(graph, iters)
            if np.linalg.norm(vector) <= 0:
                break
            directions[bucket, axis] = residual_np[members].T @ vector
            eigenvalue = float(vector @ graph @ vector)
            graph = graph - eigenvalue * np.outer(vector, vector)
            if graph.shape[0] and float(np.abs(graph).max()) < 1e-12:
                break

    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    directions[norms[..., 0] <= 1e-12] = 0.0
    return torch.from_numpy(directions).to(dtype=torch.float32)


def _branch_penalty(codes: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between each candidate's direction and the row's own axis.

    Both sides are in residual space. The candidates are the level-2 codewords,
    which live in residual space by construction (the RQ walk subtracts the L1
    codeword before handing the residual to level 2), and the axes are estimated
    from residuals as well. The previous version computed
    ``cos(code - parent, axis)``: an L1-space parent codeword subtracted from an
    L2 codeword, so the two sides of the cosine were not in one frame.

    Sign: this term is *added*, and that is deliberate, and it is the opposite of
    the fan-out arm M1. Here the claim is that items with different interests
    should take different children of the same parent: each row is matched to the
    axis it is closest to, and a child aligned with *that* axis looks closest, so
    rows with different axes diverge onto different children. Subtracting the
    term here would invert the mechanism.
    """
    unit_c = torch.nn.functional.normalize(codes, dim=-1)
    unit_axis = torch.nn.functional.normalize(axes, dim=-1)
    cosine = unit_axis @ unit_c.t()
    penalty = 1.0 - cosine.clamp(-1.0, 1.0)
    has_axis = axes.abs().sum(-1) > 0
    return torch.where(has_axis.unsqueeze(1), penalty, torch.zeros_like(penalty))


def _select_axes(latent: torch.Tensor, parent_indices: torch.Tensor,
                directions: torch.Tensor) -> torch.Tensor:
    """The bucket axis each row is closest to, shape ``(N, dim)``.

    ``parent_indices`` (integer level-1 codewords) selects the bucket; the
    geometry is the row's own residual. The previous version normalised
    ``latent - parent_codes`` -- the L1 codeword was subtracted from a residual
    that had already given it up, evaluating ``z - 2*e1``. The residual is the
    item's direction inside its parent's cell, so it is used as is.
    """
    per_row = directions[parent_indices]                   # (N, k, dim)
    unit = torch.nn.functional.normalize(latent, dim=-1)
    scores = (per_row * unit.unsqueeze(1)).sum(dim=-1)
    best = scores.argmax(dim=-1)
    return per_row[torch.arange(per_row.shape[0], device=per_row.device), best]


def install_interest_branch(layer, parent_layer, directions: torch.Tensor,
                            c: float = BRANCH_CURVATURE,
                            strength: float = BRANCH_STRENGTH) -> None:
    """Interest-aware level-2 assignment; reads the parent indices per row.

    The axes are registered as a persistent buffer: they participate in choosing
    codes, so a checkpoint that does not carry them cannot reproduce its own SIDs
    on reload.
    """
    if getattr(layer, "_interest_branch_active", False):
        return
    original = layer._distances
    axes_table = torch.as_tensor(directions, dtype=torch.float32,
                                 device=layer.get_code_embs().device)
    if not hasattr(layer, "_interest_axes"):
        layer.register_buffer("_interest_axes", axes_table)
    else:
        with torch.no_grad():
            layer._interest_axes.copy_(axes_table)
    directions = layer._interest_axes

    def distances(self, latent):
        parents = getattr(self, "_parent_indices", None)
        if parents is None:
            return original(latent)
        codes = self.get_code_embs()
        scale = shared_tangent_scale(codes, c)
        points = encode_points(latent, c, scale)
        code_points = encode_points(codes, c, scale)
        geodesic = pair_poincare_distance(points, code_points, c)
        axes = _select_axes(latent, parents, directions)
        return geodesic + float(strength) * _branch_penalty(codes, axes)

    layer._distances = MethodType(distances, layer)
    layer._interest_branch_active = True
    layer._interest_branch_strength = float(strength)
    layer._interest_branch_k = int(directions.shape[1])


def _trace_with_parents(module, embeddings: torch.Tensor, with_tokens: bool = False):
    """Residual trace that hands each level its parent indices, like RQLayer.forward."""
    residual = module.encoder(embeddings)
    traces = [residual]
    tokens = []
    parents = None
    for level, vq_layer in enumerate(module.rq.vq_layers):
        quant, _, _, indices = vq_layer(residual, parent_indices=parents if level else None)
        tokens.append(indices)
        residual = residual - quant
        traces.append(residual)
        parents = indices
    stacked = torch.stack(traces[: module.rq.codebook_num], dim=0)
    if with_tokens:
        return stacked, torch.stack(tokens, dim=0)
    return stacked
