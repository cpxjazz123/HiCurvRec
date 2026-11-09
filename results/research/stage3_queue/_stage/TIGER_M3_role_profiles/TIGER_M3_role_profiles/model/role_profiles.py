"""Issue #142 (M3): local curvature profiles as a TRAIN-only structural teacher.

A single AF3 scalar per edge does not describe what kind of node a commodity is:
a community interior node, a bridge and a hub can share an edge-AF3 mean and
differ in how their incident edges are spread. This module freezes that
description, keeps it out of the export path, and asks the quantizer to carry it.

The teacher
-----------
For every node of the TRAIN behaviour graph, over its incident-edge AF3 values
``F_ij = 4 - d_i - d_j + 3 T_ij``:

    p_i = [mean, std, q_0.1, q_0.5, q_0.9, log(1 + d_i), t_i]

with ``t_i`` the local closure ratio ``T_i / C(d_i, 2)`` (zero for ``d_i < 2``).
The teacher is computed once from the canonical TRAIN graph, with a mask for
isolated and single-edge nodes, and a per-dimension standardisation whose
location and scale come from the valid nodes of that same graph. Nothing here
reads valid or test behaviour.

The head
--------
``h_theta`` is a two-layer map from the quantized readout ``phi(q)`` to the
profile. It lives inside the model so DDP synchronises it, and it is never called
on the export path: the export is the three tokens, unchanged.

The objective
-------------
    L = L_RQ + s * [ w_profile * Huber(h_theta(phi) - p) + w_geometry * eta * Huber(d_mix(phi_i, phi_j) - (a + b ||p_i - p_j||)) ]

The second term says only that two commodities whose structural roles are alike
should not be far apart in the quantized geometry, and it is limited to the
TRAIN support's real edges. ``a`` and ``b`` are a least-squares fit on that support
and are frozen. Whether the term is kept at all is decided *before* the run by a
frozen diagnostic: if the fitted line disagrees with the sign of the support's
edges systematically, the term is removed and the mechanism is declared single
(profile-to-quantizer only). ``tests/test_math.py`` runs that diagnostic on the
real graph and records its answer.

``s`` comes from :class:`af3_hierarchy.GradientBudget`, the same frozen share the
repository already uses for auxiliary structure terms.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .af3_hierarchy import GradientBudget, gradient_norm, soft_assignments
from .structure_surface import huber_penalty, mixed_delta, soft_readout

#: Nodes per step.
NODES_PER_STEP = 256
#: Support edges per step.
EDGES_PER_STEP = 256
#: Huber width on the standardised profile residual.
HUBER_DELTA = 1.0
#: Declared unit weights; the scale is the gradient-share bound.
LOSS_WEIGHTS = {"profile": 1.0, "geometry": 0.5}
#: The frozen gradient-share bound.
GRADIENT_SHARE = 0.10
#: Profile dimensions, in the order the specification names them.
PROFILE_DIM = 7


def profiles(graph: dict) -> dict:
    """``p_i`` for every TRAIN node, with the mask and the standardisation."""
    item_ids = np.asarray(graph["item_ids"], dtype=np.int64)
    rows = np.asarray(graph["rows"], dtype=np.int64)
    cols = np.asarray(graph["cols"], dtype=np.int64)
    af3 = np.asarray(graph["weights"], dtype=np.float64)
    nodes = len(item_ids)
    incident: list[list[float]] = [[] for _ in range(nodes)]
    degree = np.zeros(nodes, dtype=np.int64)
    triangles = np.zeros(nodes, dtype=np.int64)
    adjacency = [set() for _ in range(nodes)]
    for left, right, value in zip(rows, cols, af3):
        incident[int(left)].append(float(value))
        incident[int(right)].append(float(value))
        degree[int(left)] += 1
        degree[int(right)] += 1
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    for node in range(nodes):
        for left in adjacency[node]:
            triangles[node] += len(adjacency[node] & adjacency[left])
        triangles[node] //= 2

    raw = np.zeros((nodes, PROFILE_DIM), dtype=np.float64)
    valid = np.zeros(nodes, dtype=bool)
    for node in range(nodes):
        values = np.asarray(incident[node], dtype=np.float64)
        if len(values) < 2:
            continue
        valid[node] = True
        pairings = max(1, degree[node] * (degree[node] - 1) // 2)
        raw[node] = [values.mean(), values.std(ddof=0),
                     np.quantile(values, 0.1), np.quantile(values, 0.5),
                     np.quantile(values, 0.9), np.log1p(float(degree[node])),
                     float(triangles[node]) / float(pairings)]
    if not valid.any():
        raise ValueError("no TRAIN node has a usable local curvature profile")
    location = raw[valid].mean(0)
    scale = raw[valid].std(0)
    scale[scale < 1e-9] = 1.0
    standardised = (raw - location) / scale
    return {"profiles": standardised.astype(np.float32), "valid": valid,
            "raw": raw.astype(np.float32), "location": location, "scale": scale,
            "degree": degree, "triangles": triangles}


def relational_fit(state: dict, support: dict) -> dict:
    """``a + b ||p_i - p_j||`` fitted on the TRAIN support's real edges.

    The fit is a least-squares line from the profile distance to the more
    trustworthy quantity available before training: the target distance is the
    support's own mixed-geometry distance in the *initial* geometry, which the
    specification calls the frozen scale. Because that distance is available only
    after a model exists, the fit is computed once, written down and reused.
    """
    profiles_matrix = state["profiles"]
    rows = np.asarray(support["pairs"], dtype=np.int64)
    is_edge = np.asarray(support["is_edge"], dtype=bool) if support.get("is_edge") is not None \
        else np.ones(len(rows), dtype=bool)
    pairs = rows[is_edge]
    if len(pairs) < 8:
        raise ValueError("too few TRAIN support edges to fit the relational scale")
    difference = profiles_matrix[pairs[:, 0]] - profiles_matrix[pairs[:, 1]]
    distance = np.linalg.norm(difference, axis=1).astype(np.float64)
    targets = np.asarray(support["distances"], dtype=np.float64)[is_edge] \
        if support.get("distances") is not None else distance
    design = np.stack([np.ones_like(distance), distance], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residual = design @ coefficients - targets
    # The frozen diagnostic: does the line disagree with the ordering of the edges?
    ordering = np.sign(np.corrcoef(distance, targets)[0, 1])
    return {"a": float(coefficients[0]), "b": float(coefficients[1]),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "sign_agrees": bool(ordering >= 0)}


class ProfileHead(nn.Module):
    """``h_theta``: the quantized readout to the structural profile."""

    def __init__(self, width: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                 nn.Linear(hidden, PROFILE_DIM))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)


def prepare_state(graph: dict, support: dict, all_embeddings: torch.Tensor,
                  device: torch.device) -> dict:
    teacher = profiles(graph)
    fit = relational_fit(teacher, support)
    pairs = np.asarray(support["pairs"], dtype=np.int64)
    valid = torch.as_tensor(teacher["valid"], dtype=torch.bool, device=device)
    # The head belongs to this module, not to the model: a registered submodule
    # that only the role loss uses is reported as unused by DDP's bucket
    # accounting. It is optimised through the parameter group the trainer adds.
    # The head reads the quantized readout, whose width is the joint codeword's.
    from .mixed_rq import TOTAL_DIM
    head = ProfileHead(width=int(TOTAL_DIM)).to(device)
    return {"head": head, "profiles": torch.as_tensor(teacher["profiles"], device=device),
            "valid": valid, "pairs": torch.as_tensor(pairs, dtype=torch.long, device=device),
            "fit": fit, "all_embeddings": all_embeddings, "device": device,
            "raw": teacher["raw"], "degree": teacher["degree"],
            "triangles": teacher["triangles"], "keep_geometry": bool(fit["sign_agrees"])}


def sample_batch(state: dict, rng) -> dict:
    valid = np.flatnonzero(state["valid"].cpu().numpy())
    nodes = rng.choice(valid, size=min(NODES_PER_STEP, len(valid)), replace=False)
    edges = rng.integers(len(state["pairs"]), size=EDGES_PER_STEP)
    pairs = state["pairs"].cpu().numpy()[edges]
    keep = state["valid"].cpu().numpy()[pairs].all(1)
    pairs = pairs[keep]
    return {"nodes": torch.as_tensor(nodes, dtype=torch.long, device=state["device"]),
            "pairs": torch.as_tensor(pairs, dtype=torch.long, device=state["device"])}


def profile_penalty(predicted: torch.Tensor, target: torch.Tensor,
                    huber_delta: float) -> torch.Tensor:
    """The profile term for one row, as the reference contract states it.

    ``huber(h_theta(phi) - p)`` on a single standardised coordinate, which is the
    atom the head's loss is the mean of.
    """
    value = huber_penalty(predicted - target, huber_delta)
    # The reference implementation returns one scalar per sample row.
    return value.squeeze(-1) if value.dim() > 1 and value.shape[-1] == 1 else value


def augment_loss(base_loss: torch.Tensor, raw_module, state: dict | None, epoch: int,
                 rng, curvature: float = 1.0) -> tuple[torch.Tensor, dict]:
    if state is None or not state["valid"].any():
        return base_loss, {"active": False, "epoch": epoch}
    device = next(raw_module.parameters()).device
    picked = sample_batch(state, rng)
    codebooks = [layer.get_code_embs() for layer in raw_module.rq.vq_layers]
    embeddings = state["all_embeddings"]

    def readout(nodes: torch.Tensor):
        # The embedding table lives on the host; gather there, then move the rows.
        rows = embeddings[nodes.to(embeddings.device)].to(device)
        latent = raw_module.encoder(rows)
        assignments = soft_assignments(latent, codebooks, curvature=curvature)
        return soft_readout(assignments, codebooks, curvature), assignments

    phi, _ = readout(picked["nodes"])
    predicted = state["head"](phi)
    profile_term = huber_penalty(predicted - state["profiles"][picked["nodes"]], HUBER_DELTA).mean()

    geometry_term = torch.zeros((), device=device, dtype=profile_term.dtype)
    if state["keep_geometry"] and len(picked["pairs"]):
        left, right = picked["pairs"][:, 0], picked["pairs"][:, 1]
        phi_left, _ = readout(left)
        phi_right, _ = readout(right)
        measured = mixed_delta(phi_left, phi_right, curvature)
        difference = (state["profiles"][left] - state["profiles"][right]).norm(dim=-1)
        target = state["fit"]["a"] + state["fit"]["b"] * difference
        geometry_term = huber_penalty((measured - target) / max(state["fit"]["rmse"], 1e-3),
                                      HUBER_DELTA).mean()

    auxiliary = LOSS_WEIGHTS["profile"] * profile_term + LOSS_WEIGHTS["geometry"] * geometry_term
    if not torch.isfinite(auxiliary):
        raise RuntimeError("non-finite role objective")
    encoder = list(raw_module.encoder.parameters())
    base_encoder, base_codebook = gradient_norm(base_loss, encoder), gradient_norm(base_loss, codebooks)
    new_encoder, new_codebook = gradient_norm(auxiliary, encoder), gradient_norm(auxiliary, codebooks)
    scale = GradientBudget(share=GRADIENT_SHARE).observe(base_encoder, new_encoder,
                                                        base_codebook, new_codebook)
    scale_tensor = torch.tensor(scale, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scale_tensor, op=dist.ReduceOp.MIN)
    scale = float(scale_tensor)
    log = {"active": True, "epoch": epoch, "scale": scale,
           "profile": float(profile_term.detach()), "geometry": float(geometry_term.detach()),
           "keep_geometry": state["keep_geometry"], "fit": state["fit"],
           "base_encoder_norm": base_encoder, "base_codebook_norm": base_codebook,
           "new_encoder_norm": new_encoder, "new_codebook_norm": new_codebook}
    return base_loss + scale * auxiliary, log
