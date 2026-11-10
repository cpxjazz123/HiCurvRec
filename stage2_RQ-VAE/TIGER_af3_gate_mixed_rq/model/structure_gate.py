"""The AF3-guided structure gate and the three structural terms (issue #149).

The mechanism's only behavioural input is the frozen descriptor ``s_i`` of the
TRAIN behaviour graph. A per-level head turns it into a weight

    alpha_{i,l} = sigmoid(f_l(s_i)) in (0, 1)

and that weight replaces the baseline's frozen constant in the mixed cost

    C_{ik}^(l) = 2 [ alpha_{i,l} ||rE - eE||^2 + (1 - alpha_{i,l}) d_c(rH, eH)^2 / 4 ].

The assignment rule itself is untouched: the cost matrix still goes to the
baseline's Sinkhorn on the balanced levels and to its own rule on the others, so
the mechanism's score enters the balanced allocation instead of replacing it
(CLAUDE.md §19h).

Three switches exist in this file and no arm is trained with one of them off
(§19g):

``BEHAVIOR_OFF``
    the gate reads the frozen TRAIN-mean descriptor, so ``alpha`` no longer
    depends on the item; every item in a batch shares one weight per level.
``GEOMETRY_OFF``
    the curvature is set to zero, so the hyperbolic half degenerates to the
    Euclidean one (``d_0^2 / 4`` is the squared Euclidean distance); the gate and
    everything else is unchanged.
``BOTH_ON``
    the factory configuration of this file.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .structure_table import (DESCRIPTOR_DIMS, DESCRIPTOR_NAMES, TABLE_PATH_RELATIVE,
                              load as load_arrays, manifest_path, sha256_of, table_path,
                              verify as verify_table)

#: The number of quantisation levels a gate maps the descriptor into.
LEVELS = 3
#: The width of the gate's single hidden layer: two cells per descriptor
#: dimension, which is the smallest width that can represent an interaction
#: between a dimension and its own sign without a deeper stack.
GATE_HIDDEN = 2 * DESCRIPTOR_DIMS
#: ``sigma(0)``, the weight the frozen baseline starts from. A zero-initialised
#: output layer reproduces it point for point.
ALPHA_INIT = 0.5
#: The three switches. ``BOTH_ON`` is the configuration that trains.
BEHAVIOR_OFF = False
GEOMETRY_OFF = False
#: The weights of the three structural terms. Each term is a mean over pairs or
#: codes and therefore already lies in [0, 1]: the three are dimensionally the
#: same quantity, so equal weight is the default and no weight is swept.
LAMBDA_CLOSURE = 1.0
LAMBDA_BRIDGE = 1.0
LAMBDA_USAGE = 1.0
#: The temperature of the soft assignment approximation. The baseline's own
#: centring already divides the cost by ``max - middle``, which puts the matrix
#: in [-1, 1]; one is therefore the scale the centring leaves behind, not a value
#: chosen from a result.
TEMPERATURE = 1.0


class StructureTable:
    """The frozen descriptor of every item id, with its measured/undefined mask."""

    def __init__(self, values, observed, metadata, arrays: dict | None = None,
                 sha256: str | None = None):
        values = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32)
        observed = np.asarray(observed, dtype=bool)
        if values.ndim != 2 or values.shape[1] != DESCRIPTOR_DIMS:
            raise ValueError("a structure table is [items, " + str(DESCRIPTOR_DIMS) + "]")
        if observed.shape != tuple(values.shape):
            raise ValueError("the descriptor mask and the matrix disagree in shape")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("a standardised descriptor entry must be finite")
        self.values = values
        self.observed = observed
        self.metadata = dict(metadata)
        self.arrays = arrays or {}
        #: The SHA-256 of the file this table was read from; a checkpoint records
        #: it so a restore cannot silently pair weights with another table.
        self.sha256 = sha256

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def item_count(self) -> int:
        return len(self)

    def lookup(self, item_ids) -> torch.Tensor:
        """The standardised descriptor rows of these item ids."""
        ids = torch.as_tensor(item_ids, dtype=torch.long)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(self)):
            raise IndexError("an item id is outside the frozen structure table")
        return self.values.index_select(0, ids)

    def undefined_counts(self) -> dict:
        per_dimension = {name: int((~self.observed[:, dim]).sum())
                         for dim, name in enumerate(DESCRIPTOR_NAMES)}
        return {"entries": int((~self.observed).sum()), "per_dimension": per_dimension}

    @classmethod
    def load(cls, path: Path | str) -> "StructureTable":
        """Load the frozen table, refusing one whose bytes moved."""
        path = Path(path)
        verify_table(path)
        arrays = load_arrays(path)
        return cls(arrays["values"], arrays["observed"], arrays["metadata"], arrays,
                   sha256=sha256_of(path))

    def frozen_sets(self) -> dict:
        """The triangles and the two pair sets, as item-id tensors."""
        return {name: torch.from_numpy(self.arrays[name].astype(np.int64))
                for name in ("triangles", "same_pairs", "cross_pairs")}


class StructureGate(nn.Module):
    """``f_l``: one independently parameterised head per quantisation level."""

    def __init__(self, dim: int = DESCRIPTOR_DIMS, hidden: int = GATE_HIDDEN,
                 levels: int = LEVELS):
        super().__init__()
        self.dim = int(dim)
        self.levels = int(levels)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(self.dim, int(hidden)), nn.GELU(), nn.Linear(int(hidden), 1))
            for _ in range(self.levels)
        )
        # Zero output layer: the start is sigma(0) = 1/2 at every level, which is
        # the frozen baseline's weight. The hidden layer keeps its Xavier-ish
        # default, so only the readout starts silent.
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, structure: torch.Tensor) -> list[torch.Tensor]:
        if structure.ndim != 2 or int(structure.shape[1]) != self.dim:
            raise ValueError("the gate expects [items, " + str(self.dim) + "] descriptors")
        rows = torch.zeros_like(structure) if BEHAVIOR_OFF else structure
        return [torch.sigmoid(head(rows).reshape(-1)) for head in self.heads]


# ------------------------------------------------------------------- terms
def _centred(cost: torch.Tensor) -> torch.Tensor:
    """The baseline's own centring rule, reused so the soft scale is its scale."""
    maximum = cost.max()
    minimum = cost.min()
    middle = (maximum + minimum) / 2
    amplitude = maximum - middle + 1e-5
    if not bool(amplitude > 0):
        raise ValueError("Cannot centre a constant distance matrix.")
    return (cost - middle) / amplitude


def soft_prefix(cost: torch.Tensor) -> torch.Tensor:
    """The differentiable approximation of one level's assignment.

    ``q = softmax(-centred(C) / 1)``. It is used by the three structural terms
    only; the assignment that produces the SID stays the baseline's.
    """
    return torch.softmax(-_centred(cost) / TEMPERATURE, dim=-1)


def _overlap(prefixes: list[torch.Tensor], first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    total = torch.zeros(first.shape[0], dtype=prefixes[0].dtype, device=prefixes[0].device)
    for prefix in prefixes:
        total = total + (prefix.index_select(0, first) * prefix.index_select(0, second)).sum(-1)
    return total / float(len(prefixes))


def _connected_zero(prefixes: list[torch.Tensor]) -> torch.Tensor:
    """A zero that still belongs to the graph, so the term is never dead."""
    return prefixes[0].sum() * 0.0


def closure_loss(prefixes: list[torch.Tensor], triangles: torch.Tensor) -> torch.Tensor:
    """``1 -`` the mean prefix overlap of every edge of every triangle."""
    if triangles.numel() == 0:
        return _connected_zero(prefixes)
    overlaps = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        overlaps.append(_overlap(prefixes, triangles[:, first], triangles[:, second]))
    return 1.0 - torch.stack(overlaps).mean()


def bridge_loss(prefixes: list[torch.Tensor], same_pairs: torch.Tensor,
                cross_pairs: torch.Tensor) -> torch.Tensor:
    """The hinge that keeps a real cross-community edge less similar than a peer.

    ``relu(mean_same(1 - overlap) - mean_cross(1 - overlap))``: the two pooled
    means of the same quantity, so the term is bounded by one and the two sides
    are symmetric. A cross edge is never treated as a negative sample.
    """
    if same_pairs.numel() == 0 or cross_pairs.numel() == 0:
        return _connected_zero(prefixes)
    dissimilar_same = (1.0 - _overlap(prefixes, same_pairs[:, 0], same_pairs[:, 1])).mean()
    dissimilar_cross = (1.0 - _overlap(prefixes, cross_pairs[:, 0], cross_pairs[:, 1])).mean()
    return torch.relu(dissimilar_same - dissimilar_cross)


def gate_alpha(structure: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """One hidden unit of one level's head, as the scalar surface the contract freezes.

    ``alpha = sigmoid(scale * GELU(weight . s + bias) + shift)`` with the exact
    (erf) GELU of ``nn.GELU``. A level stacks ``GATE_HIDDEN`` such units; one unit
    is what a SymPy statement can describe without inventing one symbol per matrix
    entry.
    """
    # A shared head weight is a vector; the reference sampler hands the same
    # width as one row per sampled point, and the product is elementwise either
    # way, so both shapes evaluate the frozen formula.
    weight = weight.reshape(-1) if weight.ndim == 1 else weight
    pre = (structure * weight).sum(-1) + bias.reshape(())
    hidden = 0.5 * pre * (1.0 + torch.erf(pre / math.sqrt(2.0)))
    return torch.sigmoid(scale.reshape(()) * hidden + shift.reshape(()))


def usage_loss(prefixes: list[torch.Tensor]) -> torch.Tensor:
    """The mean normalised entropy deficit of the soft code usage per level."""
    total = prefixes[0].sum() * 0.0
    for prefix in prefixes:
        probability = prefix.mean(0)
        entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum()
        codes = int(prefix.shape[-1])
        if codes < 2:
            continue
        total = total + (1.0 - entropy / math.log(codes))
    return total / float(len(prefixes))


def structural_objective(prefixes: list[torch.Tensor], item_ids, frozen: dict,
                         item_count: int) -> tuple[torch.Tensor, dict]:
    """The three terms on the part of the frozen sets this batch actually covers.

    A triangle or a pair is used only when every one of its items is in the batch,
    so the terms are computed on the very assignments the step produced; what was
    covered is reported instead of being silently dropped.
    """
    if len(prefixes) != LEVELS:
        raise ValueError("this mechanism expects one soft assignment per level")
    ids = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)
    if int(ids.numel()) != int(prefixes[0].shape[0]):
        raise ValueError("the item ids and the soft assignments disagree in length")
    count = int(item_count)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= count):
        raise IndexError("an item id is outside the frozen structure table")
    lookup = np.full(count, -1, dtype=np.int64)
    lookup[ids.detach().cpu().numpy().astype(np.int64)] = np.arange(int(ids.numel()), dtype=np.int64)

    def _local(pairs) -> tuple[torch.Tensor, int, int]:
        tensor = torch.as_tensor(pairs, dtype=torch.long)
        if tensor.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.long), 0, 0
        mapped = lookup[tensor.detach().cpu().numpy().astype(np.int64)]
        keep = (mapped >= 0).all(axis=1)
        return (torch.from_numpy(mapped[keep]).to(torch.long).to(prefixes[0].device),
                int(keep.sum()), int(tensor.shape[0]))

    triangles, triangles_used, triangles_total = _local(frozen["triangles"])
    if triangles.shape[1] != 3:
        raise ValueError("a triangle needs three items")
    same_pairs, same_used, same_total = _local(frozen["same_pairs"])
    cross_pairs, cross_used, cross_total = _local(frozen["cross_pairs"])
    closure = closure_loss(prefixes, triangles)
    bridge = bridge_loss(prefixes, same_pairs, cross_pairs)
    usage = usage_loss(prefixes)
    total = (LAMBDA_CLOSURE * closure + LAMBDA_BRIDGE * bridge + LAMBDA_USAGE * usage)
    report = {"closure": float(closure.detach()), "bridge": float(bridge.detach()),
              "usage": float(usage.detach()), "total": float(total.detach()),
              "triangles_used": triangles_used, "triangles_total": triangles_total,
              "same_pairs_used": same_used, "same_pairs_total": same_total,
              "cross_pairs_used": cross_used, "cross_pairs_total": cross_total}
    return total, report


def default_table(root: Path | str) -> Path:
    """The frozen table of this repository, resolved from the root marker."""
    return Path(root).joinpath(*TABLE_PATH_RELATIVE)


__all__ = ["StructureGate", "StructureTable", "BEHAVIOR_OFF", "GEOMETRY_OFF",
           "closure_loss", "bridge_loss", "usage_loss", "soft_prefix",
           "structural_objective", "default_table", "table_path", "manifest_path",
           "sha256_of", "DESCRIPTOR_NAMES", "DESCRIPTOR_DIMS", "LEVELS",
           "ALPHA_INIT", "LAMBDA_CLOSURE", "LAMBDA_BRIDGE", "LAMBDA_USAGE",
           "TEMPERATURE"]
