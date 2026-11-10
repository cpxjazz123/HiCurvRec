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
#: The temperature of the differentiable part of the proxy assignment. The
#: baseline's centring maps the matrix into [-1, 1], so the logits span
#: ``2 / T``. With ``T = 2 / 64 = 1/32`` that span is 64, i.e. ``exp(-32)`` at
#: the flat end -- thirteen decades above the float32 underflow line -- so no
#: entry underflows, every row keeps a gradient, and the sharpest non-underflowing
#: value is what the derivation gives, not a swept one.
PROXY_TEMPERATURE = 2.0 / 64.0
#: The uniform floor mixed into the soft part. A sharp softmax in float32 rounds
#: its maximum to exactly 1.0 once the off-mass falls below the format's
#: resolution near one (~1e-7), and a row whose soft part is exactly one-hot has
#: no derivative left. Mixing in ``S / K`` bounds the maximum by
#: ``1 - S(1 - 1/K)``, so the straight-through residual keeps ~1e-3 of dynamic
#: range -- four decades above float32's resolution -- while the argmax, and
#: therefore the hard assignment, is unchanged by a uniform shift.
PROXY_SMOOTHING = 1e-3


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
        """The probe subgraph's triangles and pair sets, as item-id tensors."""
        return {name: torch.from_numpy(self.arrays[name].astype(np.int64))
                for name in ("triangles", "same_pairs", "cross_pairs")}

    @property
    def probe_items(self) -> np.ndarray:
        """The frozen item ids the three structural terms are estimated on."""
        return np.asarray(self.arrays["probe_items"], dtype=np.int64)


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


def sinkhorn_plan(centred: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """The baseline's balanced assignment as a differentiable plan.

    The frozen ``MixedVQLevel.sinkhorn`` is ``@torch.no_grad()`` because it only
    produces an index. The proxy needs the same numbers *with* a derivative, so
    this is the same iteration written out, in double, and row-normalised at the
    end. ``test_proxy_assignment.py`` checks it against the quantiser's own plan.
    """
    q = torch.exp(-centred / float(epsilon))
    q = q / q.sum()
    batch, codes = int(q.shape[0]), int(q.shape[1])
    for _ in range(int(iterations)):
        q = q / q.sum(0, keepdim=True)
        q = q / float(codes)
        q = q / q.sum(1, keepdim=True)
        q = q / float(batch)
    q = q * float(batch)
    return q / q.sum(-1, keepdim=True)


def hard_index(cost: torch.Tensor, layer, is_last: bool) -> torch.Tensor:
    """The code the quantiser's own rule would take on this cost matrix."""
    if is_last and bool(getattr(layer, "use_sk", False)):
        centred = layer.center_distance(cost).detach().double()
        plan = layer.sinkhorn(centred, layer.sk_epsilon, layer.sk_iters)
        return (plan / plan.sum(-1, keepdim=True)).argmax(dim=-1)
    return cost.detach().argmin(dim=-1)


def proxy_soft_part(cost: torch.Tensor, layer, is_last: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The gradient carrier and the hard code it is centred on.

    * the last level carries the balanced plan itself -- the same centring, the
      same ``SK_EPSILON``, the same iteration count, row-normalised -- so the
      carrier's ``argmax`` is the token the quantiser takes; a pure per-row
      softmax cannot reproduce a balancing constraint;
    * the earlier levels carry the softmax of the same centred cost, whose
      ``argmax`` is the ``argmin`` the quantiser takes there;
    * the uniform floor keeps every entry positive.

    This is the straight-through pattern: the value is the real assignment, the
    derivative is the carrier's.
    """
    centred = layer.center_distance(cost)
    if is_last and bool(getattr(layer, "use_sk", False)):
        soft = sinkhorn_plan(centred.double(), layer.sk_epsilon, layer.sk_iters).to(cost.dtype)
        hard = soft.detach().argmax(dim=-1)
    else:
        soft = torch.softmax(-centred / PROXY_TEMPERATURE, dim=-1)
        hard = cost.detach().argmin(dim=-1)
    codes = int(soft.shape[-1])
    soft = (1.0 - PROXY_SMOOTHING) * soft + PROXY_SMOOTHING / float(codes)
    return soft, hard


def proxy_assignment(cost: torch.Tensor, layer, is_last: bool) -> torch.Tensor:
    """One level's assignment proxy: the hard rule's one-hot plus a live gradient.

    ``h = one_hot(hard) + (soft - soft.detach())``. Its forward value is exactly
    the hard assignment, so a term built on it reaches 0 and 1 the way the tokens
    do; its backward is the derivative of ``soft``, so the cost matrix, the gate
    and the encoder keep a gradient.

    * the last level's hard rule is the baseline's balanced assignment: the
      transport plan of the same centring, the same ``SK_EPSILON`` and the same
      iteration count, row-normalised, and its ``argmax``;
    * the earlier levels' hard rule is the baseline's ``argmin`` of the cost, and
      ``soft`` is the softmax of the same centred cost;
    * a uniform floor is mixed into ``soft`` so its maximum stays strictly below
      one and the straight-through residual keeps float32 dynamic range.

    This is the same straight-through pattern the frozen quantiser already uses on
    the codebook; the proxy never produces a SID.
    """
    soft, hard = proxy_soft_part(cost, layer, is_last)
    onehot = torch.zeros_like(soft).scatter_(1, hard.unsqueeze(1), 1.0)
    return onehot + (soft - soft.detach())


def _proxy_from_hard(proxy: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """The proxy rebuilt around a declared hard code; a test/derivation helper."""
    onehot = torch.zeros_like(proxy).scatter_(1, hard.reshape(-1, 1).to(proxy.device), 1.0)
    soft = proxy - proxy.detach() + onehot.detach()
    return onehot + (soft - soft.detach())


def _one_hot_proxy(reference: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """A pure one-hot proxy of the reference's shape, for endpoint constructions."""
    return torch.zeros_like(reference).scatter_(1, hard.reshape(-1, 1).to(reference.device), 1.0)


def level_diagnostics(proxies: list[torch.Tensor], records: list[dict],
                      alphas: list[torch.Tensor] | None,
                      structure: torch.Tensor | None) -> list[dict]:
    """Per-level evidence that the gate and the terms are alive (R7/R8).

    Usage is read off the proxy's hard part, the two half-distances off the pass
    that produced it, and the structure dependence as the largest absolute
    Spearman correlation between ``alpha`` and a descriptor dimension.
    """
    import numpy as np
    rows = []
    for level, proxy in enumerate(proxies):
        hard = proxy.detach().argmax(dim=-1)
        counts = torch.bincount(hard, minlength=int(proxy.shape[-1])).to(torch.float64)
        probability = counts / counts.sum()
        nonzero = probability[probability > 0]
        entropy = float(-(nonzero * nonzero.log()).sum())
        codes = int(proxy.shape[-1])
        record = records[level] if level < len(records) else {}
        entry = {
            "level": level + 1,
            "codes_used": int((counts > 0).sum()),
            "max_bucket_share": float(probability.max()),
            "usage_entropy": entropy,
            "usage_entropy_normalised": entropy / math.log(codes) if codes > 1 else 0.0,
        }
        if "residual_e" in record and "picked_e" in record:
            squared_e = (record["residual_e"] - record["picked_e"]).pow(2).sum(-1)
            from .poincare_contract import squared_distance
            squared_h = squared_distance(record["residual_h"], record["picked_h"],
                                         record["curvature"]) / 4.0
            entry["mean_squared_euclidean"] = float(squared_e.mean())
            entry["mean_squared_curved_over_four"] = float(squared_h.mean())
            ratio = float(squared_e.mean()) / max(float(squared_h.mean()), 1e-12)
            entry["euclidean_over_curved"] = ratio
        if alphas is not None and structure is not None and level < len(alphas):
            alpha = alphas[level].detach().cpu().numpy()
            descriptors = structure.detach().cpu().numpy()
            best = (0.0, None)
            for dim, name in enumerate(DESCRIPTOR_NAMES):
                column = descriptors[:, dim]
                if float(column.std()) <= 0:
                    continue
                correlation = _spearman(alpha, column)
                if abs(correlation) > abs(best[0]):
                    best = (correlation, name)
            entry["max_abs_spearman_alpha_vs_descriptor"] = float(best[0])
            entry["spearman_descriptor"] = best[1]
        rows.append(entry)
    return rows


def _spearman(first, second) -> float:
    """Rank correlation, centred; the caller has already dropped constant columns."""
    import numpy as np
    def _rank(values):
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        return ranks - ranks.mean()
    left = _rank(np.asarray(first, dtype=np.float64))
    right = _rank(np.asarray(second, dtype=np.float64))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else 0.0


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
           "closure_loss", "bridge_loss", "usage_loss", "proxy_assignment",
           "proxy_soft_part", "hard_index", "sinkhorn_plan",
           "level_diagnostics",
           "structural_objective", "default_table", "table_path", "manifest_path",
           "sha256_of", "DESCRIPTOR_NAMES", "DESCRIPTOR_DIMS", "LEVELS",
           "ALPHA_INIT", "LAMBDA_CLOSURE", "LAMBDA_BRIDGE", "LAMBDA_USAGE",
           "PROXY_TEMPERATURE", "PROXY_SMOOTHING"]
