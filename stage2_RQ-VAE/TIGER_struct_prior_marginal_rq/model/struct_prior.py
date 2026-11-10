"""The structure-stratified target marginal of the balanced assignment (issue #150).

The mechanism's only altered internal quantity is the column marginal the
balanced assignment is normalised to. Everything else -- the joint E16 x H16
codeword, the cost, the residual recursion, the readout, the export and the
assignment rule itself -- is the baseline's.

    pi_k^(l) = rho_{g(k)} / |B_{g(k)}|,        k in bucket B_g

with equal-size buckets and ``rho_g`` the frozen TRAIN share of structure layer
``g``. The layers come from the quartiles of the triangle-participation share
``tau_i`` on the TRAIN graph, so they are frozen before training and carry their
own SHA-256.

Switches (present, never trained in a closed state):

``BEHAVIOR_OFF``  the marginal is uniform, so the structure signal leaves the
                  assignment entirely;
``GEOMETRY_OFF``  curvature zero, i.e. the hyperbolic half degenerates;
``BOTH_ON``       the factory configuration of this file.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

#: The number of structure layers.
GROUPS = 4
#: The layer boundaries on tau. The layers are *thresholds*, not quantiles, because
#: quantiles force an equal population per layer and therefore a nearly uniform
#: target marginal (measured: rho within 4.19% of 1/4, pi within 2.59% of uniform,
#: 0.39% of tokens changed on a real 1024x256 cost matrix). The cuts are the
#: thirds of the participation share after the exact zero, which is its own layer:
#: an item with no triangular edge has no partial participation at all, and a
#: third is the coarsest split that keeps three distinguishable partial levels.
#: Frozen before the single run; the review asked for this decision explicitly.
TAU_ZERO_LAYER = True
TAU_CUTS = (1.0 / 3.0, 2.0 / 3.0)
#: The three term weights. Each term is normalised into [0, 1] by its own
#: definition, so the three are the same kind of quantity and equal weight is the
#: default rather than a swept value.
LAMBDA_TRIANGLE = 1.0
LAMBDA_FORK = 1.0
LAMBDA_PRIOR = 1.0
#: The differentiable carrier's temperature and uniform floor, derived exactly as
#: in issue #149: the baseline's centring spans 2, so T = 2/64 keeps every
#: exponent inside float32, and the 1e-3 floor keeps the sharp end representable
#: so no row loses its derivative.
PROXY_TEMPERATURE = 2.0 / 64.0
PROXY_SMOOTHING = 1e-3

BEHAVIOR_OFF = False
GEOMETRY_OFF = False

TABLE_PATH_RELATIVE = ("results", "research", "issue150", "structure_prior.npz")


# --------------------------------------------------------------------- table
class StructPriorTable:
    """The frozen TRAIN stratification, the graph's edges and its triangles."""

    def __init__(self, arrays: dict, sha256: str | None = None):
        self.tau = np.asarray(arrays["tau"], dtype=np.float64)
        self.layer = np.asarray(arrays["layer"], dtype=np.int64)
        self.rho = np.asarray(arrays["rho"], dtype=np.float64)
        self.item_ids = np.asarray(arrays["item_ids"], dtype=np.int64)
        self.community = np.asarray(arrays["community"], dtype=np.int64)
        self.edges = np.asarray(arrays["edges"], dtype=np.int64)
        self.triangles = np.asarray(arrays["triangles"], dtype=np.int64)
        self.edge_layer = np.asarray(arrays["edge_layer"], dtype=np.int64)
        self.metadata = dict(arrays["metadata"]) if isinstance(arrays["metadata"], dict) \
            else json.loads(str(arrays["metadata"]))
        self.sha256 = sha256

    @property
    def groups(self) -> int:
        return int(self.rho.size)

    @classmethod
    def load(cls, path: Path | str) -> "StructPriorTable":
        path = Path(path)
        manifest = verify(path)
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        return cls(arrays, sha256=manifest["table_sha256"])

    def layer_of(self, item_ids: np.ndarray) -> np.ndarray:
        """The frozen structure layer of these item ids, -1 when the item has none."""
        lookup = np.full(int(self.metadata["n_items"]), -1, dtype=np.int64)
        lookup[self.item_ids] = self.layer
        ids = np.asarray(item_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= lookup.size):
            raise IndexError("an item id is outside the frozen structure table")
        return lookup[ids]


def table_path(root: Path | str) -> Path:
    return Path(root).joinpath(*TABLE_PATH_RELATIVE)


def manifest_path(path: Path | str) -> Path:
    return Path(path).with_suffix(".json")


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path | str) -> dict:
    path = Path(path)
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    digest = sha256_of(path)
    if manifest["table_sha256"] != digest:
        raise ValueError("the frozen structure table does not match its manifest: "
                         + digest + " != " + manifest["table_sha256"])
    return manifest


# ------------------------------------------------------------------ marginal
def stratify(tau: np.ndarray, cuts=TAU_CUTS) -> tuple[np.ndarray, np.ndarray]:
    """Threshold layers of tau, plus their frozen population shares.

    ``tau = 0`` is its own layer when ``TAU_ZERO_LAYER``; the remaining items fall
    into the bands the cuts define. The shares are whatever the data gives, which
    is the point: a quantile split would force them to ``1/G`` and make the target
    marginal almost uniform.
    """
    tau = np.asarray(tau, dtype=np.float64)
    boundaries = list(cuts)
    if TAU_ZERO_LAYER:
        layer = np.zeros(tau.shape, dtype=np.int64)
        positive = tau > 0
        layer[positive] = np.searchsorted(boundaries, tau[positive], side="left") + 1
    else:
        layer = np.searchsorted(boundaries, tau, side="left").astype(np.int64)
    groups = int(layer.max()) + 1 if tau.size else 0
    counts = np.bincount(layer, minlength=groups).astype(np.float64)
    return layer, counts / counts.sum()


def equal_partition(codes: int, groups: int = GROUPS) -> list[list[int]]:
    """``groups`` buckets of equal size, covering every code exactly once."""
    if codes % groups:
        raise ValueError("the code count must divide into equal buckets")
    size = codes // groups
    return [list(range(index * size, (index + 1) * size)) for index in range(groups)]


def usage_partition(usage: torch.Tensor, groups: int, bucket_size: int) -> list[list[int]]:
    """Buckets from the previous epoch's per-(layer, code) usage mass.

    Layer ``g`` takes the ``bucket_size`` unassigned codes with the largest mass
    from that layer; ties fall to the lower code index, and layer order decides a
    contested code. The partition is therefore deterministic and every code ends
    up in exactly one bucket.
    """
    usage = torch.as_tensor(usage, dtype=torch.float64)
    codes = int(usage.shape[1])
    if usage.shape[0] != groups:
        raise ValueError("the usage matrix must have one row per structure layer")
    if bucket_size * groups != codes:
        raise ValueError("the buckets must cover the codebook exactly")
    leftover = set(range(codes))
    buckets: list[list[int]] = []
    for group in range(groups):
        ranked = sorted(leftover, key=lambda code: (-float(usage[group, code]), code))
        chosen = sorted(ranked[:bucket_size])
        buckets.append(chosen)
        leftover.difference_update(chosen)
    if leftover:
        raise RuntimeError("the partition left codes unassigned")
    return buckets


def target_marginal(partition: list[list[int]], rho, codes: int | None = None,
                    dtype=torch.float64) -> torch.Tensor:
    """``pi_k = rho_g / |B_g|`` for every code; uniform under ``BEHAVIOR_OFF``."""
    total = int(codes if codes is not None else sum(len(bucket) for bucket in partition))
    if BEHAVIOR_OFF:
        return torch.full((total,), 1.0 / float(total), dtype=dtype)
    rho = torch.as_tensor(rho, dtype=dtype).reshape(-1)
    if rho.numel() != len(partition):
        raise ValueError("rho must have one entry per bucket")
    if float(rho.sum()) <= 0:
        raise ValueError("rho must be a positive share vector")
    marginal = torch.empty(total, dtype=dtype)
    for share, bucket in zip(rho.tolist(), partition):
        value = float(share) / float(len(bucket))
        marginal[torch.as_tensor(bucket, dtype=torch.long)] = value
    return marginal / marginal.sum()


def structured_sinkhorn(centred: torch.Tensor, epsilon: float, iterations: int,
                        marginal: torch.Tensor) -> torch.Tensor:
    """The baseline's balanced assignment, normalised to ``marginal`` columns.

    The iteration is the frozen one -- ``exp(-D/eps)``, column normalisation, row
    normalisation -- with the *target* column mass replaced by ``batch * marginal``
    instead of the constant ``batch / K``, so a row-normalised plan of ``batch``
    rows carries ``marginal`` as its column share. Buckets keep equal capacity by
    construction, so the balancing the baseline relies on is preserved.
    """
    dtype = centred.dtype
    marginal = torch.as_tensor(marginal, dtype=dtype, device=centred.device).reshape(-1)
    q = torch.exp(-centred / float(epsilon))
    q = q / q.sum()
    batch, codes = int(q.shape[0]), int(q.shape[1])
    if int(marginal.numel()) != codes:
        raise ValueError("the marginal must have one entry per code")
    for _ in range(int(iterations)):
        target = (marginal * float(batch)).unsqueeze(0)
        q = q * (target / q.sum(0, keepdim=True).clamp_min(1e-300))
        q = q / q.sum(1, keepdim=True)
    return q / q.sum(-1, keepdim=True)


# ------------------------------------------------------------------- proxies
def centred_cost(cost: torch.Tensor, where: str = "") -> torch.Tensor:
    """The baseline's centring, with the degenerate cases named.

    ``MixedVQLevel.center_distance`` raises on a constant matrix because the
    baseline only ever centres a matrix it is about to balance; the proxy centres
    every level, and a level whose cost has no spread carries no preference, so a
    zero centring is the defined answer there. A non-finite matrix is a real defect
    and still raises, with the range in the message.
    """
    maximum, minimum = cost.max(), cost.min()
    spread = maximum - minimum
    if not bool(torch.isfinite(spread)):
        finite = float(torch.isfinite(cost).float().mean())
        raise ValueError("the cost matrix is not finite at " + (where or "an unnamed call")
                         + ": max=" + str(float(maximum)) + " min=" + str(float(minimum))
                         + " finite_rows=" + str(finite) + " shape=" + str(tuple(cost.shape)))
    if float(spread) <= 0:
        return torch.zeros_like(cost)
    return (cost - (maximum + minimum) / 2) / (spread / 2 + 1e-5)


def proxy_assignment(cost: torch.Tensor, plan: torch.Tensor | None,
                     where: str = "") -> torch.Tensor:
    """Straight-through proxy: the taken code in the forward, a live carrier back.

    The last level's hard code is the balanced plan's ``argmax``; the earlier
    levels take ``argmin`` of the cost and carry the centred softmax. The value is
    the one-hot of the code the quantiser takes, so the prefix terms reach 0 and
    1; the derivative is the carrier's.
    """
    codes = int(cost.shape[-1])
    if plan is not None:
        carrier = plan
        hard = carrier.detach().argmax(dim=-1)
    else:
        centred = centred_cost(cost, where=where)
        carrier = torch.softmax(-centred / PROXY_TEMPERATURE, dim=-1)
        hard = cost.detach().argmin(dim=-1)
    smoothed = (1.0 - PROXY_SMOOTHING) * carrier + PROXY_SMOOTHING / float(codes)
    onehot = torch.zeros_like(smoothed).scatter_(1, hard.unsqueeze(1), 1.0)
    return onehot + (smoothed - smoothed.detach())


# -------------------------------------------------------------------- terms
def _overlap(prefixes: list[torch.Tensor], first: torch.Tensor,
             second: torch.Tensor) -> torch.Tensor:
    total = torch.zeros(first.shape[0], dtype=prefixes[0].dtype, device=prefixes[0].device)
    for prefix in prefixes:
        total = total + (prefix.index_select(0, first) * prefix.index_select(0, second)).sum(-1)
    return total / float(len(prefixes))


def _connected_zero(prefixes: list[torch.Tensor]) -> torch.Tensor:
    return prefixes[0].sum() * 0.0


def triangle_loss(prefixes: list[torch.Tensor], triangles: torch.Tensor) -> torch.Tensor:
    """``1 -`` the mean prefix overlap of every edge of every batch triangle."""
    if triangles.numel() == 0:
        return _connected_zero(prefixes)
    overlaps = [_overlap(prefixes, triangles[:, first], triangles[:, second])
                for first, second in ((0, 1), (0, 2), (1, 2))]
    return 1.0 - torch.stack(overlaps).mean()


def fork_loss(prefixes: list[torch.Tensor], within: torch.Tensor,
              cross: torch.Tensor) -> torch.Tensor:
    """Within-layer edges must be *more* prefix-similar than cross-layer edges.

    A low-AF3 open edge is therefore never treated as a negative sample.
    """
    if within.numel() == 0 or cross.numel() == 0:
        return _connected_zero(prefixes)
    dissimilar_within = (1.0 - _overlap(prefixes, within[:, 0], within[:, 1])).mean()
    dissimilar_cross = (1.0 - _overlap(prefixes, cross[:, 0], cross[:, 1])).mean()
    return torch.relu(dissimilar_within - dissimilar_cross)


def _kl(observed: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
    observed = observed.clamp_min(1e-12)
    marginal = marginal.clamp_min(1e-12)
    if observed.shape[-1] != marginal.shape[-1]:
        raise ValueError("the usage histogram and the marginal disagree in width")
    return (observed * (observed / marginal).log()).sum()


def prior_kl(prefixes: list[torch.Tensor], marginal: torch.Tensor) -> torch.Tensor:
    """``KL(usage || pi)`` per level, normalised by its own exact upper bound.

    ``KL(q || pi) <= log(1 / min pi)`` for any distribution ``q``, so dividing by
    that constant puts every level's term inside ``[0, 1]`` without a clamp.
    """
    upper = math.log(1.0 / float(marginal.min().clamp_min(1e-12)))
    total = prefixes[0].sum() * 0.0
    for prefix in prefixes:
        total = total + _kl(prefix.mean(0), marginal) / upper
    return total / float(len(prefixes))


# ---------------------------------------------------------------- batch sets
def batch_sets(item_ids: np.ndarray, layer_of_item: np.ndarray, edges: np.ndarray,
               triangles: np.ndarray, item_count: int) -> dict:
    """The frozen triangles and edges whose endpoints are all in this batch.

    Returns local row indices plus the coverage counts a report has to state: the
    previous round's structural terms saw 12.5% of the items, so the counts are
    reported per step instead of assumed.
    """
    ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    layers = np.asarray(layer_of_item, dtype=np.int64).reshape(-1)
    if layers.shape[0] != ids.shape[0]:
        raise ValueError("one layer per item id is required")
    lookup = np.full(int(item_count), -1, dtype=np.int64)
    lookup[ids] = np.arange(ids.size, dtype=np.int64)

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    mapped_edges = lookup[edges] if edges.size else np.zeros((0, 2), dtype=np.int64)
    mapped_triangles = lookup[triangles] if triangles.size else np.zeros((0, 3), dtype=np.int64)
    edge_keep = mapped_edges >= 0 if mapped_edges.size else np.zeros(0, dtype=bool)
    edge_keep = edge_keep.all(axis=1) if mapped_edges.size else edge_keep
    triangle_keep = (mapped_triangles >= 0).all(axis=1) if mapped_triangles.size \
        else np.zeros(0, dtype=bool)
    kept_edges = mapped_edges[edge_keep]
    same_layer = (layers[kept_edges[:, 0]] == layers[kept_edges[:, 1]]) if kept_edges.size \
        else np.zeros(0, dtype=bool)
    within_local = kept_edges[same_layer] if kept_edges.size else np.zeros((0, 2), dtype=np.int64)
    cross_local = kept_edges[~same_layer] if kept_edges.size else np.zeros((0, 2), dtype=np.int64)
    return {
        "triangles": mapped_triangles[triangle_keep] if mapped_triangles.size
        else np.zeros((0, 3), dtype=np.int64),
        "within": within_local,
        "cross": cross_local,
        "coverage": {
            "triangles_used": int(triangle_keep.sum()) if triangle_keep.size else 0,
            "triangles_total": int(triangles.shape[0]),
            "edges_used": int(edge_keep.sum()) if edge_keep.size else 0,
            "edges_total": int(edges.shape[0]),
            "within_used": int(within_local.shape[0]),
            "cross_used": int(cross_local.shape[0]),
        },
    }
