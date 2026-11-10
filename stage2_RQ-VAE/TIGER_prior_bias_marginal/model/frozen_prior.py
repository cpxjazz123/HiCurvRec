"""The frozen structure prior of issue #150's corrected arm (#150 fix, arm 2).

What this arm keeps from #150: the structure layers come from the triangle
participation share tau on the TRAIN graph, and the balanced assignment of the last
level is normalised to a per-code share that follows those layers' population.

What it changes, because #150 measured the opposite of its own claim:

* the bucket partition is frozen **before training** from the initial latent space
  (each layer's mean encoder output is its centre, codes go to the nearest centre
  under an equal-quota rule), written to disk with its SHA-256, and never re-sorted
  from the run's own usage. #150 re-ran ``usage_partition`` every epoch, so the
  target chased the realisation and rewarded whichever codes were already used;
* the target marginal is applied **only on the last level**, the only level with a
  balanced assignment. Levels 1-2 stay the baseline's ``argmin`` and carry no
  usage/KL constraint: pushing their usage to a 2.88x non-uniform target has no
  realisation path except distorting the geometry, which is where #150 collapsed
  (58 of 256 codes, one code at 58.7%);
* an exposure guard runs at every evaluation epoch: fewer than 128 codes used or
  more than 10% of the items in one code is a mechanism failure, recorded and
  stopped, instead of being discovered after 3000 epochs;
* ``L_fork`` is removed (it was 0.0 throughout #150 by definition), so the
  objective is ``L_RQ + L_tri + L_prior`` with ``L_prior`` on the last level only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

#: The structure layers, the thirds of tau with the exact zero as its own layer.
GROUPS = 4
TAU_CUTS = (1.0 / 3.0, 2.0 / 3.0)
#: The two term weights; both terms are normalised into [0, 1] by definition.
LAMBDA_TRIANGLE = 1.0
LAMBDA_PRIOR = 1.0
#: The differentiable carrier's temperature and floor, derived in #150: the
#: centring spans 2, so T = 2/64 keeps every exponent inside float32 and the floor
#: keeps the sharp end representable so no row loses its derivative.
PROXY_TEMPERATURE = 2.0 / 64.0
PROXY_SMOOTHING = 1e-3
#: The dispatch-time gate, corrected by the review: a 5% relative error at the
#: training batch is arithmetically unreachable (one row is 1/B of the mass, and
#: 1/(B * min(pi)) is 23.86% at B = 1024), so the gate is expressed at a scale where
#: it can be met and in a per-row unit where it means something.
PROBE_ROWS = 16384            # >= 8192 required; at 16384 the arithmetic bound is 2.98%
PROBE_TOLERANCE = 0.05        # max relative error of the realised column mass at the probe
BATCH_TOLERANCE_ROWS = 1.5    # |realised_k - pi_k| <= 1.5 rows at the training batch
CURVE_BATCHES = (1024, 2048, 4096, 8192, 16384)
CODES_USED_MIN = 128
MAX_BUCKET_SHARE_MAX = 0.10

#: The prior's scale. Measured, not scanned: it is the median top-1/top-2 cost gap of
#: the frozen 1024-row probe (results/research/issue150/marginal_scale_probe.json,
#: key "gap.median"), so the bias competes with the cost on the cost's own scale
#: instead of on epsilon's. Fixed before the single run; never tuned.
PRIOR_T = 0.0012651681900024414

BEHAVIOR_OFF = False
GEOMETRY_OFF = False

#: The frozen structure table of the route (#150's table, SHA verified on load).
STRUCTURE_RELATIVE = ("results", "research", "issue150", "structure_prior.npz")
#: The frozen bucket partition this arm builds once, before training.
PARTITION_RELATIVE = ("results", "research", "issue150", "frozen_partition.npz")


# --------------------------------------------------------------------- inputs
def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_path(root: Path | str) -> Path:
    return Path(root).joinpath(*STRUCTURE_RELATIVE)


def partition_path(root: Path | str) -> Path:
    return Path(root).joinpath(*PARTITION_RELATIVE)


def manifest_path(path: Path | str) -> Path:
    return Path(path).with_suffix(".json")


def load_structure(root: Path | str) -> dict:
    """The TRAIN stratification (#150's table), checked against its manifest."""
    path = structure_path(root)
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    digest = sha256_of(path)
    if manifest["table_sha256"] != digest:
        raise ValueError("the structure table does not match its manifest")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["metadata"] = json.loads(str(arrays["metadata"]))
    arrays["sha256"] = digest
    return arrays


def load_partition(root: Path | str) -> dict:
    """The frozen partition and its marginal, checked against its manifest."""
    path = partition_path(root)
    if not path.is_file():
        raise FileNotFoundError("missing frozen partition: " + str(path)
                                + " (build it with build_frozen_partition.py)")
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    digest = sha256_of(path)
    if manifest["table_sha256"] != digest:
        raise ValueError("the frozen partition does not match its manifest")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["metadata"] = json.loads(str(arrays["metadata"]))
    arrays["sha256"] = digest
    arrays["partition"] = [list(map(int, bucket)) for bucket in arrays["partition"]]
    return arrays


# ------------------------------------------------------------------ partition
def stratify(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Threshold strata of tau: the exact zero, then the thirds of the share."""
    tau = np.asarray(tau, dtype=np.float64)
    layer = np.zeros(tau.shape, dtype=np.int64)
    positive = tau > 0
    layer[positive] = np.searchsorted(list(TAU_CUTS), tau[positive], side="left") + 1
    counts = np.bincount(layer, minlength=GROUPS if tau.size == 0 else int(layer.max()) + 1)
    return layer, counts / counts.sum()


def equal_partition(codes: int, groups: int = GROUPS) -> list[list[int]]:
    if codes % groups:
        raise ValueError("the code count must divide into equal buckets")
    size = codes // groups
    return [list(range(index * size, (index + 1) * size)) for index in range(groups)]


def affinity_partition(code_points: torch.Tensor, layer_centres: torch.Tensor,
                       groups: int = GROUPS) -> list[list[int]]:
    """Equal-quota buckets by affinity to each layer's centre in the initial latent.

    Every code is assigned to the layer whose centre is nearest, but each layer may
    take only ``K / G`` codes: the layer with the strongest claim picks first, ties
    fall to the lower code index, and a code already taken is never reused. With
    identical centres the affinities tie and the result is the equal split, which is
    the declared fallback rather than a silent one.
    """
    codes = int(code_points.shape[0])
    if codes % groups:
        raise ValueError("the code count must divide into equal buckets")
    if layer_centres.shape != (groups, code_points.shape[1]):
        raise ValueError("one centre per layer is required")
    distance = torch.cdist(code_points.double(), layer_centres.double())
    quota = codes // groups
    leftover = set(range(codes))
    buckets: list[list[int]] = []
    for group in range(groups):
        ranked = sorted(leftover, key=lambda code: (float(distance[code, group]), code))
        chosen = sorted(ranked[:quota])
        buckets.append(chosen)
        leftover.difference_update(chosen)
    if leftover:
        raise RuntimeError("the partition left codes unassigned")
    return buckets


def target_marginal(partition: list[list[int]], rho, codes: int | None = None,
                    dtype=torch.float64) -> torch.Tensor:
    """``pi_k = rho_{g(k)} / |B_{g(k)}|``, uniform under ``BEHAVIOR_OFF``."""
    total = int(codes if codes is not None else sum(len(bucket) for bucket in partition))
    if BEHAVIOR_OFF:
        return torch.full((total,), 1.0 / float(total), dtype=dtype)
    rho = torch.as_tensor(rho, dtype=dtype).reshape(-1)
    if rho.numel() != len(partition):
        raise ValueError("rho must have one entry per bucket")
    marginal = torch.empty(total, dtype=dtype)
    for share, bucket in zip(rho.tolist(), partition):
        marginal[torch.as_tensor(bucket, dtype=torch.long)] = float(share) / float(len(bucket))
    return marginal / marginal.sum()


# ------------------------------------------------------------- the assignment
def centred_cost(cost: torch.Tensor, where: str = "") -> torch.Tensor:
    """The baseline's centring, with the degenerate cases named.

    A flat matrix has no preference, so its centring is zero; a non-finite one is a
    defect and raises with the range in the message.
    """
    maximum, minimum = cost.max(), cost.min()
    spread = maximum - minimum
    if not bool(torch.isfinite(spread)):
        raise ValueError("the cost matrix is not finite at " + (where or "an unnamed call")
                         + ": max=" + str(float(maximum)) + " min=" + str(float(minimum))
                         + " finite_rows=" + str(float(torch.isfinite(cost).float().mean())))
    if float(spread) <= 0:
        return torch.zeros_like(cost)
    return (cost - (maximum + minimum) / 2) / (spread / 2 + 1e-5)


def structured_sinkhorn(centred: torch.Tensor, epsilon: float, iterations: int,
                        marginal: torch.Tensor) -> torch.Tensor:
    """The baseline's balanced assignment normalised to ``marginal`` columns.

    Same ``exp(-D/eps)``, same column and row normalisation, same iteration count as
    the frozen quantiser; only the column target is ``batch * marginal`` instead of a
    constant ``batch / K``. Row-normalised at the end, exactly like the baseline.
    """
    dtype = centred.dtype
    marginal = torch.as_tensor(marginal, dtype=dtype, device=centred.device).reshape(-1)
    q = torch.exp(-centred / float(epsilon))
    q = q / q.sum()
    batch, codes = int(q.shape[0]), int(q.shape[1])
    if int(marginal.numel()) != codes:
        raise ValueError("the marginal must have one entry per code")
    for _ in range(int(iterations)):
        q = q * ((marginal * float(batch)).unsqueeze(0) / q.sum(0, keepdim=True).clamp_min(1e-300))
        q = q / q.sum(1, keepdim=True)
    return q / q.sum(-1, keepdim=True)


def prior_bias(pi: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """``-T * log pi`` per code, the additive bias the prior contributes.

    ``BEHAVIOR_OFF`` zeroes it (and the caller passes the uniform pi), so the arm
    reduces to the plain balanced assignment.
    """
    if BEHAVIOR_OFF or scale is None or float(scale) == 0.0:
        return torch.zeros_like(pi)
    return -float(scale) * torch.log(pi.reshape(-1).clamp_min(1e-12))


def hard_histogram(plan: torch.Tensor, codes: int) -> dict:
    """The hard-code histogram a plan's argmax produces: used, largest share, entropy."""
    picks = plan.argmax(dim=-1)
    counts = torch.bincount(picks, minlength=int(codes)).to(torch.float64)
    share = counts / counts.sum().clamp_min(1.0)
    nonzero = share[share > 0]
    return {"codes_used": int((counts > 0).sum()), "max_bucket_share": float(share.max()),
            "usage_entropy_normalised": float(
                -(nonzero * nonzero.log()).sum() / math.log(float(codes)))}


def hard_deviation(plan: torch.Tensor, pi: torch.Tensor) -> dict:
    """The hard-object gate: per-code row deviation and max relative error vs a target."""
    rows = float(plan.shape[0])
    picks = plan.argmax(dim=-1)
    counts = torch.bincount(picks, minlength=int(pi.numel())).to(torch.float64)
    got = counts / counts.sum().clamp_min(1.0)
    target = pi.reshape(-1).clamp_min(1e-12)
    return {"max_row_deviation": float(((got - target).abs() * rows).max()),
            "max_relative_error": float(((got - target).abs() / target).max()),
            "quantisation_floor": float((0.5 / rows) / float(target.min()))}


def proxy_assignment(cost: torch.Tensor, plan: torch.Tensor | None,
                     where: str = "") -> torch.Tensor:
    """Straight-through proxy: the taken code in front, the carrier's slope behind.

    The hard code is the plan's ``argmax`` on the last level and the cost's
    ``argmin`` before it; the derivative comes from the single centred softmax, never
    from the plan, because fifty sharp normalisations amplify the backward by
    ``(1/eps)^50`` (the failure #150's arm hit inside its first epoch).
    """
    codes = int(cost.shape[-1])
    centred = centred_cost(cost, where=where)
    carrier = torch.softmax(-centred / PROXY_TEMPERATURE, dim=-1)
    hard = (plan.detach().argmax(dim=-1) if plan is not None
            else cost.detach().argmin(dim=-1))
    smoothed = (1.0 - PROXY_SMOOTHING) * carrier + PROXY_SMOOTHING / float(codes)
    onehot = torch.zeros_like(smoothed).scatter_(1, hard.unsqueeze(1), 1.0)
    return onehot + (smoothed - smoothed.detach())


# ---------------------------------------------------------------------- terms
def _overlap(prefixes: list[torch.Tensor], first: torch.Tensor,
             second: torch.Tensor) -> torch.Tensor:
    total = torch.zeros(first.shape[0], dtype=prefixes[0].dtype, device=prefixes[0].device)
    for prefix in prefixes:
        total = total + (prefix.index_select(0, first) * prefix.index_select(0, second)).sum(-1)
    return total / float(len(prefixes))


def triangle_loss(prefixes: list[torch.Tensor], triangles: torch.Tensor) -> torch.Tensor:
    """``1 -`` the mean prefix overlap of every edge of every batch triangle."""
    if triangles.numel() == 0:
        return prefixes[0].sum() * 0.0
    overlaps = [_overlap(prefixes, triangles[:, first], triangles[:, second])
                for first, second in ((0, 1), (0, 2), (1, 2))]
    return 1.0 - torch.stack(overlaps).mean()


def prior_kl(plan: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
    """``KL(column mass || pi)`` for the last level, normalised by its exact bound.

    ``KL(q || pi) <= log(1 / min pi)`` for any distribution ``q``, so the divisor puts
    the term in ``[0, 1]`` without a clamp.
    """
    marginal = marginal.reshape(-1).clamp_min(1e-12)
    observed = plan.mean(0).clamp_min(1e-12)
    upper = math.log(1.0 / float(marginal.min()))
    return (observed * (observed / marginal).log()).sum() / upper


def realised_error(plan: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
    """The largest relative deviation of the realised column share from ``pi``."""
    marginal = marginal.reshape(-1)
    got = plan.mean(0)
    got = got / got.sum().clamp_min(1e-12)
    return ((got - marginal).abs() / marginal.clamp_min(1e-12)).max()


def realised_rows(plan: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
    """``|realised_k - pi_k|`` expressed in rows of this batch, per code."""
    rows = float(plan.shape[0])
    marginal = marginal.reshape(-1)
    got = plan.mean(0)
    got = got / got.sum().clamp_min(1e-12)
    return (got - marginal).abs() * rows


def realisability(plan_of_batch, marginal: torch.Tensor) -> dict:
    """The three checks the review replaced the single threshold with.

    ``plan_of_batch`` is called with a row count and must return the row-normalised
    plan of that many rows. The report carries the probe's relative error, the
    per-code row deviation at the training batch, and the whole curve, so a reader
    can see the 1/B decay instead of one number that was never reachable.
    """
    curve = []
    for rows in CURVE_BATCHES:
        plan = plan_of_batch(rows)
        curve.append({"rows": int(rows),
                      "max_relative_error": float(realised_error(plan, marginal)),
                      "max_row_deviation": float(realised_rows(plan, marginal).max())})
    probe = next(entry for entry in curve if entry["rows"] == PROBE_ROWS)
    batch = next(entry for entry in curve if entry["rows"] == 4096)
    return {
        "probe": {"rows": probe["rows"], "max_relative_error": probe["max_relative_error"],
                  "tolerance": PROBE_TOLERANCE,
                  "arithmetic_bound": 1.0 / (probe["rows"] * float(marginal.min())),
                  "passed": probe["max_relative_error"] <= PROBE_TOLERANCE},
        "batch": {"rows": batch["rows"], "max_row_deviation": batch["max_row_deviation"],
                  "tolerance_rows": BATCH_TOLERANCE_ROWS,
                  "passed": batch["max_row_deviation"] <= BATCH_TOLERANCE_ROWS},
        "curve": curve,
        "curve_decays": all(curve[index]["max_relative_error"]
                            >= curve[index + 1]["max_relative_error"]
                            for index in range(len(curve) - 1)),
    }


def batch_triangles(item_ids: np.ndarray, triangles: np.ndarray,
                    item_count: int) -> tuple[np.ndarray, int, int]:
    """The frozen triangles whose three items are all in this batch."""
    ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    lookup = np.full(int(item_count), -1, dtype=np.int64)
    lookup[ids] = np.arange(ids.size, dtype=np.int64)
    triples = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    mapped = lookup[triples] if triples.size else np.zeros((0, 3), dtype=np.int64)
    keep = (mapped >= 0).all(axis=1) if mapped.size else np.zeros(0, dtype=bool)
    return mapped[keep], int(keep.sum()), int(triples.shape[0])


def level_usage(prefixes: list[torch.Tensor]) -> list[dict]:
    """Per-level codes used, largest code share and normalised entropy."""
    rows = []
    for level, prefix in enumerate(prefixes, start=1):
        hard = prefix.detach().argmax(dim=-1)
        codes = int(prefix.shape[-1])
        counts = torch.bincount(hard, minlength=codes).to(torch.float64)
        share = counts / counts.sum().clamp_min(1.0)
        nonzero = share[share > 0]
        rows.append({"level": level, "codes_used": int((counts > 0).sum()),
                     "max_bucket_share": float(share.max()),
                     "usage_entropy_normalised": float(
                         -(nonzero * nonzero.log()).sum() / math.log(codes))})
    return rows


def guard_violations(usage: list[dict], codes_min: int = CODES_USED_MIN,
                     share_max: float = MAX_BUCKET_SHARE_MAX) -> list[dict]:
    """The exposure guard: a level outside these bounds is a mechanism failure."""
    violations = []
    for row in usage:
        reasons = []
        if row["codes_used"] < codes_min:
            reasons.append("codes_used " + str(row["codes_used"]) + " < " + str(codes_min))
        if row["max_bucket_share"] > share_max:
            reasons.append("max_bucket_share " + str(round(row["max_bucket_share"], 4))
                           + " > " + str(share_max))
        if reasons:
            violations.append({"level": row["level"], "reasons": reasons})
    return violations
