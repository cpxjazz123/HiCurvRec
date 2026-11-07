"""The outer loop: choose the geometry weight per prefix, accept only a real gain.

Nothing here trains. The loop freezes the model, re-encodes the items of one
prefix under each candidate weight, measures the exact structural distortion of
the resulting hard SIDs, and accepts the candidate only when it both improves
the distortion by more than the tolerance and keeps every quantity of the
feasible region inside its budget. Acceptance is sequential and deterministic:
level by level, prefix by prefix in the stable order, one candidate at a time,
with the state updated before the next prefix is looked at.

The feasible region is what stops the loop from buying structure with semantics
or with resolution: a candidate that drops the distortion but coarsens the codes,
collides more identifiers or worsens the degree error is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy

import numpy as np
import torch

from .mixed_rq import ALPHA_CANDIDATES, ALPHA_TOLERANCE, AlphaTable, encode_tokens

#: The smallest prefix the table may hold an entry for.
MIN_PREFIX_ITEMS = 32
#: The smallest number of observed edges and unobserved pairs a prefix needs.
MIN_PREFIX_EDGES = 64
MIN_PREFIX_UNOBSERVED = 64
#: The feasible region of one accepted step, relative to the round's start.
SEMANTIC_SLACK = 1.01
COLLISION_SLACK = 0.005
CODE_USAGE_FLOOR = 0.95
BUCKET_SLACK = 1.10


@dataclass
class RoundRecord:
    """Everything one outer-loop round did, so the report can be checked."""

    round_index: int
    epoch: int
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    distortion_start: float = 0.0
    distortion_end: float = 0.0
    seconds: float = 0.0
    changed_sids: int = 0

    def to_dict(self) -> dict:
        return {"round": self.round_index, "epoch": self.epoch,
                "distortion_start": self.distortion_start,
                "distortion_end": self.distortion_end,
                "accepted": self.accepted, "rejected": self.rejected,
                "skipped": self.skipped, "seconds": round(self.seconds, 3),
                "changed_sids": self.changed_sids}


def prefix_groups(tokens: torch.Tensor, level: int) -> dict[tuple, np.ndarray]:
    """The item indices of every prefix at one level, in the stable dictionary order."""
    keys = tokens[:, :level].cpu().numpy()
    groups: dict[tuple, list[int]] = {}
    for index, key in enumerate(map(tuple, keys.tolist())):
        groups.setdefault(key, []).append(index)
    return {key: np.asarray(sorted(items), dtype=np.int64) for key, items in sorted(groups.items())}


def prefix_probe_counts(group: np.ndarray, state) -> tuple[int, int]:
    """The observed and unobserved supported pairs that touch this prefix."""
    inside = np.zeros(state.n_nodes, dtype=bool)
    inside[group] = True
    positions = np.unique(np.concatenate([state.incident[int(item)] for item in group]))
    touches = inside[state.pairs[positions, 0]] | inside[state.pairs[positions, 1]]
    positions = positions[touches]
    return int(state.is_edge[positions].sum()), int((~state.is_edge[positions]).sum())


def prefix_eligibility_error(table: AlphaTable, tokens: torch.Tensor, state) -> str:
    """Reject stale explicit local entries; do not invent a pruning policy.

    SGD or an earlier-level change can shrink a previously supported group.
    Merely skipping it does NOT make AlphaTable.get inherit its ancestor. A
    mandatory deletion can also change SIDs and the round's guard baseline;
    that transition needs a reviewed specification, not a silent mutation.
    """
    for level, entries in sorted(table.levels.items()):
        if level <= 1:
            continue
        groups = prefix_groups(tokens, level - 1)
        for prefix in sorted(entries):
            if not prefix:  # A global coordinate, not a local-prefix entry.
                continue
            items = groups.get(prefix)
            if items is None or len(items) < MIN_PREFIX_ITEMS:
                return (f"unsupported explicit prefix at level {level}: {prefix}; "
                        "ancestor fallback requires a reviewed table transition")
            edges, unobserved = prefix_probe_counts(items, state)
            if edges < MIN_PREFIX_EDGES or unobserved < MIN_PREFIX_UNOBSERVED:
                return (f"insufficient support for explicit prefix at level {level}: "
                        f"{prefix} ({edges} edges, {unobserved} unobserved); "
                        "ancestor fallback requires a reviewed table transition")
    return ""


def resolution_report(tokens: torch.Tensor, n_codes: int) -> dict:
    """Used codes, occupied prefixes, the full-SID bucket and the excess collision."""
    levels = tokens.shape[1]
    used = [len(torch.unique(tokens[:, level])) for level in range(levels)]
    prefixes = [len(torch.unique(tokens[:, :level], dim=0)) for level in range(1, levels + 1)]
    # Count occupied SIDs only: a 256^3 dense bincount allocates 128 MiB for
    # each tiny candidate although at most N complete SIDs can be occupied.
    _, counts = torch.unique(tokens, dim=0, return_counts=True)
    largest = int(counts.max())
    unique = int(len(counts))
    total = int(tokens.shape[0])
    return {"used_codes": used, "occupied_prefixes": prefixes,
            "max_bucket": largest, "unique_sids": unique,
            "excess_collision": 1.0 - unique / total}


def within_feasible_region(start: dict, candidate: dict, semantic: float,
                           adjacency_brier: float, degree_error: float) -> tuple[bool, str]:
    """Whether a step may be accepted, and why not when it may not."""
    metrics = [semantic, adjacency_brier, degree_error,
               start["semantic"], start["adjacency_brier"], start["degree_error"]]
    for resolution in (start["resolution"], candidate):
        metrics.extend([resolution["excess_collision"], resolution["max_bucket"]])
        metrics.extend(resolution["used_codes"])
        metrics.extend(resolution["occupied_prefixes"])
    if not np.isfinite(np.asarray(metrics, dtype=np.float64)).all():
        return False, "non-finite start or candidate guard metric"
    if np.any(np.asarray(metrics, dtype=np.float64) < 0):
        return False, "negative guard metric"
    for key in ("used_codes", "occupied_prefixes"):
        if len(start["resolution"][key]) != len(candidate[key]):
            return False, "mismatched resolution levels"
    if semantic > start["semantic"] * SEMANTIC_SLACK + 1e-12:
        return False, "semantic reconstruction worsened"
    if adjacency_brier > start["adjacency_brier"] * SEMANTIC_SLACK + 1e-12:
        return False, "adjacency Brier worsened"
    if degree_error > start["degree_error"] * SEMANTIC_SLACK + 1e-12:
        return False, "degree error worsened"
    if candidate["excess_collision"] > start["resolution"]["excess_collision"] + COLLISION_SLACK:
        return False, "excess collision grew past its allowance"
    for level, (was, now) in enumerate(zip(start["resolution"]["used_codes"],
                                           candidate["used_codes"])):
        if now < was * CODE_USAGE_FLOOR:
            return False, f"level {level + 1} code usage fell below 95% of the start"
    for level, (was, now) in enumerate(zip(start["resolution"]["occupied_prefixes"],
                                           candidate["occupied_prefixes"])):
        if now < was * CODE_USAGE_FLOOR:
            return False, f"level {level + 1} prefix count fell below 95% of the start"
    if candidate["max_bucket"] > start["resolution"]["max_bucket"] * BUCKET_SLACK:
        return False, "the largest SID bucket grew past its allowance"
    return True, ""


def structural_guard_metrics(probability: np.ndarray, pairs: np.ndarray,
                             is_edge: np.ndarray, original_degree: np.ndarray) -> dict:
    """The SAME class-balanced Brier and node-degree error for every candidate.

    These take probabilities, not edge-weighted AF3 expectations. The degree
    target is the original TRAIN graph, never the larger candidate support.
    """
    p = np.asarray(probability, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    edge = np.asarray(is_edge, dtype=bool)
    original = np.asarray(original_degree, dtype=np.float64)
    if (p.ndim != 1 or pairs.shape != (len(p), 2) or edge.shape != p.shape
            or original.ndim != 1 or not len(original)):
        raise ValueError("invalid structural guard shapes")
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.isfinite(original).all() or np.any(original < 0)):
        raise ValueError("structural guards require finite probabilities and degrees")
    if not edge.any() or edge.all():
        raise ValueError("class-balanced Brier requires edges and non-edges")
    if pairs.min() < 0 or pairs.max() >= len(original):
        raise ValueError("support endpoints do not match original node degrees")
    soft_degree = np.zeros_like(original)
    np.add.at(soft_degree, pairs[:, 0], p)
    np.add.at(soft_degree, pairs[:, 1], p)
    brier = 0.5 * np.square(p[edge] - 1.0).mean() + 0.5 * np.square(p[~edge]).mean()
    error = np.abs(soft_degree - original).mean() / max(float(original.mean()), 1e-12)
    return {"adjacency_brier": float(brier), "degree_error": float(error)}


@torch.no_grad()
def choose_weight(z: torch.Tensor, rq, table: AlphaTable, state, *, level: int,
                  prefix: tuple, items: np.ndarray, current_tokens: torch.Tensor,
                  start: dict, evaluate_feasibility, tolerance: float | None = None) -> dict:
    """Evaluate the candidates for one prefix and accept at most one."""
    tolerance = ALPHA_TOLERANCE if tolerance is None else tolerance
    incumbent = state.summary()["distortion"]
    if not np.isfinite(incumbent):
        raise ValueError("the incumbent AF3 distortion is non-finite")
    original_phi = _readout_of(z, current_tokens, rq)
    attempts = []
    best = None
    for candidate in ALPHA_CANDIDATES:
        if abs(candidate - table.get(level, prefix)) < 1e-12:
            attempts.append({"alpha": candidate, "distortion": incumbent,
                             "verdict": "incumbent"})
            continue
        override = {(level, prefix): candidate}
        tokens = encode_tokens(z, rq, override=override)
        if torch.equal(tokens, current_tokens):
            attempts.append({"alpha": candidate, "distortion": incumbent,
                             "verdict": "no code changed, so no geometric effect"})
            continue
        phi = _readout_of(z, tokens, rq)
        changed = torch.nonzero((tokens != current_tokens).any(dim=1)).flatten().cpu().numpy()
        try:
            state.update(phi, changed)
            outcome = state.summary()
            if np.isfinite(outcome["distortion"]):
                feasible, reason = evaluate_feasibility(tokens, outcome)
            else:
                feasible, reason = False, "non-finite candidate AF3 distortion"
        finally:
            # Also restore when the guard evaluator raises: never leave a trial
            # in the cache after an aborted outer-loop step.
            state.update(original_phi, changed)
        moved = int(len(changed))
        entry = {"alpha": candidate, "distortion": outcome["distortion"],
                 "changed_sids": moved, "feasible": feasible, "reason": reason,
                 "constraints": outcome.get("guard_metrics", {})}
        attempts.append(entry)
        better = outcome["distortion"] < incumbent - tolerance * max(1.0, abs(incumbent))
        if feasible and better and (best is None or outcome["distortion"] < best["distortion"]):
            best = entry
    if best is None:
        return {"accepted": None, "attempts": attempts, "distortion": incumbent}
    original_table = copy.deepcopy(table.levels)
    try:
        table.set(level, prefix, best["alpha"])
        tokens = encode_tokens(z, rq)
        changed = torch.nonzero((tokens != current_tokens).any(dim=1)).flatten().cpu().numpy()
        state.update(_readout_of(z, tokens, rq), changed)
    except Exception:
        table.levels = original_table
        # A failed update may have modified only part of the cache. Rebuild
        # from the incumbent, rather than trusting an incomplete delta.
        state.full(original_phi)
        raise
    return {"accepted": best, "attempts": attempts,
            "distortion": state.summary()["distortion"]}


def _readout_of(z: torch.Tensor, tokens: torch.Tensor, rq) -> torch.Tensor:
    """The readout ``phi`` of a complete token set, from the codebooks."""
    from .mixed_rq import quantized_readout
    from .poincare_contract import expmap0
    euclidean, curved = [], []
    for level, layer in enumerate(rq.levels):
        index = tokens[:, level].to(layer.embed.weight.device)
        picked = layer.embed.weight[index]
        left, right = torch.split(picked, [16, 16], dim=-1)
        euclidean.append(left)
        # The curved half of the table is a tangent-space parameter, so the ball
        # point the readout composes is exp_0 of it.
        curved.append(expmap0(right, rq.curvature) if rq.curvature > 0 else right)
    return quantized_readout(euclidean, curved, rq.curvature)

