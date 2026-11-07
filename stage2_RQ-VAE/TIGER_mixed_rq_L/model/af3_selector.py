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

import numpy as np
import torch

from .af3_structure import TAU_FLOOR
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


def resolution_report(tokens: torch.Tensor, n_codes: int) -> dict:
    """Used codes, occupied prefixes, the full-SID bucket and the excess collision."""
    levels = tokens.shape[1]
    used = [len(torch.unique(tokens[:, level])) for level in range(levels)]
    prefixes = [len(torch.unique(tokens[:, :level], dim=0)) for level in range(1, levels + 1)]
    counts = torch.bincount(tokens[:, 0] * (n_codes ** (levels - 1))
                            + tokens[:, 1] * (n_codes ** (levels - 2)) + tokens[:, 2],
                            minlength=n_codes ** levels)
    largest = int(counts.max())
    unique = int((counts > 0).sum())
    total = int(tokens.shape[0])
    return {"used_codes": used, "occupied_prefixes": prefixes,
            "max_bucket": largest, "unique_sids": unique,
            "excess_collision": 1.0 - unique / total}


def within_feasible_region(start: dict, candidate: dict, semantic: float,
                           adjacency_brier: float, degree_error: float) -> tuple[bool, str]:
    """Whether a step may be accepted, and why not when it may not."""
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


def choose_weight(z: torch.Tensor, rq, table: AlphaTable, state, *, level: int,
                  prefix: tuple, items: np.ndarray, current_tokens: torch.Tensor,
                  start: dict, evaluate_feasibility, tolerance: float | None = None) -> dict:
    """Evaluate the candidates for one prefix and accept at most one."""
    tolerance = ALPHA_TOLERANCE if tolerance is None else tolerance
    incumbent = state.summary()["distortion"]
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
        state.update(phi, items)
        outcome = state.summary()
        moved = int((tokens[items] != current_tokens[items]).any(dim=1).sum())
        feasible, reason = evaluate_feasibility(tokens, outcome)
        entry = {"alpha": candidate, "distortion": outcome["distortion"],
                 "changed_sids": moved, "feasible": feasible, "reason": reason}
        attempts.append(entry)
        better = outcome["distortion"] < incumbent - tolerance * max(1.0, abs(incumbent))
        if feasible and better and (best is None or outcome["distortion"] < best["distortion"]):
            best = entry
        # Restore the state for the next candidate: every candidate starts from
        # exactly the same model state, as the specification requires.
        state.update(_readout_of(z, current_tokens, rq), items)
    if best is None:
        return {"accepted": None, "attempts": attempts, "distortion": incumbent}
    table.set(level, prefix, best["alpha"])
    tokens = encode_tokens(z, rq)
    state.update(_readout_of(z, tokens, rq), items)
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
