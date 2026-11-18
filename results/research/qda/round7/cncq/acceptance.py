"""The acceptance gate: a checkpoint is certified only if every layer passes.

Three layers, in the order they can fail:

    capacity      min occupancy >= m for every codeword       (construction, layer 1)
    geometry      codewords separated by the analytic bound   (construction, layer 2)
    representation effective rank of the quantised readout     (measurement)
    fidelity      L3 behaviour AUC vs the certified reference  (measurement)
    strata        bridge / intra-community AUC vs the reference(measurement)

The last three cannot be guaranteed by construction: a quantiser cannot invent semantics an
encoder never produced. What the gate can guarantee is that such a model is never *accepted*
-- the caller keeps the previous certified checkpoint instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Gate:
    name: str
    actual: float
    threshold: float
    ok: bool
    note: str = ""


@dataclass
class Policy:
    """Thresholds, with where each number comes from."""

    #: Every codeword must hold at least this share of the balanced assignment.  A quarter
    #: of N/K is the weakest floor that still forbids "a handful of codes carry everything":
    #: for N=24587, K=256 it is 24 items per code (the balanced share is 96).
    minimum_share: float = 0.25
    #: The quantised readout must keep at least this effective rank.  Round 1 measured a
    #: healthy L1 readout at 12-13 and a collapsed one at 2-7, so 8 sits between them.
    min_effective_rank: float = 8.0
    #: Behaviour fidelity is judged against the certified reference, not in the abstract.
    fidelity_tolerance: float = 0.0005      # the measured screen-to-screen repeat difference
    stratum_tolerance: float = 0.0010


def minimum_count(items: int, codewords: int, policy: Policy) -> int:
    return max(1, int((items / max(codewords, 1)) * policy.minimum_share))


def certify_capacity(counts, policy: Policy = Policy()) -> Gate:
    counts = [int(c) for c in counts]
    items, codewords = sum(counts), len(counts)
    floor = minimum_count(items, codewords, policy)
    return Gate("capacity", float(min(counts)), float(floor), min(counts) >= floor,
                f"{sum(1 for c in counts if c == 0)} empty codeword(s) of {codewords}")


def certify_geometry(bound: dict) -> Gate:
    return Gate("geometry", float(bound["margin"]), 0.0, bool(bound["ok"]),
                f"delta={bound['delta']:.4g} 2rho={bound['two_rho']:.4g}")


def certify_representation(rank: float, policy: Policy = Policy()) -> Gate:
    return Gate("representation", float(rank), policy.min_effective_rank,
                rank >= policy.min_effective_rank, "effective rank of the quantised readout")


def certify_fidelity(actual: float, reference: float, policy: Policy = Policy()) -> Gate:
    threshold = reference - policy.fidelity_tolerance
    return Gate("fidelity", float(actual), threshold, actual >= threshold,
                f"reference L3 AUC {reference:.4f}")


def certify_strata(strata: dict, reference_strata: dict, policy: Policy = Policy()) -> list[Gate]:
    gates = []
    for name, value in strata.items():
        if name not in reference_strata:
            continue
        threshold = reference_strata[name] - policy.stratum_tolerance
        gates.append(Gate(f"stratum:{name}", float(value), threshold, value >= threshold,
                          "bridge edges are the stratum that fails first"))
    return gates


def certify(metrics: dict, reference: dict, policy: Policy = Policy()) -> dict:
    """`metrics` and `reference` share the keys produced by the round analysis."""
    gates = [
        certify_capacity(metrics["counts"], policy),
        certify_geometry(metrics["geometry"]),
        certify_representation(metrics["effective_rank"], policy),
        certify_fidelity(metrics["fidelity"], reference["fidelity"], policy),
    ]
    gates += certify_strata(metrics.get("strata", {}), reference.get("strata", {}), policy)
    failed = [gate.name for gate in gates if not gate.ok]
    return {"certified": not failed, "failed": failed,
            "gates": [{"name": g.name, "actual": g.actual, "threshold": g.threshold,
                       "ok": g.ok, "note": g.note} for g in gates]}
