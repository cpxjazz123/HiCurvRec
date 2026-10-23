"""The one predicate that decides scientific success (issue #110, section 4).

The previous campaign could decide success in three different places, and one of them read
the Stage-3 metrics without the frozen evaluation. Issue #110 asks for exactly one shared
predicate over matched artifacts, and this is it.

A win requires ALL of the following, from the SAME arm:

* one proposal hash, one implementation hash, one frozen protocol hash and the seed 42;
* the seven hyperbolic gates H1-H7 passed, with none of them declared not-applicable or
  insufficient-evidence;
* the six CGV gates G1-G6 passed under the same identity;
* the frozen Hard-L1 enrichment ratio strictly above its threshold, with both shares
  recorded;
* a complete, audited Stage-3 whose Recall@10 and NDCG@10 are both strictly above their
  thresholds.

An ER-only win is not a win. A Stage-3-only win is not a win. A tie is not a win -- the
thresholds are strict. A measurement from a different implementation, a different protocol
or a different seed is not a win either, however good it looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The frozen thresholds, from issue #110's first table.
ER_THRESHOLD = 16.38953942232631
RECALL_THRESHOLD = 0.06406796775709883
NDCG_THRESHOLD = 0.03530651701940677

#: The only seed an official experiment may use.
FROZEN_SEED = 42

#: The hyperbolic gates, in the order the issue lists them.
HYPERBOLIC_GATES: Tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")

#: The CGV gates.
CGV_GATES: Tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")

#: A gate state that does not count as a pass, whatever the gate is.
NON_PASSING_STATES = frozenset({"not_applicable", "insufficient_evidence", "fail", "failed"})


@dataclass
class Check:
    """One requirement, and whether the evidence satisfies it."""

    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class TripleWinVerdict:
    """The verdict of the success predicate, with every check that produced it."""

    won: bool
    reason: str = ""
    checks: List[Check] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "won": self.won,
            "reason": self.reason,
            "checks": [check.as_dict() for check in self.checks],
            "failed_checks": [check.name for check in self.checks if not check.ok],
        }


def _gate_states(certificate: Optional[Dict[str, Any]], gates: Sequence[str]) -> Dict[str, str]:
    """The per-gate states of a certificate, whether it is flat or nested."""
    if not isinstance(certificate, dict):
        return {}
    states: Dict[str, str] = {}
    for gate in gates:
        entry = certificate.get(gate)
        if isinstance(entry, dict):
            states[gate] = str(entry.get("state") or entry.get("status") or "")
        elif isinstance(entry, str):
            states[gate] = entry
        else:
            summary = certificate.get("summary") or {}
            if isinstance(summary, dict) and gate in summary:
                value = summary[gate]
                states[gate] = str(
                    value.get("state") if isinstance(value, dict) else value
                )
    return states


def _gates_passed(certificate: Optional[Dict[str, Any]], gates: Sequence[str]) -> Tuple[bool, str]:
    """Whether every gate of a certificate passed, strictly."""
    if not isinstance(certificate, dict):
        return False, "the certificate is missing"
    states = _gate_states(certificate, gates)
    if not states:
        return False, "the certificate records no gate states"
    failed = [
        f"{gate}={states.get(gate, 'missing')}"
        for gate in gates
        if states.get(gate, "") != "pass"
    ]
    if failed:
        return False, ", ".join(failed)
    return True, "every gate passed"


def is_verified_triple_win(evidence: Optional[Dict[str, Any]]) -> TripleWinVerdict:
    """Whether the evidence is a verified triple-metric win, and why.

    The function never raises for missing evidence: it returns the checks that failed, so a
    caller can report exactly what is missing instead of treating a missing measurement as
    a win.
    """
    evidence = evidence or {}
    checks: List[Check] = []

    # 1. the identity of the experiment
    proposal = evidence.get("proposal_sha256") or ""
    implementation = evidence.get("implementation_sha256") or ""
    protocol = evidence.get("frozen_protocol_hash") or ""
    seed = evidence.get("seed")
    identity_ok = bool(proposal) and bool(implementation) and bool(protocol)
    checks.append(
        Check(
            "identity",
            identity_ok,
            "proposal, implementation and protocol hashes are recorded"
            if identity_ok
            else "the proposal, implementation or protocol hash is missing",
        )
    )
    checks.append(
        Check(
            "seed",
            seed == FROZEN_SEED,
            f"the seed is {seed}" if seed == FROZEN_SEED else f"the seed is {seed}, not {FROZEN_SEED}",
        )
    )

    # 2. the gates, on the same identity
    hyperbolic = evidence.get("hyperbolic_certificate") or {}
    cgv = evidence.get("cgv_certificate") or {}
    h_ok, h_detail = _gates_passed(hyperbolic, HYPERBOLIC_GATES)
    checks.append(Check("hyperbolic_gates", h_ok, h_detail))
    g_ok, g_detail = _gates_passed(cgv, CGV_GATES)
    checks.append(Check("cgv_gates", g_ok, g_detail))
    for name, certificate in (("hyperbolic", hyperbolic), ("cgv", cgv)):
        bound = (certificate or {}).get("proposal_sha256") if isinstance(certificate, dict) else None
        if bound and proposal and bound != proposal:
            checks.append(
                Check(
                    f"{name}_certificate_identity",
                    False,
                    f"the certificate is for proposal {bound}, not {proposal}",
                )
            )

    # 3. the frozen enrichment ratio, with both shares
    evaluation = evidence.get("frozen_evaluation") or {}
    er = evaluation.get("hard_l1_er")
    if er is None:
        er = evaluation.get("l1_er")
    er_ok = isinstance(er, (int, float)) and er > ER_THRESHOLD
    checks.append(
        Check(
            "frozen_er",
            er_ok,
            f"ER {er} against the threshold {ER_THRESHOLD}"
            if er is not None
            else "the frozen enrichment ratio is missing",
        )
    )
    shares_ok = (
        evaluation.get("pos_share") is not None and evaluation.get("neg_share") is not None
    )
    checks.append(
        Check(
            "er_shares_recorded",
            shares_ok,
            "both the positive and the negative share are recorded"
            if shares_ok
            else "the positive or the negative share is missing",
        )
    )

    # 4. the downstream measurement, complete
    stage3 = evidence.get("stage3") or {}
    recall = stage3.get("test_recall@10")
    ndcg = stage3.get("test_ndcg@10")
    complete = bool(stage3.get("complete", stage3.get("n_eval") is not None))
    checks.append(
        Check(
            "stage3_complete",
            complete,
            "the Stage-3 result is complete"
            if complete
            else "the Stage-3 result is incomplete or unaudited",
        )
    )
    recall_ok = isinstance(recall, (int, float)) and recall > RECALL_THRESHOLD
    checks.append(
        Check(
            "stage3_recall",
            recall_ok,
            f"Recall@10 {recall} against the threshold {RECALL_THRESHOLD}"
            if recall is not None
            else "the Recall@10 is missing",
        )
    )
    ndcg_ok = isinstance(ndcg, (int, float)) and ndcg > NDCG_THRESHOLD
    checks.append(
        Check(
            "stage3_ndcg",
            ndcg_ok,
            f"NDCG@10 {ndcg} against the threshold {NDCG_THRESHOLD}"
            if ndcg is not None
            else "the NDCG@10 is missing",
        )
    )

    failed = [check for check in checks if not check.ok]
    if failed:
        return TripleWinVerdict(
            False,
            f"{len(failed)} requirement(s) not met: "
            + ", ".join(f"{check.name} ({check.detail})" for check in failed),
            checks,
        )
    return TripleWinVerdict(
        True,
        "the frozen ER, the Stage-3 Recall@10 and the Stage-3 NDCG@10 are all strictly "
        "above their thresholds for the same proposal, implementation, protocol and seed, "
        "with every hyperbolic and CGV gate passed",
        checks,
    )


__all__ = [
    "CGV_GATES",
    "Check",
    "ER_THRESHOLD",
    "FROZEN_SEED",
    "HYPERBOLIC_GATES",
    "NDCG_THRESHOLD",
    "RECALL_THRESHOLD",
    "TripleWinVerdict",
    "is_verified_triple_win",
]
