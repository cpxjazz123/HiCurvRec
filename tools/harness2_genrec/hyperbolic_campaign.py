"""The durable state machine for issue #110's hyperbolic-only campaign.

This is a NEW machine, not the #109 one resumed: the issue says so explicitly, and the two
campaigns have different states, different namespaces and different rules. The states are the
ones the issue lists, and each candidate is an independently durable state machine of its
own -- a research planning cycle is not a measure of completed experiments, and a resource
pause is not a failed science.

The transitions are the only way a candidate moves. Each one is guarded by what the issue
requires at that point: the mathematics is reviewed before there is code, the geometry gates
and the CGV gates both pass before an official cold run, the frozen evaluation is read once
after the export is final, and the Stage-3 is queued only for an arm that cleared the gate.
Every transition is written to an append-only audit log, and a candidate that cannot move
records a BLOCKED state with the operator action it needs, rather than spinning.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The new campaign's namespace. Nothing from the earlier campaigns is written here.
NAMESPACE = REPO_ROOT / "results" / "hyperbolic_research_runs"

#: The states a candidate can be in. The order is the order of the issue's flow.
STATES: Tuple[str, ...] = (
    "proposed",
    "reviewed",
    "implemented",
    "screened",
    "sealed",
    "stage2_reserved",
    "stage2_running",
    "stage2_complete",
    "evaluated",
    "stage3_reserved",
    "stage3_running",
    "stage3_complete",
    "analysed",
)

#: The states that are not progress but a stop, with the reason they carry.
BLOCKING_STATES: Tuple[str, ...] = (
    "rejected",
    "novelty_unknown",
    "blocked_literature",
    "blocked_resources",
    "reconciliation_needed",
)

#: The state a candidate starts in.
INITIAL_STATE = "proposed"

#: The legal transitions, as the issue's flow states them.
LEGAL_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "proposed": ("reviewed", "rejected", "novelty_unknown", "blocked_literature"),
    "reviewed": ("implemented", "rejected", "novelty_unknown", "blocked_literature"),
    "implemented": ("screened", "rejected", "reconciliation_needed"),
    "screened": ("sealed", "rejected", "blocked_resources", "reconciliation_needed"),
    "sealed": ("stage2_reserved", "rejected", "blocked_resources"),
    "stage2_reserved": ("stage2_running", "blocked_resources", "reconciliation_needed"),
    "stage2_running": ("stage2_complete", "reconciliation_needed"),
    "stage2_complete": ("evaluated", "reconciliation_needed"),
    "evaluated": ("stage3_reserved", "analysed"),
    "stage3_reserved": ("stage3_running", "blocked_resources", "reconciliation_needed"),
    "stage3_running": ("stage3_complete", "reconciliation_needed"),
    "stage3_complete": ("analysed",),
    "analysed": (),
    "rejected": (),
    "novelty_unknown": ("proposed",),
    "blocked_literature": ("proposed", "rejected"),
    "blocked_resources": ("proposed", "rejected"),
    "reconciliation_needed": ("stage2_complete", "stage3_complete", "rejected"),
}

#: The identity of an official experiment: the tuple that must never repeat.
EXPERIMENT_IDENTITY = ("proposal_sha256", "implementation_sha256", "frozen_protocol_hash", "seed", "kind")


class TransitionError(RuntimeError):
    """The transition is not legal from the candidate's current state."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_id(
    *,
    proposal_sha256: str,
    implementation_sha256: str,
    frozen_protocol_hash: str,
    seed: int,
    kind: str,
) -> str:
    """The identity of an official experiment, as a digest of its tuple."""
    payload = json.dumps(
        {
            "proposal_sha256": proposal_sha256,
            "implementation_sha256": implementation_sha256,
            "frozen_protocol_hash": frozen_protocol_hash,
            "seed": seed,
            "kind": kind,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Candidate:
    """One candidate, and the state machine it moves through."""

    mechanism_id: str
    proposal_sha256: str
    family: str = ""
    state: str = INITIAL_STATE
    implementation_sha256: str = ""
    frozen_protocol_hash: str = ""
    seed: int = 42
    certificates: Dict[str, str] = field(default_factory=dict)
    measurements: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    blocked_reason: str = ""
    next_action: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.state in BLOCKING_STATES

    @property
    def is_finished(self) -> bool:
        return self.state in ("analysed", "rejected")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "proposal_sha256": self.proposal_sha256,
            "implementation_sha256": self.implementation_sha256,
            "frozen_protocol_hash": self.frozen_protocol_hash,
            "family": self.family,
            "seed": self.seed,
            "state": self.state,
            "certificates": dict(self.certificates),
            "measurements": dict(self.measurements),
            "blocked_reason": self.blocked_reason,
            "next_action": self.next_action,
            "history": list(self.history),
        }


class Campaign:
    """The durable campaign: the candidates, the audit log, and the transitions."""

    def __init__(self, namespace: Path = NAMESPACE) -> None:
        self.namespace = Path(namespace)
        self.candidates_dir = self.namespace / "candidates"
        self.audit_path = self.namespace / "audit.jsonl"
        self._candidates: Dict[str, Candidate] = {}
        self._experiments: Dict[str, str] = {}
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> None:
        if not self.candidates_dir.is_dir():
            return
        for path in sorted(self.candidates_dir.glob("*/candidate.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate = Candidate(
                mechanism_id=document["mechanism_id"],
                proposal_sha256=document.get("proposal_sha256", ""),
                family=document.get("family", ""),
                state=document.get("state", INITIAL_STATE),
                implementation_sha256=document.get("implementation_sha256", ""),
                frozen_protocol_hash=document.get("frozen_protocol_hash", ""),
                seed=int(document.get("seed", 42)),
                certificates=document.get("certificates") or {},
                measurements=document.get("measurements") or {},
                history=document.get("history") or [],
                blocked_reason=document.get("blocked_reason", ""),
                next_action=document.get("next_action", ""),
            )
            self._candidates[candidate.mechanism_id] = candidate
        experiments = self.namespace / "experiments.json"
        if experiments.is_file():
            try:
                self._experiments = json.loads(experiments.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._experiments = {}

    def _persist_candidate(self, candidate: Candidate) -> None:
        directory = self.candidates_dir / candidate.mechanism_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "candidate.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(candidate.as_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def _persist_experiments(self) -> None:
        path = self.namespace / "experiments.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._experiments, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def _audit(self, event: str, **fields: Any) -> Dict[str, Any]:
        record = {"utc": now_utc(), "event": event, **fields}
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    # ---------------------------------------------------------------- the candidates
    def add(self, candidate: Candidate) -> Candidate:
        """Register a candidate, or return the one already registered."""
        if candidate.mechanism_id in self._candidates:
            return self._candidates[candidate.mechanism_id]
        self._candidates[candidate.mechanism_id] = candidate
        self._persist_candidate(candidate)
        self._audit(
            "proposed",
            mechanism_id=candidate.mechanism_id,
            proposal_sha256=candidate.proposal_sha256,
            family=candidate.family,
        )
        return candidate

    def candidate(self, mechanism_id: str) -> Optional[Candidate]:
        return self._candidates.get(mechanism_id)

    def candidates(self, *, state: Optional[str] = None) -> List[Candidate]:
        rows = list(self._candidates.values())
        if state is not None:
            rows = [row for row in rows if row.state == state]
        return sorted(rows, key=lambda row: row.mechanism_id)

    # ---------------------------------------------------------------- the transitions
    def transition(
        self,
        mechanism_id: str,
        to_state: str,
        *,
        reason: str = "",
        next_action: str = "",
        certificates: Optional[Dict[str, str]] = None,
        measurements: Optional[Dict[str, Any]] = None,
        implementation_sha256: str = "",
        frozen_protocol_hash: str = "",
    ) -> Candidate:
        """Move a candidate to a new state, if the transition is legal."""
        candidate = self._candidates.get(mechanism_id)
        if candidate is None:
            raise TransitionError(f"there is no candidate {mechanism_id!r}")
        allowed = LEGAL_TRANSITIONS.get(candidate.state, ())
        if to_state not in allowed:
            raise TransitionError(
                f"{mechanism_id!r} cannot move from {candidate.state!r} to {to_state!r}; the "
                f"legal moves are {list(allowed)}"
            )
        previous = candidate.state
        candidate.state = to_state
        candidate.blocked_reason = reason if to_state in BLOCKING_STATES else ""
        candidate.next_action = next_action
        if certificates:
            candidate.certificates.update(certificates)
        if measurements:
            candidate.measurements.update(measurements)
        if implementation_sha256:
            candidate.implementation_sha256 = implementation_sha256
        if frozen_protocol_hash:
            candidate.frozen_protocol_hash = frozen_protocol_hash
        candidate.history.append(
            {"utc": now_utc(), "from": previous, "to": to_state, "reason": reason}
        )
        self._persist_candidate(candidate)
        self._audit(
            "transition",
            mechanism_id=mechanism_id,
            **{"from": previous, "to": to_state, "reason": reason},
        )
        return candidate

    # ---------------------------------------------------------------- the transactions
    def reserve_experiment(
        self,
        mechanism_id: str,
        kind: str,
        *,
        implementation_sha256: str = "",
        frozen_protocol_hash: str = "",
    ) -> str:
        """Reserve the one official experiment of this kind for this mechanism.

        The identity is the tuple the issue names, and a second reservation of the same tuple
        is refused: two official cold trainings of one scientific mechanism are not allowed,
        not even under a renamed proposal.
        """
        candidate = self._candidates.get(mechanism_id)
        if candidate is None:
            raise TransitionError(f"there is no candidate {mechanism_id!r}")
        implementation = implementation_sha256 or candidate.implementation_sha256
        protocol = frozen_protocol_hash or candidate.frozen_protocol_hash
        if not candidate.proposal_sha256 or not implementation or not protocol:
            raise TransitionError(
                f"{mechanism_id!r} cannot run an official experiment without its proposal "
                "hash, its implementation hash and its frozen protocol hash"
            )
        identity = experiment_id(
            proposal_sha256=candidate.proposal_sha256,
            implementation_sha256=implementation,
            frozen_protocol_hash=protocol,
            seed=candidate.seed,
            kind=kind,
        )
        if identity in self._experiments:
            raise TransitionError(
                f"the official {kind} experiment of {mechanism_id!r} is already reserved as "
                f"{identity[:16]}; a second official run of the same mechanism is not allowed"
            )
        self._experiments[identity] = mechanism_id
        self._persist_experiments()
        self._audit(
            "experiment_reserved",
            mechanism_id=mechanism_id,
            kind=kind,
            identity=identity,
        )
        return identity

    def reserved_experiments(self) -> Dict[str, str]:
        return dict(self._experiments)

    # ---------------------------------------------------------------- the report
    def summary(self) -> Dict[str, Any]:
        """The campaign's counts, with the kinds of progress kept apart."""
        states: Dict[str, int] = {}
        for candidate in self._candidates.values():
            states[candidate.state] = states.get(candidate.state, 0) + 1
        return {
            "candidates": len(self._candidates),
            "states": dict(sorted(states.items())),
            "official_experiments": len(self._experiments),
            "by_kind": {
                kind: sum(1 for value in self._experiments.values() if value)
                for kind in ("stage2", "stage3")
            },
            # a blocked candidate needs an operator action; a rejected one is a decision
            # that has been made, and the states count it separately
            "blocked": [
                {
                    "mechanism_id": candidate.mechanism_id,
                    "state": candidate.state,
                    "reason": candidate.blocked_reason,
                    "next_action": candidate.next_action,
                }
                for candidate in sorted(
                    self._candidates.values(), key=lambda row: row.mechanism_id
                )
                if candidate.state
                in ("blocked_literature", "blocked_resources", "reconciliation_needed")
            ],
            "rejected": [
                candidate.mechanism_id
                for candidate in sorted(
                    self._candidates.values(), key=lambda row: row.mechanism_id
                )
                if candidate.state == "rejected"
            ],
        }


__all__ = [
    "BLOCKING_STATES",
    "Campaign",
    "Candidate",
    "EXPERIMENT_IDENTITY",
    "INITIAL_STATE",
    "LEGAL_TRANSITIONS",
    "NAMESPACE",
    "STATES",
    "TransitionError",
    "experiment_id",
    "now_utc",
]
