"""The planner of the hyperbolic-only campaign (issue #110, section 4 and acceptance 11).

The planner is the loop that decides what happens next. It is deliberately small: the
campaign's state machine owns the candidates and the audit log, the agents own the work, and
the planner owns the decision.

The rule that matters is the one about not spinning. When there is no novel candidate and no
new evidence, the planner does NOT create a round, does NOT dispatch training, does NOT
advance the round counter, and preserves its status: it says why it is blocked and waits. A
planner that manufactures a round per tick is how the old loop burned the machine, and the
acceptance test for that is seventy-two ticks with no new evidence, zero rounds and zero
training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .hyperbolic_campaign import Campaign, Candidate, NAMESPACE

#: The states of the planner itself.
IDLE = "idle"
WORKING = "working"
BLOCKED = "blocked"

#: The reason the planner is blocked when there is nothing to do.
NO_WORK = "no novel candidate and no new evidence"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlannerStatus:
    """What one tick did."""

    tick: int
    state: str
    action: str = ""
    reason: str = ""
    round: int = 0
    round_created: bool = False
    training_dispatched: bool = False
    candidate_id: str = ""
    next_action: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "state": self.state,
            "action": self.action,
            "reason": self.reason,
            "round": self.round,
            "round_created": self.round_created,
            "training_dispatched": self.training_dispatched,
            "candidate_id": self.candidate_id,
            "next_action": self.next_action,
        }


@dataclass
class PlannerState:
    """The planner's own durable state: the tick counter, the round counter and the note."""

    ticks: int = 0
    round: int = 0
    blocked_reason: str = ""
    blocked_since: str = ""
    last_action: str = ""
    rounds_created: int = 0
    trainings_dispatched: int = 0
    status: str = IDLE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticks": self.ticks,
            "round": self.round,
            "blocked_reason": self.blocked_reason,
            "blocked_since": self.blocked_since,
            "last_action": self.last_action,
            "rounds_created": self.rounds_created,
            "trainings_dispatched": self.trainings_dispatched,
            "status": self.status,
        }


class Planner:
    """The loop that decides what happens next, and that refuses to spin."""

    def __init__(
        self,
        campaign: Optional[Campaign] = None,
        *,
        namespace: Path = NAMESPACE,
        proposals: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        measurements: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        advance: Optional[Callable[[Candidate, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.campaign = campaign or Campaign(namespace=namespace)
        self.proposals = proposals or (lambda: [])
        self.measurements = measurements or (lambda: [])
        self.advance = advance or (lambda candidate, action: {"action": "noop"})
        self.state_path = Path(namespace) / "planner" / "state.json"
        self.state = PlannerState()
        self._load()

    # ------------------------------------------------------------------ the durable state

    def _load(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            document = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, value in document.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.state_path)

    # ------------------------------------------------------------------------- the round

    def _start_round(self, action: str) -> int:
        """Start a round, and only when there is real work to do."""
        self.state.round += 1
        self.state.rounds_created += 1
        round_dir = Path(self.campaign.namespace) / "rounds" / f"round_{self.state.round:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "round.json").write_text(
            json.dumps(
                {
                    "round": self.state.round,
                    "started_utc": now_utc(),
                    "first_action": action,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.state.round

    # ---------------------------------------------------------------------------- the tick

    def tick(self) -> PlannerStatus:
        """One decision. It never creates a round without work, and never spins."""
        self.state.ticks += 1
        tick = self.state.ticks

        # 1. the new evidence: the results that arrived since the last tick
        arrived = list(self.measurements() or ())
        if arrived:
            candidate = self.campaign.candidates()[0] if self.campaign.candidates() else None
            if candidate is not None:
                result = self.advance(candidate, {"action": "record_measurement", "records": arrived})
                self.state.last_action = "recorded a new measurement"
                self.state.status = WORKING
                self.save()
                return PlannerStatus(
                    tick=tick,
                    state=WORKING,
                    action="record_measurement",
                    round=self.state.round,
                    round_created=False,
                    candidate_id=candidate.mechanism_id,
                    next_action=str(result.get("next_action", "")),
                )

        # 2. the candidates that are ready to move on
        ready = [
            candidate
            for candidate in self.campaign.candidates()
            if not candidate.is_finished and not candidate.is_blocked and candidate.next_action
        ]
        if ready:
            candidate = ready[0]
            created = self._start_round(f"advance {candidate.mechanism_id}")
            result = self.advance(candidate, {"action": candidate.next_action})
            training = str(result.get("kind", "")) in ("stage2", "stage3")
            if training:
                self.state.trainings_dispatched += 1
            self.state.last_action = f"advanced {candidate.mechanism_id}"
            self.state.status = WORKING
            self.state.blocked_reason = ""
            self.save()
            return PlannerStatus(
                tick=tick,
                state=WORKING,
                action=candidate.next_action,
                round=created,
                round_created=True,
                training_dispatched=training,
                candidate_id=candidate.mechanism_id,
                next_action=str(result.get("next_action", "")),
            )

        # 3. the novel candidates
        fresh = list(self.proposals() or ())
        if fresh:
            created = self._start_round("admit a novel candidate")
            first = fresh[0]
            candidate = self.campaign.add(
                Candidate(
                    mechanism_id=first["mechanism_id"],
                    proposal_sha256=first.get("proposal_sha256", ""),
                    family=first.get("family", ""),
                )
            )
            self.state.last_action = f"admitted {candidate.mechanism_id}"
            self.state.status = WORKING
            self.state.blocked_reason = ""
            self.save()
            return PlannerStatus(
                tick=tick,
                state=WORKING,
                action="admit",
                round=created,
                round_created=True,
                candidate_id=candidate.mechanism_id,
                next_action="review the proposal",
            )

        # 4. nothing to do: say so, and wait. No round, no training, no busy loop.
        if not self.state.blocked_reason:
            self.state.blocked_since = now_utc()
        self.state.blocked_reason = NO_WORK
        self.state.last_action = "no work"
        self.state.status = BLOCKED
        self.save()
        return PlannerStatus(
            tick=tick,
            state=BLOCKED,
            action="wait",
            reason=NO_WORK,
            round=self.state.round,
            round_created=False,
            training_dispatched=False,
            next_action=(
                "wait for a novel candidate or a new measurement; the status is preserved and "
                "no round is created"
            ),
        )

    def run(self, ticks: int) -> List[PlannerStatus]:
        """Run the loop for a bounded number of ticks."""
        statuses: List[PlannerStatus] = []
        for _ in range(int(ticks)):
            statuses.append(self.tick())
        return statuses


__all__ = [
    "BLOCKED",
    "IDLE",
    "NO_WORK",
    "Planner",
    "PlannerState",
    "PlannerStatus",
    "WORKING",
]
