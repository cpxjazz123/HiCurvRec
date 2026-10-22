"""The guardrails (issue #106).

The issue names four executions that are prohibited, and says the restrictions override any
older issue, CLI default, prompt or agent instruction:

1. **no multi-seed experiments** -- one frozen seed, one official cold run per candidate;
2. **no re-running the old baseline policy** -- historical policy records are read-only;
3. **no control re-training** -- the archived control evidence is the comparison, and a lost
   historical artifact is never recreated and called the original;
4. **no silent rerun of an arm that failed**, no hidden extra training, and no repeated use
   of the frozen final evaluation as a search oracle.

A prohibition that lives only in a prompt is not a prohibition. This module makes each one
a refusal with a reason, and the refusals happen BEFORE any process is launched or any
artifact is written. The store is what makes them checkable: a run that already exists, an
arm that already failed, an evaluation that was already consumed, are all facts the store
knows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .evidence import EvidenceStore, now_utc

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The project's single frozen seed. A candidate that asks for anything else is refused.
FROZEN_SEED = 42

#: The archived control. It is evidence, not a thing to run.
CONTROL_MECHANISMS = ("r8_control", "r8_control_seed43", "r8_control_restore")

#: The policies that have already been executed. Their records are read-only.
ARCHIVED_POLICIES = ("old", "learned", "harness2_baseline", "harness3_baseline")


class GuardrailRefusal(RuntimeError):
    """A prohibited execution was refused. The reason says which rule and why."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"[{rule}] {detail}")
        self.rule = rule
        self.detail = detail


@dataclass
class ExecutionIntent:
    """What a caller intends to run. The guardrails read this and nothing else."""

    kind: str = "stage2"                       # stage2 | stage3 | probe | experiment | policy
    mechanism_id: str = ""
    seed: int = FROZEN_SEED
    policy: str = "current"
    tree: Optional[Path] = None
    rerun_of: Optional[str] = None             # the arm this run repeats
    consumes_frozen_evaluation: bool = False
    semantic_key: str = ""                     # the normalised specification digest
    sid_sha256: str = ""                       # the export this run would produce
    fingerprint: str = ""                      # the mechanism's mathematics digest
    sid_effect_fingerprint: str = ""           # the mechanism's declared effect on the codes
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "mechanism_id": self.mechanism_id,
            "seed": self.seed,
            "policy": self.policy,
            "tree": str(self.tree) if self.tree else None,
            "rerun_of": self.rerun_of,
            "consumes_frozen_evaluation": self.consumes_frozen_evaluation,
            "semantic_key": self.semantic_key,
            "sid_sha256": self.sid_sha256,
            "fingerprint": self.fingerprint,
            "sid_effect_fingerprint": self.sid_effect_fingerprint,
            "note": self.note,
        }


@dataclass
class GuardrailReport:
    """What the guardrails decided, for the manifest."""

    allowed: bool
    rule: str = ""
    detail: str = ""
    checked: List[str] = field(default_factory=list)
    refused_utc: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "rule": self.rule,
            "detail": self.detail,
            "checked": list(self.checked),
            "refused_utc": self.refused_utc,
        }


def check_intent(
    intent: ExecutionIntent,
    store: Optional[EvidenceStore] = None,
) -> GuardrailReport:
    """Decide whether an execution is allowed. It is refused BEFORE anything starts."""
    checked: List[str] = []

    # 1. the seed is the project's frozen one
    checked.append("single_seed")
    if intent.seed != FROZEN_SEED:
        return GuardrailReport(
            False,
            "single_seed",
            f"the project's frozen seed is {FROZEN_SEED} and this run asks for {intent.seed}; "
            "multi-seed experiments are prohibited by issue #106 and by issue #109, which "
            "requires one stochastic realisation per genuinely new mechanism",
            checked,
            now_utc(),
        )

    # 2. the control is evidence, never a thing to train
    checked.append("no_control_retrain")
    if intent.mechanism_id in CONTROL_MECHANISMS:
        return GuardrailReport(
            False,
            "no_control_retrain",
            f"{intent.mechanism_id!r} is the archived control: it is evidence, and issue #106 "
            "prohibits re-training it; compare against the archived control artifacts",
            checked,
            now_utc(),
        )

    # 3. an archived policy is read-only
    checked.append("no_archived_policy_execution")
    if intent.policy in ARCHIVED_POLICIES:
        return GuardrailReport(
            False,
            "no_archived_policy_execution",
            f"the {intent.policy!r} policy has already been executed; issue #106 prohibits "
            "running it again, and its records are read-only",
            checked,
            now_utc(),
        )

    # 4. no silent rerun of an arm, and no hidden extra training
    checked.append("no_failed_arm_rerun")
    if store is not None:
        failed = [
            manifest
            for manifest in store.runs()
            if manifest.get("mechanism_id") == intent.mechanism_id
            and manifest.get("status") in ("failed", "refused")
        ]
        if failed and intent.kind in ("stage2", "stage3"):
            return GuardrailReport(
                False,
                "no_failed_arm_rerun",
                f"{intent.mechanism_id!r} already has a {failed[0].get('status')} run in the "
                f"store ({failed[0].get('run_id')}); issue #106 forbids rerunning an arm after "
                "a failure without a new, separately justified candidate",
                checked,
                now_utc(),
            )
        # 5. the frozen final evaluation is a measurement, not a search oracle
        checked.append("no_repeated_frozen_evaluation")
        if intent.consumes_frozen_evaluation and store is not None:
            consumed = [
                manifest
                for manifest in store.runs()
                if manifest.get("mechanism_id") == intent.mechanism_id
                and ((manifest.get("metrics") or {}).get("frozen_evaluation") or {})
            ]
            if consumed:
                return GuardrailReport(
                    False,
                    "no_repeated_frozen_evaluation",
                    f"the frozen held-out evaluation was already consumed for "
                    f"{intent.mechanism_id!r} in {consumed[0].get('run_id')}; it is a "
                    "measurement, not a search oracle, and it is read once per candidate",
                    checked,
                    now_utc(),
                )

    # 6. Stage-3 is read once per arm, and only for an arm that cleared the ER gate
    checked.append("no_repeated_stage3")
    if intent.kind == "stage3" and store is not None:
        evaluated = [
            manifest
            for manifest in store.runs()
            if manifest.get("mechanism_id") == intent.mechanism_id
            and (manifest.get("metrics") or {}).get("stage3")
        ]
        if evaluated:
            return GuardrailReport(
                False,
                "no_repeated_stage3",
                f"Stage-3 was already measured for {intent.mechanism_id!r} in "
                f"{evaluated[0].get('run_id')}; it is read once per arm",
                checked,
                now_utc(),
            )

    # 7. a renamed arm is the same arm: the semantic key is the specification's digest
    checked.append("no_semantic_duplicate")
    if store is not None and intent.semantic_key:
        duplicates = [
            manifest
            for manifest in store.runs()
            if (manifest.get("extra") or {}).get("semantic_key") == intent.semantic_key
            and manifest.get("mechanism_id") != intent.mechanism_id
        ]
        if duplicates:
            return GuardrailReport(
                False,
                "no_semantic_duplicate",
                f"the specification digest {intent.semantic_key[:16]} is already the "
                f"mechanism of {duplicates[0].get('mechanism_id')!r}; a renamed copy of an "
                "unchanged mechanism is the same mechanism",
                checked,
                now_utc(),
            )

    # 8. exactly one official cold Stage-2 run per novel arm
    checked.append("no_repeated_cold_run")
    if intent.kind == "stage2" and store is not None:
        trained = [
            manifest
            for manifest in store.runs()
            if manifest.get("mechanism_id") == intent.mechanism_id
            and manifest.get("kind") == "stage2"
            and manifest.get("status") == "succeeded"
        ]
        if trained:
            return GuardrailReport(
                False,
                "no_repeated_cold_run",
                f"{intent.mechanism_id!r} already has an official cold Stage-2 run "
                f"({trained[0].get('run_id')}); issue #109 allows exactly one fixed-seed "
                "3000-epoch cold run per novel arm, and a further run is a new experiment "
                "that needs a new mechanism",
                checked,
                now_utc(),
            )

    # 9. exactly one probe per arm, and none under a renamed arm
    checked.append("no_repeated_probe")
    if intent.kind == "probe" and store is not None:
        probed = [
            manifest
            for manifest in store.runs()
            if manifest.get("kind") == "probe"
            and (
                manifest.get("mechanism_id") == intent.mechanism_id
                or (
                    intent.sid_sha256
                    and (manifest.get("extra") or {}).get("sid_sha256") == intent.sid_sha256
                )
            )
        ]
        if probed:
            return GuardrailReport(
                False,
                "no_repeated_probe",
                f"a probe was already run for {probed[0].get('mechanism_id')!r} "
                f"({probed[0].get('run_id')}); issue #109 allows one lightweight probe per "
                "genuinely new arm, and a probe under a renamed arm is the same probe",
                checked,
                now_utc(),
            )

    # 10. the mechanism must be new with respect to the whole historical record
    checked.append("no_historical_repetition")
    if intent.kind in ("stage2", "probe") and intent.fingerprint:
        verdict = _novelty_verdict(intent)
        if verdict is not None and not verdict.novel:
            return GuardrailReport(
                False,
                "no_historical_repetition",
                f"the mechanism repeats one that already exists: {verdict.detail} "
                f"(matched {verdict.matched_component} {verdict.matched_id!r}); issue #109 "
                "requires a mathematically substantive, independently auditable change",
                checked,
                now_utc(),
            )

    return GuardrailReport(True, "", "", checked, "")


def _novelty_verdict(intent: ExecutionIntent):
    """The novelty verdict against the historical fingerprint registry, if it is available.

    The registry is built from the archives, so it may legitimately be absent on a machine
    that has not built it yet; in that case the check is skipped and the controller's own
    startup builds it, so the rule is enforced for every run that is actually dispatched.
    """
    try:
        from . import historical_fingerprints

        registry = historical_fingerprints.load_registry(historical_fingerprints.DEFAULT_REGISTRY)
    except Exception:  # noqa: BLE001
        return None
    for entry in registry.get("entries") or ():
        if entry.get("primary") and entry.get("primary") == intent.fingerprint:
            return historical_fingerprints.NoveltyResult(
                False,
                "primary",
                entry.get("id", ""),
                "the mechanism's mathematics matches a registered one",
                intent.fingerprint,
                intent.sid_effect_fingerprint,
            )
    return historical_fingerprints.NoveltyResult(
        True,
        primary=intent.fingerprint,
        sid_effect=intent.sid_effect_fingerprint,
    )


def refuse_if_prohibited(
    intent: ExecutionIntent,
    store: Optional[EvidenceStore] = None,
) -> GuardrailReport:
    """The enforcing wrapper: it raises when the intent is prohibited."""
    report = check_intent(intent, store)
    if not report.allowed:
        raise GuardrailRefusal(report.rule, report.detail)
    return report


__all__ = [
    "ARCHIVED_POLICIES",
    "CONTROL_MECHANISMS",
    "ExecutionIntent",
    "FROZEN_SEED",
    "GuardrailRefusal",
    "GuardrailReport",
    "check_intent",
    "refuse_if_prohibited",
]
