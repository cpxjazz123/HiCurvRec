"""The hypothesis ledger (issue #105, workstream B).

A ledger entry is a chain of reasoning that a reader can check:

    observation -> competing explanations -> falsifiable predictions ->
    discriminating experiment -> results -> supported / refuted / inconclusive ->
    the next experiment

Two rules make it more than a diary. First, every claim names the **artifacts** it rests
on, and those artifacts live in the immutable evidence store, so a claim can be checked
against the run that produced it rather than against the summary of it. Second, a
hypothesis may not be "supported" by results that its own experiments produced: the store
knows which run produced what, and the ledger refuses that circularity.

The ledger is also the research system's only input. The agent that proposes the next
mechanism is briefed from the ledger, which is built from the store, which is built from
runs. Nothing in the prompt is a hardcoded prior outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .evidence import EvidenceStore, canonical_json, now_utc, sha256_bytes

#: The status vocabulary a hypothesis may end in.
HYPOTHESIS_STATUS = ("supported", "refuted", "inconclusive", "open")

#: The status vocabulary an experiment may end in.
EXPERIMENT_STATUS = ("planned", "executed", "failed", "abandoned")


@dataclass
class Prediction:
    """A falsifiable statement, with the measurement that would refute it."""

    id: str
    statement: str
    falsifier: str
    measurement: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "falsifier": self.falsifier,
            "measurement": self.measurement,
        }


@dataclass
class Explanation:
    """One rival account of an observation."""

    id: str
    statement: str
    predictions: List[Prediction] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "predictions": [prediction.as_dict() for prediction in self.predictions],
        }


@dataclass
class Experiment:
    """A discriminating experiment, its control, and what came out."""

    id: str
    tests: List[str] = field(default_factory=list)
    control_run: Optional[str] = None
    treatment_runs: List[str] = field(default_factory=list)
    status: str = "planned"
    result: str = ""
    verdict: str = ""
    produced_runs: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tests": list(self.tests),
            "control_run": self.control_run,
            "treatment_runs": list(self.treatment_runs),
            "status": self.status,
            "result": self.result,
            "verdict": self.verdict,
            "produced_runs": list(self.produced_runs),
        }


@dataclass
class Hypothesis:
    """One observation, its rival explanations, and the experiments that decided them."""

    id: str
    observation: str
    evidence_runs: List[str] = field(default_factory=list)
    evidence_paths: List[str] = field(default_factory=list)
    explanations: List[Explanation] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    status: str = "open"
    next_experiment: str = ""
    created_utc: str = field(default_factory=now_utc)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "observation": self.observation,
            "evidence_runs": list(self.evidence_runs),
            "evidence_paths": list(self.evidence_paths),
            "explanations": [explanation.as_dict() for explanation in self.explanations],
            "experiments": [experiment.as_dict() for experiment in self.experiments],
            "status": self.status,
            "next_experiment": self.next_experiment,
            "created_utc": self.created_utc,
            "notes": self.notes,
        }

    def experiment(self, experiment_id: str) -> Optional[Experiment]:
        for experiment in self.experiments:
            if experiment.id == experiment_id:
                return experiment
        return None


class Ledger:
    """The ledger itself: entries, their links, and the checks that keep it honest."""

    def __init__(self, path: Path, store: Optional[EvidenceStore] = None) -> None:
        self.path = Path(path)
        self.store = store
        self.hypotheses: List[Hypothesis] = []

    # -- persistence ------------------------------------------------------------------
    def load(self) -> "Ledger":
        if not self.path.is_file():
            raise FileNotFoundError(f"no ledger at {self.path}")
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.hypotheses = []
        for entry in document.get("hypotheses") or ():
            hypothesis = Hypothesis(
                id=entry["id"],
                observation=entry["observation"],
                evidence_runs=list(entry.get("evidence_runs") or ()),
                evidence_paths=list(entry.get("evidence_paths") or ()),
                explanations=[
                    Explanation(
                        id=item["id"],
                        statement=item["statement"],
                        predictions=[
                            Prediction(
                                id=prediction["id"],
                                statement=prediction["statement"],
                                falsifier=prediction["falsifier"],
                                measurement=prediction.get("measurement", ""),
                            )
                            for prediction in item.get("predictions") or ()
                        ],
                    )
                    for item in entry.get("explanations") or ()
                ],
                experiments=[
                    Experiment(
                        id=item["id"],
                        tests=list(item.get("tests") or ()),
                        control_run=item.get("control_run"),
                        treatment_runs=list(item.get("treatment_runs") or ()),
                        status=item.get("status", "planned"),
                        result=item.get("result", ""),
                        verdict=item.get("verdict", ""),
                        produced_runs=list(item.get("produced_runs") or ()),
                    )
                    for item in entry.get("experiments") or ()
                ],
                status=entry.get("status", "open"),
                next_experiment=entry.get("next_experiment", ""),
                created_utc=entry.get("created_utc", now_utc()),
                notes=entry.get("notes", ""),
            )
            self.hypotheses.append(hypothesis)
        return self

    def save(self) -> str:
        document = {
            "ledger_version": 1,
            "generated_utc": now_utc(),
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
        }
        document["ledger_sha256"] = sha256_bytes(canonical_json(document).encode("utf-8"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return document["ledger_sha256"]

    # -- content ----------------------------------------------------------------------
    def add(self, hypothesis: Hypothesis) -> Hypothesis:
        self.hypotheses.append(hypothesis)
        return hypothesis

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        for hypothesis in self.hypotheses:
            if hypothesis.id == hypothesis_id:
                return hypothesis
        return None

    # -- the checks -------------------------------------------------------------------
    def problems(self) -> List[str]:
        """Everything wrong with the ledger, in the order it should be fixed."""
        problems: List[str] = []
        seen_ids = set()
        for hypothesis in self.hypotheses:
            if hypothesis.id in seen_ids:
                problems.append(f"the hypothesis id {hypothesis.id!r} is used twice")
            seen_ids.add(hypothesis.id)
            if hypothesis.status not in HYPOTHESIS_STATUS:
                problems.append(
                    f"{hypothesis.id}: status {hypothesis.status!r} is not one of "
                    f"{list(HYPOTHESIS_STATUS)}"
                )
            if not hypothesis.explanations:
                problems.append(f"{hypothesis.id}: no competing explanations")
            for explanation in hypothesis.explanations:
                if not explanation.predictions:
                    problems.append(
                        f"{hypothesis.id}/{explanation.id}: no falsifiable prediction"
                    )
                for prediction in explanation.predictions:
                    if not prediction.falsifier:
                        problems.append(
                            f"{hypothesis.id}/{explanation.id}/{prediction.id}: no falsifier"
                        )
            for experiment in hypothesis.experiments:
                if experiment.status not in EXPERIMENT_STATUS:
                    problems.append(
                        f"{hypothesis.id}/{experiment.id}: status {experiment.status!r} is not "
                        f"one of {list(EXPERIMENT_STATUS)}"
                    )
                # The circularity guard: a hypothesis may not be decided by the results of
                # the very experiments it asked for.
                produced = set(experiment.produced_runs)
                used = set(hypothesis.evidence_runs)
                if produced & used:
                    problems.append(
                        f"{hypothesis.id}: the evidence {sorted(produced & used)} was produced "
                        f"by its own experiment {experiment.id}; that is circular"
                    )
            if self.store is not None:
                for run_id in hypothesis.evidence_runs:
                    try:
                        self.store.load(run_id)
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{hypothesis.id}: the evidence run {run_id!r} is not in "
                                        f"the store: {exc}")
        return problems

    def digest(self) -> str:
        return sha256_bytes(
            canonical_json([hypothesis.as_dict() for hypothesis in self.hypotheses]).encode(
                "utf-8"
            )
        )

    # -- the brief --------------------------------------------------------------------
    def brief(self, *, max_chars: int = 24000) -> str:
        """The research system's input, rendered for the agent.

        This replaces the hardcoded outcome summary that used to live in the code: every
        number here comes from a run in the store, and every run names its artifacts.
        """
        lines: List[str] = []
        for hypothesis in self.hypotheses:
            lines.append(f"## {hypothesis.id}: {hypothesis.observation}")
            if hypothesis.evidence_runs:
                lines.append(f"evidence runs: {', '.join(hypothesis.evidence_runs)}")
            for explanation in hypothesis.explanations:
                lines.append(f"* explanation {explanation.id}: {explanation.statement}")
                for prediction in explanation.predictions:
                    lines.append(
                        f"  * prediction {prediction.id}: {prediction.statement} "
                        f"[falsified by: {prediction.falsifier}]"
                    )
            for experiment in hypothesis.experiments:
                lines.append(
                    f"* experiment {experiment.id} ({experiment.status}) tests "
                    f"{', '.join(experiment.tests)}: {experiment.result or 'not executed'} "
                    f"-> {experiment.verdict or 'no verdict'}"
                )
            lines.append(f"status: {hypothesis.status}")
            if hypothesis.next_experiment:
                lines.append(f"next experiment: {hypothesis.next_experiment}")
            lines.append("")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[truncated]\n"
        return text

    def measurements(self) -> List[Dict[str, Any]]:
        """The ledger's numbers, for the tables in the report."""
        rows: List[Dict[str, Any]] = []
        for hypothesis in self.hypotheses:
            for experiment in hypothesis.experiments:
                rows.append(
                    {
                        "hypothesis": hypothesis.id,
                        "experiment": experiment.id,
                        "status": experiment.status,
                        "verdict": experiment.verdict,
                        "result": experiment.result,
                        "control_run": experiment.control_run,
                        "treatment_runs": experiment.treatment_runs,
                    }
                )
        return rows


__all__ = [
    "EXPERIMENT_STATUS",
    "Experiment",
    "Explanation",
    "HYPOTHESIS_STATUS",
    "Hypothesis",
    "Ledger",
    "Prediction",
]
