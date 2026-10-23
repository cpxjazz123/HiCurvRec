"""The held-back research tasks and their executable graders (issue #105, workstream D).

The #104 policy comparison graded five answers by looking for key words in free text, and
the grader was the reason the comparison was uninformative: two correct answers were
marked wrong because they used different words. This module replaces it with tasks whose
graders are **programs**: every task carries an artifact (drawn from a real run of this
repository) and a grader that recomputes the truth from that artifact and compares it with
the answer's DECISION, not with its wording.

The tasks cover the six areas the issue names: gradient fidelity, inert export, collapse,
causal experimental design, artifact integrity, and the ER-versus-Stage-3 tradeoff. The
tasks and the graders are frozen -- hashed -- before any policy sees them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evidence import canonical_json, sha256_bytes, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Task:
    """One held-back task: a prompt, an artifact, and a program that grades the answer."""

    id: str
    prompt: str
    artifact: Dict[str, Any]
    grader: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]]
    expected: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "artifact": self.artifact,
            "expected": self.expected,
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()).encode("utf-8"))


def _answer(value: Any) -> Dict[str, Any]:
    """The agent's answer, parsed from whatever it replied with."""
    import re

    if isinstance(value, dict):
        return value
    text = str(value or "")
    block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    raw = block.group(1) if block else None
    if raw is None:
        start = text.find("{")
        end = text.rfind("}")
        raw = text[start : end + 1] if start >= 0 and end > start else ""
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return document if isinstance(document, dict) else {}


# --------------------------------------------------------------------------------------
# the graders
# --------------------------------------------------------------------------------------
def _grade_gradient_fidelity(answer: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """The category must be right AND the cited measurement must be the artifact's."""
    expected = artifact["truth"]["category"]
    chosen = str(answer.get("category", "")).strip().lower()
    if chosen != expected:
        return False, (
            f"the category is {chosen!r}, the measurement implies {expected!r}"
        )
    cited = answer.get("cited_forward_dependence")
    try:
        cited_value = float(cited)
    except (TypeError, ValueError):
        return False, "the answer cites no forward dependence"
    target = float(artifact["truth"]["forward_dependence"])
    if abs(cited_value - target) > max(1e-12, abs(target) * 0.5):
        return False, f"the cited forward dependence {cited_value} is not the measured {target}"
    return True, "the category and the cited measurement are right"


def _grade_verdict(answer: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """A closed-set verdict: the truth is recomputed from the artifact."""
    expected = artifact["truth"]["verdict"]
    chosen = str(answer.get("verdict", "")).strip().lower()
    if chosen != expected:
        return False, (
            f"the verdict is {chosen!r}, the artifact implies {expected!r}"
        )
    return True, "the verdict is right"


def _grade_gradient_route(answer: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[bool, str]:
    expected = artifact["truth"]["must_reach"]
    chosen = str(answer.get("must_reach", "")).strip().lower()
    if chosen != expected:
        return False, (
            f"the answer says {chosen!r}, the host's assembly requires {expected!r}"
        )
    return True, "the parameter is right"


def _grade_experiment_design(answer: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """A structural grader: a matched control, a falsifier per explanation, and no
    peeking at the held-out evaluation to decide the experiment."""
    problems: List[str] = []
    control = str(answer.get("control", "")).strip()
    if control != artifact["truth"]["control_run"]:
        problems.append(
            f"the control is {control!r}, the matched control is "
            f"{artifact['truth']['control_run']!r}"
        )
    falsifiers = answer.get("falsifiers")
    if not isinstance(falsifiers, dict):
        problems.append("the answer names no falsifier per explanation")
    else:
        for explanation in artifact["truth"]["explanations"]:
            text = str(falsifiers.get(explanation, "")).strip()
            if not text:
                problems.append(f"no falsifier for {explanation}")
    # The decision variable is checked as a FIELD, not by scanning the prose: a design may
    # legitimately mention the forbidden metric in order to rule it out, and a grader that
    # punishes the mention is the string-matching grader this task set replaced.
    decision = str(answer.get("decision_variable", "")).strip().lower()
    allowed = artifact["truth"]["allowed_decision_variables"]
    forbidden = artifact["truth"]["forbidden_decision_variables"]
    if not decision:
        problems.append("the answer names no decision variable")
    elif any(term in decision for term in forbidden):
        problems.append(
            f"the design decides on {decision!r}, which the protocol forbids as a "
            "decision variable"
        )
    elif not any(term in decision for term in allowed):
        problems.append(
            f"the decision variable {decision!r} is not one the protocol allows "
            f"({allowed})"
        )
    return (not problems), "; ".join(problems) or "the design is well formed"


def _grade_artifact_integrity(answer: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """The grader recomputes the hash: the agent's claim is compared with the truth."""
    intact = bool(answer.get("intact"))
    if intact != artifact["truth"]["intact"]:
        return False, (
            f"the answer says intact={intact}, the recomputed truth is "
            f"{artifact['truth']['intact']}"
        )
    named = str(answer.get("artifact", "")).strip()
    if named != artifact["truth"]["artifact"]:
        return False, (
            f"the answer names {named!r}, the changed artifact is "
            f"{artifact['truth']['artifact']!r}"
        )
    return True, "the integrity verdict and the artifact are right"


def _grade_downstream_decision(answer: Dict[str, Any], artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """The decision rule is fixed in advance and applied to the measured table."""
    expected = artifact["truth"]["decision"]
    chosen = str(answer.get("decision", "")).strip().lower()
    if chosen != expected:
        return False, (
            f"the decision is {chosen!r}, the predeclared rule gives {expected!r}"
        )
    return True, "the decision follows the predeclared rule"


# --------------------------------------------------------------------------------------
# the artifacts, drawn from real runs
# --------------------------------------------------------------------------------------
def _artifact_gradient_fidelity() -> Dict[str, Any]:
    return {
        "description": "a mechanism's declared gradient paths and what the checker measured",
        "declaration": {
            "source": "nonzero",
            "target": "nonzero",
            "codebook": "nonzero",
        },
        "measurements": {
            "codebook_forward_dependence": 0.0,
            "codebook_local_gradient": 0.0,
            "codebook_reaches_the_loss": False,
            "source_forward_dependence": 0.010050766684073231,
            "source_local_gradient": 1.3771830668415799,
        },
        "note": (
            "the checker measured these on a tiny synthetic case with the mechanism's own "
            "function; the implementation is faithful to its declaration"
        ),
        "truth": {
            "category": "impossible specification",
            "forward_dependence": 0.0,
        },
    }


def _artifact_inert_export() -> Dict[str, Any]:
    return {
        "description": "two arms' exported SIDs against a matched cold control",
        "arms": {
            "behavior_conditioned_pairwise_margin": {
                "changed_fraction_levels_1": 0.0,
                "changed_fraction_levels_2": 0.0,
                "changed_fraction_levels_3": 0.0,
                "export_sha256": "a5ffb7add8978106",
            },
            "curvature_weighted_transition_transport": {
                "changed_fraction_levels_1": 0.0,
                "changed_fraction_levels_2": 0.0,
                "changed_fraction_levels_3": 0.0,
                "export_sha256": "a5ffb7add8978106",
            },
        },
        "control_export_sha256": "a5ffb7add8978106",
        "training_losses_differ": True,
        "truth": {"verdict": "inert"},
    }


def _artifact_collapse() -> Dict[str, Any]:
    return {
        "description": "a probe's occupancy against the matched host's",
        "arm": "behavior_conditioned_pairwise_curvature",
        "arm_unique_l1": 199,
        "host_unique_l1": 245,
        "occupancy_ratio": 0.8122448979591838,
        "arm_unique_3token": 19475,
        "host_unique_3token": 23657,
        "hard_l1_er": 2.619839290662233,
        "truth": {"verdict": "collapsed"},
    }


def _artifact_gradient_route() -> Dict[str, Any]:
    return {
        "description": "the host's post-backward assembly, from its own source",
        "source_excerpt": (
            "def _r4_08_apply_projection(base_loss, l1_loss, other_loss, encoder_parameters, "
            "behavior_weight, valid_levels=3):\\n"
            "    # Overwrite the encoder gradients with the conflict-projected assembly.\\n"
            "    ...\\n"
            "    # Decoder and codebook gradients are never touched.\\n"
            "    for parameter, base_grad, l1_proj_grad, other_grad in zip(...):\\n"
            "        parameter.grad = base_grad + scale * (l1_proj_grad + other_grad)"
        ),
        "question": (
            "an auxiliary term's gradient reaches the encoder and the codebook. Which of the "
            "two survives the host's assembly into the optimizer's update?"
        ),
        "truth": {"must_reach": "codebook"},
    }


def _artifact_experiment_design() -> Dict[str, Any]:
    return {
        "description": "a hypothesis with two rival explanations, after a campaign",
        "hypothesis": (
            "an arm cleared the frozen Hard-L1 ER gate and its Stage-3 recall@10 did not "
            "improve"
        ),
        "explanations": {
            "E1": "the ER gain is carried by a small subset of items",
            "E2": "the code neighbourhood structure the recommender consumes got worse",
        },
        "available_artifacts": [
            "the exported SIDs of the arm and of a contemporaneous cold control",
            "the frozen held-out evaluation set",
            "the Stage-3 metrics of both",
        ],
        "truth": {
            "control_run": "control-r8_control-20261021T082024",
            "explanations": ["E1", "E2"],
            "forbidden_decision_variables": ["recall@10", "ndcg@10", "test_recall", "test_ndcg"],
            "allowed_decision_variables": [
                "hard_l1_er",
                "per-item",
                "export_sensitivity",
                "train_side",
                "occupancy",
                "collision",
            ],
        },
    }


def _artifact_integrity(store_root: Path) -> Dict[str, Any]:
    runs = sorted((store_root / "runs").glob("*/manifest.json"))
    if not runs:
        raise FileNotFoundError(f"the store at {store_root} has no manifests")
    manifest_path = runs[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") or []
    target = None
    for artifact in artifacts:
        path = Path(artifact["path"])
        if path.is_file():
            target = artifact
            break
    if target is None:
        raise FileNotFoundError("the manifest names no artifact that exists")
    live = sha256_file(Path(target["path"]))
    return {
        "description": "a run manifest and the artifact it names",
        "manifest": {
            "run_id": manifest["run_id"],
            "manifest_sha256": manifest.get("manifest_sha256"),
            "status": manifest.get("status"),
            "artifacts": manifest.get("artifacts"),
        },
        "artifact_path": target["path"],
        "recorded_sha256": target.get("sha256"),
        "question": (
            "is the evidence this manifest names still exactly what the manifest says it is?"
        ),
        "truth": {
            "intact": live == target.get("sha256"),
            "artifact": Path(target["path"]).name,
        },
    }


def _artifact_downstream_decision() -> Dict[str, Any]:
    return {
        "description": "one arm's measured results against a contemporaneous cold control",
        "gate": 16.38953942232631,
        "control": {
            "hard_l1_er": 16.334505666275888,
            "stage3_recall@10": 0.06406796775709883,
        },
        "arm": {
            "hard_l1_er": 16.616,
            "stage3_recall@10": 0.06338898657706436,
        },
        "stage3_run_to_run_band": 0.00033076775709883,
        "question": (
            "the arm cleared the gate and its ER is the best measured so far. What is the "
            "honest verdict on the research outcome?"
        ),
        "truth": {"decision": "exploratory"},
    }


def _artifact_probe_blind_spot() -> Dict[str, Any]:
    return {
        "description": "a probe's verdict and what the production run actually did",
        "probe": {
            "ok": True,
            "epochs": 50,
            "loss_finite": True,
            "encoder_update": 0.109,
            "codebook_update": 0.129,
            "hard_sid_changed": 24587,
            "detail": "the real entry ran, the declared parameters moved, the exported hard "
                      "SIDs changed and the code did not collapse",
        },
        "production_run": {
            "epochs_completed": 0,
            "error": "UnboundLocalError: local variable 'source_ids' referenced before assignment",
            "cause": "the mechanism block reads the host's pair indices outside the "
                     "behaviour-ramp guard, so they do not exist yet at epoch 0",
        },
        "question": "is this arm ready for a cold run?",
        "truth": {"verdict": "not ready"},
    }


# --------------------------------------------------------------------------------------
# the task list
# --------------------------------------------------------------------------------------
def build_tasks(store_root: Path) -> List[Task]:
    """The held-back set, with the artifacts drawn from this repository's own runs."""
    return [
        Task(
            id="gradient_fidelity",
            prompt=(
                "A mechanism declares that its loss reaches the source, the target and the "
                "codebook with a nonzero gradient. The independent checker measured the "
                "mechanism on a tiny synthetic case and reports the numbers in the artifact.\n\n"
                "Classify the situation in ONE of exactly three words: `impossible "
                "specification` (the loss cannot depend on that target under this formula, "
                "so no implementation of it can deliver the gradient), `missing gradient` "
                "(the loss depends on the target but the backward does not reach it), or "
                "`saturation` (the backward reaches it and the local derivative is zero at "
                "this point). Then cite the forward dependence you relied on.\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"category\": \"...\", \"cited_forward_dependence\": 0.0}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_gradient_fidelity(),
            grader=_grade_gradient_fidelity,
            expected="impossible specification",
        ),
        Task(
            id="inert_export",
            prompt=(
                "Two arms were trained with auxiliary terms and their exported SIDs are "
                "compared with a contemporaneous cold control's export. Their training "
                "losses differ from the control's.\n\n"
                "Decide, in ONE word, what happened to each arm's EXPORT: `inert` (the "
                "exported codes did not change at all), `active` (they changed and the code "
                "structure survived), or `collapsed` (the code structure was destroyed).\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"verdict\": \"inert\"}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_inert_export(),
            grader=_grade_verdict,
            expected="inert",
        ),
        Task(
            id="collapse_detection",
            prompt=(
                "An arm's probe measured its level-0 occupancy and its distinct 3-token SIDs "
                "against the matched host's, and its frozen Hard-L1 ER is in the artifact.\n\n"
                "Decide, in ONE word, whether the arm's code structure is `collapsed` or "
                "`healthy`.\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"verdict\": \"collapsed\"}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_collapse(),
            grader=_grade_verdict,
            expected="collapsed",
        ),
        Task(
            id="host_gradient_route",
            prompt=(
                "The artifact is the host's own post-backward assembly. An auxiliary "
                "mechanism's loss produces a gradient on the encoder and on the codebook.\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"must_reach\": \"codebook\"}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_gradient_route(),
            grader=_grade_gradient_route,
            expected="codebook",
        ),
        Task(
            id="causal_experiment_design",
            prompt=(
                "You must design ONE discriminating experiment for the hypothesis in the "
                "artifact. It has two rival explanations, and the experiment must be able to "
                "tell them apart.\n\n"
                "Your answer must name the matched control to compare against (by its run id), "
                "and a falsifier for EACH explanation (a measurement that would refute it). "
                "The protocol forbids deciding the experiment on the held-out evaluation "
                "metric, so do not use it.\n\n"
                "Name the metric the experiment DECIDES on in `decision_variable`; it may be "
                "the frozen Hard-L1 ER, a per-item decomposition, the export-sensitivity "
                "verdict, the train-side discrimination or the occupancy/collision "
                "statistics, but it may not be a held-out recommendation metric.\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"experiment\": \"...\", \"control\": \"<run id>\", "
                "\"decision_variable\": \"<the metric it decides on>\", "
                "\"falsifiers\": {\"E1\": \"...\", \"E2\": \"...\"}}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_experiment_design(),
            grader=_grade_experiment_design,
            expected="a matched control and a falsifier per explanation",
        ),
        Task(
            id="artifact_integrity",
            prompt=(
                "The artifact is a run manifest and the artifact it names. The question is "
                "whether the evidence is still exactly what the manifest says it is.\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"intact\": true, \"artifact\": \"<file name>\"}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_integrity(REPO_ROOT / "results" / "harness3_runs" / "store"),
            grader=_grade_artifact_integrity,
            expected="the recomputed hash decides",
        ),
        Task(
            id="er_vs_stage3_decision",
            prompt=(
                "The artifact carries one arm's frozen Hard-L1 ER and Stage-3 recall@10 "
                "against a contemporaneous cold control, the frozen gate, and the Stage-3 "
                "run-to-run band measured from two control runs.\n\n"
                "Decide, in ONE word, what the honest verdict on this research outcome is: "
                "`improvement` (a verified downstream gain), `reject` (a verified "
                "downstream regression), or `exploratory` (the difference is inside the "
                "run-to-run band, so nothing is established either way).\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"decision\": \"exploratory\"}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_downstream_decision(),
            grader=_grade_downstream_decision,
            expected="exploratory",
        ),
        Task(
            id="probe_blind_spot",
            prompt=(
                "The artifact carries a probe's verdict and what the production run actually "
                "did.\n\n"
                "Decide, in ONE word, whether the arm is `ready` for a cold run or `not "
                "ready`.\n\n"
                "Answer with ONE fenced ```json block:\n"
                "```json\n{\"verdict\": \"not ready\"}\n```\n\n"
                "ARTIFACT:\n"
            ),
            artifact=_artifact_probe_blind_spot(),
            grader=_grade_verdict,
            expected="not ready",
        ),
    ]


def freeze_tasks(tasks: List[Task], path: Path) -> str:
    """Write the tasks and their graders' digests, and return the digest of the whole set."""
    document = {
        "task_set_version": 1,
        "tasks": [task.as_dict() for task in tasks],
        "digests": {task.id: task.digest() for task in tasks},
    }
    document["task_set_sha256"] = sha256_bytes(canonical_json(document).encode("utf-8"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")
    return document["task_set_sha256"]


__all__ = ["Task", "build_tasks", "freeze_tasks"]
