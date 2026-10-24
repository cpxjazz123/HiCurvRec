"""One canonical Issue117 DAG; each research run owns a separate evidence directory."""
import json
import os
import shutil
from pathlib import Path

from .. import settings
from ..agents.prompts import build_prompt
from ..agents.runner import run_agent
from ..core.dag import StateError, TaskStore
from ..core.evidence import atomic_json, digest_bytes, digest_file, read_json
from ..integrations.preflight import audit_surviving_mainline
from ..science.behavior import run as analyze_train_behavior
from ..science.contracts import check_candidate, check_review_identity

TASKS = (
    ("behavior", "DATA", ()),
    ("literature", "LITERATURE", ()),
    ("geometry", "GEOMETRY", ("behavior", "literature")),
    ("review", "REVIEW", ("geometry",)),
)
OUTPUTS = {"literature": "literature.json", "geometry": "candidate.json",
           "review": "review.json"}
AGENT_ROLES = ("LITERATURE", "GEOMETRY", "REVIEW")


def store():
    return TaskStore(settings.STATE_ROOT)


def run_dir(run_id):
    if not isinstance(run_id, str) or len(run_id) != 20 or not run_id.startswith("run_"):
        raise ValueError("unknown research run identifier")
    if not all(c in "0123456789abcdef" for c in run_id[4:]):
        raise ValueError("unsafe research run identifier")
    return settings.STATE_ROOT / "runs" / run_id


def bootstrap_run(run_id):
    dag = store()
    for name, role, deps in TASKS:
        dag.add(run_id + "__" + name, role, [run_id + "__" + d for d in deps])
    return dag


def manifest_for(task, root):
    behavior = root / "behavior.json"
    literature = root / "agents" / "literature" / "literature.json"
    proposal = root / "agents" / "geometry" / "candidate.json"
    manifest = {"task_id": task, "research": "issue117-behavior-first"}
    if task == "literature":
        manifest["research_question"] = (
            "Research verified literature where user next-item behavior, "
            "hyperbolic geometry and generative semantic IDs intersect."
        )
    if task in ("geometry", "review"):
        report = read_json(behavior)
        lit = read_json(literature)
        manifest.update(behavior_evidence_sha256=digest_file(behavior),
                        literature_evidence_sha256=digest_file(literature),
                        behavior_evidence=report, literature_evidence=lit)
        if task == "review":
            candidate = read_json(proposal)
            check_candidate(candidate)
            manifest.update(candidate_id=candidate["candidate_id"],
                            candidate_sha256=digest_file(proposal),
                            candidate=candidate)
        if task == "geometry":
            feedback = root / "agents" / "geometry" / "revision_feedback.json"
            if feedback.is_file():
                manifest.update(revision_feedback=read_json(feedback),
                                revision_feedback_sha256=digest_file(feedback))
    manifest["inputs_sha256"] = digest_bytes(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())
    return manifest


def verify_science_output(task, root, manifest):
    output = read_json(root / "agents" / task / OUTPUTS[task])
    if task == "literature":
        if not isinstance(output.get("sources"), list) or not output["sources"]:
            raise ValueError("literature requires real cited/qualified sources")
    elif task == "geometry":
        check_candidate(output)
        if (output["behavior_evidence_sha256"] != manifest["behavior_evidence_sha256"]
                or output["literature_evidence_sha256"] != manifest["literature_evidence_sha256"]):
            raise ValueError("candidate ignores frozen DATA/LITERATURE inputs")
    elif task == "review":
        if (output.get("candidate_id") != manifest["candidate_id"]
                or output.get("candidate_sha256") != manifest["candidate_sha256"]):
            raise ValueError("review targets another candidate")
        if output.get("verdict") not in ("FATAL", "MATERIAL", "CPU_ELIGIBLE", "PARKED"):
            raise ValueError("review verdict absent/unknown")
        # Scientific outcomes are NOT machine-output errors; the DAG handles
        # MATERIAL revision and FATAL/PARKED rejection separately.


def process_next(run_id):
    """Claim one eligible task; callers may run separate workers concurrently."""
    root = run_dir(run_id)
    if not (root / "run.json").is_file() or (root / "cancel.json").exists():
        return None
    dag = store()
    owner = "research-worker-pid-" + str(os.getpid())
    permitted = None if settings.LIVE_AGENTS_ALLOWED else ("DATA",)
    claimed = dag.claim(owner, settings.LEASE_SECONDS, settings.MAX_ATTEMPTS,
                        roles=permitted, prefix=run_id + "__")
    if claimed is None:
        return None
    task_id, role = claimed
    task = task_id.split("__", 1)[1]
    try:
        if (root / "cancel.json").exists():
            dag.finish_cancel(task_id, owner)
            return None
        if role == "DATA":
            evidence = analyze_train_behavior(settings.TRAIN, root / "behavior.json")
        else:
            manifest = manifest_for(task, root)
            prompt = build_prompt(role, manifest, OUTPUTS[task], root=root)
            if role == "GEOMETRY" and manifest.get("revision_feedback"):
                prompt = (prompt + "\n\nAn independent reviewer REJECTED your previous "
                          "proposal with these findings. Correct exactly them, keep every "
                          "requirement you already met, and write the output file again:\n"
                          + json.dumps(manifest["revision_feedback"], ensure_ascii=False))
            if role == "REVIEW":
                check_review_identity(
                    root / "agents" / "geometry" / "candidate.json", manifest, prompt)
            atomic_json(root / "manifests" / (task + ".json"), manifest)
            evidence = run_agent_until_valid(
                task, root, manifest, prompt, task_id=task_id, owner=owner, dag=dag)
            if role == "REVIEW":
                verdict = read_json(root / "agents" / "review" / "review.json")["verdict"]
                if verdict == "MATERIAL":
                    if not _send_back_for_revision(dag, task_id, root, ValueError(verdict)):
                        raise StateError("could not reopen fixable proposal")
                    dag.fail(task_id, owner, "scientific revision: MATERIAL", retry=True)
                    return {"task": task, "role": role, "revision": True, "evidence": False}
                if verdict in ("FATAL", "PARKED"):
                    dag.fail(task_id, owner, "scientific rejection: " + verdict)
                    return {"task": task, "role": role, "blocked": True,
                            "review_verdict": verdict, "evidence": False}
        if (root / "cancel.json").exists():
            dag.finish_cancel(task_id, owner)
            return None
        dag.complete(task_id, owner, evidence)
    except Exception as exc:
        if (root / "cancel.json").exists():
            dag.finish_cancel(task_id, owner)
            return None
        try:
            dag.fail(task_id, owner, type(exc).__name__ + ": " + str(exc))
        except StateError:
            pass
        raise
    return {"task": task, "role": role, "evidence": True}


def inspect():
    return {"system": "research-issue117", "live_agents_allowed": settings.LIVE_AGENTS_ALLOWED,
            "training_launch_available": False,
            "repository": audit_surviving_mainline(),
            "tasks": store().snapshot()["tasks"]}


def step():
    """Compatibility for the source-configured no-argument local entry."""
    return inspect()


def _archive_attempt(root: Path, task: str, attempt: int = 1) -> Path | None:
    """Archive every attempted implementation with non-colliding evidence."""
    workspace = root / "agents" / task
    if not workspace.is_dir():
        return None
    while (workspace / ("attempt-%d" % attempt)).exists():
        attempt += 1
    target = workspace / ("attempt-%d" % attempt)
    target.mkdir(parents=True, exist_ok=False)
    for item in list(workspace.iterdir()):
        if item == target or item.name.startswith("attempt-"):
            continue
        if item.name == "revision_feedback.json":
            continue
        shutil.move(str(item), str(target / item.name))
    return target


def run_agent_until_valid(task: str, root: Path, manifest: dict, prompt: str,
                          *, task_id=None, owner=None, dag=None) -> Path:
    """No attempt limit for recoverable errors; still enforce cancellation/leases.

    Every failure is preserved and repaired with concrete checker feedback.
    Frozen scientific FATAL/PARKED verdicts never enter this repair branch.
    """
    import time

    root = Path(root)
    original_prompt = prompt
    workspace = root / "agents" / task
    if workspace.is_dir() and any(
            (workspace / x).exists()
            for x in (OUTPUTS[task], "session.jsonl", "execution.json")):
        _archive_attempt(root, task)  # recover from a prior interrupted worker
    attempt = 0
    while True:
        if (root / "cancel.json").is_file():
            raise RuntimeError("research run cancelled before retry")
        if dag is not None:
            if not task_id or not owner:
                raise ValueError("lease owner/task id required")
            dag.renew(task_id, owner, settings.LEASE_SECONDS)
        attempt += 1
        try:
            evidence = run_agent(task, prompt, OUTPUTS[task], root,
                                 binary=settings.AGENT_BINARY,
                                 model=settings.AGENT_MODEL,
                                 timeout=settings.AGENT_TIMEOUT_SECONDS)
            verify_science_output(task, root, manifest)
            return evidence
        except (ValueError, RuntimeError) as exc:
            if (root / "cancel.json").is_file():
                raise
            archived = _archive_attempt(root, task)
            if archived is None:
                raise
            atomic_json(archived / "repair_failure.json", {
                "attempt_in_invocation": attempt, "error_type": type(exc).__name__,
                "error": str(exc), "at": time.time(),
                "scientific_review_overridden": False})
            # Only the current error is appended: the prompt never grows without
            # bound, and previous outputs remain in the archived directory.
            prompt = (original_prompt +
                      "\n\nYour previous answer was rejected by the automated checker "
                      "with this exact message:\n" + str(exc) +
                      "\nExamine the archived attempt at " +
                      str(archived / OUTPUTS[task]) +
                      ", fix this specific defect, preserve the frozen protocol "
                      "and every other satisfied requirement, then write a new output.")
            continue


def _send_back_for_revision(dag, review_task_id, root: Path, error: Exception) -> bool:
    """Route a rejected proposal back to the geometry task with the findings.

    The review's own verdict and findings are written next to the geometry
    output, the previous geometry attempt is archived with its receipts, and the
    geometry task is reopened so the next agent call can correct exactly what
    the reviewer rejected.
    """
    review = root / "agents" / "review" / "review.json"
    if not review.is_file():
        return False
    try:
        payload = read_json(review)
    except Exception:
        return False
    verdict = str(payload.get("verdict") or "")
    # MATERIAL can be repaired with a new candidate; FATAL/PARKED is a
    # scientific stop, not an instruction to re-ask the same reviewer.
    if verdict != "MATERIAL":
        return False
    workspace = root / "agents" / "geometry"
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_json(workspace / "revision_feedback.json", {
        "review_verdict": verdict,
        "review_sha256": digest_file(review),
        "blockers": payload.get("blockers"),
        "reproducible_evidence": payload.get("reproducible_evidence"),
    })
    _archive_attempt(root, "geometry")
    _archive_attempt(root, "review")  # next REVIEW must not read stale evidence
    dag.reopen(review_task_id.split("__", 1)[0] + "__geometry",
               "review blocked the proposal: " + verdict)
    return True
