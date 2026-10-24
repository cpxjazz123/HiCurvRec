"""One native issue116 path: train evidence and literature before geometry."""
import json
import os

from .. import settings
from ..agents.prompts import build_prompt
from ..agents.runner import run_agent
from ..core.dag import TaskStore
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
OWNER = "issue116-native-" + str(os.getpid())


def store():
    return TaskStore(settings.STATE_ROOT)


def bootstrap():
    dag = store()
    for ident, role, dependencies in TASKS:
        dag.add(ident, role, dependencies)
    return dag


def inspect():
    state = bootstrap().snapshot()
    return {"system": "research-issue116", "mode": settings.ACTION,
            "live_agents_allowed": settings.LIVE_AGENTS_ALLOWED,
            "training_launch_available": False,
            "surviving_mainline": audit_surviving_mainline(),
            "tasks": {ident: {"role": task["role"], "status": task["status"],
                              "attempts": task["attempts"]}
                      for ident, task in state["tasks"].items()}}


def manifest_for(task, root):
    behavior = root / "behavior.json"
    literature = root / "agents" / "literature" / "literature.json"
    candidate_path = root / "agents" / "geometry" / "candidate.json"
    manifest = {"task_id": task, "research": "issue116-behavior-first"}
    if task == "literature":
        manifest["research_question"] = (
            "Measure exact primary-source research overlap for user next-item behavior "
            "x negative-curvature geometry x generative SID; use no future interactions."
        )
    if task in ("geometry", "review"):
        report = read_json(behavior)
        lit = read_json(literature)
        manifest.update(behavior_evidence_sha256=digest_file(behavior),
                        literature_evidence_sha256=digest_file(literature),
                        behavior_evidence=report, literature_evidence=lit)
        if task == "review":
            candidate = read_json(candidate_path)
            check_candidate(candidate)
            manifest.update(candidate_id=candidate["candidate_id"],
                            candidate_sha256=digest_file(candidate_path),
                            candidate=candidate)
    manifest["inputs_sha256"] = digest_bytes(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return manifest


def verify_science_output(task, root, manifest):
    path = root / "agents" / task / OUTPUTS[task]
    output = read_json(path)
    if task == "literature":
        if not isinstance(output.get("sources"), list) or not output["sources"]:
            raise ValueError("literature output must include verified or qualified sources")
    elif task == "geometry":
        check_candidate(output)
        if (output["behavior_evidence_sha256"] != manifest["behavior_evidence_sha256"]
                or output["literature_evidence_sha256"] != manifest["literature_evidence_sha256"]):
            raise ValueError("candidate does not bind current frozen DATA and LITERATURE")
    elif task == "review":
        if (output.get("candidate_id") != manifest["candidate_id"]
                or output.get("candidate_sha256") != manifest["candidate_sha256"]):
            raise ValueError("review verdict pertains to a different candidate")
        if output.get("verdict") not in ("FATAL", "MATERIAL", "CPU_ELIGIBLE", "PARKED"):
            raise ValueError("review must emit a current structured verdict")
        if output["verdict"] != "CPU_ELIGIBLE":
            raise ValueError("scientific admission blocked: " + output["verdict"])


def step():
    """Exactly one eligible task per invocation. Multiple workers share atomic leases."""
    dag = bootstrap()
    permitted = None if settings.LIVE_AGENTS_ALLOWED else ("DATA",)
    claim = dag.claim(OWNER, settings.LEASE_SECONDS, settings.MAX_ATTEMPTS,
                      roles=permitted)
    if not claim:
        return inspect()
    task, role = claim
    root = settings.STATE_ROOT
    try:
        if role == "DATA":
            evidence = analyze_train_behavior(settings.TRAIN, root / "behavior.json")
        else:
            manifest = manifest_for(task, root)
            prompt = build_prompt(role, manifest, OUTPUTS[task])
            if role == "REVIEW":
                check_review_identity(
                    root / "agents" / "geometry" / "candidate.json", manifest, prompt)
            atomic_json(root / "manifests" / (task + ".json"), manifest)
            evidence = run_agent(task, prompt, OUTPUTS[task], root,
                                 binary=settings.AGENT_BINARY,
                                 model=settings.AGENT_MODEL,
                                 timeout=settings.AGENT_TIMEOUT_SECONDS)
            verify_science_output(task, root, manifest)
        dag.complete(task, OWNER, evidence)
    except Exception as exc:
        dag.fail(task, OWNER, type(exc).__name__ + ": " + str(exc))
        raise
    return inspect()
