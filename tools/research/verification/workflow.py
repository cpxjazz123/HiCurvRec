"""Independent OpenCode test authors and evidence freeze for Issue116.

Three verifier agents may author test sources in parallel; they do NOT install
their own code as passing tests or auto-approve it. Reviewed tests are promoted
into a candidate-specific tracked source directory by the repository operator.
The independent review agent then examines that exact source, and the manifest
can be frozen for the MCP's repeated deterministic test execution.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import settings
from ..agents.prompts import build_prompt
from ..agents.runner import run_agent
from ..core.evidence import atomic_json, digest_bytes, digest_file, read_json
from .admission import (AGENT_ROLES, CASES_ROOT, EVIDENCE_ROOT, SUITES,
                        AdmissionError, source_digest, validate_evidence, sha256)

AUTHOR_SPEC = {
    "math": ("MATH_VERIFY", "math.json", {"math": "test_source"}),
    "gradient": ("GRADIENT_VERIFY", "gradient.json", {"gradient": "test_source"}),
    "runtime": ("RUNTIME_VERIFY", "runtime.json",
                {"ddp": "ddp_test_source", "entrypoint": "entrypoint_test_source"}),
}


def _require_live():
    if not settings.LIVE_AGENTS_ALLOWED:
        raise PermissionError("real OpenCode Agents disabled; check sandbox and settings.py first")


def _inputs(candidate_id: str, tree: Path) -> dict:
    proposal = EVIDENCE_ROOT / "agents" / "geometry" / "candidate.json"
    candidate = read_json(proposal)
    if candidate.get("candidate_id") != candidate_id:
        raise AdmissionError("proposal ID mismatch")
    tree = Path(tree).resolve()
    if not (tree / "train_rqvae.py").is_file():
        raise AdmissionError("no candidate Stage2 trainer")
    return {
        "candidate_id": candidate_id,
        "proposal_sha256": sha256(proposal),
        "source_sha256": source_digest(tree),
        "source_tree": str(tree),
        "candidate": candidate,
        "test_policy": "Use independent mathematical oracle; really call trainer; DDP needs >=2 actual ranks.",
    }


def _prompt(role, ident, data, output):
    frozen = {"task_id": ident, **data}
    frozen["inputs_sha256"] = digest_bytes(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode())
    return build_prompt(role, frozen, output)


def draft_test_suites(candidate_id: str, tree: Path) -> dict:
    """Write untrusted test drafts only to results/, never into tracked cases."""
    _require_live()
    data = _inputs(candidate_id, tree)
    draft_dir = EVIDENCE_ROOT / "verification" / candidate_id / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)

    def author(role_key: str):
        role, output, mapping = AUTHOR_SPEC[role_key]
        task_id = candidate_id + "_" + role_key
        prompt = _prompt(role, task_id, data, output)
        execution = run_agent(task_id, prompt, output, EVIDENCE_ROOT,
                              binary=settings.AGENT_BINARY,
                              model=settings.AGENT_MODEL,
                              timeout=settings.AGENT_TIMEOUT_SECONDS)
        payload = read_json(execution.parent / output)
        if payload.get("candidate_id") != candidate_id or payload.get("source_sha256") != data["source_sha256"]:
            raise AdmissionError("independent author changed candidate/source identity")
        emitted = {}
        for suite, field in mapping.items():
            code = payload.get(field)
            if not isinstance(code, str) or "unittest" not in code or "test_" not in code:
                raise AdmissionError("missing executable test draft: " + suite)
            target = draft_dir / ("test_" + suite + ".py")
            target.write_text(code, encoding="utf-8")
            emitted[suite] = {"path": str(target), "sha256": digest_file(target)}
        return role_key, {"execution": str(execution), "drafts": emitted}

    with ThreadPoolExecutor(max_workers=3) as pool:
        pairs = list(pool.map(author, AUTHOR_SPEC))
    return {"source_sha256": data["source_sha256"],
            "proposal_sha256": data["proposal_sha256"], "authors": dict(pairs),
            "next_action": "Review and commit independent cases into tools/research/verification/cases/<candidate_id>/; do NOT auto-install drafts."}


def review_committed_suites(candidate_id: str, tree: Path) -> Path:
    """Read actual committed files, then run a distinct independent review Agent."""
    _require_live()
    data = _inputs(candidate_id, tree)
    case_dir = CASES_ROOT / candidate_id
    tests = {}
    for suite in SUITES:
        path = case_dir / ("test_" + suite + ".py")
        if not path.is_file() or path.is_symlink():
            raise AdmissionError("missing reviewed test in source tree: " + str(path))
        tests[suite] = {"sha256": sha256(path), "source": path.read_text(encoding="utf-8")}
    data["tests"] = tests
    data["author_reports"] = {}
    for role in ("math", "gradient", "runtime"):
        output = EVIDENCE_ROOT / "agents" / (candidate_id + "_" + role) / (role + ".json")
        data["author_reports"][role] = read_json(output)
    prompt = _prompt("VERIFY_REVIEW", candidate_id + "_review", data, "review.json")
    return run_agent(candidate_id + "_review", prompt, "review.json",
                     EVIDENCE_ROOT, binary=settings.AGENT_BINARY,
                     model=settings.AGENT_MODEL,
                     timeout=settings.AGENT_TIMEOUT_SECONDS)


def freeze_reviewed_manifest(candidate_id: str, tree: Path) -> Path:
    """Build a candidate-specific content address only; never claim tests ran."""
    data = _inputs(candidate_id, tree)
    test_paths = [CASES_ROOT / candidate_id / ("test_" + role + ".py") for role in SUITES]
    if any(not path.is_file() or path.is_symlink() for path in test_paths):
        raise AdmissionError("four reviewed test modules are required")
    manifest = {
        "schema": 1, "candidate_id": candidate_id, "source_tree": Path(tree).name,
        "source_sha256": data["source_sha256"],
        "proposal_sha256": data["proposal_sha256"],
        "tests": dict(zip(SUITES, (sha256(path) for path in test_paths))),
    }
    path = EVIDENCE_ROOT / "verification" / candidate_id / "manifest.json"
    atomic_json(path, manifest)
    # Non-test identity check; will fail closed if review evidence is incomplete.
    validate_evidence(tree, candidate_id)
    return path
