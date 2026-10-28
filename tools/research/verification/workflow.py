"""Independent OpenCode test authors and evidence freeze for Issue117.

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
from ..stage2 import contract
from ..core.evidence import atomic_json, digest_bytes, digest_file, read_json
from .admission import (AGENT_ROLES, CASES_ROOT, EVIDENCE_ROOT, SUITES,
                        AdmissionError, source_digest, source_digest_recipe, validate_evidence,
                        sha256)

AUTHOR_SPEC = {
    "math": ("MATH_VERIFY", "math.json", {"math": "test_source"}),
    "gradient": ("GRADIENT_VERIFY", "gradient.json", {"gradient": "test_source"}),
    "runtime": ("RUNTIME_VERIFY", "runtime.json",
                {"ddp": "ddp_test_source", "entrypoint": "entrypoint_test_source"}),
}


def _require_live():
    if not settings.LIVE_AGENTS_ALLOWED:
        raise PermissionError("real OpenCode Agents disabled; check sandbox and settings.py first")


def _inputs(candidate_id: str, tree: Path, run_id: str) -> dict:
    from ..workflows.pipeline import run_dir
    root = run_dir(run_id)
    if not (root / "run.json").is_file():
        raise AdmissionError("run_id does not reference a recorded research run")
    proposal = root / "agents" / "geometry" / "candidate.json"
    candidate = read_json(proposal)
    if candidate.get("candidate_id") != candidate_id:
        raise AdmissionError("proposal ID mismatch")
    tree = Path(tree).resolve()
    if not (tree / "train_rqvae.py").is_file():
        raise AdmissionError("no candidate Stage2 trainer")
    source_sha, source_files = source_digest_recipe(tree)
    # The authoring roles receive the mechanism, not the whole proposal dump: the
    # full candidate text made the math role return an empty answer three times
    # on a verbose proposal.
    mechanism = {
        "candidate_id": candidate.get("candidate_id"),
        "behavior_problem_id": candidate.get("behavior_problem_id"),
        "next_item_hypothesis": candidate.get("next_item_hypothesis"),
        "math_definition": candidate.get("math_definition"),
        "hyperbolic_necessity": candidate.get("hyperbolic_necessity"),
        "euclidean_control": candidate.get("euclidean_control"),
        "effect_site": candidate.get("effect_site"),
        "real_trigger_measurement": candidate.get("real_trigger_measurement"),
        "training_only_inputs": candidate.get("training_only_inputs"),
        "stable_global_sid": candidate.get("stable_global_sid"),
        "protocol": candidate.get("protocol"),
        "falsifier": candidate.get("falsifier"),
    }
    return {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "proposal_sha256": sha256(proposal),
        "source_sha256": source_sha,
        "candidate": mechanism,
        "source_digest_recipe": {
            "algorithm": "sha256 over the sorted list of source files: for each file, the "
                         "relative path, a NUL byte, and the file's own sha256 in ascii",
            "file_suffixes": [".py", ".json", ".yaml", ".yml", ".toml"],
            "excluded_directories": ["logs", "results", "__pycache__", ".git"],
            "files": source_files,
        },
        "source_tree": str(tree),
        "candidate": candidate,
        "test_policy": "Use independent mathematical oracle; really call trainer; DDP needs >=2 actual ranks.",
    }


def _prompt(role, ident, data, output, root=None):
    frozen = {"task_id": ident, **data}
    frozen["inputs_sha256"] = digest_bytes(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode())
    return build_prompt(role, frozen, output, root=root)


def sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _archive_workspace(workspace: Path, request_id: str = "") -> Path | None:
    """Move a previous role workspace aside so a re-authoring can write there."""
    if not workspace.is_dir():
        return None
    suffix = (request_id[:8] if request_id else "old")
    target = workspace.with_name(workspace.name + "." + suffix)
    index = 1
    while target.exists():
        index += 1
        target = workspace.with_name(workspace.name + "." + suffix + "." + str(index))
    workspace.rename(target)
    return target


def draft_test_suites(candidate_id: str, tree: Path, run_id: str,
                      request_id: str = "") -> dict:
    """Write untrusted test drafts only to results/, never into tracked cases."""
    _require_live()
    data = _inputs(candidate_id, tree, run_id)
    from ..workflows.pipeline import run_dir
    run_root = run_dir(run_id)
    draft_dir = EVIDENCE_ROOT / "verification" / candidate_id / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)

    def author(role_key: str):
        role, output, mapping = AUTHOR_SPEC[role_key]
        # The receipt path is part of the evidence contract: the admission
        # checker looks the role receipts up at agents/<candidate_id>_<role>.
        # A re-authored role therefore archives its previous workspace instead
        # of moving to a different path, so the contract holds and the earlier
        # evidence is kept.
        task_id = candidate_id + "_" + role_key
        _archive_workspace(run_root / "agents" / task_id, request_id)
        prompt = _prompt(role, task_id, data, output, root=run_root)
        # The Issue #117 surface runs under the same boundary as a round: the
        # candidate tree it reviews and the Stage 2 interpreter are the only
        # research files it may read, and the run's other agents' answers are
        # not among them.
        execution = run_agent(task_id, prompt, output, run_root,
                              binary=settings.AGENT_BINARY,
                              model=settings.AGENT_MODEL,
                              timeout=settings.AGENT_TIMEOUT_SECONDS,
                              read_paths=[tree, Path(contract.STAGE2_PYTHON).parent.parent],
                              allowed_roots=[settings.ROOT, Path(contract.STAGE2_PYTHON).parent.parent],
                              candidate_tree=tree,
                              allow_shell=True, allow_write=True)
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
    authors = dict(pairs)
    # The index tells the review stage where each author wrote its output, so
    # the reviewer reads the same sessions the draft stage actually produced.
    atomic_json(draft_dir / "authors.json",
                {key: {"execution": value["execution"], "output_name": AUTHOR_SPEC[key][1]}
                 for key, value in authors.items()})
    return {"source_sha256": data["source_sha256"],
            "proposal_sha256": data["proposal_sha256"], "authors": authors,
            "next_action": "Review and commit independent cases into tools/research/verification/cases/<candidate_id>/; do NOT auto-install drafts."}


def review_committed_suites(candidate_id: str, tree: Path, run_id: str,
                        request_id: str = "") -> Path:
    """Read actual committed files, then run a distinct independent review Agent."""
    _require_live()
    from ..workflows.pipeline import run_dir
    run_root = run_dir(run_id)
    data = _inputs(candidate_id, tree, run_id)
    case_dir = CASES_ROOT / candidate_id
    tests = {}
    # The reviewed suites are materialized in the run's evidence space as well,
    # so a reviewer can match concrete bytes to the hashes it is given instead
    # of taking the committed files on trust.
    reviewed = EVIDENCE_ROOT / "verification" / candidate_id / "reviewed"
    reviewed.mkdir(parents=True, exist_ok=True)
    for suite in SUITES:
        path = case_dir / ("test_" + suite + ".py")
        if not path.is_file() or path.is_symlink():
            raise AdmissionError("missing reviewed test in source tree: " + str(path))
        source = path.read_text(encoding="utf-8")
        materialised = reviewed / ("test_" + suite + ".py")
        materialised.write_text(source, encoding="utf-8")
        tests[suite] = {"sha256": sha256(path), "source": source,
                        "committed_path": str(path), "materialised_path": str(materialised),
                        "materialised_sha256": sha256(materialised)}
    data["tests"] = tests
    # The draft author reports are receipts of the authoring step, checked by
    # salt-and-hash admission; they are NOT evidence about the committed tests.
    # Carrying their stale prose into the review made a reviewer block two
    # corrected suites because a draft still quoted the pre-correction expected
    # value. The committed, materialised tests are the only authoritative text.
    data["author_reports"] = {}
    prompt = _prompt("VERIFY_REVIEW", candidate_id + "_review", data, "review.json",
                     root=run_root)
    _archive_workspace(run_root / "agents" / (candidate_id + "_review"), request_id)
    return run_agent(candidate_id + "_review", prompt, "review.json",
                     run_root, binary=settings.AGENT_BINARY,
                     model=settings.AGENT_MODEL,
                     timeout=settings.AGENT_TIMEOUT_SECONDS,
                     read_paths=[tree, Path(contract.STAGE2_PYTHON).parent.parent],
                     allowed_roots=[settings.ROOT, Path(contract.STAGE2_PYTHON).parent.parent],
                     candidate_tree=tree,
                     allow_shell=True, allow_write=True)


def freeze_reviewed_manifest(candidate_id: str, tree: Path, run_id: str) -> Path:
    """Build a candidate-specific content address only; never claim tests ran."""
    data = _inputs(candidate_id, tree, run_id)
    test_paths = [CASES_ROOT / candidate_id / ("test_" + role + ".py") for role in SUITES]
    if any(not path.is_file() or path.is_symlink() for path in test_paths):
        raise AdmissionError("four reviewed test modules are required")
    manifest = {
        "schema": 1, "candidate_id": candidate_id, "run_id": run_id,
        "source_tree": Path(tree).name,
        "source_sha256": data["source_sha256"],
        "proposal_sha256": data["proposal_sha256"],
        "tests": dict(zip(SUITES, (sha256(path) for path in test_paths))),
    }
    path = EVIDENCE_ROOT / "verification" / candidate_id / "manifest.json"
    if path.exists() and read_json(path).get("run_id") != run_id:
        raise AdmissionError("candidate verification identity is owned by another run")
    atomic_json(path, manifest)
    # Non-test identity check; will fail closed if review evidence is incomplete.
    # A blocked review must leave NO manifest behind, otherwise a later
    # verification_check could find a manifest whose review never approved it.
    try:
        validate_evidence(tree, candidate_id)
    except AdmissionError:
        path.unlink(missing_ok=True)
        raise
    return path
