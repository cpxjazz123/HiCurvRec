"""Deterministic Stage2 preflight for independent OpenCode verification evidence.

This is not an LLM judge and does not import retired CGV code. It accepts a
candidate identifier, resolves all evidence under the Issue116 root, verifies
frozen source and test hashes, and RE-RUNS independent tests before every GPU
launch. Agent prose alone never authorizes training.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = REPO / "results" / "research" / "issue116"
CASES_ROOT = Path(__file__).resolve().parent / "cases"
SUITES = ("math", "gradient", "ddp", "entrypoint")
AGENT_ROLES = ("math", "gradient", "runtime", "review")
TIMEOUT_SECONDS = 900
SAFE_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{2,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AdmissionError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(tree: Path) -> str:
    """Bind the complete active trainer Python/config source, not just its launcher."""
    tree = Path(tree)
    if not tree.is_dir() or not (tree / "train_rqvae.py").is_file():
        raise AdmissionError("Stage2 tree has no real trainer")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if set(path.relative_to(tree).parts) & {"logs", "results", "__pycache__", ".git"}:
            continue
        if path.suffix not in {".py", ".json", ".yaml", ".yml", ".toml"}:
            continue
        digest.update(str(path.relative_to(tree)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        count += 1
    if not count:
        raise AdmissionError("no source files available for digest")
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError("missing/invalid evidence: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise AdmissionError("evidence must be JSON object: " + str(path))
    return payload


def _require_sha(actual: str, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or not SHA256.fullmatch(expected) or actual != expected:
        raise AdmissionError(label + " differs from frozen evidence")


def validate_evidence(tree: Path, candidate_id: str) -> tuple[Path, dict, dict]:
    """Check all immutable byte identities without launching any processes."""
    if not isinstance(candidate_id, str) or not SAFE_ID.fullmatch(candidate_id):
        raise AdmissionError("candidate_id must be a safe, nonempty identifier")
    tree = Path(tree).resolve(strict=True)
    # Research candidates are new source trees; never relaunch the two baselines.
    if tree.name in {"TIGER_RQ-VAE", "TIGER_BEHAVIOR_SAMPLE_RQ-VAE"}:
        raise AdmissionError("research verification must target a distinct candidate source tree")
    path = EVIDENCE_ROOT / "verification" / candidate_id / "manifest.json"
    manifest = _read_json(path)
    if manifest.get("schema") != 1 or manifest.get("candidate_id") != candidate_id:
        raise AdmissionError("manifest schema/candidate identity mismatch")
    if manifest.get("source_tree") != tree.name:
        raise AdmissionError("manifest targets a different Stage2 tree")
    source = source_digest(tree)
    _require_sha(source, manifest.get("source_sha256"), "trainer source")
    proposal = EVIDENCE_ROOT / "agents" / "geometry" / "candidate.json"
    _require_sha(sha256(proposal), manifest.get("proposal_sha256"), "research proposal")
    candidate = _read_json(proposal)
    if candidate.get("candidate_id") != candidate_id:
        raise AdmissionError("research proposal names a different candidate")
    tests = manifest.get("tests")
    if not isinstance(tests, dict) or set(tests) != set(SUITES):
        raise AdmissionError("all four independent test suites are mandatory")
    case_dir = CASES_ROOT / candidate_id
    if not case_dir.is_dir() or case_dir.is_symlink():
        raise AdmissionError("missing reviewed test case directory")
    for suite in SUITES:
        target = case_dir / ("test_" + suite + ".py")
        if target.is_symlink() or not target.is_file():
            raise AdmissionError("missing regular test file: " + suite)
        _require_sha(sha256(target), tests[suite], "test suite " + suite)

    # The receipts attest actual subprocess activity; they are not scientific verdicts.
    receipts = {}
    pids = set()
    for role in AGENT_ROLES:
        session = EVIDENCE_ROOT / "agents" / (candidate_id + "_" + role)
        record = _read_json(session / "execution.json")
        if (record.get("verified") is not True or record.get("exit_code") != 0
                or record.get("timed_out") is not False or not isinstance(record.get("pid"), int)
                or record["pid"] < 1):
            raise AdmissionError("independent Agent did not complete: " + role)
        if record["pid"] in pids:
            raise AdmissionError("Agent roles do not have distinct process evidence")
        pids.add(record["pid"])
        name = record.get("output_name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+\.json", name):
            raise AdmissionError("unsafe Agent output filename")
        output = session / name
        _require_sha(sha256(output), record.get("output_sha256"),
                     role + " Agent output")
        _require_sha(sha256(session / "session.jsonl"),
                     record.get("session_sha256"), role + " Agent log")
        receipts[role] = _read_json(output)
        if receipts[role].get("candidate_id") != candidate_id:
            raise AdmissionError("Agent evaluated another candidate: " + role)
        if role != "review" and receipts[role].get("source_sha256") != source:
            raise AdmissionError("Agent evaluated different source: " + role)
    verdict = receipts["review"]
    if (verdict.get("verdict") != "APPROVE_TESTS"
            or verdict.get("source_sha256") != source
            or verdict.get("proposal_sha256") != manifest["proposal_sha256"]
            or verdict.get("test_sha256") != tests):
        raise AdmissionError("independent Reviewer did not approve these exact tests and source")
    return case_dir, manifest, {"candidate_id": candidate_id,
                                "source_sha256": source, "review": "APPROVE_TESTS"}


def run_preflight(tree: Path, candidate_id: str, python: str) -> dict:
    """Run independently authored test files using the Stage2 interpreter.

    Code review of these tests is required before merging them into main.
    A passing unittest result alone cannot prove a mathematically sound test.
    """
    try:
        case_dir, manifest, context = validate_evidence(tree, candidate_id)
    except (AdmissionError, OSError) as exc:
        return {"passed": False, "refused": str(exc), "candidate_id": candidate_id}
    result = {"passed": False, **context, "suites": {}}
    for suite in SUITES:
        # Discover exactly one reviewed and content-addressed suite.
        command = [python, "-m", "unittest", "discover", "-s",
                   str(case_dir), "-p", "test_" + suite + ".py", "-v"]
        try:
            done = subprocess.run(command, cwd=str(REPO), capture_output=True,
                                  text=True, timeout=TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["refused"] = suite + " did not execute: " + type(exc).__name__
            return result
        text = (done.stdout or "") + "\n" + (done.stderr or "")
        ran = re.search(r"\bRan\s+(\d+)\s+tests?\b", text)
        count = int(ran.group(1)) if ran else 0
        passed = (done.returncode == 0 and count >= 1
                  and bool(re.search(r"(?m)^OK(?:\s|$)", text)))
        result["suites"][suite] = {
            "passed": passed, "tests_run": count, "exit_code": done.returncode,
            "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "output_tail": text[-1200:],
        }
        if not passed:
            result["refused"] = suite + " independent tests failed or ran zero tests"
            return result
    result["passed"] = True
    result["verified_at_epoch"] = time.time()
    result["note"] = ("All four reviewed test suites actually ran; the existing "
                      "MCP one-epoch trainer smoke still follows before official training.")
    return result
