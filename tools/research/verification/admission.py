"""Deterministic Stage2 preflight for independent OpenCode verification evidence.

This is not an LLM judge and does not import retired CGV code. It accepts a
candidate identifier, resolves all evidence under the Issue117 root, verifies
frozen source and test hashes, and RE-RUNS independent tests before every GPU
launch. Agent prose alone never authorizes training.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = REPO / "results" / "research" / "issue117"
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
    return source_digest_recipe(tree)[0]


def source_digest_recipe(tree: Path) -> tuple[str, list]:
    """The tree digest together with the per-file hashes it was computed from.

    The recipe makes the digest reproducible by an independent party: it lists
    every file that went into the hash, with that file's own sha256, in the
    order the digest consumed them.
    """
    tree = Path(tree)
    if not tree.is_dir() or not (tree / "train_rqvae.py").is_file():
        raise AdmissionError("Stage2 tree has no real trainer")
    digest = hashlib.sha256()
    entries: list = []
    count = 0
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if set(path.relative_to(tree).parts) & {"logs", "results", "__pycache__", ".git"}:
            continue
        if path.suffix not in {".py", ".json", ".yaml", ".yml", ".toml"}:
            continue
        relative = str(path.relative_to(tree))
        file_sha = sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        entries.append({"path": relative, "sha256": file_sha})
        count += 1
    if not count:
        raise AdmissionError("no source files available for digest")
    return digest.hexdigest(), entries


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
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]{16}", run_id):
        raise AdmissionError("verification evidence must identify a real research run")
    run_root = EVIDENCE_ROOT / "runs" / run_id
    run_record = _read_json(run_root / "run.json")
    if run_record.get("run_id") != run_id or (run_root / "cancel.json").is_file():
        raise AdmissionError("research run identity mismatch or cancelled")
    _require_sha(source, manifest.get("source_sha256"), "trainer source")
    proposal = run_root / "agents" / "geometry" / "candidate.json"
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
        session = run_root / "agents" / (candidate_id + "_" + role)
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
            or _normalise_suite_keys(verdict.get("test_sha256")) != tests):
        raise AdmissionError("independent Reviewer did not approve these exact tests and source")
    return case_dir, manifest, {"candidate_id": candidate_id,
                                "source_sha256": source, "review": "APPROVE_TESTS"}


def run_preflight(tree: Path, candidate_id: str, python: str) -> dict:
    """Run independently authored test files using the Stage2 interpreter.

    Code review of these tests is required before merging them into main.
    A passing unittest result alone cannot prove a mathematically sound test.
    """
    outcome_path = EVIDENCE_ROOT / "verification" / candidate_id / "preflight.json"

    def _persist(payload: dict) -> dict:
        """Record the execution outcome on disk.

        The MCP client's request deadline is shorter than four separate test
        processes pay in interpreter and CUDA start-up, so the reply can be lost
        after the suites really ran. The artifact keeps the proof.
        """
        try:
            outcome_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = outcome_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps({**payload, "recorded_at_epoch": time.time()},
                           indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, outcome_path)
        except OSError:
            pass
        return payload

    try:
        case_dir, manifest, context = validate_evidence(tree, candidate_id)
    except (AdmissionError, OSError) as exc:
        return _persist({"passed": False, "refused": str(exc), "candidate_id": candidate_id})
    result = {"passed": False, **context, "suites": {}}
    # Three process groups, run concurrently. Grouping is the isolation
    # boundary, not an optimisation detail: math and gradient share one
    # interpreter (neither spawns ranks nor drives the trainer) so PyTorch is
    # imported once for both, while the two-rank DDP suite and the real
    # trainer-entrypoint suite each keep their own process.
    groups = (("math", "gradient"), ("ddp",), ("entrypoint",))
    runner = Path(__file__).resolve().parent / "run_suites.py"
    deadline = time.monotonic() + TIMEOUT_SECONDS
    started = []
    for group in groups:
        try:
            started.append((group, subprocess.Popen(
                [python, str(runner), str(case_dir), *group],
                cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True)))
        except OSError as exc:
            result["refused"] = " / ".join(group) + " did not execute: " + type(exc).__name__
            for _, proc in started:
                proc.kill()
            return _persist(result)
    for group, proc in started:
        remaining = max(1.0, deadline - time.monotonic())
        try:
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            result["refused"] = " / ".join(group) + " exceeded the execution budget"
            for _, other in started:
                if other.poll() is None:
                    other.kill()
            return _persist(result)
        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {"ok": False, "suites": {}}
            for name in group:
                payload["suites"][name] = {"ok": False, "tests_run": 0,
                                           "output": (stdout or "") + (stderr or "")}
        for name in group:
            record = payload["suites"].get(name) or {"ok": False, "tests_run": 0,
                                                     "output": (stdout or "") + (stderr or "")}
            text = record.pop("output", "") or ""
            ran = int(record.get("tests_run") or 0)
            clean = bool(record.get("ok")) and ran >= 1 and proc.returncode == 0
            result["suites"][name] = {
                "passed": clean, "tests_run": ran, "exit_code": proc.returncode,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "output_tail": text[-1200:],
                "failures": record.get("failures"), "errors": record.get("errors"),
                "skipped": record.get("skipped"),
                "expected_failures": record.get("expected_failures"),
                "unexpected_successes": record.get("unexpected_successes"),
                "process_group": list(group),
            }
            if not clean:
                result["refused"] = (name + " independent tests failed, were skipped, "
                                     "or ran zero tests")
                return _persist(result)
    result["passed"] = True
    result["verified_at_epoch"] = time.time()
    result["note"] = ("All four reviewed test suites actually ran; the existing "
                      "MCP one-epoch trainer smoke still follows before official training.")
    return _persist(result)

def _normalise_suite_keys(mapping):
    """Accept a report that names suites by file instead of by suite name.

    The manifest keys the reviewed suites by suite name ("math"), while a
    reviewer may write the filename ("test_math.py"). The hashes are the same
    evidence either way, so normalise the names and keep the values strict.
    """
    if not isinstance(mapping, dict):
        return mapping
    normalised = {}
    for key, value in mapping.items():
        name = str(key)
        if name.startswith("test_"):
            name = name[len("test_"):]
        if name.endswith(".py"):
            name = name[:-len(".py")]
        normalised[name] = value
    return normalised
