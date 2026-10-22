"""The immutable evidence store (issue #105, workstream A).

Every run this system performs leaves a manifest that names what ran, against which frozen
declaration, with which seeds, and what came out -- and that manifest is written once, in
full, with the digest of everything it points at. Two properties make the rest of the
system trustworthy:

* **A run is identified by its content, not by its path.** The manifest carries a digest
  over the canonical JSON of the run, and every artifact it names is named with its own
  SHA-256. A later reader can check that the evidence still is what the manifest says it is.
* **A run cannot be quietly reused.** The store refuses to overwrite a run id, refuses
  output paths that overlap each other, refuses to start a run whose target artifacts
  already exist, and refuses to start a run whose source tree changed after the verdict
  that admitted it.

The subprocess helper is fail-closed by construction: a non-zero exit, a missing report, a
report that does not match its schema, or a report with fewer samples than the protocol
demands are all *failures with a reason*, never a silent success.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The manifest format version. A bump means the store refuses to read older manifests
#: rather than guessing what their fields meant.
MANIFEST_VERSION = 1

#: The kinds of run the store knows about. A kind outside this list is a typo, not a new
#: kind of evidence.
RUN_KINDS = (
    "control",
    "arm",
    "probe",
    "stage2",
    "stage3",
    "experiment",
    "analysis",
    "policy_session",
    "campaign",
)

#: The file suffixes that make up a source snapshot. Logs, checkpoints and caches are not
#: source and are excluded on purpose.
SOURCE_SUFFIXES = (".py", ".json", ".sh", ".md")


class EvidenceError(RuntimeError):
    """A refusal. The store never works around one."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)


def manifest_digest(manifest: Dict[str, Any]) -> str:
    """The digest over the manifest's own content, excluding the digest field itself."""
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def tree_digest(tree: Path) -> Tuple[str, int]:
    """The digest of a source tree: every source file's path and content, sorted.

    Two trees with the same digest are the same source, whatever they are called. The
    second value is the number of files that went into it.
    """
    tree = Path(tree)
    if not tree.is_dir():
        return "", 0
    entries: List[str] = []
    count = 0
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix not in SOURCE_SUFFIXES:
            continue
        digest = sha256_file(path)
        if digest is None:
            continue
        entries.append(f"{path.relative_to(tree).as_posix()}:{digest}")
        count += 1
    return sha256_bytes("\n".join(entries).encode("utf-8")), count


def _atomic_write(path: Path, text: str) -> None:
    """Write once, fully, or not at all: a half-written manifest is a broken manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=f".{path.name}.", delete=False
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


@dataclass
class RunPlan:
    """What a run declares BEFORE it runs. The manifest is built from this."""

    kind: str
    mechanism_id: str
    tree: Optional[Path] = None
    stage: str = ""
    seed: int = 42
    proposal_sha256: Optional[str] = None
    parent_runs: Tuple[str, ...] = ()
    output_dir: Optional[Path] = None
    command: Tuple[str, ...] = ()
    gpu_budget_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in RUN_KINDS:
            raise EvidenceError(
                f"unknown run kind {self.kind!r}; known kinds are {list(RUN_KINDS)}"
            )
        if not self.mechanism_id:
            raise EvidenceError("a run must name the mechanism it is about")


@dataclass
class Run:
    """A run that has been admitted and is now executing."""

    run_id: str
    kind: str
    mechanism_id: str
    directory: Path
    output_dir: Path
    plan: RunPlan
    tree_sha256: str = ""
    tree_files: int = 0


class EvidenceStore:
    """The store: a directory of immutable run manifests and their artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"

    # -- opening ---------------------------------------------------------------------
    def open(self) -> "EvidenceStore":
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    # -- admitting a run ------------------------------------------------------------
    def preflight(
        self,
        plan: RunPlan,
        *,
        verdict_tree_digest: Optional[str] = None,
        allow_existing_output: bool = False,
    ) -> None:
        """Refuse the run for every reason the store knows about, before anything starts.

        The empty-output-directory rule stops a run from overwriting what is already there.
        It must not stop the store from RECORDING a run that has already finished: by then
        the run's own products are in that directory, and refusing to record them would
        leave the system of record blind to exactly the arms that produced evidence.
        """
        plan.validate()
        if plan.output_dir is None:
            raise EvidenceError("a run must declare its output directory")
        output = Path(plan.output_dir).resolve()
        if not allow_existing_output and output.exists() and any(output.iterdir()):
            raise EvidenceError(
                f"the target output directory already exists and is not empty: {output}"
            )
        if plan.tree is not None:
            tree = Path(plan.tree)
            if not tree.is_dir():
                raise EvidenceError(f"the source tree does not exist: {tree}")
            digest, _files = tree_digest(tree)
            if verdict_tree_digest and verdict_tree_digest != digest:
                raise EvidenceError(
                    "the source tree changed after the verdict that admitted this run: the "
                    f"verdict was for {verdict_tree_digest[:16]} and the tree is now "
                    f"{digest[:16]}"
                )

    def admit(
        self,
        plan: RunPlan,
        *,
        verdict_tree_digest: Optional[str] = None,
        allow_existing_output: bool = False,
    ) -> Run:
        """Preflight, allocate an isolated directory and a unique run id."""
        self.preflight(
            plan,
            verdict_tree_digest=verdict_tree_digest,
            allow_existing_output=allow_existing_output,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = f"{plan.kind}-{plan.mechanism_id}-{stamp}"
        run_id = base
        counter = 1
        while self.run_dir(run_id).exists():
            counter += 1
            run_id = f"{base}-{counter}"
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        output = Path(plan.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        digest = ""
        files = 0
        if plan.tree is not None:
            digest, files = tree_digest(Path(plan.tree))
        return Run(
            run_id=run_id,
            kind=plan.kind,
            mechanism_id=plan.mechanism_id,
            directory=directory,
            output_dir=output,
            plan=plan,
            tree_sha256=digest,
            tree_files=files,
        )

    # -- closing a run ---------------------------------------------------------------
    def close(
        self,
        run: Run,
        *,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Iterable[Path]] = None,
        verdicts: Optional[Dict[str, Any]] = None,
        failure: Optional[Dict[str, Any]] = None,
        lineage: Optional[Iterable[str]] = None,
        parent_runs: Optional[Iterable[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write the manifest, once, and never again."""
        if status not in ("succeeded", "failed", "refused", "inconclusive"):
            raise EvidenceError(f"unknown run status {status!r}")
        manifest_path = self.manifest_path(run.run_id)
        if manifest_path.exists():
            raise EvidenceError(
                f"the manifest for {run.run_id} already exists; a run is immutable, so the "
                "correction is a NEW run, not an edit"
            )
        artifact_records = []
        for path in artifacts or ():
            path = Path(path)
            artifact_records.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                    "bytes": path.stat().st_size if path.is_file() else None,
                }
            )
        plan = run.plan
        manifest: Dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": run.run_id,
            "kind": run.kind,
            "stage": plan.stage,
            "mechanism_id": run.mechanism_id,
            "status": status,
            "created_utc": now_utc(),
            "seed": plan.seed,
            "tree": str(plan.tree) if plan.tree else None,
            "tree_sha256": run.tree_sha256,
            "tree_files": run.tree_files,
            "output_dir": str(run.output_dir),
            "command": list(plan.command),
            "proposal_sha256": plan.proposal_sha256,
            "parent_runs": sorted(
                set(list(plan.parent_runs) + list(lineage or ()) + list(parent_runs or ()))
            ),
            "gpu_budget_s": plan.gpu_budget_s,
            "metrics": metrics or {},
            "verdicts": verdicts or {},
            "artifacts": artifact_records,
            "failure": failure,
            "extra": {**plan.extra, **(extra or {})},
        }
        manifest["manifest_sha256"] = manifest_digest(manifest)
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        return manifest

    # -- reading ---------------------------------------------------------------------
    def load(self, run_id: str) -> Dict[str, Any]:
        path = self.manifest_path(run_id)
        if not path.is_file():
            raise EvidenceError(f"no manifest for run {run_id!r}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            raise EvidenceError(
                f"manifest {run_id} has version {manifest.get('manifest_version')}, this store "
                f"reads version {MANIFEST_VERSION}"
            )
        return manifest

    def runs(self) -> List[Dict[str, Any]]:
        manifests = []
        if not self.runs_dir.is_dir():
            return manifests
        for path in sorted(self.runs_dir.glob("*/manifest.json")):
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        return manifests

    def verify(self, run_id: str) -> Tuple[bool, List[str]]:
        """Re-hash everything the manifest names, and re-check the manifest's own digest."""
        problems: List[str] = []
        try:
            manifest = self.load(run_id)
        except EvidenceError as exc:
            return False, [str(exc)]
        expected = manifest.get("manifest_sha256")
        actual = manifest_digest(manifest)
        if expected != actual:
            problems.append(
                f"the manifest's own digest does not match its content: {expected} != {actual}"
            )
        for artifact in manifest.get("artifacts") or ():
            path = Path(artifact["path"])
            if not artifact.get("exists"):
                problems.append(f"the artifact is gone: {path}")
                continue
            digest = sha256_file(path)
            if digest != artifact.get("sha256"):
                problems.append(
                    f"the artifact's content changed since the manifest: {path} "
                    f"({artifact.get('sha256')} != {digest})"
                )
        return not problems, problems

    def lineage(self, run_id: str) -> List[str]:
        """The run's ancestry, oldest first, as a list of run ids."""
        chain: List[str] = []
        seen = set()
        stack = [run_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            chain.append(current)
            try:
                manifest = self.load(current)
            except EvidenceError:
                continue
            stack.extend(manifest.get("parent_runs") or ())
        chain.reverse()
        return chain


# --------------------------------------------------------------------------------------
# fail-closed subprocess execution
# --------------------------------------------------------------------------------------
@dataclass
class CheckedRun:
    """What a command did, and whether that counts as evidence."""

    cmd: List[str]
    returncode: int
    timed_out: bool
    duration_s: float
    stdout_tail: str
    stderr_tail: str
    report: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cmd": self.cmd,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "stdout_tail": self.stdout_tail[-2000:],
            "stderr_tail": self.stderr_tail[-2000:],
            "report": self.report,
            "artifacts": self.artifacts,
            "problems": list(self.problems),
            "failure_reason": self.failure_reason,
            "ok": self.ok,
        }


def run_checked(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = 3600,
    report_path: Optional[Path] = None,
    report_keys: Sequence[str] = (),
    min_samples: Optional[int] = None,
    samples_key: str = "samples",
    artifact_paths: Sequence[Path] = (),
    tail_chars: int = 4000,
) -> CheckedRun:
    """Run a command and decide, in one place, whether it produced usable evidence.

    The decision is deliberately conservative: a non-zero exit, a timeout, a missing
    report, a report missing a required key, or a report with too few samples are all
    failures, and the failure carries a reason a reader can act on. Nothing here decides
    whether a *mechanism* is good; it only decides whether the evidence exists.
    """
    cmd = [str(part) for part in cmd]
    environment = dict(os.environ)
    environment.update(env or {})
    started = time.time()
    stdout = ""
    stderr = ""
    returncode = -1
    timed_out = False
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr or "" if isinstance(exc.stderr, str) else ""
    except OSError as exc:
        stderr = f"{type(exc).__name__}: {exc}"
    duration = time.time() - started

    problems: List[str] = []
    report: Optional[Dict[str, Any]] = None
    if timed_out:
        problems.append(f"the command timed out after {timeout}s")
    elif returncode != 0:
        problems.append(f"the command exited with status {returncode}")
    if report_path is not None:
        path = Path(report_path)
        if not path.is_file():
            problems.append(f"the command wrote no report at {path}")
        else:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"the report at {path} is not valid JSON: {exc}")
                report = None
            if isinstance(report, dict):
                for key in report_keys:
                    if key not in report:
                        problems.append(f"the report at {path} has no {key!r}")
                if min_samples is not None and samples_key in report:
                    value = report.get(samples_key)
                    if not isinstance(value, (int, float)) or int(value) < int(min_samples):
                        problems.append(
                            f"the report carries {report.get(samples_key)!r} {samples_key}, "
                            f"fewer than the protocol's {min_samples}"
                        )
    artifacts = []
    for path in artifact_paths:
        path = Path(path)
        artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
        if not path.exists():
            problems.append(f"the expected artifact is missing: {path}")

    failure_reason = None
    if problems:
        failure_reason = "; ".join(problems)
    return CheckedRun(
        cmd=cmd,
        returncode=returncode,
        timed_out=timed_out,
        duration_s=duration,
        stdout_tail=stdout[-tail_chars:],
        stderr_tail=stderr[-tail_chars:],
        report=report,
        artifacts=artifacts,
        problems=problems,
        failure_reason=failure_reason,
    )


def refuse_overlapping_paths(paths: Iterable[Path]) -> None:
    """Refuse a set of output paths that overlap each other.

    Two runs writing into the same directory is the collision that destroyed the #104
    control's artifacts; this is the check that would have stopped it.
    """
    resolved = [Path(path).resolve() for path in paths]
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or second.is_relative_to(first) or first.is_relative_to(second):
                raise EvidenceError(
                    f"the output paths overlap: {first} and {second}; two runs must never "
                    "write into the same tree"
                )


def archive_existing(path: Path, archive_root: Path) -> Optional[Path]:
    """Move an existing artifact out of the way, into the archive, with its digest."""
    path = Path(path)
    if not path.exists():
        return None
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = archive_root / f"{path.name}.{stamp}"
    shutil.move(str(path), str(destination))
    return destination


__all__ = [
    "CheckedRun",
    "EvidenceError",
    "EvidenceStore",
    "MANIFEST_VERSION",
    "RUN_KINDS",
    "Run",
    "RunPlan",
    "archive_existing",
    "canonical_json",
    "manifest_digest",
    "now_utc",
    "refuse_overlapping_paths",
    "run_checked",
    "sha256_bytes",
    "sha256_file",
    "tree_digest",
]
