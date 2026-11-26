"""Node-local Stage3 dependency staging.

The Stage3 trainer imports torch and several Hugging Face packages.  On this
host the Python environment lives on Lustre; a cold import of the small
``transformers`` dependency graph is the measured startup bottleneck.  This
module copies only those pure-Python packages to a node-local directory, with a
content manifest, an atomic publication and a cross-process lock.  The source
environment is never modified and no training state or SID is cached.

A missing or unreadable cache falls back to the source environment; correctness
is never traded for startup speed.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

DEFAULT_PACKAGES = ("transformers", "tokenizers", "safetensors",
                    "huggingface_hub", "sentencepiece", "accelerate")
DEFAULT_ROOT = Path(os.environ.get("GENREC_STAGE3_DEP_CACHE", "/dev/shm/genrec-stage3-deps"))
SCHEMA = "genrec.stage3.dependency_cache/1"
MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state(source: Path, packages) -> dict:
    """Cheap identity used to select a cached generation.

    Content hashes are recorded in the built manifest and verified in the
    node-local target. Requiring a full source re-read on every launch was the
    dominant cold-start cost on Lustre; size+mtime detects normal package
    changes without reading every source byte.
    """
    state = {}
    for name in packages:
        package = source / name
        if not package.is_dir():
            continue
        for path in sorted(package.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            state[path.relative_to(source).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return state


def _tree(source: Path, packages) -> tuple[list[str], dict]:
    files = []
    details = {}
    for name in packages:
        package = source / name
        if not package.is_dir():
            continue
        for path in sorted(package.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source).as_posix()
            files.append(relative)
            details[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return files, details


def _fingerprint(source: Path, packages) -> tuple[list[str], dict, str]:
    files, details = _tree(source, packages)
    payload = json.dumps({"source": str(source), "packages": list(packages),
                          "files": details}, sort_keys=True).encode("utf-8")
    return files, details, hashlib.sha256(payload).hexdigest()


def _valid(target: Path, expected: dict) -> bool:
    manifest = target / MANIFEST
    if not manifest.is_file():
        return False
    try:
        stored = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if stored.get("schema") != SCHEMA or stored.get("files") != expected:
        return False
    for relative, item in expected.items():
        path = target / relative
        if not path.is_file() or path.stat().st_size != item.get("size"):
            return False
    # Hash verification protects against a truncated/corrupted shared cache.
    for relative, item in expected.items():
        if _sha256(target / relative) != item.get("sha256"):
            return False
    return True


def prepare(source, cache_root=None, packages=DEFAULT_PACKAGES):
    """Return a local site-packages-like directory for the requested packages.

    ``source`` is the interpreter's purelib directory.  The returned directory
    can be prepended to ``PYTHONPATH``; only packages that actually exist in the
    source are included.  On any cache failure the source path is returned so a
    run still starts.
    """
    source = Path(source)
    cache_root = Path(cache_root) if cache_root is not None else DEFAULT_ROOT
    if not source.is_dir():
        return None
    packages = tuple(str(name) for name in packages)
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        lock_path = cache_root / ".lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            source_state = _source_state(source, packages)
            state_digest = hashlib.sha256(json.dumps(source_state, sort_keys=True).encode()).hexdigest()
            target = cache_root / ("site-" + state_digest[:24])
            if target.is_dir():
                try:
                    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    manifest = None
                if (manifest and manifest.get("source_state") == source_state
                        and _valid(target, manifest.get("files", {}))):
                    return target
            files, details, digest = _fingerprint(source, packages)
            if not files:
                return None
            # Publication is one rename.  A reader either sees a complete
            # directory or no directory, never a half-copied package tree.
            temp = Path(tempfile.mkdtemp(prefix=".stage3-deps-", dir=cache_root))
            try:
                for name in packages:
                    source_package = source / name
                    if source_package.is_dir():
                        shutil.copytree(source_package, temp / name,
                                        symlinks=False, dirs_exist_ok=False)
                manifest = {"schema": SCHEMA, "source": str(source),
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "packages": list(packages), "source_state": source_state,
                            "files": details, "source_digest": digest}
                temp.joinpath(MANIFEST).write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if not _valid(temp, details):
                    raise RuntimeError("the copied dependency cache did not verify")
                if target.exists():
                    if _valid(target, details):
                        shutil.rmtree(temp)
                        return target
                    shutil.rmtree(target)
                os.replace(temp, target)
            except Exception:
                shutil.rmtree(temp, ignore_errors=True)
                return source
            return target
    except (OSError, RuntimeError, ValueError):
        return source
