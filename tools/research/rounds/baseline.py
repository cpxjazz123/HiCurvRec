"""The one permitted starting tree, and the copies made from it (Issue #121).

Issue #121 fixes the baseline: every new mechanism starts from
``/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE`` and from
nothing else. This module copies that tree for a candidate, records the
provenance of both the baseline and the copy, refuses to continue when the
source is missing or unreadable, and refuses a candidate whose output paths
would write into the baseline or into another candidate.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

#: The only tree a new mechanism may be built from.
CANONICAL_BASELINE = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE")

#: Files that belong to a run, not to a source tree.
_RUN_ARTIFACTS = ("logs", "results", "dataset", "__pycache__", "ckpt")

#: The output constants a trainer may point somewhere else.
_OUTPUT_CONSTANTS = ("OUTPUT_SID", "OUTPUT_JSON", "CHECKPOINT", "METRICS_PATH",
                     "LOG_DIR", "RESULT_DIR", "OUT_ROOT", "SID_OUTPUT")


class BaselineError(RuntimeError):
    """The baseline is missing, unreadable, or a candidate would overwrite it."""


def baseline_path() -> Path:
    """The canonical baseline, resolved.

    The issue names an absolute path, so the check is on that path and not on
    whatever happens to sit in the working tree.
    """
    if not CANONICAL_BASELINE.is_dir():
        raise BaselineError("the canonical baseline does not exist: " + str(CANONICAL_BASELINE))
    if not (CANONICAL_BASELINE / "train_rqvae.py").is_file():
        raise BaselineError("the canonical baseline has no trainer: " + str(CANONICAL_BASELINE))
    return CANONICAL_BASELINE.resolve()


def source_digest(tree: Path) -> str:
    """A digest of the source files of a tree, for the provenance record."""
    digest = hashlib.sha256()
    tree = Path(tree)
    for path in sorted(tree.rglob("*.py")):
        # The parts that matter are the ones inside the tree: a tree that lives
        # under results/ must not look like it holds no source at all.
        if any(part in _RUN_ARTIFACTS for part in path.relative_to(tree).parts):
            continue
        digest.update(str(path.relative_to(tree)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def copy_baseline(destination: Path, *, baseline: Path = None) -> dict:
    """Copy the canonical baseline to ``destination`` and record the provenance.

    The copy is a fresh copy of the baseline for every candidate: nothing is
    carried over from another candidate, and no run artifact travels with it.
    """
    source = Path(baseline) if baseline is not None else baseline_path()
    source = source.resolve()
    destination = Path(destination)
    if not source.is_dir() or not (source / "train_rqvae.py").is_file():
        raise BaselineError("the baseline is missing or has no trainer: " + str(source))
    if destination.exists():
        # A stale directory is not a fresh copy: the issue asks for a new one
        # per candidate, so the old one is removed instead of merged.
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns(*_RUN_ARTIFACTS))
    provenance = {
        "baseline": str(source),
        "baseline_digest": source_digest(source),
        "candidate": str(destination.resolve()),
        "candidate_digest": source_digest(destination),
    }
    if provenance["candidate_digest"] != provenance["baseline_digest"]:
        # A copy that changed the baseline's own source is not a copy of it.
        raise BaselineError("the candidate copy does not match the baseline digest")
    return provenance


def _split_path_parts(expr: str) -> list:
    """Split a path expression on ``/`` outside quotes."""
    parts, current, quote = [], [], None
    for char in str(expr):
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == "/":
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _eval_path_expr(expr: str, anchors: dict, tree: Path):
    """Evaluate a path expression made of Path literals, anchors and ``/`` joins."""
    parts = _split_path_parts(expr)
    value = None
    for part in parts:
        piece = None
        literal = re.fullmatch(r'Path\(\s*["\']([^"\']+)["\']\s*\)', part)
        if literal:
            piece = literal.group(1)
        elif re.fullmatch(r'Path\(__file__\)(?:\.resolve\(\))?\.parent', part):
            piece = str(tree)
        elif re.fullmatch(r'f["\']([^"\']+)["\']', part):
            template = re.fullmatch(r'f["\']([^"\']+)["\']', part).group(1)
            filled = re.sub(r"\{(\w+)\}", lambda m: str(anchors.get(m.group(1), m.group(0))),
                            template)
            piece = filled
        elif re.fullmatch(r'["\']([^"\']+)["\']', part):
            piece = re.fullmatch(r'["\']([^"\']+)["\']', part).group(1)
        elif re.fullmatch(r"[A-Za-z_]\w*", part) and part in anchors:
            piece = anchors[part]
        if piece is None:
            return None
        value = Path(piece) if value is None else value / piece
    return str(value) if value is not None else None


def _declared_paths(tree: Path, *, location: Path = None) -> dict:
    """The output paths a trainer declares, resolved through its own anchors.

    A trainer writes chains such as
    ``RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_x"`` and then
    ``OUTPUT_JSON = RESULT_DIR / "item_sids.json"``, so reading only literal
    ``Path("...")`` assignments would miss every output it declares.
    """
    entry = Path(tree) / "train_rqvae.py"
    if not entry.is_file():
        return {}
    text = entry.read_text(encoding="utf-8", errors="replace")
    # The paths are evaluated where the trainer will actually run: the promoted
    # tree under stage2_RQ-VAE/, whose depth differs from the round tree's.
    here = Path(location) if location is not None else Path(tree)
    # An output path anchored on the number of parents of this file works in the
    # round tree and breaks after promotion, where the tree sits one level
    # higher. That is a defect, not a style choice.
    for match in re.finditer(r"^(\w+)\s*=\s*Path\(__file__\)(?:\.resolve\(\))?"
                             r"\.parents\[\d+\]", text, re.M):
        name = match.group(1)
        path_like = (name in _OUTPUT_CONSTANTS
                     or re.search(r"(_ROOT|ROOT|_DIR|DIR|_PATH|PATH|_HERE|_TREE)$", name))
        if path_like:
            raise BaselineError("the output path " + match.group(1) + " depends on the "
                                "source tree's depth (parents[N]); use REPO_ROOT or an "
                                "absolute path under results/stage2_RQ-VAE/")
    # The names a trainer may use for the two anchors it is allowed to start
    # from. Anything else has to be resolvable from these.
    anchors: dict[str, str] = {}
    assignments = re.findall(r"^(\w+)\s*=\s*(.+)$", text, re.M)
    assigned = {name for name, _ in assignments}
    # The file's own definition wins: a REPO_ROOT computed from parents[N] must
    # be evaluated, not shadowed by the fallback.
    for alias in ("REPO_ROOT", "_REPO_ROOT", "ROOT", "_ROOT", "PROJECT_ROOT",
                  "BASE_DIR", "REPO"):
        if alias not in assigned:
            anchors[alias] = "/home/wlia0047/ar57/wenyu/GeneRec"
    for alias in ("TREE_ROOT", "_TREE", "_HERE", "TREE_DIR", "THIS_DIR"):
        if alias not in assigned:
            anchors[alias] = str(here)
    # A plain string constant may be interpolated into an f-string path.
    for name, value in re.findall(r"^(\w+)\s*=\s*[\"']([^\"']*)[\"']\s*$", text, re.M):
        anchors.setdefault(name, value)
    for _ in range(6):                       # resolve chains in dependency order
        changed = False
        for name, expr in assignments:
            if name in anchors:
                continue
            value = _eval_path_expr(expr.strip(), anchors, here)
            if value is not None:
                anchors[name] = value
                changed = True
        if not changed:
            break
    declared = {name: value for name, value in anchors.items()
                if name in _OUTPUT_CONSTANTS or name.endswith("DIR")}
    if not any(name in declared for name in _OUTPUT_CONSTANTS):
        # The preflight cannot check what it cannot resolve: a trainer whose
        # output constants are opaque is refused, not assumed safe.
        raise BaselineError("the preflight cannot resolve any output path of " +
                            str(tree) + "; declare RESULT_DIR as an absolute path "
                            "under results/stage2_RQ-VAE/TIGER_<idea>/")
    return declared


def _resolve(path_text: str, tree: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return (Path(tree) / candidate).resolve()


def results_root(baseline: Path) -> Path:
    """Where the results of the trees under ``stage2_RQ-VAE/`` are written."""
    parent = Path(baseline).resolve().parent
    if parent.name == "stage2_RQ-VAE":
        return parent
    return Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE")


def arm_name(tree: Path) -> str:
    """The name this tree gets when it enters the Stage 2 allowlist."""
    name = Path(tree).name
    return name if name.startswith("TIGER_") else "TIGER_" + name


def guard_output_paths(tree: Path, *, baseline: Path = None,
                       other_candidates=(), arm: str = None,
                       location: Path = None) -> dict:
    """Refuse a candidate that would write into the baseline or a sibling.

    Returns the resolved output paths when the candidate is isolated, and
    raises :class:`BaselineError` when it is not.
    """
    tree = Path(tree).resolve()
    base = (Path(baseline).resolve() if baseline is not None
            else baseline_path().resolve())
    promoted = Path(location) if location is not None else (
        Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE") / (arm or arm_name(tree)))
    declared_all = _declared_paths(tree, location=promoted)
    # A repository root that depends on the file's depth is correct in the round
    # tree and wrong after promotion; it is checked where the trainer will run.
    for name, value in declared_all.items():
        if name in ("REPO_ROOT", "_REPO_ROOT", "ROOT", "_ROOT", "PROJECT_ROOT", "REPO"):
            if Path(value).resolve() != Path("/home/wlia0047/ar57/wenyu/GeneRec").resolve():
                raise BaselineError("the trainer's " + name + " resolves to " + str(value) +
                                    " instead of the repository root once the tree is "
                                    "promoted; use the absolute repository path")
    declared = {name: value for name, value in declared_all.items()
                if name in _OUTPUT_CONSTANTS}
    if not declared:
        raise BaselineError("the candidate declares no output path: " + str(tree))
    # The baseline's own results live next to the other trees, so a candidate
    # that writes into that directory overwrites the baseline's artifacts.
    protected = [base, results_root(base) / base.name]
    for other in other_candidates:
        other = Path(other).resolve()
        protected.append(other)
        protected.append(results_root(base) / other.name)
    # The one place a candidate may write: the repository's Stage 2 results,
    # resolved so the /home and /fs04 spellings of the same directory agree.
    approved = (Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE")).resolve()
    resolved = {}
    for name, text in declared.items():
        path = _resolve(text, tree).resolve()
        resolved[name] = str(path)
        if tree in path.parents:
            # The source tree is read-only: run artifacts inside it would be
            # carried into the next promotion and read as this run's evidence.
            raise BaselineError("the candidate writes into its own source tree: "
                                + name + " -> " + str(path))
        for target in protected:
            target = Path(target).resolve()
            if path == target or target in path.parents:
                raise BaselineError("the candidate would write into the baseline or "
                                    "another candidate: " + name + " -> " + str(path))
        arm_dir = approved / (arm or arm_name(tree))
        if path != arm_dir and arm_dir not in path.parents:
            raise BaselineError("the candidate writes outside its arm directory "
                                + str(arm_dir) + ": " + name + " -> " + str(path))
    return resolved


def guard_baseline_untouched(baseline: Path, digest_before: str) -> None:
    """Fail closed when the baseline changed while a candidate was built."""
    current = source_digest(Path(baseline))
    if current != digest_before:
        raise BaselineError("the baseline changed: " + str(baseline) + " was " +
                            digest_before + " and is now " + current)
