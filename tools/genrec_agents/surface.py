"""The candidate-editable surface of the GeneRec iteration harness, and the tree helpers.

Issue #101 fixes the three editable kinds of file:

* ``CLAUDE.md`` -- the always-on protocol (c1),
* ``.agent/skills/<name>/SKILL.md`` -- the reusable workflows (c6),
* ``scripts/<name>.py`` -- the helper scripts the skills invoke (c7).

Frozen math, thresholds, oracles, judges and trainer evaluation code are NOT on this
surface: the harness reads a tree, validates it, and only the paths declared here can
ever reach a candidate.

The surface specification and the tree helpers used to come from the third-party
framework (``harness2.core.tree``). They are declared here instead, with the same
fields and the same semantics, so that no active module imports anything outside this
repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

__all__ = ["GENREC_SURFACE", "SurfaceSpec", "normalize_rel", "is_allowed", "read_tree",
           "is_tree", "write_tree", "always_on_text", "skill_names", "subagent_names"]

#: A tree is a mapping from a repository-relative path to the file's content.
Tree = Dict[str, str]

_NAME = r"[a-z0-9][a-z0-9_\-]*"


@dataclass(frozen=True)
class SurfaceSpec:
    """The editable surface of one agent framework: paths, caps and per-surface rules."""

    name: str
    always_on: Tuple[str, ...]
    always_on_max: int
    always_on_cap_reason: str
    skill_dir: str
    skill_name_rule: str
    trigger_hint: str
    autotrigger_file: str
    trigger_line: str
    subagent_requires_mode: bool
    plugin_dir: Optional[str]
    hooks_file: Optional[str]
    hook_events: FrozenSet[str] = frozenset()
    compile_skill_scripts: bool = False
    reject_empty_always_on: bool = False
    stubs: Tuple[Tuple[str, str], ...] = ()


GENREC_SURFACE = SurfaceSpec(
    name="genrec",
    always_on=("CLAUDE.md",),
    always_on_max=60_000,
    always_on_cap_reason=(
        "the protocol competes with the agent's own reasoning budget; keep it under the cap"
    ),
    skill_dir=".agent/skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Add a trigger line naming `{name}` to CLAUDE.md, e.g. "
        '"When <the situation this skill handles>, load the `{name}` skill."'
    ),
    autotrigger_file="CLAUDE.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    compile_skill_scripts=False,
    reject_empty_always_on=True,
)


@lru_cache(maxsize=8)
def _res(surface_name: str, skill_dir: str, plugin_dir: Optional[str]) -> Dict[str, re.Pattern]:
    sd = re.escape(skill_dir)
    out = {
        "skill": re.compile(rf"{sd}/({_NAME})/SKILL\.md"),
        "skill_script": re.compile(rf"{sd}/{_NAME}/scripts/[A-Za-z0-9_.\-]+"),
        "subagent": re.compile(rf"sub_agents/({_NAME})/AGENT\.md"),
        "script": re.compile(r"scripts/[a-z0-9_][a-z0-9_]*\.py"),
        "script_ref": re.compile(rf"(?:python3?\s+)((?:{sd}/[\w\-]+/)?scripts/[\w.\-]+\.py)"),
    }
    if plugin_dir is not None:
        out["plugin"] = re.compile(rf"{re.escape(plugin_dir)}/({_NAME})\.ts")
    return out


def _rx(surface: SurfaceSpec) -> Dict[str, re.Pattern]:
    return _res(surface.name, surface.skill_dir, surface.plugin_dir)


def normalize_rel(rel: str) -> str:
    """Normalise a relative path, rejecting escapes before it is joined to a directory."""
    r = str(rel).strip().replace("\\", "/")
    if r.startswith("/") or ".." in r.split("/"):
        raise ValueError(f"path escapes the harness: {rel!r}")
    while r.startswith("./"):
        r = r[2:]
    return r.strip("/")


def is_allowed(surface: SurfaceSpec, rel: str) -> bool:
    """Whether this path is on the editable surface. A predicate: it never raises."""
    try:
        rel = normalize_rel(rel)
    except ValueError:
        return False
    if rel in surface.always_on:
        return True
    if surface.hooks_file is not None and rel == surface.hooks_file:
        return True
    r = _rx(surface)
    if (
        r["skill"].fullmatch(rel)
        or r["skill_script"].fullmatch(rel)
        or r["subagent"].fullmatch(rel)
        or r["script"].fullmatch(rel)
    ):
        return True
    return "plugin" in r and bool(r["plugin"].fullmatch(rel))


def skill_names(surface: SurfaceSpec, tree: Tree) -> List[str]:
    rx = _rx(surface)["skill"]
    return sorted({m.group(1) for rel in tree if (m := rx.fullmatch(rel))})


def subagent_names(surface: SurfaceSpec, tree: Tree) -> List[str]:
    rx = _rx(surface)["subagent"]
    return sorted({m.group(1) for rel in tree if (m := rx.fullmatch(rel))})


def _stub_paths(surface: SurfaceSpec) -> FrozenSet[str]:
    return frozenset(rel for rel, _ in surface.stubs)


def read_tree(surface: SurfaceSpec, d: Path | str, *, rejected: Optional[List[str]] = None) -> Tree:
    """rel -> content for every file on the editable surface, README stubs excluded.

    Pass `rejected` to collect paths that exist on disk but are off the surface.
    """
    d = Path(d)
    out: Tree = {}
    if not d.is_dir():
        return out
    stubs = _stub_paths(surface)
    for p in sorted(d.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(d))
        if rel in stubs:
            continue
        if not is_allowed(surface, rel):
            if rejected is not None:
                rejected.append(rel)
            continue
        out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def is_tree(surface: SurfaceSpec, d: Path | str) -> bool:
    d = Path(d)
    return d.is_dir() and any((d / a).is_file() for a in surface.always_on)


def write_tree(surface: SurfaceSpec, tree: Tree, dest: Path | str, *, stubs: bool = True) -> Path:
    """Materialise an exact editable tree, with always-on files and optional stubs present.

    Files outside the editable surface are preserved; editable files already there are not.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stub_set = _stub_paths(surface)
    for p in sorted(dest.rglob("*"), reverse=True):
        if not p.is_file():
            continue
        rel = str(p.relative_to(dest))
        if is_allowed(surface, rel) or rel in stub_set:
            p.unlink()
    for a in surface.always_on:
        p = dest / a
        p.parent.mkdir(parents=True, exist_ok=True)
        if a not in tree and not p.exists():
            p.write_text("", encoding="utf-8")
    for rel, content in tree.items():
        p = dest / normalize_rel(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    if stubs:
        for rel, body in surface.stubs:
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text(body, encoding="utf-8")
    return dest


def always_on_text(surface: SurfaceSpec, tree: Tree) -> str:
    return "\n\n".join(tree.get(a, "") for a in surface.always_on)
