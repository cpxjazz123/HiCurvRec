"""The improvement library: what one task taught the next one.

A sweep is only a sweep if the harness that won one task is available to the next. This module
owns that record, in the same layout the third-party framework used, because the layout is the
useful part and it costs nothing to keep:

    <library_root>/<benchmark>/k<k>-<variant>/<mode>/<domain>/
        INDEX.md                       every task, the label that won it and why
        harnesses/<task_id>/EXPERIENCE.md   the lesson, with the evidence behind it
        harnesses/<task_id>/<editable files> the winning harness itself

``variant`` names the substrate (``local``, ``opencode``), so two substrates never share a
library by accident.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import surface as surface_mod

__all__ = ["root_for", "Library"]


def root_for(base: Path | str, *, benchmark: str, k: int, variant: str = "",
             tag: str = "") -> Path:
    """``<base>/<benchmark>/k<k>-<variant>``; an explicit tag opts into a shared library."""
    if tag:
        return Path(base) / benchmark / tag
    parts = [f"k{int(k)}"] + ([variant] if variant else [])
    return Path(base) / benchmark / "-".join(parts)


class Library:
    """One mode's library for one domain: the lessons and the harnesses that earned them."""

    def __init__(self, root: Path | str, *, mode: str, domain: str) -> None:
        self.root = Path(root) / mode / domain
        self.mode = mode
        self.domain = domain

    # -- writing ----------------------------------------------------------
    def add(self, task_id: str, tree: Dict[str, str], *, label: str, why: str = "",
            run_id: str = "", experience: Optional[Dict[str, Any]] = None) -> Path:
        """Record one task's winning harness and the lesson it carries."""
        entry = self.root / "harnesses" / task_id
        surface_mod.write_tree(surface_mod.GENREC_SURFACE, tree, entry, stubs=False)
        lesson = (experience or {}).get("lesson") or why or "the harness transferred"
        evidence = (experience or {}).get("evidence") or run_id or label
        entry.joinpath("EXPERIENCE.md").write_text(
            f"# {task_id}\n\n"
            f"- lesson: {lesson}\n"
            f"- evidence: {evidence}\n"
            f"- chosen label: {label}\n"
            f"- run: {run_id}\n",
            encoding="utf-8",
        )
        entry.joinpath("experience.json").write_text(
            json.dumps({"task_id": task_id, "label": label, "why": why, "run_id": run_id,
                        "lesson": lesson, "evidence": evidence,
                        "experience": experience or {}}, indent=2) + "\n", encoding="utf-8")
        self._write_index()
        return entry

    def _entries(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        harnesses = self.root / "harnesses"
        if not harnesses.is_dir():
            return out
        for path in sorted(harnesses.iterdir()):
            record = path / "experience.json"
            if record.is_file():
                out.append(json.loads(record.read_text(encoding="utf-8")))
        return out

    def _write_index(self) -> None:
        lines = [f"# {self.domain} -- mode {self.mode}", "",
                 "| task | chosen label | lesson | evidence |", "|---|---|---|---|"]
        for entry in self._entries():
            lines.append(
                f"| {entry['task_id']} | {entry.get('label', '')} | "
                f"{entry.get('lesson', '')} | {entry.get('evidence', '')} |")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- reading ----------------------------------------------------------
    def lessons(self) -> List[Dict[str, Any]]:
        return self._entries()

    def has_experience(self, task_id: str) -> bool:
        return (self.root / "harnesses" / task_id / "EXPERIENCE.md").is_file()

    def copy_into(self, destination: Path) -> Optional[Path]:
        """Make the library available to the next task's evidence tree."""
        if not self.root.is_dir():
            return None
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "library"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(self.root, target)
        return target
