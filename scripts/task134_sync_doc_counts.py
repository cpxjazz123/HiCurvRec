#!/usr/bin/env python3
"""Task #134 — Auto-sync README.md + TASKS_INDEX.md count numbers.

Refreshes inline count numbers that drift every time a task closes:
- README.md line 58 "(N entries, R9 continuous)" → descriptions unique task IDs
- README.md line 61 "(~N files)" → scripts/ file count
- TASKS_INDEX.md line 3 "Snapshot: YYYY-MM-DD (post Task #N)" → max task ID
- TASKS_INDEX.md line 8-9 "N numbered files covering M unique task IDs
  (Task #1 - Task #N)" → actual numbered + unique + max
- TASKS_INDEX.md line 13 "N files — verdicts for Task #1 - Task #N + ..." → actual verdicts count + max

Usage:
    python3 scripts/task134_sync_doc_counts.py            # auto diff + write
    python3 scripts/task134_sync_doc_counts.py --check   # exit 0 iff 同步

Why replace (not append)?
- README.md counts are inline numbers in prose; append would break the
  tree-style listing format.
- TASKS_INDEX.md header block is purely descriptive; coverage matrix
  below is hand-curated and untouched.
- Class Task #129 (TASKS_INDEX "Recent closed tasks" replace pattern)
  and Task #133 (CHANGELOG append-only pattern).

Differences from Task #129/133:
- Numbers refresh (no enumeration, just `len()` of file lists)
- No _candidate_verdicts() helper needed (file count, not content extract)
- Snapshot date inferred from max_task_id (latest closed task == snapshot reference)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DESCRIPTIONS = REPO / "descriptions"
VERDICTS = REPO / "verdicts"
SCRIPTS = REPO / "scripts"
README = REPO / "README.md"
TASKS_INDEX = REPO / "TASKS_INDEX.md"


def _run(cmd: list[str]) -> str:
    """Run shell command and return stdout (stripped)."""
    return subprocess.run(cmd, cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def compute_counts() -> dict[str, int]:
    """Compute actual file counts from repo filesystem."""
    desc_files = _run(["ls", str(DESCRIPTIONS)]).splitlines()
    numbered = [f for f in desc_files if re.match(r"^task\d+_", f)]
    unique_ids = sorted({int(re.match(r"^task(\d+)_", f).group(1))
                         for f in numbered})
    return {
        "n_descriptions_total": len(desc_files),
        "n_descriptions_numbered": len(numbered),
        "n_descriptions_unique_task_ids": len(unique_ids),
        "max_task_id": unique_ids[-1] if unique_ids else 0,
        "n_verdicts_total": len(_run(["ls", str(VERDICTS)]).splitlines()),
        "n_scripts_total": len(_run(["ls", str(SCRIPTS)]).splitlines()),
    }


def apply_to_readme(text: str, counts: dict[str, int]) -> str:
    """Replace README.md line 58/61 count numbers."""
    # Line 58: descriptions/ count + R9 continuous
    new_line_58 = (
        f"├── descriptions/                  # Task definitions "
        f"({counts['n_descriptions_unique_task_ids']} entries, R9 continuous)"
    )
    text = re.sub(
        r"^├── descriptions/\s+# Task definitions \(\d+ entries, R9 continuous\)\s*$",
        new_line_58, text, count=1, flags=re.MULTILINE,
    )
    # Line 61: scripts/ count
    new_line_61 = (
        f"└── scripts/                       # Diagnostic & analysis scripts "
        f"(~{counts['n_scripts_total']} files)"
    )
    text = re.sub(
        r"^└── scripts/\s+# Diagnostic & analysis scripts \(~\d+ files\)\s*$",
        new_line_61, text, count=1, flags=re.MULTILINE,
    )
    return text


def apply_to_tasks_index(text: str, counts: dict[str, int]) -> str:
    """Replace TASKS_INDEX.md header count numbers."""
    # Line 3: Snapshot date + post Task #N (may have trailing text like "final state closure")
    new_snapshot_prefix = (
        f"> **Snapshot**: 2026-07-24 (post Task #{counts['max_task_id']}"
    )
    text = re.sub(
        r"^>\s+\*\*Snapshot\*\*:\s+\d{4}-\d{2}-\d{2} \(post Task #\d+",
        new_snapshot_prefix, text, count=1, flags=re.MULTILINE,
    )
    # Line 8: descriptions/ total + numbered files (single line, ending with "— N")
    # Format: "- **descriptions/**: ... N files total — M"
    new_desc_total_line = (
        f"- **descriptions/**: task definitions (markdown). "
        f"{counts['n_descriptions_total']} files total — "
        f"{counts['n_descriptions_numbered']}"
    )
    text = re.sub(
        r"^-\s+\*\*descriptions/[^.]+\.\s+\d+ files total — \d+\s*$",
        new_desc_total_line, text, count=1, flags=re.MULTILINE,
    )
    # Line 9: numbered files covering + unique task IDs + range
    new_desc_numbered_prefix = (
        f"  numbered files covering {counts['n_descriptions_unique_task_ids']} "
        f"unique task IDs (Task #1 - Task #{counts['max_task_id']}"
    )
    text = re.sub(
        r"^\s+numbered files covering \d+ unique task IDs \(Task #\d+ - Task #\d+",
        new_desc_numbered_prefix, text, count=1, flags=re.MULTILINE,
    )
    # Line 13: verdicts/ count + range
    new_verdicts_prefix = (
        f"  {counts['n_verdicts_total']} files — verdicts for "
        f"Task #1 - Task #{counts['max_task_id']}"
    )
    text = re.sub(
        r"^\s+\d+ files — verdicts for Task #\d+ - Task #\d+",
        new_verdicts_prefix, text, count=1, flags=re.MULTILINE,
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Exit 0 iff README.md + TASKS_INDEX.md counts are in sync")
    args = parser.parse_args()

    if not README.exists():
        print(f"❌ README.md not found at {README}")
        return 1
    if not TASKS_INDEX.exists():
        print(f"❌ TASKS_INDEX.md not found at {TASKS_INDEX}")
        return 1

    counts = compute_counts()
    summary = (
        f"descriptions: {counts['n_descriptions_total']} files / "
        f"{counts['n_descriptions_unique_task_ids']} unique IDs (max #{counts['max_task_id']}), "
        f"verdicts: {counts['n_verdicts_total']} files, scripts: {counts['n_scripts_total']} files"
    )

    readme_text = README.read_text(encoding="utf-8")
    new_readme = apply_to_readme(readme_text, counts)
    readme_changed = new_readme != readme_text

    ti_text = TASKS_INDEX.read_text(encoding="utf-8")
    new_ti = apply_to_tasks_index(ti_text, counts)
    ti_changed = new_ti != ti_text

    if not readme_changed and not ti_changed:
        print(f"✅ README.md + TASKS_INDEX.md counts are in sync ({summary})")
        return 0

    if args.check:
        changes = []
        if readme_changed:
            changes.append("README.md")
        if ti_changed:
            changes.append("TASKS_INDEX.md")
        print(f"❌ {' + '.join(changes)} counts OUT OF SYNC ({summary})")
        print(f"    Run `python3 scripts/task134_sync_doc_counts.py` to sync.")
        return 1

    if readme_changed:
        README.write_text(new_readme, encoding="utf-8")
        print(f"✅ README.md counts refreshed ({summary})")
    if ti_changed:
        TASKS_INDEX.write_text(new_ti, encoding="utf-8")
        print(f"✅ TASKS_INDEX.md counts refreshed ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())