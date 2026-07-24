#!/usr/bin/env python3
"""Task #133 — Auto-sync CHANGELOG.md [Unreleased] section.

Scans verdicts/task<N>_*.md (or *_result.md) and appends a
"### Recent housekeeping (auto-generated)" subsection at the end of
the [Unreleased] block. Preserves all hand-curated sections
(Added/Changed/Verified/Notes). Reuses Task #129 _candidate_verdicts()
+ H1 regex pattern for consistency.

Usage:
    python3 scripts/task133_sync_changelog.py            # auto append + write
    python3 scripts/task133_sync_changelog.py --check   # exit 0 iff 同步
    python3 scripts/task133_sync_changelog.py --from-id 121  # post-submission start

Why append (not replace)?
- [Unreleased] already contains hand-curated Added/Changed/Verified/Notes
  subsections (Task #121 synthesis). Replacing them would lose that
  semantic context. Appending "Recent housekeeping (auto-generated)"
  keeps hand content + adds machine-generated completeness check.
- Class TASKS_INDEX "Recent closed tasks" pattern (Task #129) — auto-gen
  only the enumerative part.

Orphan verdicts note:
- discover_task_ids() includes all verdict files (e.g. task134 R9-Enforce
  legacy verdict, task16_* legacy verdicts) even when descriptions/ has no
  matching entry (R9 only enforces descriptions/ contiguous 1..N; verdicts/
  orphan is warning-only). This matches Task #129 sync behaviour for
  TASKS_INDEX. To clean up orphan verdicts, file a separate renumbering
  task (out of scope for Task #133).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DESCRIPTIONS = REPO / "descriptions"
VERDICTS = REPO / "verdicts"
CHANGELOG = REPO / "CHANGELOG.md"

AUTO_SUBSECTION_HEADER = "### Recent housekeeping (auto-generated)"


def discover_task_ids() -> list[int]:
    """Scan verdicts/ for task<N>_*.md (or *_result.md) and return sorted N."""
    ids: set[int] = set()
    for f in VERDICTS.glob("task[0-9]*_*.md"):
        m = re.match(r"task(\d+)_", f.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _candidate_verdicts(task_id: int) -> list[Path]:
    """Glob + sort verdict candidates.

    Sort priority:
    1. Verdict filename shares slug with `descriptions/task<N>_<slug>*.md` (newest)
    2. `*_result.md` suffix preferred (canonical verdict files)
    3. Alphabetical by name (final tiebreaker)

    Why description-slug match priority?
    - For tasks with multiple verdicts (e.g. #116, #134), the verdict whose
      filename matches the current description's slug is the canonical one.
      Legacy orphan verdicts (e.g. verdicts/task134_r9_enforce_audit_result.md
      from renumbering) lose to the live verdict (task134_readme_tasks_*
      _sync_result.md) when slug match takes priority.
    - Without this, lexically first orphan wins (e.g. "r9" < "re" sorts first),
      causing CHANGELOG/TASKS_INDEX to display stale orphan text instead of
      the current Task #N's subject.
    """
    candidates = list(VERDICTS.glob(f"task{task_id}_*_result.md"))
    if not candidates:
        candidates = list(VERDICTS.glob(f"task{task_id}_*.md"))

    # Find description slug for this task (if any)
    desc_files = list(DESCRIPTIONS.glob(f"task{task_id}_*.md"))
    desc_slugs = {f.stem.replace(f"task{task_id}_", "") for f in desc_files}

    def sort_key(p: Path) -> tuple:
        stem = p.stem
        # Strip "_result" suffix for slug comparison
        if stem.endswith("_result"):
            stem = stem[:-len("_result")]
        slug = stem.replace(f"task{task_id}_", "")
        slug_match = 0 if slug in desc_slugs else 1
        result_pref = 0 if p.name.endswith("_result.md") else 1
        return (slug_match, result_pref, p.name)

    candidates.sort(key=sort_key)
    return candidates


def extract_subject(task_id: int) -> str:
    """Read H1 from verdict file and extract subject after task number."""
    candidates = _candidate_verdicts(task_id)
    if not candidates:
        return "(no verdict file)"
    text = candidates[0].read_text(encoding="utf-8")
    for line in text.split("\n"):
        m = re.match(rf"^#\s*Task\s*#{task_id}\b.*?[—\-]\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return "(no H1 found)"


def extract_verdict_filename(task_id: int) -> str:
    """Return canonical verdict filename for the row."""
    candidates = _candidate_verdicts(task_id)
    if candidates:
        return candidates[0].name
    return f"task{task_id}_*.md (missing)"


def build_subsection(task_ids: list[int]) -> str:
    """Build Recent housekeeping auto-gen markdown subsection."""
    lines = [
        AUTO_SUBSECTION_HEADER,
        "",
        "| Task | Subject | Verdict |",
        "|------|---------|---------|",
    ]
    for tid in task_ids:
        subject = extract_subject(tid)
        verdict_fn = extract_verdict_filename(tid)
        lines.append(f"| #{tid} | {subject} | `verdicts/{verdict_fn}` |")
    lines.append("")
    lines.append("_Auto-generated by `scripts/task133_sync_changelog.py` — "
                 "DO NOT hand-edit. Run `python3 scripts/task133_sync_changelog.py` "
                 "after closing a task._")
    lines.append("")
    return "\n".join(lines)


def find_unreleased_section(text: str) -> tuple[int, int] | None:
    """Find (start, end) line offsets of [Unreleased] section.

    start: line index of "## [Unreleased]"
    end: line index of next "## [" header (e.g., [1.0.0]) or EOF
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## [Unreleased]"):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## ["):
            return (start, j)
    return (start, len(lines))


def find_existing_subsection(text: str, unreleased_start: int, unreleased_end: int) -> int | None:
    """Find line index of existing auto-gen subsection, or None.

    Scans only within [Unreleased] section (between unreleased_start and
    unreleased_end) to avoid matching similar headers elsewhere.
    """
    lines = text.split("\n")
    for i in range(unreleased_start, unreleased_end):
        if lines[i].startswith(AUTO_SUBSECTION_HEADER):
            return i
    return None


def replace_or_append_subsection(text: str, new_subsection: str) -> tuple[str, bool]:
    """Replace existing auto-gen subsection or append to [Unreleased].

    Returns (new_text, replaced_existing).
    """
    span = find_unreleased_section(text)
    if span is None:
        # No [Unreleased] section — append at end (graceful fallback)
        if not text.endswith("\n"):
            text += "\n"
        if not text.endswith("\n\n"):
            text += "\n"
        text += "## [Unreleased]\n\n" + new_subsection + "\n"
        return text, False

    start, end = span
    lines = text.split("\n")
    existing_idx = find_existing_subsection(text, start, end)
    if existing_idx is not None:
        # Find end of existing subsection (next ### or ## or end of [Unreleased])
        subsection_end = end
        for j in range(existing_idx + 1, end):
            if lines[j].startswith("### ") or lines[j].startswith("## "):
                subsection_end = j
                break
        new_lines = lines[:existing_idx] + new_subsection.split("\n") + lines[subsection_end:]
        return "\n".join(new_lines), True

    # Append at end of [Unreleased] section (before ## [1.0.0])
    new_lines = lines[:end] + ["", ""] + new_subsection.split("\n") + lines[end:]
    return "\n".join(new_lines), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Exit 0 iff CHANGELOG [Unreleased] auto-subsection is in sync, 1 otherwise")
    parser.add_argument("--from-id", type=int, default=121,
                        help="Show tasks from this ID onwards (default 121 = start of post-submission housekeeping)")
    args = parser.parse_args()

    if not CHANGELOG.exists():
        print(f"❌ CHANGELOG.md not found at {CHANGELOG}")
        return 1
    if not VERDICTS.exists():
        print(f"❌ verdicts/ not found at {VERDICTS}")
        return 1

    task_ids = discover_task_ids()
    if not task_ids:
        print("⚠️  No task IDs found in verdicts/")
        return 0

    window = [t for t in task_ids if t >= args.from_id]
    if not window:
        print(f"⚠️  No tasks >= #{args.from_id}")
        return 0
    new_subsection = build_subsection(window)
    current_text = CHANGELOG.read_text(encoding="utf-8")
    new_text, replaced = replace_or_append_subsection(current_text, new_subsection)

    if new_text == current_text:
        action = "replaced" if replaced else "appended"
        print(f"✅ CHANGELOG.md [Unreleased] auto-subsection is in sync "
              f"(action={action}, {len(window)} tasks, "
              f"Task #{window[0]} - Task #{window[-1]})")
        return 0

    if args.check:
        print(f"❌ CHANGELOG.md [Unreleased] auto-subsection is OUT OF SYNC "
              f"({len(window)} tasks, Task #{window[0]} - Task #{window[-1]})")
        print("    Run `python3 scripts/task133_sync_changelog.py` to sync.")
        return 1

    CHANGELOG.write_text(new_text, encoding="utf-8")
    action = "Replaced" if replaced else "Appended"
    print(f"✅ {action} CHANGELOG.md [Unreleased] auto-subsection "
          f"({len(window)} tasks, Task #{window[0]} - Task #{window[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())