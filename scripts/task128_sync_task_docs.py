#!/usr/bin/env python3
"""Task #128/129 — Auto-sync TASKS_INDEX.md Recent closed tasks section.

Scans verdicts/task<N>_*.md (or *_result.md) and regenerates the
"## Recent closed tasks" table in TASKS_INDEX.md. Preserves all other
hand-curated sections (Coverage matrix, Conventions, etc.).

Usage:
    python3 scripts/task128_sync_task_docs.py            # auto diff + write
    python3 scripts/task128_sync_task_docs.py --check   # exit 0 iff 同步
    python3 scripts/task128_sync_task_docs.py --window 50  # show last 50 tasks

Why only Recent closed tasks?
- Coverage matrix groups tasks by semantic range (baselines, paper-defense,
  housekeeping). These groupings are hand-curated and informative; auto-gen
  would lose the semantics.
- Recent closed tasks is purely enumerative; auto-gen keeps it in sync.
- Other sections (intro, three categories, post-submission artifacts,
  conventions) are static and shouldn't be regenerated.

Design:
- Find `## Recent closed tasks (Task #101 - Task #XXX)` header line.
- Replace content until next `## ` header or EOF.
- Default `--window 27` (matches current Task #101 - Task #127 range).
- Subject extraction: read first heading from each verdict file's H1 (`# Task #N — <subject>`).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DESCRIPTIONS = REPO / "descriptions"
VERDICTS = REPO / "verdicts"
TASKS_INDEX = REPO / "TASKS_INDEX.md"


def discover_task_ids() -> list[int]:
    """Scan verdicts/ for task<N>_*.md (or *_result.md) and return sorted N.

    Skips reference files (those without task<N>_ prefix).
    """
    ids: set[int] = set()
    for f in VERDICTS.glob("task[0-9]*_*.md"):
        m = re.match(r"task(\d+)_", f.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def extract_subject(task_id: int) -> str:
    """Read H1 from verdict file and extract subject after task number.

    Format: `# Task #N [optional text] — <subject>` (em-dash or hyphen).
    """
    candidates = _candidate_verdicts(task_id)
    if not candidates:
        return "(no verdict file)"
    text = candidates[0].read_text(encoding="utf-8")
    for line in text.split("\n"):
        # Allow optional words between "Task #N" and the em-dash/hyphen.
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


def _candidate_verdicts(task_id: int) -> list[Path]:
    """Glob + sort verdict candidates: prefer *_result.md, then by name.

    Sort key ensures extract_subject and extract_verdict_filename pick the
    same canonical file (lexically first *_result.md). For tasks with
    multiple verdicts (e.g. #116 has both #116_hgrec_delta and
    #116_track_untracked), this picks alphabetically first.
    """
    candidates = list(VERDICTS.glob(f"task{task_id}_*_result.md"))
    if not candidates:
        candidates = list(VERDICTS.glob(f"task{task_id}_*.md"))
    candidates.sort(key=lambda p: (not p.name.endswith("_result.md"), p.name))
    return candidates


def build_table(task_ids: list[int]) -> str:
    """Build Recent closed tasks markdown table."""
    lines = [
        f"## Recent closed tasks (Task #{task_ids[0]} - Task #{task_ids[-1]})",
        "",
        "| Task | Subject | Status | Verdict |",
        "|------|---------|--------|---------|",
    ]
    for tid in task_ids:
        subject = extract_subject(tid)
        verdict_fn = extract_verdict_filename(tid)
        lines.append(f"| #{tid} | {subject} | ✅ | `verdicts/{verdict_fn}` |")
    lines.append("")
    return "\n".join(lines)


def find_recent_section(text: str) -> tuple[int, int] | None:
    """Find (start, end) line offsets of "## Recent closed tasks" section.

    start: line index of "## Recent closed tasks ..."
    end: line index of next "## " header or EOF
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## Recent closed tasks"):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            return (start, j)
    return (start, len(lines))


def replace_section(text: str, new_section: str) -> str:
    """Replace the Recent closed tasks section with new_section."""
    span = find_recent_section(text)
    if span is None:
        # No existing section — append at end
        if not text.endswith("\n"):
            text += "\n"
        if not text.endswith("\n\n"):
            text += "\n"
        return text + "\n" + new_section + "\n"
    start, end = span
    lines = text.split("\n")
    new_lines = lines[:start] + new_section.split("\n") + lines[end:]
    return "\n".join(new_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Exit 0 iff TASKS_INDEX is in sync, 1 otherwise (no write)")
    parser.add_argument("--from-id", type=int, default=101,
                        help="Show tasks from this ID onwards (default 101 = start of paper-defense period)")
    args = parser.parse_args()

    if not TASKS_INDEX.exists():
        print(f"❌ TASKS_INDEX.md not found at {TASKS_INDEX}")
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
    new_section = build_table(window)
    current_text = TASKS_INDEX.read_text(encoding="utf-8")
    new_text = replace_section(current_text, new_section)

    if new_text == current_text:
        print(f"✅ TASKS_INDEX.md is in sync (last {len(window)} tasks, "
              f"Task #{window[0]} - Task #{window[-1]})")
        return 0

    if args.check:
        print(f"❌ TASKS_INDEX.md is OUT OF SYNC (last {len(window)} tasks, "
              f"Task #{window[0]} - Task #{window[-1]})")
        print("    Run `python3 scripts/task128_sync_task_docs.py` to sync.")
        return 1

    TASKS_INDEX.write_text(new_text, encoding="utf-8")
    print(f"✅ TASKS_INDEX.md updated (last {len(window)} tasks, "
          f"Task #{window[0]} - Task #{window[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
