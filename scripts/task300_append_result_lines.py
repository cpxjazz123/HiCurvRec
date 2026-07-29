#!/usr/bin/env python3
"""Append `result: Task #N — <subject>` line to verdicts missing it.

R8 §9.3 mandate: every verdict ends with `result: Task #X — ...` line.
Per task114 A1 audit: 68 verdicts missing this line.

Logic:
- For each *_result.md in verdicts/ that lacks `result:` line:
  - Extract task ID from filename
  - Extract subject from H1 line (`# Task #N — <subject>`)
  - Append `result: Task #N — <subject>` at end

Usage:
    python3 scripts/task300_append_result_lines.py [--check] [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path

VERDICTS = Path("/home/wlia0047/ar57/wenyu/GeneRec/verdicts")
TASK_ID_RE = re.compile(r"task(\d+)")


def extract_subject(text: str, task_id: int) -> str:
    """Extract subject from H1 line: '# Task #N — <subject>' or '# Task #N — <subject> ...'"""
    # Look for H1 line containing Task #N
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# ") and f"Task #{task_id}" in line:
            # Remove leading '#' and split on '—'
            if "—" in line:
                subject = line.split("—", 1)[1].strip()
                # Clean trailing markdown cruft
                return subject
    return f"Task #{task_id} (auto-extracted fallback)"


def needs_result_line(text: str) -> bool:
    return not any(line.startswith("result:") for line in text.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Check only, exit 0 iff all have result: line")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be appended, don't write")
    args = ap.parse_args()

    if not VERDICTS.exists():
        print(f"❌ verdicts/ not found at {VERDICTS}")
        sys.exit(1)

    md_verdicts = sorted(VERDICTS.glob("*_result.md"))
    total = len(md_verdicts)
    missing = []

    for v in md_verdicts:
        text = v.read_text(encoding="utf-8", errors="replace")
        if needs_result_line(text):
            m = TASK_ID_RE.search(v.name)
            task_id = int(m.group(1)) if m else 0
            subject = extract_subject(text, task_id)
            missing.append((v, task_id, subject))

    if args.check:
        if missing:
            print(f"❌ {len(missing)} verdicts missing result: line")
            sys.exit(1)
        print(f"✅ all {total} verdicts have result: line")
        sys.exit(0)

    if not missing:
        print(f"✅ all {total} verdicts already have result: line")
        return

    print(f"Found {len(missing)} verdicts missing result: line")
    for v, tid, subject in missing:
        result_line = f"\nresult: Task #{tid} — {subject}\n"
        if args.dry_run:
            print(f"[DRY-RUN] {v.name}: append '{result_line.strip()}'")
        else:
            with v.open("a", encoding="utf-8") as f:
                f.write(result_line)
            print(f"✓ {v.name}: appended result: Task #{tid} — {subject[:60]}...")


if __name__ == "__main__":
    main()
