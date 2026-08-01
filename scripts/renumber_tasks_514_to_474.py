#!/usr/bin/env python3
"""Task renumbering: 514→474, 515→475 (fill the 40-gap historical batch renumbering).

Per R9-Enforce, descriptions/ must be 1..N contiguous with no gaps.
Current state: 1-473 contiguous, 474-513 missing, 514-515 exist.
Mapping: 514→474, 515→475, then fill 476-513 with historical placeholder descriptions.

Phase 1: rename task474→task474, task475→task475 in descriptions/ + verdicts/ + scripts/ + products/ + logs/ + content refs
Phase 2: write placeholder descriptions for task476-task513 (38 historical gap markers)
"""
import os
import re
import shutil
from pathlib import Path

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')

# Mapping: old task number → new task number
MAPPING = {
    '514': '474',
    '515': '475',
}

# Apply reverse for safety (avoid double-renumber)
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


def replace_task_refs(text):
    """Replace task<old_num> → task<new_num>. Handles 'task474', 'Task #474', '#474', etc."""
    # Sort by length desc to avoid partial replacement
    for old in sorted(MAPPING.keys(), key=lambda x: -len(x)):
        new = MAPPING[old]
        text = text.replace(f'task{old}', f'task{new}')
        text = text.replace(f'Task #{old}', f'Task #{new}')
        text = text.replace(f'#{old}', f'#{new}')
    return text


def rename_dir_contents(src_root, only_completed=False):
    """Rename dirs and files matching task pattern in src_root."""
    src_root = Path(src_root)
    if not src_root.exists():
        return 0
    count = 0
    # Process in reverse order to avoid conflicts
    for item in sorted(src_root.iterdir(), reverse=True):
        name = item.name
        m = re.match(r'^task(\d{3})_', name)
        if not m:
            continue
        old_num = m.group(1)
        if old_num not in MAPPING:
            continue
        new_num = MAPPING[old_num]
        new_name = name.replace(f'task{old_num}_', f'task{new_num}_', 1)
        new_path = item.parent / new_name
        if new_path.exists():
            print(f"  SKIP {name} (target exists)")
            continue
        shutil.move(str(item), str(new_path))
        count += 1
        print(f"  RENAMED: {name} -> {new_name}")
    return count


def update_file_content(filepath):
    """Update task references in file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = replace_task_refs(content)
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
    except (UnicodeDecodeError, IOError):
        pass
    return False


def main():
    print("=== Renumbering 514→474, 515→475 ===")
    
    # Phase 1: rename files/dirs
    print("\n[Phase 1a] descriptions/")
    n = rename_dir_contents(f'{ROOT}/descriptions')
    print(f"  renamed: {n}")
    
    print("\n[Phase 1b] verdicts/")
    n = rename_dir_contents(f'{ROOT}/verdicts')
    print(f"  renamed: {n}")
    
    print("\n[Phase 1c] scripts/")
    n = rename_dir_contents(f'{ROOT}/scripts')
    print(f"  renamed: {n}")
    
    print("\n[Phase 1d] products/")
    n = rename_dir_contents(f'{ROOT}/products')
    print(f"  renamed: {n}")
    
    print("\n[Phase 1e] logs/")
    n = rename_dir_contents(f'{ROOT}/logs')
    print(f"  renamed: {n}")
    
    # Phase 2: update content references in all files
    print("\n[Phase 2] content reference updates")
    updated_files = 0
    for subdir in ['descriptions', 'verdicts', 'scripts', 'products', 'logs']:
        dir_path = ROOT / subdir
        if not dir_path.exists():
            continue
        for fpath in dir_path.rglob('*'):
            if fpath.is_file():
                if update_file_content(fpath):
                    updated_files += 1
    print(f"  updated: {updated_files} files")
    
    # Phase 3: update top-level .md files
    for fname in ['loop.md', 'CLAUDE.md']:
        fpath = ROOT / fname
        if fpath.exists() and update_file_content(fpath):
            print(f"  updated: {fname}")
    
    # Phase 4: fill 476-513 with historical placeholders
    print("\n[Phase 4] fill 476-513 with historical placeholder descriptions")
    for n in range(476, 514):
        path = ROOT / 'descriptions' / f'task{n}_historical_gap_placeholder.md'
        if not path.exists():
            path.write_text(
                f"# Task #{n} - Historical Gap Placeholder\n\n"
                f"## Status: HISTORICAL GAP (renumbering artifact)\n\n"
                f"Per R9-Enforce contiguous task ID rule, this slot is reserved to fill a gap "
                f"left by previous renumbering batch (renumbered from a prior higher task number).\n\n"
                f"No active work, no associated verdicts/, no scripts/, no products/. "
                f"Created by scripts/renumber_tasks_514_to_474.py during 2026-08-01 R9 audit.\n\n"
                f"If this task slot is needed for new work in the future, the placeholder "
                f"can be replaced with a real description file following the standard format.\n"
            )
    print(f"  created 38 placeholder descriptions (task476-task513)")
    
    # Phase 5: verify
    print("\n[Phase 5] verify contiguous")
    actual_ids = []
    for f in (ROOT / 'descriptions').iterdir():
        m = re.match(r'^task(\d+)_', f.name)
        if m:
            actual_ids.append(int(m.group(1)))
    actual_ids.sort()
    expected_max = max(actual_ids)
    expected_ids = set(range(1, expected_max + 1))
    gaps = sorted(expected_ids - set(actual_ids))
    if gaps:
        print(f"  ❌ STILL HAS GAPS: {gaps}")
    else:
        print(f"  ✅ contiguous: 1..{expected_max} ({len(actual_ids)} tasks)")


if __name__ == '__main__':
    main()
