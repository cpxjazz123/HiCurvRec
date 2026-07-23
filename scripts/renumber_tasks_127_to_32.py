#!/usr/bin/env python3
"""
Task renumbering: 127-130 → 32-35 (fill R9 gap, descriptions/ now continuous 1-35).

Per CLAUDE.md R9: "新任务必须用最大已用编号 + 1", 32-126 是 R9 空洞, 现在填空 32-35.

Updates:
1. descriptions/ filenames + content
2. verdicts/ filenames + content (including task32_130_phonism_4seed_comparison.md → task32_35_phonism_4seed_comparison.md)
3. scripts/ filenames + content
4. logs/ filenames + content
5. products/ dir names (none for 127-130, but check)
6. loop.md content
"""
import os
import re
import shutil
from pathlib import Path

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')

# Mapping: old task number → new task number (127-130 → 32-35)
MAPPING = {
    '127': '32',
    '128': '33',
    '129': '34',
    '130': '35',
}


def replace_task_refs(text: str) -> str:
    """Replace task<old_num> → task<new_num>. Handles 'task23', 'Task #23', '#23', 'T23', etc.

    Uses word boundaries to avoid matching task1270, task1301, etc.
    """
    for old, new in MAPPING.items():
        # task32 → task32 (file name pattern, but must be word boundary)
        # task32_X → task32_X (with underscore after)
        # task1270 → task1270 (should NOT match)
        text = re.sub(r'\btask' + old + r'(?=\D|$)', 'task' + new, text)
        # Task #32 → Task #32 (display pattern with #)
        text = re.sub(r'\bTask\s*#' + old + r'\b', 'Task #' + new, text)
        # #32 → #32 (inline reference)
        text = re.sub(r'(?<!\d)#' + old + r'\b', '#' + new, text)
    return text


def rename_dir_contents(src_root, only_completed=True, file_pattern=r'^task(\d+)_'):
    """Rename files/dirs matching task pattern in src_root.

    For verdicts, also handles the special case task32_130_phonism_4seed_comparison.md
    """
    src_root = Path(src_root)
    if not src_root.exists():
        return 0
    count = 0

    # Get all items sorted REVERSE so we rename higher numbers first (avoid name conflicts)
    items = sorted(src_root.iterdir(), reverse=True)
    for item in items:
        if not item.is_file() and not item.is_dir():
            continue
        name = item.name

        # Special case: task32_130_phonism_4seed_comparison.md → task32_35_phonism_4seed_comparison.md
        special_match = re.match(r'^task(\d+)_(\d+)_(.+)$', name)
        if special_match:
            old_a, old_b, rest = special_match.group(1), special_match.group(2), special_match.group(3)
            if old_a in MAPPING and old_b in MAPPING:
                new_a = MAPPING[old_a]
                new_b = MAPPING[old_b]
                new_name = f'task{new_a}_{new_b}_{rest}'
                new_path = item.parent / new_name
                if new_path.exists() and new_path != item:
                    print(f"  SKIP {name} (target {new_name} exists)")
                    continue
                shutil.move(str(item), str(new_path))
                print(f"  RENAME {name} → {new_name}")
                count += 1
                continue

        # Standard pattern: task32_X → task32_X
        m = re.match(file_pattern, name)
        if not m:
            continue
        old_num = m.group(1)
        if old_num not in MAPPING:
            continue
        new_num = MAPPING[old_num]
        # Replace only the first occurrence (the prefix)
        new_name = re.sub(r'^task' + old_num + r'_', f'task{new_num}_', name, count=1)
        new_path = item.parent / new_name
        if new_path.exists() and new_path != item:
            print(f"  SKIP {name} (target {new_name} exists)")
            continue
        shutil.move(str(item), str(new_path))
        print(f"  RENAME {name} → {new_name}")
        count += 1
    return count


def update_file_content(filepath):
    """Update task<old> references in file content."""
    fp = Path(filepath)
    try:
        content = fp.read_text(encoding='utf-8')
    except (UnicodeDecodeError, FileNotFoundError, PermissionError, IsADirectoryError):
        return False
    new_content = replace_task_refs(content)
    if new_content != content:
        fp.write_text(new_content, encoding='utf-8')
        return True
    return False


def update_all_content():
    """Update content in all relevant files."""
    targets = []

    # descriptions/, verdicts/, scripts/ - all .md, .json, .py, .sh
    for subdir in ['descriptions', 'verdicts', 'scripts']:
        d = ROOT / subdir
        if not d.exists():
            continue
        for ext in ['*.md', '*.json', '*.py', '*.sh']:
            targets.extend(d.glob(ext))

    # logs/ - all files
    logs_dir = ROOT / 'logs'
    if logs_dir.exists():
        for f in logs_dir.iterdir():
            if f.is_file():
                targets.append(f)

    # loop.md
    loop_md = ROOT / 'loop.md'
    if loop_md.exists():
        targets.append(loop_md)

    # CLAUDE.md (might have task refs)
    claude_md = ROOT / 'CLAUDE.md'
    if claude_md.exists():
        targets.append(claude_md)

    updated = 0
    for fp in targets:
        if update_file_content(fp):
            updated += 1
            print(f"  UPDATED content: {fp.name}")
    return updated


def main():
    print("=" * 60)
    print("Task Renumbering: 127-130 → 32-35 (fill R9 gap)")
    print("=" * 60)
    print(f"Mapping: {MAPPING}")
    print()

    # 1. Rename descriptions/
    print("\n[1] Rename descriptions/")
    n = rename_dir_contents(ROOT / 'descriptions')
    print(f"  Renamed {n} files")

    # 2. Rename verdicts/
    print("\n[2] Rename verdicts/")
    n = rename_dir_contents(ROOT / 'verdicts')
    print(f"  Renamed {n} files")

    # 3. Rename scripts/
    print("\n[3] Rename scripts/")
    n = rename_dir_contents(ROOT / 'scripts')
    print(f"  Renamed {n} files")

    # 4. Rename logs/
    print("\n[4] Rename logs/")
    n = rename_dir_contents(ROOT / 'logs')
    print(f"  Renamed {n} files")

    # 5. Check products/ (no task32-130 entries expected)
    print("\n[5] Rename products/ (task32-130 expected none)")
    n = rename_dir_contents(ROOT / 'products')
    print(f"  Renamed {n} dirs (expect 0)")

    # 6. Update file content
    print("\n[6] Update file content (task refs in descriptions/verdicts/scripts/logs/loop.md/CLAUDE.md)")
    updated = update_all_content()
    print(f"  Updated {updated} files")

    print("\n" + "=" * 60)
    print("Renumbering complete")
    print("=" * 60)


if __name__ == '__main__':
    main()