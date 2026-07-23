#!/usr/bin/env python3
"""
Task renumbering: 104-117 → 23-35 (continuous from 23)
Phase 1: descriptions/, verdicts/, scripts/, completed products/logs/, content refs
Phase 2 (deferred): active RQ-VAE jobs (#26/#28/#29/#30 = 108/110/111/112) — wait until Stage 2.1 done
"""
import os
import re
import shutil
from pathlib import Path

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')

# Mapping: old task number → new task number (104-117 → 23-35)
MAPPING = {
    '104': '23', '105': '24',
    '107': '25',
    '108': '26', '109': '27', '110': '28', '111': '29', '112': '30',
    '113': '31', '114': '32', '115': '33', '116': '34', '117': '35',
}

# Active jobs to skip in Phase 1 (logs/products dirs for these will be renumbered in Phase 2)
ACTIVE_JOBS = {'108', '110', '111', '112'}  # = 26, 28, 29, 30 in new numbering

def replace_task_refs(text):
    """Replace task<old_num> → task<new_num>. Handles 'task23', 'Task #23', '#23', 'T23', etc."""
    for old, new in MAPPING.items():
        # task23 → task23 (file name pattern)
        text = text.replace(f'task{old}', f'task{new}')
        # Task #23 → Task #23 (display pattern with #)
        text = text.replace(f'Task #{old}', f'Task #{new}')
        # #23 → #23 (inline reference)
        text = text.replace(f'#{old}', f'#{new}')
        # T23 prefix (sometimes used in plots/labels)
        text = text.replace(f'T{old}', f'T{new}')
    return text

def rename_dir_contents(src_root, only_completed=True):
    """Rename dirs and files matching task pattern in src_root."""
    src_root = Path(src_root)
    if not src_root.exists():
        return 0
    count = 0
    for item in sorted(src_root.iterdir(), reverse=True):
        name = item.name
        m = re.match(r'^task(\d{3})_', name)
        if not m:
            continue
        old_num = m.group(1)
        if old_num not in MAPPING:
            continue
        if only_completed and old_num in ACTIVE_JOBS:
            continue
        new_num = MAPPING[old_num]
        new_name = name.replace(f'task{old_num}_', f'task{new_num}_', 1)
        new_path = item.parent / new_name
        if new_path.exists():
            print(f"  SKIP {name} (target exists)")
            continue
        shutil.move(str(item), str(new_path))
        print(f"  RENAME {name} → {new_name}")
        count += 1
    return count

def update_file_content(filepath, skip_active=False):
    """Update task<old> references in file content."""
    fp = Path(filepath)
    try:
        content = fp.read_text(encoding='utf-8')
    except (UnicodeDecodeError, FileNotFoundError, PermissionError):
        return False
    new_content = replace_task_refs(content)
    if new_content != content:
        fp.write_text(new_content, encoding='utf-8')
        return True
    return False

def main():
    print("=" * 60)
    print("Phase 1 Task Renumbering: 104-117 → 23-35")
    print("=" * 60)

    # 1. Rename descriptions/
    print("\n[1] Rename descriptions/")
    n = rename_dir_contents(ROOT / 'descriptions', only_completed=False)
    print(f"  Renamed {n} files")

    # 2. Rename verdicts/
    print("\n[2] Rename verdicts/")
    n = rename_dir_contents(ROOT / 'verdicts', only_completed=False)
    print(f"  Renamed {n} files")

    # 3. Rename scripts/
    print("\n[3] Rename scripts/")
    n = rename_dir_contents(ROOT / 'scripts', only_completed=False)
    print(f"  Renamed {n} files")

    # 4. Rename completed products/
    print("\n[4] Rename completed products/")
    n = rename_dir_contents(ROOT / 'products', only_completed=True)
    print(f"  Renamed {n} dirs")

    # 5. Rename completed logs/
    print("\n[5] Rename completed logs/ (skip active jobs)")
    n = rename_dir_contents(ROOT / 'logs', only_completed=True)
    print(f"  Renamed {n} items")

    # 6. Update content in all markdown/json/python files
    print("\n[6] Update file content (task refs)")
    targets = []
    for subdir in ['descriptions', 'verdicts', 'scripts']:
        d = ROOT / subdir
        if d.exists():
            targets.extend(d.glob('*.md'))
            targets.extend(d.glob('*.json'))
            targets.extend(d.glob('*.py'))
    loop_md = ROOT / 'loop.md'
    if loop_md.exists():
        targets.append(loop_md)

    updated = 0
    for fp in targets:
        if update_file_content(fp):
            updated += 1
    print(f"  Updated {updated} files")

    # 7. Update content in completed logs (not active)
    print("\n[7] Update content in completed logs (skip active)")
    log_targets = []
    for item in (ROOT / 'logs').iterdir():
        if not item.is_file() or not item.suffix == '.log':
            continue
        m = re.match(r'^task(\d{3})_', item.name)
        if not m or m.group(1) not in MAPPING:
            continue
        if m.group(1) in ACTIVE_JOBS:
            continue
        log_targets.append(item)
    n = sum(1 for fp in log_targets if update_file_content(fp))
    print(f"  Updated {n} log files")

    print("\n" + "=" * 60)
    print("Phase 1 complete")
    print(f"  Active jobs (deferred to Phase 2): {sorted(ACTIVE_JOBS)}")
    print(f"  Mapping: {MAPPING}")
    print("=" * 60)

if __name__ == '__main__':
    main()
