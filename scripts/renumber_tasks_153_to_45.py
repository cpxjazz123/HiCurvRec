#!/usr/bin/env python3
"""R9 renumbering: 153-164 → 45-56 (填补 descriptions/ 45-152 空洞)

Per R9: "新任务必须用最大已用编号 +1, 保证 1, 2, 3, ..., N-1, N 严格连续无空洞"
发现 descriptions/ 45-152 空洞 (108 个), 必须 renumber.

Mapping (156 = 48 等):
  153 → 45 (Joint-Flat-3seg)
  154 → 46 (L1 fixes)
  155 → 47 (L1 GSRQ)
  156 → 48 (S4 AE)
  157 → 49 (QMP)
  158 → 50 (S6 item2vec)
  159 → 51 (TIGER A vs B)
  160 → 52 (G1 verdict)
  161 → 53 (Real TIGER)
  162 → 54 (L3 norm)
  163 → 55 (OPQ)
  164 → 56 (Curvature)

Renamed locations:
  - descriptions/task<N>_*.md (12 files)
  - verdicts/task<N>_*.md (2 files for task52)
  - products/task<N>/* (8 dirs for task45-160)
  - scripts/task<N>_*.py (8 files for task45-160)
  - logs/task<N>_* (历史已存, 不动; 已完成的产物)
  - verdicts/task<old_N> 内容引用更新到新编号
  - loop.md §16 引用更新
"""
import os
import re
import shutil
from pathlib import Path

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')

MAPPING = {
    153: 45, 154: 46, 155: 47, 156: 48, 157: 49, 158: 50,
    159: 51, 160: 52, 161: 53, 162: 54, 163: 55, 164: 56,
}

# 不要 rename log 路径 (历史已跑完的产物), 仅 rename 描述/产物
RENAMED_DIRS = {
    'descriptions': True,
    'verdicts': True,
    'products': True,
    'scripts': True,
}

# 不 rename logs/ (历史产物, 跨任务共享, 改路径会破坏引用)
SKIP_DIRS = {'logs', 'GRID/logs', 'data'}


def rename_path(old: Path, new: Path, dry_run=False):
    """Rename a single file or directory."""
    if new.exists():
        print(f'  ⚠️  target already exists: {new}')
        return False
    if dry_run:
        print(f'  DRY: {old} → {new}')
        return True
    old.rename(new)
    return True


def update_content(path: Path, mapping: dict, dry_run=False):
    """Update content references inside a file."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f'  ⚠️  read failed: {path} ({e})')
        return
    new_text = text
    # Replace in descending order (largest IDs first) to avoid double-replacement
    for old_id in sorted(mapping.keys(), reverse=True):
        new_id = mapping[old_id]
        # Match task<N> (word boundary)
        new_text = re.sub(rf'\btask{old_id}\b', f'task{new_id}', new_text)
        new_text = re.sub(rf'\bTask #{old_id}\b', f'Task #{new_id}', new_text)
        new_text = re.sub(rf'Task #<old_id>'.replace('<old_id>', str(old_id)),
                          f'Task #{new_id}', new_text)
    if new_text != text:
        if dry_run:
            print(f'  DRY content: {path}')
        else:
            path.write_text(new_text, encoding='utf-8')


def main(dry_run=False):
    print(f'{"=" * 70}')
    print(f'  Renumbering 153-164 → 45-56 ({"DRY RUN" if dry_run else "ACTIVE"})')
    print(f'{"=" * 70}')

    # Step 1: rename files/dirs in each location
    for sub in RENAMED_DIRS:
        base = ROOT / sub
        if not base.exists():
            continue
        print(f'\n--- {sub}/ ---')
        for old_id, new_id in MAPPING.items():
            # Find all entries starting with task<old_id>
            for entry in list(base.iterdir()):
                name = entry.name
                if re.match(rf'^task{old_id}([_\.].*)?$', name):
                    new_name = re.sub(rf'^task{old_id}', f'task{new_id}', name)
                    new_path = entry.parent / new_name
                    if rename_path(entry, new_path, dry_run):
                        print(f'  ✓ {sub}/{name} → {new_name}')

    # Step 2: update content in renamable files
    print(f'\n--- content updates ---')
    content_dirs = [
        ROOT / 'descriptions',
        ROOT / 'verdicts',
        ROOT / 'products',
        ROOT / 'scripts',
    ]
    for cd in content_dirs:
        if not cd.exists():
            continue
        for f in cd.rglob('*'):
            if f.is_file() and f.suffix in {'.md', '.py', '.json', '.txt', '.yaml', '.yml'}:
                update_content(f, MAPPING, dry_run)

    # Step 3: update loop.md §16 reference (Task #53 → Task #53)
    loop_md = ROOT / 'loop.md'
    if loop_md.exists():
        update_content(loop_md, MAPPING, dry_run)

    # Step 4: update CLAUDE.md references
    claude_md = ROOT / 'CLAUDE.md'
    if claude_md.exists():
        update_content(claude_md, MAPPING, dry_run)

    print(f'\n{"=" * 70}')
    print(f'  Done.')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    import sys
    dry = '--dry' in sys.argv
    main(dry_run=dry)