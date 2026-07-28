"""
rename_artifacts_fix.py — 单次严格不 chain rename 的版本。

核心原则:每个路径最多处理一次。
判断标准:文件/目录当前名中的 id 若不在 mapping 的 `old_to_new` keys 中,或者已经是 NEW id(== o2n[itself]),跳过。
"""

import json
import pathlib
import re
import shutil
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
MAPPING = REPO / "tools/renumber/mapping.json"

ROOTS = [
    REPO / "GRID/task_artifacts/results",
    REPO / "GRID/logs/train/runs",
    REPO / "GRID/logs/inference/runs",
    REPO / "GRID/result",
    REPO / "result",
    REPO / "GRID/task_artifacts/verdicts",
]

TASK_FILE_RE = re.compile(r'^task(\d+)(.+)$')
EXP_RE = re.compile(r'^exp(\d+)(v\d+)?(.+)?$')
EXP388_RE = re.compile(r'^exp388(v\d+)?(.*)?$')


def load_mapping():
    m = json.loads(MAPPING.read_text())
    o2n = {k: str(v) for k, v in m["old_to_new"].items()}
    # OLD id set:真实 OLD id 列表(从 m["old_ids"])
    old_ids = set(str(x) for x in m["old_ids"])
    return o2n, old_ids


def safe_move(src, dst):
    pending = src.parent / f"__pending__{src.name}"
    shutil.move(str(src), str(pending))
    if dst.exists():
        shutil.move(str(pending), str(src))
        raise FileExistsError(f"目标已存在: {dst}")
    shutil.move(str(pending), str(dst))


def rename_in_root(root, o2n, old_ids, dry_run, scope, max_depth=3, _depth=0):
    if not root.exists() or _depth > max_depth:
        return 0
    count = 0
    for p in sorted(root.iterdir()):
        name = p.name
        if name.startswith('__pending__'):
            continue
        if EXP388_RE.match(name):
            continue
        if scope == 'exp' and _depth == 0:
            m = EXP_RE.match(name)
            if m:
                old_id = m.group(1)
                if old_id in old_ids:
                    new_id = o2n[old_id]
                    if new_id == old_id:
                        continue
                    new_name = f"exp{new_id}{m.group(2) or ''}{m.group(3) or ''}"
                    if not dry_run:
                        safe_move(p, p.parent / new_name)
                        print(f"  ✓ {name} -> {new_name}")
                    count += 1
                continue
        m = TASK_FILE_RE.match(name)
        if not m:
            # 可能是目录,递归(只对 train/inference timestamp 目录和 exp 顶层)
            if p.is_dir() and _depth < max_depth:
                count += rename_in_root(p, o2n, old_ids, dry_run, 'task', max_depth, _depth + 1)
            continue
        old_id = m.group(1)
        rest = m.group(2)
        if old_id not in old_ids:
            # 不是 OLD id,跳过(不在 mapping 中如 388)
            continue
        new_id = o2n[old_id]
        if new_id == old_id:
            # 自我映射(old 1 → new 1)— 已经是 NEW id 状态,跳过
            continue
        new_name = f"task{new_id}{rest}"
        if new_name == name:
            continue
        if not dry_run:
            if (p.parent / new_name).exists():
                # 目标已存在,跳过(idempotent)
                continue
            safe_move(p, p.parent / new_name)
            print(f"  ✓ {name} -> {new_name}")
        count += 1
    return count


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    o2n, old_ids = load_mapping()
    print(f"Mapping: {len(o2n)} OLD→NEW, {len(old_ids)} OLD ids, dry_run={args.dry_run}")

    total = 0
    # 1. results/ 顶层(只改 exp<OLD>)
    print("\n=== results/ 顶层 exp ===")
    cnt = rename_in_root(REPO / "GRID/task_artifacts/results", o2n, old_ids, args.dry_run, 'exp', max_depth=0)
    total += cnt
    print(f"  ({cnt} exp 顶层)")

    # 2. results/exp*/ 内 task<OLD>_* 文件
    print("\n=== results/exp*/ task<OLD>_* ===")
    cnt = 0
    for ed in sorted((REPO / "GRID/task_artifacts/results").iterdir()):
        if ed.is_dir() and ed.name.startswith('exp') and not EXP388_RE.match(ed.name):
            cnt += rename_in_root(ed, o2n, old_ids, args.dry_run, 'task', max_depth=2)
    total += cnt
    print(f"  ({cnt} 文件)")

    # 3. logs/{train,inference}/runs/<timestamp>/task<OLD>_*/  内
    for sub in ['train', 'inference']:
        print(f"\n=== logs/{sub}/runs/ 内 ===")
        runs = REPO / f"GRID/logs/{sub}/runs"
        if not runs.exists(): continue
        cnt = 0
        for ts in sorted(runs.iterdir()):
            if ts.is_dir():
                cnt += rename_in_root(ts, o2n, old_ids, args.dry_run, 'task', max_depth=1)
        total += cnt
        print(f"  ({cnt})")

    # 4. GRID/result/ 顶层 + 子目录
    print("\n=== GRID/result/ ===")
    cnt = rename_in_root(REPO / "GRID/result", o2n, old_ids, args.dry_run, 'task', max_depth=1)
    total += cnt
    print(f"  ({cnt})")

    # 5. result/ 顶层
    print("\n=== result/ ===")
    cnt = rename_in_root(REPO / "result", o2n, old_ids, args.dry_run, 'task', max_depth=1)
    total += cnt
    print(f"  ({cnt})")

    # 6. verdicts/
    print("\n=== verdicts/ ===")
    cnt = rename_in_root(REPO / "GRID/task_artifacts/verdicts", o2n, old_ids, args.dry_run, 'task', max_depth=1)
    total += cnt
    print(f"  ({cnt})")

    print(f"\n=== Total: {total} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())