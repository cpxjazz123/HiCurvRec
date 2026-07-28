"""
rename_artifacts.py — Step 4 of GRID organize-dir.

物理重命名产物文件中的 OLD task id 为 NEW task id。
两阶段 rename(临时前缀 __pending__)防撞名。
覆盖范围:
  - GRID/task_artifacts/results/exp*/task<OLD>_*  → exp*/task<NEW>_*
  - GRID/logs/{train,inference}/runs/task<OLD>_*  → runs/task<NEW>_*
  - GRID/result/task<OLD>_*                       → task<NEW>_*
  - result/task<OLD>_*                            → task<NEW>_*

跳过:
  - exp388* (不在 60 mapping)
  - launch_task*, taskA*, task.md 等非 task<N>_*_*.md 格式

用法:
  python rename_artifacts.py [--dry-run]
"""
import json
import pathlib
import re
import shutil
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
MAPPING = REPO / "tools/renumber/mapping.json"

# 待重命名根目录
ROOTS = [
    REPO / "GRID/task_artifacts/results",
    REPO / "GRID/logs/train/runs",
    REPO / "GRID/logs/inference/runs",
    REPO / "GRID/result",
    REPO / "result",
]

# 匹配 task<OLD>_<rest> 或 task<OLD> (仅文件名)
TASK_FILE_RE = re.compile(r'^task(\d+)(.+)$')
# 匹配 exp<OLD>_(suffix) 或 exp<OLD>
EXP_RE = re.compile(r'^exp(\d+)(v\d+)?(.+)?$')

# 388 体系跳过(exp388v4_s4_v1, exp388v5)
EXP388_RE = re.compile(r'^exp388(v\d+)?(.*)?$')


def load_mapping() -> tuple[dict[str, str], set[str]]:
    """返回 (old_to_new, set_of_old_ids)。"""
    m = json.loads(MAPPING.read_text())
    o2n = {k: str(v) for k, v in m["old_to_new"].items()}
    # OLD id set:o2n keys 即 OLD id 集合
    old_ids = set(o2n.keys())
    return o2n, old_ids


def safe_move(src: pathlib.Path, dst: pathlib.Path) -> None:
    """原子两阶段 rename。"""
    pending = src.parent / f"__pending__{src.name}"
    shutil.move(str(src), str(pending))
    if dst.exists():
        # 目标已存在:回滚 pending
        shutil.move(str(pending), str(src))
        raise FileExistsError(f"目标已存在: {dst}")
    shutil.move(str(pending), str(dst))


def rename_in_root(root: pathlib.Path, o2n: dict[str, str], old_ids: set[str], dry_run: bool, scope: str) -> int:
    """递归 rename root 下所有 task<OLD>_* 文件 / task<OLD>_* 目录 / exp<OLD>_* 目录。"""
    count = 0
    if not root.exists():
        return 0
    for p in sorted(root.iterdir()):
        name = p.name
        if name.startswith('__pending__'):
            print(f"  ⚠️  残留 pending: {p}", file=sys.stderr)
            continue
        # exp388* 跳过
        if EXP388_RE.match(name):
            continue
        # exp<OLD> 目录改名
        if scope == 'exp':
            m = EXP_RE.match(name)
            if m:
                old_id = m.group(1)
                if old_id in old_ids:
                    new_id = o2n[old_id]
                    new_name = f"exp{new_id}{m.group(2) or ''}{m.group(3) or ''}"
                    if new_name == name:
                        continue
                    if dry_run:
                        print(f"  [DRY] {p} -> {new_name}")
                    else:
                        safe_move(p, p.parent / new_name)
                        print(f"  ✓ {name} -> {new_name}")
                    count += 1
            continue
        # task<OLD>_* 文件 / 目录改名
        m = TASK_FILE_RE.match(name)
        if not m:
            continue
        old_id = m.group(1)
        rest = m.group(2)
        if old_id not in old_ids:
            # 不是 OLD id(可能是 NEW id 已被改过,或不在 mapping 的 388/426 等)— 跳过
            continue
        new_id = o2n[old_id]
        new_name = f"task{new_id}{rest}"
        if new_name == name:
            continue
        if dry_run:
            print(f"  [DRY] {p} -> {new_name}")
        else:
            safe_move(p, p.parent / new_name)
            print(f"  ✓ {name} -> {new_name}")
        count += 1

        # 如果是目录,递归 rename 子目录里 task<OLD>_* 文件
        # (已经在 iterdir 里处理:目录改名后它的子目录需要单独遍历)
    return count


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    o2n, old_ids = load_mapping()
    print(f"Mapping: {len(o2n)} OLD→NEW id entries, dry_run={args.dry_run}")

    total = 0
    # 1. results/exp<OLD>_<rest> 目录
    print(f"\n=== Phase 4a: results/exp* 目录 ===")
    cnt = rename_in_root(REPO / "GRID/task_artifacts/results", o2n, old_ids, args.dry_run, scope='exp')
    print(f"  ({cnt} exp 目录)")
    total += cnt

    # 2. results/exp<NEW>/task<OLD>_* 文件 (递归)
    print(f"\n=== Phase 4b: results/exp*/task* 文件(递归)===")
    cnt = 0
    for ed in sorted((REPO / "GRID/task_artifacts/results").iterdir()):
        if not ed.is_dir() or not ed.name.startswith('exp'): continue
        cnt += rename_in_root(ed, o2n, old_ids, args.dry_run, scope='task')
    print(f"  ({cnt} 文件)")
    total += cnt

    # 3. logs/{train,inference}/runs/task<OLD>_* 目录
    for sub in ['train', 'inference']:
        print(f"\n=== Phase 4c: logs/{sub}/runs/task* 目录 ===")
        runs = REPO / f"GRID/logs/{sub}/runs"
        if not runs.exists(): continue
        for ts in sorted(runs.iterdir()):
            if not ts.is_dir(): continue
            cnt_l = rename_in_root(ts, o2n, old_ids, args.dry_run, scope='task')
            total += cnt_l

    # 4. GRID/result/task<OLD>_* 文件/目录(顶层)
    print(f"\n=== Phase 4d: GRID/result/task* 顶层 ===")
    cnt = rename_in_root(REPO / "GRID/result", o2n, old_ids, args.dry_run, scope='task')
    print(f"  ({cnt})")
    total += cnt

    # 5. result/task<OLD>_* 顶层
    print(f"\n=== Phase 4e: result/task* 顶层 ===")
    cnt = rename_in_root(REPO / "result", o2n, old_ids, args.dry_run, scope='task')
    print(f"  ({cnt})")
    total += cnt

    # 6. verdicts/ 目录(Phase 6 后建)
    print(f"\n=== Phase 4f: verdicts/ ===")
    cnt = rename_in_root(REPO / "GRID/task_artifacts/verdicts", o2n, old_ids, args.dry_run, scope='task')
    print(f"  ({cnt})")
    total += cnt

    print(f"\n=== Total: {total} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())