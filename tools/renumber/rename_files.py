"""
rename_files.py — Step 2-6 of GRID task renumbering.

物理 rename 任务文件/目录。支持 5 个 phase:
  - task_defs: GRID/task_artifacts/task_definitions/taskXXX_*.md
  - scripts:   GRID/task_artifacts/scripts/taskXXX_*.{py,sh}
  - result:    result/taskXXX_* 和 GRID/result/taskXXX_*
  - exp:       GRID/task_artifacts/results/expXXX(_suffix) (跳过不在 mapping 的)
  - logs:      GRID/logs/{train,inference}/runs/taskXXX_*/

设计要点:
  - 两阶段 rename(临时前缀 __pending_<old>__ 防重叠)
  - 保护白名单:.ckpt/.pt/.npy/.pkl 文件名(只改父目录)
  - 支持 --reverse 回滚(从 new_to_old 重建 old)
  - 支持 --dry-run 只打印计划
  - exp phase 自动跳过不在 mapping 的 exp(如 exp388v4/v5)

用法:
  python rename_files.py --phase task_defs [--dry-run]
  python rename_files.py --phase all [--dry-run]
  python rename_files.py --phase all --reverse
"""

import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Optional

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
MAPPING = REPO / "tools/renumber/mapping.json"

# 各 phase 涉及路径
PATHS = {
    "task_defs": REPO / "GRID/task_artifacts/task_definitions",
    "scripts": REPO / "GRID/task_artifacts/scripts",
    "result_top": REPO / "result",
    "result_grid": REPO / "GRID/result",
    "exp": REPO / "GRID/task_artifacts/results",
    "logs_train": REPO / "GRID/logs/train/runs",
    "logs_inference": REPO / "GRID/logs/inference/runs",
}

# 受保护的文件后缀(不参与 rename,只改其所在目录)
PROTECTED_SUFFIXES = {".ckpt", ".pt", ".npy", ".pkl", ".bin", ".safetensors"}

# task.md 模板名(不参与编号)
TASK_TEMPLATE = "task.md"

# taskXXX 提取正则
TASK_FILE_RE = re.compile(r"^task(\d+)(_.+)$")
TASK_DIR_RE = re.compile(r"^task(\d+)(_.+)?$")
EXP_RE = re.compile(r"^exp(\d+)(_.+)?$")


def load_mapping(reverse: bool = False) -> dict[str, str]:
    """加载映射。reverse=True 时返回 new->old 方向"""
    m = json.loads(MAPPING.read_text())
    return m["new_to_old"] if reverse else m["old_to_new"]


def load_full_mapping() -> tuple[dict[str, str], dict[str, str]]:
    """加载双向映射"""
    m = json.loads(MAPPING.read_text())
    return m["old_to_new"], m["new_to_old"]


def pending_name(name: str, old_id: str) -> str:
    """生成临时前缀名:task478_xxx.md -> __pending_478__task478_xxx.md"""
    return f"__pending_{old_id}__{name}"


def safe_rename(src: pathlib.Path, dst: pathlib.Path, dry_run: bool) -> bool:
    """安全 rename。dst 已存在则报错"""
    if not dry_run:
        if dst.exists():
            print(f"  ❌ 目标已存在: {dst.name}", file=sys.stderr)
            return False
        # 用 shutil.move 而不是 rename,跨设备也能 work
        shutil.move(str(src), str(dst))
    return True


def rename_task_path(p: pathlib.Path, old_to_new: dict[str, str],
                     new_to_old: dict[str, str], reverse: bool,
                     dry_run: bool) -> Optional[pathlib.Path]:
    """rename taskXXX_xxx 或 taskXXX_xxx/ 形式的路径。

    用 new_to_old 反向检测"文件已是新 ID,无需再改"。
    返回新路径或 None(跳过)。
    """
    name = p.name
    if name == TASK_TEMPLATE:
        return None

    if p.is_file():
        m = TASK_FILE_RE.match(name)
    else:
        m = TASK_DIR_RE.match(name)

    if not m:
        return None

    old_id = m.group(1)
    suffix = m.group(2) or ""

    if reverse:
        # 回滚模式:从 new_id 找 old_id 还原
        if name.startswith("__pending_"):
            pm = re.match(r"__pending_(\d+)__task\d+(.+)$", name)
            if not pm:
                return None
            actual_old = pm.group(1)
            actual_suffix = pm.group(2)
            new_name = f"task{actual_old}{actual_suffix}"
        elif old_id in new_to_old:
            # 正常命名文件:用 new_to_old 反向
            actual_old = new_to_old[old_id]
            new_name = f"task{actual_old}{suffix}"
        else:
            return None
    else:
        # forward 模式:old_id → new_id
        if old_id not in old_to_new:
            return None
        new_id = old_to_new[old_id]
        if new_id == old_id:
            # ID 已是新号(task1,2,3 等映射到自身),无需改
            return None
        new_name = f"task{new_id}{suffix}"

    if dry_run:
        print(f"  [DRY] {p.parent.name}/{name} -> {new_name}")
        return p.parent / new_name

    pending = p.parent / f"__pending_{old_id}__{name}"
    shutil.move(str(p), str(pending))
    final = p.parent / new_name
    if final.exists():
        print(f"  ❌ 目标已存在: {final}", file=sys.stderr)
        # 回滚 pending
        shutil.move(str(pending), str(p))
        return None
    shutil.move(str(pending), str(final))
    print(f"  ✓ {name} -> {final.name}")
    return final


def phase_task_defs(old_to_new: dict[str, str], new_to_old: dict[str, str],
                    reverse: bool, dry_run: bool) -> int:
    """task_definitions/taskXXX_*.md -> taskYYY_*.md"""
    print(f"\n=== Phase task_defs ({PATHS['task_defs']}) ===")
    count = 0
    for f in sorted(PATHS["task_defs"].iterdir()):
        if not f.is_file():
            continue
        if rename_task_path(f, old_to_new, new_to_old, reverse, dry_run):
            count += 1
    return count


def phase_scripts(old_to_new: dict[str, str], new_to_old: dict[str, str],
                  reverse: bool, dry_run: bool) -> int:
    """scripts/taskXXX_*.{py,sh} + launch_taskXXX_*.sh -> taskYYY_*.{py,sh}"""
    print(f"\n=== Phase scripts ({PATHS['scripts']}) ===")
    count = 0
    # 1. 普通 task*_*.{py,sh}
    for ext in ("py", "sh"):
        for f in sorted(PATHS["scripts"].glob(f"task*_*.{ext}")):
            if rename_task_path(f, old_to_new, new_to_old, reverse, dry_run):
                count += 1
    # 2. launch_task*_*.sh(launch 前缀需特殊处理)
    LAUNCH_TASK_RE = re.compile(r"^launch_task(\d+)(_.+)$")
    for f in sorted(PATHS["scripts"].glob("launch_task*_*.sh")):
        m = LAUNCH_TASK_RE.match(f.name)
        if not m:
            continue
        old_id = m.group(1)
        if reverse:
            if f.name.startswith("launch_task") and "__pending_" in f.name:
                pm = re.match(r"__pending_(\d+)__launch_task\d+(_.*)$", f.name)
                if not pm:
                    continue
                actual_old = pm.group(1)
                actual_suffix = pm.group(2)
                new_name = f"launch_task{actual_old}{actual_suffix}"
            elif old_id in new_to_old:
                actual_old = new_to_old[old_id]
                new_name = f"launch_task{actual_old}{suffix}"
            else:
                continue
        else:
            if old_id not in old_to_new:
                continue
            new_id = old_to_new[old_id]
            if new_id == old_id:
                continue
            suffix = m.group(2)
            new_name = f"launch_task{new_id}{suffix}"
        if dry_run:
            print(f"  [DRY] scripts/{f.name} -> {new_name}")
        else:
            pending = f.parent / f"__pending_{old_id}__{f.name}"
            shutil.move(str(f), str(pending))
            final = f.parent / new_name
            if final.exists():
                print(f"  ❌ 目标已存在: {final}", file=sys.stderr)
                shutil.move(str(pending), str(f))
                continue
            shutil.move(str(pending), str(final))
            print(f"  ✓ {f.name} -> {final.name}")
        count += 1
    return count


def phase_result(old_to_new: dict[str, str], new_to_old: dict[str, str],
                 reverse: bool, dry_run: bool) -> int:
    """result/taskXXX_* 和 GRID/result/taskXXX_*"""
    total = 0
    for key in ("result_top", "result_grid"):
        root = PATHS[key]
        print(f"\n=== Phase result/{key} ({root}) ===")
        count = 0
        for p in sorted(root.iterdir()):
            if not p.name.startswith("task"):
                continue
            if p.name == TASK_TEMPLATE:
                continue
            # 文件和目录都处理
            if p.is_file() or p.is_dir():
                if rename_task_path(p, old_to_new, new_to_old, reverse, dry_run):
                    count += 1
        total += count
        print(f"  ({count} renamed in {key})")
    return total


def phase_exp(old_to_new: dict[str, str], new_to_old: dict[str, str],
              reverse: bool, dry_run: bool) -> int:
    """expXXX(_suffix) -> expYYY(_suffix),跳过不在 mapping 的"""
    print(f"\n=== Phase exp ({PATHS['exp']}) ===")
    count = 0
    for p in sorted(PATHS["exp"].iterdir()):
        if not p.name.startswith("exp"):
            continue
        m = EXP_RE.match(p.name)
        if not m:
            continue
        old_id = m.group(1)
        suffix = m.group(2) or ""
        if reverse:
            if p.name.startswith("__pending_"):
                pm = re.match(r"__pending_(\d+)__exp\d+(_.*)?$", p.name)
                if not pm:
                    continue
                actual_old = pm.group(1)
                actual_suffix = pm.group(2) or ""
                new_name = f"exp{actual_old}{actual_suffix}"
            elif old_id in new_to_old:
                actual_old = new_to_old[old_id]
                new_name = f"exp{actual_old}{suffix}"
            else:
                continue
        else:
            if old_id not in old_to_new:
                if not dry_run:
                    print(f"  ⏭️  {p.name} (old_id={old_id} 不在 mapping,跳过)")
                continue
            new_id = old_to_new[old_id]
            if new_id == old_id:
                continue
            new_name = f"exp{new_id}{suffix}"
        if dry_run:
            print(f"  [DRY] {p.name} -> {new_name}")
        else:
            pending = p.parent / f"__pending_{old_id}__{p.name}"
            shutil.move(str(p), str(pending))
            final = p.parent / new_name
            if final.exists():
                print(f"  ❌ 目标已存在: {final}", file=sys.stderr)
                shutil.move(str(pending), str(p))
                continue
            shutil.move(str(pending), str(final))
            print(f"  ✓ {p.name} -> {final.name}")
        count += 1
    return count


def phase_logs(old_to_new: dict[str, str], new_to_old: dict[str, str],
               reverse: bool, dry_run: bool) -> int:
    """GRID/logs/{train,inference}/runs/taskXXX_*/"""
    total = 0
    for key in ("logs_train", "logs_inference"):
        root = PATHS[key]
        print(f"\n=== Phase logs/{key} ({root}) ===")
        count = 0
        if not root.exists():
            print(f"  ⚠️  路径不存在: {root}")
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            if not p.name.startswith("task"):
                continue
            if rename_task_path(p, old_to_new, new_to_old, reverse, dry_run):
                count += 1
        total += count
        print(f"  ({count} renamed in {key})")
    return total


PHASES = {
    "task_defs": phase_task_defs,
    "scripts": phase_scripts,
    "result": phase_result,
    "exp": phase_exp,
    "logs": phase_logs,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=list(PHASES.keys()) + ["all"],
                        help="要执行的 phase")
    parser.add_argument("--reverse", action="store_true",
                        help="回滚模式(new_id -> old_id)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划,不实际 rename")
    args = parser.parse_args()

    if not MAPPING.exists():
        print(f"ERROR: mapping.json 不存在: {MAPPING}", file=sys.stderr)
        print(f"请先运行 build_mapping.py", file=sys.stderr)
        return 1

    old_to_new, new_to_old = load_full_mapping()
    print(f"Mapping loaded: {len(old_to_new)} entries, reverse={args.reverse}")
    print(f"Dry run: {args.dry_run}")

    phases = list(PHASES.keys()) if args.phase == "all" else [args.phase]

    total = 0
    for phase in phases:
        total += PHASES[phase](old_to_new, new_to_old, args.reverse, args.dry_run)

    print(f"\n=== Total renamed: {total} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())