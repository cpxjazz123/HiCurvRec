"""
replace_ids.py — Step 7-8 of GRID task renumbering.

全仓文本 ID 替换。扫所有 .md/.py/.sh/.json/.yaml/.txt 文件,
按 mapping 替换 task<old_id> 和 exp<old_id> 为新 ID。

正则边界:
  task(\d{1,4})(?=[_\b]|$)  — 排除 task388v5 中 388 被误伤
  exp(\d{1,4})(?=[_\b]|$)

不在 mapping 中的 ID 保留原样(如 task388 是辅助脚本中的命名,
不在 60 个 mapping 内,不改)。

支持:
  --reverse 回滚(new_id -> old_id)
  --dry-run 只统计 + 打印样例,不写文件
  --whitelist 跳过特定文件(如 launch_task60_s3.sh 的 SID_PATH 行)
"""

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Optional

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
MAPPING = REPO / "tools/renumber/mapping.json"

# 文本后缀
TEXT_EXTS = {".md", ".py", ".sh", ".json", ".yaml", ".yml", ".txt",
             ".toml", ".cfg", ".ini", ".log", ".csv"}

# 排除目录(相对路径前缀)
EXCLUDE_DIRS = {
    ".git",
    "GRID/data/amazon_data",  # 数据集
    "GRID/data/Beauty",       # 早期 Beauty 数据集
    "GRID/data/Sports",       # 早期 Sports 数据集
    ".claude/worktrees",
    "__pycache__",
    "tools/renumber",         # 自身脚本 + mapping.json
    "backup", "backups", ".backup",
}

# 白名单:相对路径 -> set of line patterns to skip (substring match)
# 这些文件/行不会被替换
WHITELIST = {
    # launch_task60_s3.sh (原 launch_task479_s3.sh) 含 SID_PATH 路径,
    # .pt 文件未改名,路径保持原样避免失效
    "GRID/task_artifacts/scripts/launch_task60_s3.sh": {"task478_hhhh_l4_sid_tensor.pt"},
    # mapping.json 自身
    "tools/renumber/mapping.json": set(),  # 整个文件跳过(下文特殊处理)
}

# 正则边界:
#   task<digits{1-3}> 后跟 _ / 单词边界 / 行尾
# 限制 \d{1,3}:本仓库最大 task ID 是 479(3 位),用位数限制避免
#   "task4789" 中贪婪匹配到 4789 而非 478 的问题。
# 用 (?=_|\b|$) 明确表达"后跟下划线 或 单词边界 或 行尾"。
# 注:Python re 字符类里 \b 是 backspace(ASCII 8),不是 word boundary。
TASK_RE = re.compile(r"task(\d{1,3})(?=_|\b|$)")
EXP_RE = re.compile(r"exp(\d{1,3})(?=_|\b|$)")


def load_mapping(reverse: bool = False) -> dict[str, str]:
    m = json.loads(MAPPING.read_text())
    return m["new_to_old"] if reverse else m["old_to_new"]


def should_skip(path: pathlib.Path) -> bool:
    """是否跳过此文件"""
    rel = str(path.relative_to(REPO)) if path.is_absolute() else str(path)
    if rel == "tools/renumber/mapping.json":
        return True
    for ex in EXCLUDE_DIRS:
        if rel.startswith(ex) or f"/{ex}/" in rel:
            return True
    return False


def replace_in_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """替换文本中的 task/exp ID。返回 (新文本, 替换次数)"""
    count = 0

    def task_sub(m):
        nonlocal count
        old = m.group(1)
        if old in mapping:
            count += 1
            return f"task{mapping[old]}"
        return m.group(0)

    def exp_sub(m):
        nonlocal count
        old = m.group(1)
        if old in mapping:
            count += 1
            return f"exp{mapping[old]}"
        return m.group(0)

    text = TASK_RE.sub(task_sub, text)
    text = EXP_RE.sub(exp_sub, text)
    return text, count


def should_skip_line(rel_path: str, line: str) -> bool:
    """白名单行级 skip"""
    if rel_path in WHITELIST:
        for pattern in WHITELIST[rel_path]:
            if pattern in line:
                return True
    return False


def process_file(path: pathlib.Path, mapping: dict[str, str], dry_run: bool,
                 stats: dict, sample_limit: int = 10) -> None:
    """处理单个文件"""
    rel = str(path.relative_to(REPO))
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        stats["errors"].append((rel, str(e)))
        return

    # 行级白名单(对个别行不替换,但其他行仍替换)
    lines = original.splitlines(keepends=True)
    new_lines = []
    file_subs = 0
    samples = []
    for line in lines:
        if should_skip_line(rel, line):
            new_lines.append(line)
            continue
        new_line, n_subs = replace_in_text(line, mapping)
        if n_subs > 0:
            file_subs += n_subs
            if len(samples) < sample_limit and not dry_run:
                pass  # execute 模式不打印样例
            elif dry_run and len(samples) < sample_limit:
                samples.append((line[:100].rstrip(), new_line[:100].rstrip()))
        new_lines.append(new_line)
    new_text = "".join(new_lines)

    if file_subs > 0:
        stats["files_changed"] += 1
        stats["total_subs"] += file_subs
        if rel not in stats["file_subs_map"]:
            stats["file_subs_map"][rel] = file_subs
        if not dry_run and new_text != original:
            path.write_text(new_text, encoding="utf-8")
        elif dry_run and samples:
            stats["samples"][rel] = samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true",
                        help="回滚(new_id -> old_id)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计,不写文件")
    parser.add_argument("--roots", nargs="+",
                        default=["GRID", "result"],
                        help="要扫描的根目录(相对 REPO)")
    args = parser.parse_args()

    if not MAPPING.exists():
        print(f"ERROR: mapping.json 不存在: {MAPPING}", file=sys.stderr)
        return 1

    mapping = load_mapping(reverse=args.reverse)
    print(f"Mapping loaded: {len(mapping)} entries, reverse={args.reverse}, dry_run={args.dry_run}")

    stats = {
        "files_scanned": 0,
        "files_changed": 0,
        "total_subs": 0,
        "errors": [],
        "samples": {},
        "file_subs_map": {},
    }

    for root in args.roots:
        root_path = REPO / root
        if not root_path.exists():
            print(f"  ⚠️  根目录不存在: {root}")
            continue
        print(f"\n=== 扫描 {root} ===")
        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            if should_skip(p):
                continue
            if p.suffix not in TEXT_EXTS:
                continue
            stats["files_scanned"] += 1
            process_file(p, mapping, args.dry_run, stats)

    print(f"\n=== 统计 ===")
    print(f"  扫描文件: {stats['files_scanned']}")
    print(f"  改动文件: {stats['files_changed']}")
    print(f"  替换次数: {stats['total_subs']}")
    if stats["errors"]:
        print(f"  错误数: {len(stats['errors'])}")
        for rel, err in stats["errors"][:5]:
            print(f"    {rel}: {err}")
    if args.dry_run and stats["samples"]:
        print(f"\n=== 样例(每个文件最多 {10} 条) ===")
        for rel, samples in list(stats["samples"].items())[:15]:
            print(f"  {rel}:")
            for old, new in samples[:3]:
                print(f"    OLD: {old}")
                print(f"    NEW: {new}")
                print()

    return 0


if __name__ == "__main__":
    sys.exit(main())