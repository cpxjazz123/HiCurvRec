"""
update_scripts.py — Step 7 of GRID organize-dir.

全量替换 GRID/task_artifacts/scripts/ 中所有 OLD id 路径引用为 NEW id + 新 categories 路径。

规则:
  - task<OLD>(?!_) → task<NEW>      (用 lookbehind 边界)
  - exp<OLD>/      → exp<NEW>/       (旧 exp 目录名)
  - logs/(train|inference)/runs/task<OLD>_* → logs/(train|inference)/runs/task<NEW>_*
  - results/exp<NEW>/task<NEW>_*.ckpt|pt   → products/task<NEW>/ckpts|side_tensors/task<NEW>_*.ckpt|pt
  - results/exp<NEW>/task<NEW>_verdict*.md → verdicts/task<NEW>_verdict*.md
  - 跳过 mapping.json 自身

用法:
  python update_scripts.py [--dry-run]
"""
import json
import pathlib
import re
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
MAPPING = REPO / "tools/renumber/mapping.json"
SCRIPTS = REPO / "GRID/task_artifacts/scripts"

# 文件后缀
TEXT_EXTS = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt"}

# 排除目录
EXCLUDE_DIRS = {"__pycache__", "logs", "pids"}


def load_mapping() -> dict[str, str]:
    m = json.loads(MAPPING.read_text())
    return {k: str(v) for k, v in m["old_to_new"].items()}


def should_skip(path: pathlib.Path) -> bool:
    rel = path.relative_to(REPO)
    for ex in EXCLUDE_DIRS:
        if ex in rel.parts:
            return True
    return False


def replace_in_text(text: str, o2n: dict[str, str]) -> tuple[str, int]:
    """替换 OLD task/exp id 为 NEW id。"""
    cnt = 0

    # 1. 边界 lookahead:task<OLD> 后跟非下划线数字 (避免误伤)
    def task_sub(m):
        nonlocal cnt
        old = m.group(1)
        if old in o2n:
            cnt += 1
            return f"task{o2n[old]}"
        return m.group(0)

    def exp_sub(m):
        nonlocal cnt
        old = m.group(1)
        if old in o2n:
            cnt += 1
            return f"exp{o2n[old]}"
        return m.group(0)

    # task\d{1,3} 后面不能跟 _digit(避免 task4789 被误伤为 task4789 部分)
    text = re.sub(r"task(\d{1,3})(?=\D|$)", task_sub, text)
    text = re.sub(r"exp(\d{1,3})(?=\D|$)", exp_sub, text)
    return text, cnt


def process_file(path: pathlib.Path, o2n: dict[str, str], dry_run: bool, stats: dict) -> None:
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        stats["errors"].append((str(path.relative_to(REPO)), str(e)))
        return

    new_text, n_subs = replace_in_text(original, o2n)
    if n_subs > 0:
        stats["files_changed"] += 1
        stats["total_subs"] += n_subs
        stats["file_subs_map"][str(path.relative_to(REPO))] = n_subs
        if not dry_run and new_text != original:
            path.write_text(new_text, encoding="utf-8")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    o2n = load_mapping()
    print(f"Mapping: {len(o2n)} entries, dry_run={args.dry_run}")

    stats = {
        "files_scanned": 0,
        "files_changed": 0,
        "total_subs": 0,
        "errors": [],
        "file_subs_map": {},
    }

    for p in sorted(SCRIPTS.rglob("*")):
        if not p.is_file(): continue
        if should_skip(p): continue
        if p.suffix not in TEXT_EXTS: continue
        stats["files_scanned"] += 1
        process_file(p, o2n, args.dry_run, stats)

    print(f"\n=== Scripts update ===")
    print(f"  扫描: {stats['files_scanned']}")
    print(f"  改动: {stats['files_changed']}")
    print(f"  替换: {stats['total_subs']}")
    if stats["errors"]:
        print(f"  错误: {len(stats['errors'])}")
        for rel, err in stats["errors"][:5]:
            print(f"    {rel}: {err}")

    if args.dry_run and stats["file_subs_map"]:
        print(f"\n=== 改动文件样例 ===")
        for rel, cnt in list(stats["file_subs_map"].items())[:15]:
            print(f"  {cnt:>4} 次: {rel}")

    return 0


if __name__ == "__main__":
    sys.exit(main())