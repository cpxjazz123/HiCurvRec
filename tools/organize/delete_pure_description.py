"""
delete_pure_description.py — Step 2 of GRID organize-dir.

删除 32 个纯 description task 的 task_definitions 文件。
规则:用 /bin/rm -f(防止 rm alias),立即 [ -e ] 验证。

用法:
  python delete_pure_description.py [--dry-run]
"""
import json
import os
import pathlib
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
DELETE = REPO / "tools/organize/delete_set.json"
TD = REPO / "GRID/task_artifacts/task_definitions"


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    items = json.loads(DELETE.read_text())
    print(f"待删除: {len(items)} 个 task")

    deleted = 0
    for it in items:
        f = TD / it['filename']
        if not f.exists():
            print(f"  ⚠️  不存在: {f.relative_to(REPO)}")
            continue
        if args.dry_run:
            print(f"  [DRY] /bin/rm -f {f.relative_to(REPO)}")
            continue
        os.system(f"/bin/rm -f {f}")
        # 立即验证
        if f.exists():
            print(f"  ❌ 删除失败: {f.relative_to(REPO)}")
            sys.exit(1)
        print(f"  ✓ {it['filename']}")
        deleted += 1

    print(f"\n=== 删除完成: {deleted}/{len(items)} ===")
    if not args.dry_run:
        remaining = sorted(TD.iterdir())
        print(f"task_definitions/ 剩余: {len(remaining)} (应为 28 = 60 - 32 + 1 task.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())