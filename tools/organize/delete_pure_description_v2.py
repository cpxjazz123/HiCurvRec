"""delete_pure_description_v2.py — Block 2 专用:从 descriptions/ 删 32 个纯 description 任务文件。

与 v1 区别:TD (task_definitions/) 已空,文件在 descriptions/。
"""
import json
import os
import pathlib
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
DELETE = REPO / "tools/organize/delete_set.json"
DESC = REPO / "GRID/task_artifacts/descriptions"


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    items = json.loads(DELETE.read_text())
    print(f"待删除: {len(items)} 个 task md")

    deleted = 0
    not_found = 0
    for it in items:
        f = DESC / it['filename']
        if not f.exists():
            print(f"  ⚠️  不存在: {f.relative_to(REPO)}")
            not_found += 1
            continue
        if args.dry_run:
            print(f"  [DRY] /bin/rm -f {f.relative_to(REPO)}")
            continue
        os.system(f"/bin/rm -f {f}")
        if f.exists():
            print(f"  ❌ 删除失败: {f.relative_to(REPO)}")
            sys.exit(1)
        print(f"  ✓ {it['filename']}")
        deleted += 1

    print(f"\n=== 删除完成: {deleted}/{len(items)}, 不存在 {not_found} ===")
    if not args.dry_run:
        remaining = sorted(p.name for p in DESC.iterdir() if p.is_file())
        print(f"descriptions/ 剩余文件 ({len(remaining)}):")
        for r in remaining:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
