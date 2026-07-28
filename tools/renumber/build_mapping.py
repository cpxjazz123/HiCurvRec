"""
build_mapping.py — Step 1 of GRID task renumbering.

Scans GRID/task_artifacts/task_definitions/, extracts old IDs from filenames
(taskXXX_*.md), sorts ascending, and assigns new IDs 1..N.

Output: tools/renumber/mapping.json with:
  - old_to_new: {"1": 1, "2": 2, ..., "479": 60}
  - new_to_old: inverse mapping
  - old_ids: sorted list of old IDs
  - count: number of task files
  - new_to_slug: {"1": "task1_v3_norm_false", ...} (filename mapping)
"""

import json
import pathlib
import re
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
TD = REPO / "GRID/task_artifacts/task_definitions"
OUT = REPO / "tools/renumber/mapping.json"


def main() -> int:
    if not TD.is_dir():
        print(f"ERROR: task_definitions 目录不存在: {TD}", file=sys.stderr)
        return 1

    # 扫所有 task<digits>_<slug>.md 文件,排除 task.md 模板
    pattern = re.compile(r"^task(\d+)_([a-zA-Z0-9_]+)\.md$")
    entries = []  # [(old_id, filename), ...]
    skipped = []
    for p in sorted(TD.iterdir()):
        if not p.is_file():
            continue
        if p.name == "task.md":
            continue
        m = pattern.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        entries.append((int(m.group(1)), p.name, m.group(2)))

    if not entries:
        print("ERROR: 未找到任何 taskXXX_*.md 文件", file=sys.stderr)
        return 1

    # 按 old_id 升序排序
    entries.sort(key=lambda x: x[0])

    # 验证 ID 唯一
    ids = [e[0] for e in entries]
    if len(set(ids)) != len(ids):
        dup = [i for i in ids if ids.count(i) > 1]
        print(f"ERROR: 发现重复 old_id: {set(dup)}", file=sys.stderr)
        return 1

    # 旧 ID 列表(升序,排序后即可)
    old_ids = sorted(ids)

    # 生成映射
    old_to_new = {str(o): i + 1 for i, o in enumerate(old_ids)}
    new_to_old = {str(v): str(k) for k, v in old_to_new.items()}
    new_to_slug = {
        str(new_id): entries[i][1].replace(f"task{entries[i][0]}_", f"task{new_id}_")
        for i, new_id in enumerate(old_to_new.values())
    }

    # 写 mapping.json
    payload = {
        "count": len(old_ids),
        "old_ids": old_ids,
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "new_to_slug": new_to_slug,
        "skipped": skipped,
        "generated_at": "2026-07-16",
        "rule": "old_id ascending -> 1..N",
    }

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # 摘要输出
    print(f"✅ mapping.json 已生成: {OUT}")
    print(f"   count: {len(old_ids)}")
    print(f"   old_ids: [{old_ids[0]}..{old_ids[-1]}], gaps={len(old_ids) - (old_ids[-1] - old_ids[0] + 1)}")
    print(f"   first 5 mappings: {list(old_to_new.items())[:5]}")
    print(f"   last 5 mappings: {list(old_to_new.items())[-5:]}")
    if skipped:
        print(f"   ⚠️  skipped (non-matching): {skipped}")

    return 0


if __name__ == "__main__":
    sys.exit(main())