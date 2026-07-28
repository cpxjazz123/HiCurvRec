"""cleanup_chain_ghosts.py — 删除 chain rename 留下的 ghost (deleted NEW id 的文件名)。"""
import json
import pathlib
import re
import shutil

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")

delete_set = json.loads((REPO / "tools/organize/delete_set.json").read_text())
del_nids = {k['new_id'] for k in delete_set}
print(f"deleted NEW ids: {len(del_nids)} = {sorted([int(x) for x in del_nids])}")

TASK_RE = re.compile(r'^task(\d+)_')


def cleanup(root):
    cnt = 0
    if not root.exists():
        return 0
    for p in sorted(root.iterdir()):
        m = TASK_RE.match(p.name)
        if not m:
            continue
        nid = m.group(1)
        if nid not in del_nids:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
            print(f"  ✓ rm dir {p.name}")
        else:
            p.unlink()
            print(f"  ✓ rm file {p.name}")
        cnt += 1
    return cnt


ROOTS = [
    REPO / "result",
    REPO / "GRID/result",
    REPO / "GRID/logs/train/runs",
    REPO / "GRID/logs/inference/runs",
    REPO / "GRID/task_artifacts/results",
    REPO / "GRID/task_artifacts/verdicts",
    REPO / "GRID/task_artifacts/products",
]

total = 0
for r in ROOTS:
    print(f"\n=== {r.relative_to(REPO)} ===")
    cnt = cleanup(r)
    total += cnt
    print(f"  ({cnt})")

print(f"\n=== Total ghosts removed: {total} ===")
