"""
classify_tasks.py — Step 1 of GRID organize-dir.

扫描 60 个 task,根据产物位置分类:
  - keep: 有 ckpt/verdict/log/result 任一类产物
  - pure_description: 完全无产物

输出:
  - tools/organize/keep_set.json: 保留的 task 列表(28 个) + 每个的产物路径
  - tools/organize/delete_set.json: 删除的 task 列表(32 个)
"""
import json
import pathlib
import re

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
MAPPING = REPO / "tools/renumber/mapping.json"
TD = REPO / "GRID/task_artifacts/task_definitions"

EXP = REPO / "GRID/task_artifacts/results"
LOG_TRAIN = REPO / "GRID/logs/train/runs"
LOG_INF = REPO / "GRID/logs/inference/runs"
RESULT_TOP = REPO / "result"
RESULT_GRID = REPO / "GRID/result"


def main() -> int:
    m = json.loads(MAPPING.read_text())
    o2n = {k: str(v) for k, v in m["old_to_new"].items()}

    def all_old_for_new(nid):
        return [oid for oid, new in o2n.items() if new == nid]

    # 收集产物(按 OLD id)
    train_oids = set()
    for ts in LOG_TRAIN.iterdir():
        if not ts.is_dir(): continue
        for r in ts.iterdir():
            if r.is_dir() and (rm := re.match(r'task(\d+)_', r.name)):
                train_oids.add(rm.group(1))
    inf_oids = set()
    for ts in LOG_INF.iterdir():
        if not ts.is_dir(): continue
        for r in ts.iterdir():
            if r.is_dir() and (rm := re.match(r'task(\d+)_', r.name)):
                inf_oids.add(rm.group(1))

    verdict_oids = set()
    ckpt_oids = set()
    for ed in EXP.iterdir():
        if not ed.is_dir() or not ed.name.startswith('exp'): continue
        for f in ed.iterdir():
            fm = re.match(r'task(\d+)_', f.name)
            if not fm: continue
            oid = fm.group(1)
            if 'verdict' in f.name.lower() and f.name.endswith('.md'):
                verdict_oids.add(oid)
            if f.name.endswith('.ckpt'):
                ckpt_oids.add(oid)

    rT_oids = set()
    for f in RESULT_TOP.iterdir():
        if f.name.startswith('task') and f.name != 'task.md':
            fm = re.match(r'task(\d+)_?', f.name)
            if fm: rT_oids.add(fm.group(1))
    rG_oids = set()
    for f in RESULT_GRID.iterdir():
        if f.name.startswith('task') and f.name != 'task.md':
            fm = re.match(r'task(\d+)_?', f.name)
            if fm: rG_oids.add(fm.group(1))

    # 60 task 分类
    td_files = sorted(p for p in TD.iterdir() if p.is_file() and re.match(r'task\d+_', p.name))
    keep = []
    delete = []
    for f in td_files:
        nid = re.match(r'task(\d+)_', f.name).group(1)
        olds = all_old_for_new(nid)
        has_prod = any([
            any(o in train_oids for o in olds),
            any(o in inf_oids for o in olds),
            any(o in verdict_oids for o in olds),
            any(o in ckpt_oids for o in olds),
            any(o in rT_oids for o in olds),
            any(o in rG_oids for o in olds),
        ])
        if has_prod:
            keep.append({
                "new_id": nid,
                "filename": f.name,
                "train": [o for o in olds if o in train_oids],
                "inference": [o for o in olds if o in inf_oids],
                "verdict": [o for o in olds if o in verdict_oids],
                "ckpt": [o for o in olds if o in ckpt_oids],
                "rT": [o for o in olds if o in rT_oids],
                "rG": [o for o in olds if o in rG_oids],
            })
        else:
            delete.append({"new_id": nid, "filename": f.name})

    out_dir = REPO / "tools/organize"
    (out_dir / "keep_set.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False))
    (out_dir / "delete_set.json").write_text(json.dumps(delete, indent=2, ensure_ascii=False))

    print(f"=== 分类结果 ===")
    print(f"  keep (有产物):       {len(keep)} 个")
    print(f"  delete (纯 description): {len(delete)} 个")
    print(f"  输出: {out_dir}/keep_set.json, delete_set.json")

    print(f"\n=== delete_set (将被删除) ===")
    for d in delete:
        print(f"  new_id={d['new_id']}: {d['filename']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())