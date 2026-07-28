"""
rename_chain_resolve.py — 反复迭代 rename 直到每个文件名稳定。

对每个路径,反复应用 mapping 直到 id 不再变:
  1. 提取当前 id
  2. 看 o2n[id]: 如果 == id 自身,稳定;否则改名
  3. 重复直到稳定
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

TASK_RE = re.compile(r'^task(\d+)(.+)$')
EXP_RE = re.compile(r'^exp(\d+)(v\d+)?(.+)?$')
EXP388_RE = re.compile(r'^exp388(v\d+)?(.*)?$')


def load_mapping():
    m = json.loads(MAPPING.read_text())
    o2n = {k: str(v) for k, v in m["old_to_new"].items()}
    old_ids = set(str(x) for x in m["old_ids"])
    return o2n, old_ids


def resolve_id_chain(id_str, o2n, old_ids):
    """反复迭代直到 id 稳定。"""
    seen = set()
    cur = id_str
    while True:
        if cur not in o2n:
            return cur
        nxt = o2n[cur]
        if nxt == cur:
            return cur
        if cur in seen:
            # cycle — 终止
            return cur
        seen.add(cur)
        cur = nxt


def safe_move(src, dst):
    pending = src.parent / f"__pending__{src.name}"
    shutil.move(str(src), str(pending))
    if dst.exists():
        shutil.move(str(pending), str(src))
        raise FileExistsError(f"目标已存在: {dst}")
    shutil.move(str(pending), str(dst))


def rename_chain_in_root(root, o2n, old_ids, dry_run, max_depth=3, _depth=0):
    if not root.exists() or _depth > max_depth:
        return 0
    count = 0
    for p in sorted(root.iterdir()):
        name = p.name
        if name.startswith('__pending__'):
            continue
        if EXP388_RE.match(name):
            continue
        # exp 形式
        em = EXP_RE.match(name)
        tm = TASK_RE.match(name)
        old_id = None
        rest = ''
        if em:
            old_id = em.group(1)
            rest = (em.group(2) or '') + (em.group(3) or '')
        elif tm:
            old_id = tm.group(1)
            rest = tm.group(2)
        if old_id is None:
            if p.is_dir() and _depth < max_depth:
                count += rename_chain_in_root(p, o2n, old_ids, dry_run, max_depth, _depth + 1)
            continue
        if old_id not in old_ids:
            continue
        # 反复迭代到稳定
        final_id = resolve_id_chain(old_id, o2n, old_ids)
        if final_id == old_id:
            continue
        # 计算中间步骤 (记录链)
        chain = []
        cur = old_id
        while cur != final_id:
            nxt = o2n[cur]
            chain.append((cur, nxt))
            cur = nxt
        # 执行链式 rename:old → final (走中间节点会撞名,所以直接跳到 final)
        prefix = 'exp' if em else 'task'
        new_name = f"{prefix}{final_id}{rest}"
        if new_name == name:
            continue
        if not dry_run:
            if (p.parent / new_name).exists():
                # 目标已存在:大概率是 chain rename 已经走到 final 的"幽灵"文件
                # 跳过(实际场景:target 已经被同一 process 改过一次)
                continue
            safe_move(p, p.parent / new_name)
            if chain:
                chain_str = " → ".join(f"{a}→{b}" for a, b in chain)
                print(f"  ✓ {name} -> {new_name} ({chain_str})")
            else:
                print(f"  ✓ {name} -> {new_name}")
        count += 1
    return count


def main():
    o2n, old_ids = load_mapping()
    print(f"Mapping: {len(o2n)} entries, {len(old_ids)} OLD ids")

    total = 0
    # 1. results/ 顶层 exp
    print("\n=== results/ 顶层 exp ===")
    cnt = rename_chain_in_root(REPO / "GRID/task_artifacts/results", o2n, old_ids, False, max_depth=0)
    total += cnt
    print(f"  ({cnt})")

    # 2. results/exp*/ 内
    print("\n=== results/exp*/ ===")
    cnt = 0
    for ed in sorted((REPO / "GRID/task_artifacts/results").iterdir()):
        if ed.is_dir() and ed.name.startswith('exp') and not EXP388_RE.match(ed.name):
            cnt += rename_chain_in_root(ed, o2n, old_ids, False, max_depth=2)
    total += cnt
    print(f"  ({cnt})")

    # 3. logs/{train,inference}/runs/
    for sub in ['train', 'inference']:
        print(f"\n=== logs/{sub}/runs/ ===")
        runs = REPO / f"GRID/logs/{sub}/runs"
        if not runs.exists(): continue
        cnt = 0
        for ts in sorted(runs.iterdir()):
            if ts.is_dir():
                cnt += rename_chain_in_root(ts, o2n, old_ids, False, max_depth=1)
        total += cnt
        print(f"  ({cnt})")

    # 4. GRID/result/
    print("\n=== GRID/result/ ===")
    cnt = rename_chain_in_root(REPO / "GRID/result", o2n, old_ids, False, max_depth=3)
    total += cnt
    print(f"  ({cnt})")

    # 5. result/
    print("\n=== result/ ===")
    cnt = rename_chain_in_root(REPO / "result", o2n, old_ids, False, max_depth=3)
    total += cnt
    print(f"  ({cnt})")

    # 6. verdicts/
    print("\n=== verdicts/ ===")
    cnt = rename_chain_in_root(REPO / "GRID/task_artifacts/verdicts", o2n, old_ids, False, max_depth=1)
    total += cnt
    print(f"  ({cnt})")

    print(f"\n=== Total: {total} ===")

    # 再次跑应该全 0
    print("\n--- 2nd run (should be 0) ---")
    cnt2 = 0
    for r in [REPO / "GRID/task_artifacts/results", REPO / "GRID/logs/train/runs",
              REPO / "GRID/logs/inference/runs", REPO / "GRID/result",
              REPO / "result", REPO / "GRID/task_artifacts/verdicts"]:
        for p in r.rglob("*"):
            if not p.is_file() and not p.is_dir(): continue
            n = p.name
            if n.startswith('__pending__'): continue
            em = EXP_RE.match(n)
            tm = TASK_RE.match(n)
            oid = (em and em.group(1)) or (tm and tm.group(1))
            if oid is None: continue
            if oid not in old_ids: continue
            final = resolve_id_chain(oid, o2n, old_ids)
            if final != oid:
                print(f"  ⚠️ still not stable: {p} (id={oid}, final={final})")
                cnt2 += 1
    print(f"  unstable: {cnt2}")


if __name__ == "__main__":
    sys.exit(main() or 0)