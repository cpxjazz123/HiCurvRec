"""chain_resolve.py - 把所有 path 中的 id chain resolve 到 NEW 空间。"""
import pathlib, json, re, shutil

REPO = pathlib.Path('.')
m = json.load(open('tools/renumber/mapping.json'))
o2n = {k: str(v) for k, v in m['old_to_new'].items()}
old_ids = set(str(x) for x in m['old_ids'])


def resolve(id_str):
    """迭代直到 id 不在 OLD set(到 NEW id 空间)。"""
    seen = set()
    cur = id_str
    while cur in old_ids:
        if cur in seen:
            return cur
        seen.add(cur)
        nxt = o2n.get(cur)
        if nxt is None or nxt == cur:
            return cur
        cur = nxt
    return cur


TASK_RE = re.compile(r'^(task|exp)(\d+)(v\d+)?(.*)$')


def rename_one(p):
    m = TASK_RE.match(p.name)
    if not m:
        return False
    prefix, oid, vsuffix, rest = m.group(1), m.group(2), m.group(3) or '', m.group(4)
    if oid not in old_ids:
        return False
    final = resolve(oid)
    if final == oid:
        return False
    new_name = f'{prefix}{final}{vsuffix}{rest}'
    if new_name == p.name:
        return False
    if (p.parent / new_name).exists():
        return False
    pending = p.parent / f'__pending__{p.name}'
    shutil.move(str(p), str(pending))
    if (p.parent / new_name).exists():
        shutil.move(str(pending), str(p))
        return False
    shutil.move(str(pending), str(p.parent / new_name))
    return True


ROOTS = [
    REPO / 'GRID/task_artifacts/results',
    REPO / 'GRID/logs/train/runs',
    REPO / 'GRID/logs/inference/runs',
    REPO / 'GRID/result',
    REPO / 'result',
    REPO / 'GRID/task_artifacts/verdicts',
    REPO / 'GRID/task_artifacts/products',
]

for round_n in range(20):
    changed = 0
    for r in ROOTS:
        if not r.exists():
            continue
        all_paths = sorted(r.rglob('*'), key=lambda x: -len(str(x)))
        for p in all_paths:
            if not p.exists():
                continue
            if p.name.startswith('__pending__'):
                continue
            if not (p.is_file() or p.is_dir()):
                continue
            if p.is_symlink():
                continue
            try:
                if rename_one(p):
                    changed += 1
            except Exception:
                pass
    print(f'round {round_n+1}: changed {changed}')
    if changed == 0:
        break