"""
move_to_categories.py — Step 5+6 of GRID organize-dir.

把产物物理移动到 3 个顶层类别目录:
  - GRID/task_artifacts/descriptions/  : task_definitions/*.md
  - GRID/task_artifacts/verdicts/      : *verdict*.md
  - GRID/task_artifacts/products/task<N>/{ckpts,sid_tensors,eval,train,inference,paper}/

train/inference 子目录是软链接,指向原 GRID/logs/{train,inference}/runs/task<NEW>_*。
其它子目录是物理搬运。

用法:
  python move_to_categories.py [--dry-run]
"""
import json
import pathlib
import re
import shutil
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
KEEP = REPO / "tools/organize/keep_set.json"
TD = REPO / "GRID/task_artifacts/task_definitions"
EXP = REPO / "GRID/task_artifacts/results"
LOG_TRAIN = REPO / "GRID/logs/train/runs"
LOG_INF = REPO / "GRID/logs/inference/runs"
RESULT_TOP = REPO / "result"
RESULT_GRID = REPO / "GRID/result"

# 目标根
DESCRIPTIONS = REPO / "GRID/task_artifacts/descriptions"
VERDICTS = REPO / "GRID/task_artifacts/verdicts"
PRODUCTS = REPO / "GRID/task_artifacts/products"


def safe_move(src: pathlib.Path, dst: pathlib.Path) -> None:
    """shutil.move 到 dst。dst 父目录若不存在,自动创建。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")
    shutil.move(str(src), str(dst))


def move_descriptions(keep: list, dry_run: bool) -> int:
    """Phase 3a: 把 28 个 keep task 的 task_definitions 移到 descriptions/。"""
    cnt = 0
    print(f"\n=== Move descriptions ===")
    DESCRIPTIONS.mkdir(parents=True, exist_ok=True)
    for k in keep:
        src = TD / k['filename']
        if not src.exists():
            print(f"  ⚠️  not found: {src}")
            continue
        dst = DESCRIPTIONS / k['filename']
        if dry_run:
            print(f"  [DRY] {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
        else:
            if dst.exists():
                pass
            else:
                safe_move(src, dst)
                print(f"  ✓ {src.name} -> descriptions/")
            cnt += 1
    # task.md 模板也搬
    tpl = TD / 'task.md'
    if tpl.exists():
        if dry_run:
            print(f"  [DRY] task.md -> descriptions/task.md")
        else:
            shutil.move(str(tpl), DESCRIPTIONS / 'task.md')
            print(f"  ✓ task.md -> descriptions/")
        cnt += 1
    return cnt


def move_verdicts(keep: list, dry_run: bool, o2n: dict[str, str]) -> int:
    """Phase 6: 从 results/exp*/task<OLD_or_NEW>_*verdict*.md 物理搬到 verdicts/。

    文件名可能嵌 OLD id(还没被 Phase 4 改)或 NEW id(已改)。两者都要识别。"""
    cnt = 0
    print(f"\n=== Move verdicts ===")
    VERDICTS.mkdir(parents=True, exist_ok=True)
    keep_nids = {k['new_id'] for k in keep}
    for ed in sorted(EXP.iterdir()):
        if not ed.is_dir() or not ed.name.startswith('exp'): continue
        if re.match(r'exp388', ed.name):
            continue
        for f in sorted(ed.iterdir()):
            if not (f.is_file() and 'verdict' in f.name.lower() and f.name.endswith('.md')):
                continue
            fm = re.match(r'task(\d+)_', f.name)
            if not fm: continue
            raw_id = fm.group(1)
            # OLD id 转 NEW id
            nid = o2n.get(raw_id, raw_id)
            if nid not in keep_nids:
                continue
            dst = VERDICTS / f.name
            if dry_run:
                print(f"  [DRY] {f.relative_to(REPO)} -> verdicts/{f.name}")
            else:
                if dst.exists():
                    pass
                else:
                    safe_move(f, dst)
                    print(f"  ✓ {f.name}")
                cnt += 1
    return cnt


def move_products(keep: list, dry_run: bool, o2n: dict[str, str]) -> int:
    """Phase 5: 把 ckpt/pt/eval/log/paper 按 task 聚合到 products/task<N>/。"""
    cnt = 0
    print(f"\n=== Move products ===")
    PRODUCTS.mkdir(parents=True, exist_ok=True)
    keep_nids = {k['new_id'] for k in keep}

    for k in keep:
        nid = k['new_id']
        task_dir = PRODUCTS / f"task{nid}"
        for sub in ['ckpts', 'sid_tensors', 'eval', 'paper']:
            (task_dir / sub).mkdir(parents=True, exist_ok=True)

        # 1. ckpts/ — 从 results/exp<NEW>/task<NEW>_*.ckpt
        # 2. sid_tensors/ — 从 results/exp<NEW>/task<NEW>_*.pt
        # 3. eval/ — 从 results/exp<NEW>/task<NEW>_*_eval.json
        # 4. paper/ — 从 results/exp<NEW>/p5_paper_section_*.md (按 NEW id)
        for ed in sorted(EXP.iterdir()):
            if not ed.is_dir() or not ed.name.startswith('exp'): continue
            if re.match(r'exp388', ed.name): continue
            for f in sorted(ed.iterdir()):
                if not f.is_file(): continue
                # 必须是 task<OLD_or_NEW>_<rest>;OLD 转 NEW 后比对
                fm = re.match(r'task(\d+)_', f.name)
                if not fm: continue
                raw_id = fm.group(1)
                if o2n.get(raw_id, raw_id) != nid: continue
                # 分类
                if f.name.endswith('.ckpt'):
                    dst = task_dir / 'ckpts' / f.name
                elif f.name.endswith('.pt'):
                    dst = task_dir / 'sid_tensors' / f.name
                elif f.name.endswith('.json') and 'eval' in f.name:
                    dst = task_dir / 'eval' / f.name
                else:
                    continue
                if dry_run:
                    print(f"  [DRY] {f.relative_to(REPO)} -> {dst.relative_to(REPO)}")
                else:
                    if dst.exists():
                        # 目标已存在(idempotent skip)
                        pass
                    else:
                        safe_move(f, dst)
                        print(f"  ✓ {f.name} -> products/task{nid}/{dst.parent.name}/")
                    cnt += 1

        # 5. paper/ — exp<NEW>/p5_paper_section_v*.md (按 NEW id)
        for ed in sorted(EXP.iterdir()):
            if not ed.is_dir() or not ed.name.startswith('exp'): continue
            if re.match(r'exp388', ed.name): continue
            em = re.match(r'exp(\d+)(v\d+)?(_.*)?$', ed.name)
            if not em or em.group(1) != nid: continue
            for f in sorted(ed.iterdir()):
                if not f.is_file(): continue
                if 'paper' not in f.name.lower(): continue
                if not (f.name.endswith('.md') or f.name.endswith('.tex')): continue
                dst = task_dir / 'paper' / f"task{nid}_{f.name}"
                if dry_run:
                    print(f"  [DRY] {f.relative_to(REPO)} -> {dst.relative_to(REPO)}")
                else:
                    if dst.exists():
                        pass
                    else:
                        safe_move(f, dst)
                        print(f"  ✓ {f.name} -> products/task{nid}/paper/")
                    cnt += 1

        # 6. train/ 和 inference/ — 软链接到 GRID/logs/{train,inference}/runs/task<nid>_*
        for log_type, log_root in [('train', LOG_TRAIN), ('inference', LOG_INF)]:
            link_dir = task_dir / log_type
            link_dir.mkdir(exist_ok=True)
            if not log_root.exists(): continue
            # 遍历 timestamp/ 或顶层
            for ts_or_dir in sorted(log_root.iterdir()):
                if not ts_or_dir.is_dir(): continue
                # 顶层 run(无 timestamp)的处理
                if re.match(r'task\d+_', ts_or_dir.name):
                    rn = ts_or_dir.name
                    rm = re.match(r'task(\d+)_', rn)
                    if rm and rm.group(1) == nid:
                        target = REPO / 'GRID/logs' / log_type / 'runs' / rn
                        link = link_dir / rn
                        if link.exists() or link.is_symlink():
                            pass  # idempotent
                        else:
                            if dry_run:
                                print(f"  [DRY] ln -s {target} -> {link.relative_to(REPO)}")
                            else:
                                rel = pathlib.Path('../../../../../logs') / log_type / 'runs' / rn
                                link.symlink_to(rel)
                                print(f"  ✓ symlink {link.relative_to(REPO)} -> {rel}")
                            cnt += 1
                    continue
                # timestamp/<run_name>
                for rn_dir in sorted(ts_or_dir.iterdir()):
                    if not rn_dir.is_dir(): continue
                    rm = re.match(r'task(\d+)_', rn_dir.name)
                    if not rm or rm.group(1) != nid: continue
                    target = rn_dir
                    link_name = f"{ts_or_dir.name}_{rn_dir.name}"
                    link = link_dir / link_name
                    if link.exists() or link.is_symlink():
                        pass
                    else:
                        if dry_run:
                            print(f"  [DRY] ln -s {target} -> {link.relative_to(REPO)}")
                        else:
                            rel = pathlib.Path('../../../../../logs') / log_type / 'runs' / ts_or_dir.name / rn_dir.name
                            link.symlink_to(rel)
                            print(f"  ✓ symlink {link.relative_to(REPO)} -> {rel}")
                        cnt += 1

    return cnt


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    keep = json.loads(KEEP.read_text())
    print(f"keep tasks: {len(keep)}")

    # 加载 OLD→NEW 映射
    o2n = {k: str(v) for k, v in json.loads((REPO / "tools/renumber/mapping.json").read_text())["old_to_new"].items()}

    n1 = move_descriptions(keep, args.dry_run)
    n2 = move_verdicts(keep, args.dry_run, o2n)
    n3 = move_products(keep, args.dry_run, o2n)

    print(f"\n=== Total moved: descriptions={n1} verdicts={n2} products={n3} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())