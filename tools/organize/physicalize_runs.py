"""physicalize_runs.py — 把 products/ 下所有软链接替换为物理复制。

逻辑:
  1. 遍历 products/**/{train,inference} 下的 symlink
  2. 解析 readlink 找到物理源(在 GRID/logs/{train,inference}/runs/)
  3. 删 symlink
  4. cp -r 源到目标位置(目标位置是 products/.../train/<name> 或 inference/<name>)
  5. 验证大小一致
"""
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
PRODUCTS = REPO / "products"


def main() -> int:
    symlinks = []
    for sub in ['train', 'inference']:
        for p in PRODUCTS.rglob(f"*/{sub}/*"):
            if p.is_symlink():
                symlinks.append(p)

    print(f"Found {len(symlinks)} symlinks to physicalize")
    total_size = 0
    success = 0
    failed = []

    for sl in sorted(symlinks):
        target = sl.resolve()  # 跟随 symlink 到真实路径
        if not target.exists():
            print(f"  ⚠️  {sl.relative_to(REPO)} -> {target} (target gone)")
            failed.append(sl)
            continue
        sz = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
        total_size += sz
        # 删 symlink
        sl.unlink()
        # cp -r 源到目标位置
        try:
            shutil.copytree(target, sl, symlinks=False)
            actual = sum(f.stat().st_size for f in sl.rglob('*') if f.is_file())
            if actual != sz:
                print(f"  ❌ size mismatch: {sl.relative_to(REPO)} src={sz} dst={actual}")
                failed.append(sl)
            else:
                success += 1
        except Exception as e:
            print(f"  ❌ {sl.relative_to(REPO)}: {e}")
            failed.append(sl)

    print(f"\n=== Done: {success}/{len(symlinks)} physicalized, {len(failed)} failed ===")
    print(f"Total data physically moved: {total_size/1024/1024/1024:.2f} GB")
    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"  {f.relative_to(REPO)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
