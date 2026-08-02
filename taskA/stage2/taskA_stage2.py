#!/usr/bin/env python3
"""taskA Stage 2 统一入口. 每 stage 顶层只保留本脚本; 历史实现归档在 _archive/.

用法:
  python3 taskA/stage2/taskA_stage2.py --mode <name> [透传参数...]
  python3 taskA/stage2/taskA_stage2.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "_archive"
PROJECT = STAGE.parent.parent

MODES = {
    'kappa_sync': ('taskA_stage2_kappa_sync.py', 'Task #448 / Issue #157 [方向A Gate2] κ同步重校准的RQ-VAE代码本与完整SID链路验证  R18 4 维度路径对比 vs I'),
    'kappa_vq_fix': ('taskA_stage2_kappa_vq_fix.py', 'Task #468 / Issue #175 [方向A Gate2] κ 经代码本量化距离直接进入 VQ 损失的 forward-path 修复  Per Is'),
    'mckg_codebook': ('taskA_stage2_mckg_codebook.py', 'Task #174 Stage 2 fork: 用 MCKGGatingHRQVAE 加载 D 臂 ckpt (含 gate_net 参数).  Mirror')
}
DEFAULT_MODE = 'kappa_vq_fix.py'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=DEFAULT_MODE, help=f"实验变体 (默认 {DEFAULT_MODE})")
    ap.add_argument("--list", action="store_true", help="列出所有可用 mode")
    args, rest = ap.parse_known_args()
    if args.list:
        for m, (f, d) in MODES.items():
            print(f"  {m:36s} -> _archive/{f}  ({d})")
        return
    if args.mode not in MODES:
        raise KeyError(f"未知 mode={args.mode!r}, 可用: {sorted(MODES)}")
    script, desc = MODES[args.mode]
    target = ARCHIVE / script
    if not target.exists():
        raise FileNotFoundError(f"归档脚本缺失: {target}")
    print(f"[taskA_stage2] mode={args.mode} -> _archive/{script} ({desc})", flush=True)
    cmd = [sys.executable, str(target)] + rest
    raise SystemExit(subprocess.run(cmd, cwd=str(PROJECT)).returncode)


if __name__ == "__main__":
    main()
