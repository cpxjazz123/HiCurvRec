#!/usr/bin/env python3
"""taskA Stage 1 统一入口. 每 stage 顶层只保留本脚本; 历史实现归档在 _archive/.

用法:
  python3 taskA/stage1/taskA_stage1.py --mode <name> [透传参数...]
  python3 taskA/stage1/taskA_stage1.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "_archive"
PROJECT = STAGE.parent.parent

MODES = {
    'mckg_gating': ('taskA_stage1_mckg_gating.py', 'Task #174 Stage 1 — MCKG门控融合HRQVAE (用户 design §5 D 臂).  D 臂 = M=2 components + 门'),
    'orc_locked': ('taskA_stage1_orc_locked.py', 'Task #175 Stage 1 — κ-Stereographic + κ LOCKED at Ollivier ORC 实测值.  Hypothesis'),
    'riemannian_adamw': ('taskA_stage1_riemannian_adamw.py', 'Task #157 — Issue #55 Stage 1 训练 (Riemannian AdamW + per-layer s_l, GPU 3).  跟 T')
}
DEFAULT_MODE = 'riemannian_adamw.py'


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
    print(f"[taskA_stage1] mode={args.mode} -> _archive/{script} ({desc})", flush=True)
    cmd = [sys.executable, str(target)] + rest
    raise SystemExit(subprocess.run(cmd, cwd=str(PROJECT)).returncode)


if __name__ == "__main__":
    main()
