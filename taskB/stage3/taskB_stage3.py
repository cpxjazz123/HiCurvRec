#!/usr/bin/env python3
"""taskB Stage 3 统一入口. 每 stage 顶层只保留本脚本; 历史实现归档在 _archive/.

用法:
  python3 taskB/stage3/taskB_stage3.py --mode <name> [透传参数...]
  python3 taskB/stage3/taskB_stage3.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "_archive"
PROJECT = STAGE.parent.parent

MODES = {
    'issue193_long_run': ('taskB_stage3_issue193_long_run.py', 'Issue #193 [方向B Gate4] 几何混合分量单seed正式长训+Task84六项评估.  承接 Issue #191 commit 2368e44'),
    'issue23_option_d': ('taskB_stage3_issue23_option_d.py', 'Issue #23 [B Gate3] option_D  - conditioner  kappa  ( #20 ).  per #23 spec: - :'),
    'issue29_gate2_diag': ('taskB_stage3_issue29_gate2_diag.py', 'Issue #29 [方向B Gate2] 混合曲率训练稳定性与表示流收口 — 短程单 seed 诊断.  Per #29 spec: - 诊断固定双曲分量、欧'),
    'mixed_curv_recontinue': ('taskB_stage3_mixed_curv_recontinue.py', 'Task #471 / Issue #178 [方向B Gate3续] 加权混合曲率元数据 T5 零中心有界混合残差验证.  R18 + Issue #178')
}
DEFAULT_MODE = 'issue29_gate2_diag.py'


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
    print(f"[taskB_stage3] mode={args.mode} -> _archive/{script} ({desc})", flush=True)
    cmd = [sys.executable, str(target)] + rest
    raise SystemExit(subprocess.run(cmd, cwd=str(PROJECT)).returncode)


if __name__ == "__main__":
    main()
