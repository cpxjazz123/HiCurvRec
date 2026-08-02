#!/usr/bin/env python3
"""taskA Stage 3 统一入口. 每 stage 顶层只保留本脚本; 历史实现归档在 _archive/.

用法:
  python3 taskA/stage3/taskA_stage3.py --mode <name> [透传参数...]
  python3 taskA/stage3/taskA_stage3.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "_archive"
PROJECT = STAGE.parent.parent

MODES = {
    'issue192_long_run': ('taskA_stage3_issue192_long_run.py', 'Issue #192 [方向A Gate4] κ感知优化器单seed正式长训+Task84六项评估.  承接 Issue #190 commit 2368e44'),
    'issue24_precheck_v3': ('taskA_stage3_issue24_precheck_v3.py', 'Issue #24 [A precheck v3] kappa :  layer  mean(dim=1) .   #22 (commit 0a3da44c)'),
    'issue26_t2_lr_scan': ('taskA_stage3_issue26_t2_lr_scan.py', 'Issue #26 T2  kappa_logits  param group + LR .  Per #26 spec (): - kappa_logits'),
    'issue28_gate2_diag': ('taskA_stage3_issue28_gate2_diag.py', 'Issue #28 [方向A Gate2] κ更新后重校准与训练稳定性收口 — 短程单 seed 诊断.  Per #28 spec: - 验证 L0 K64'),
    'kappa_scale_recontinue': ('taskA_stage3_kappa_scale_recontinue.py', 'Task #470 / Issue #177 [方向A Gate3续] κ同步scale元数据 T5 零中心有界残差验证.  R18 + Issue #177')
}
DEFAULT_MODE = 'issue28_gate2_diag.py'


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
    print(f"[taskA_stage3] mode={args.mode} -> _archive/{script} ({desc})", flush=True)
    cmd = [sys.executable, str(target)] + rest
    raise SystemExit(subprocess.run(cmd, cwd=str(PROJECT)).returncode)


if __name__ == "__main__":
    main()
