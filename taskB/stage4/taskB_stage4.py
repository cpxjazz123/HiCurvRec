#!/usr/bin/env python3
"""taskB Stage 4 统一入口. 每 stage 顶层只保留本脚本; 历史实现归档在 _archive/.

用法:
  python3 taskB/stage4/taskB_stage4.py --mode <name> [透传参数...]
  python3 taskB/stage4/taskB_stage4.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "_archive"
PROJECT = STAGE.parent.parent

MODES = {
    'beam20_reeval': ('taskB_stage4_beam20_reeval.py', 'Issue #30 [方向B]: baseline 同协议 beam20 重测现有 adapter.  与 taskA_stage4_beam20_reeval'),
    'canary_argmax': ('taskB_stage4_canary_argmax.py', 'Task #475 / Issue #187 [方向B Gate3→Gate4] 协议对齐 canary 真实 argmax sanity check.  Is')
}
DEFAULT_MODE = 'beam20_reeval.py'


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
    print(f"[taskB_stage4] mode={args.mode} -> _archive/{script} ({desc})", flush=True)
    cmd = [sys.executable, str(target)] + rest
    raise SystemExit(subprocess.run(cmd, cwd=str(PROJECT)).returncode)


if __name__ == "__main__":
    main()
