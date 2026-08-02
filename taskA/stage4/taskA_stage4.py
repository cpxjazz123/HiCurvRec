#!/usr/bin/env python3
"""taskA Stage 4 统一入口. 每 stage 顶层只保留本脚本; 历史实现归档在 _archive/.

用法:
  python3 taskA/stage4/taskA_stage4.py --mode <name> [透传参数...]
  python3 taskA/stage4/taskA_stage4.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "_archive"
PROJECT = STAGE.parent.parent

MODES = {
    'beam20_reeval': ('taskA_stage4_beam20_reeval.py', 'Issue #30 [方向A]: baseline 同协议 beam20 重测现有 adapter.  根因修复验证 (verdicts/_analysis_w'),
    'canary_issue12_step2_step3': ('taskA_stage4_canary_issue12_step2_step3.py', 'Issue #12 [方向A P2/P4 Step 2+3] canary 验证 autoregressive_predict 修复 P4.  跟 taskA_'),
    'canary_issue4_step1': ('taskA_stage4_canary_issue4_step1.py', 'Issue #4 [方向A Gate4 Step 1] canary 验证修复后 compute_r_at_k on long-run best_adapter')
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
    print(f"[taskA_stage4] mode={args.mode} -> _archive/{script} ({desc})", flush=True)
    cmd = [sys.executable, str(target)] + rest
    raise SystemExit(subprocess.run(cmd, cwd=str(PROJECT)).returncode)


if __name__ == "__main__":
    main()
