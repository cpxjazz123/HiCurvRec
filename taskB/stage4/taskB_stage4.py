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
    'beam_search_issue13_step6': ('taskB_stage4_beam_search_issue13_step6.py', 'Issue #13 [方向B Step 6] beam search K=20 eval — 让 6 指标互不恒等.  Per-sample 4-step be'),
    'canary_argmax': ('taskB_stage4_canary_argmax.py', 'Task #475 / Issue #187 [方向B Gate3→Gate4] 协议对齐 canary 真实 argmax sanity check.  Is'),
    'canary_autoregressive': ('taskB_stage4_canary_autoregressive.py', 'Issue #191 [方向B Gate3协议对齐] 自回归逐步 argmax + 逐层合法SID约束 + mixing weights 非退化核查.  vs'),
    'canary_issue13_step1': ('taskB_stage4_canary_issue13_step1.py', 'Issue #13 [方向B Step 1] protocol audit on taskB with long-run best_adapter.pt.  跟'),
    'canary_issue13_step2_step3': ('taskB_stage4_canary_issue13_step2_step3.py', 'Issue #13 [方向B Step 2+3] canary 验证 autoregressive_predict 修复 taskB P4.  跟 taskA_'),
    'canary_issue5_step1': ('taskB_stage4_canary_issue5_step1.py', 'Issue #5 [方向B Gate4 Step 1] canary 验证修复后 compute_r_at_k on long-run best_adapter'),
    'curvature_fix_verify_issue15_step1': ('taskB_stage4_curvature_fix_verify_issue15_step1.py', 'Issue #15 [方向B Step 1] 数值验证 Stage3 curvature_meta spec/impl mismatch 修复.  任务 (pe'),
    'mixing_diagnostic_issue13_step5': ('taskB_stage4_mixing_diagnostic_issue13_step5.py', 'Issue #13 [方向B Step 5] 混合分量可审计诊断.  任务 (Issue #13 spec): - 三层独立 learnable κ + 固定双')
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
