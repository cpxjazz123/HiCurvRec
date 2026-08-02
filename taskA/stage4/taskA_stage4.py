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
    'beam_search_issue12_step4': ('taskA_stage4_beam_search_issue12_step4.py', 'Issue #12 [方向A Step 4] beam search K=20 eval — 让 6 指标互不恒等.  Per-sample 4-step be'),
    'canary_argmax': ('taskA_stage4_canary_argmax.py', 'Task #474 / Issue #186 [方向A Gate3→Gate4] 协议对齐 canary 真实 argmax sanity check.  Is'),
    'canary_autoregressive': ('taskA_stage4_canary_autoregressive.py', 'Issue #190 [方向A Gate3协议对齐] 自回归逐步 argmax + 逐层合法SID约束 canary.  vs Issue #188 (task'),
    'canary_issue12_step2_step3': ('taskA_stage4_canary_issue12_step2_step3.py', 'Issue #12 [方向A P2/P4 Step 2+3] canary 验证 autoregressive_predict 修复 P4.  跟 taskA_'),
    'canary_issue4_step1': ('taskA_stage4_canary_issue4_step1.py', 'Issue #4 [方向A Gate4 Step 1] canary 验证修复后 compute_r_at_k on long-run best_adapter'),
    'full_eval_issue14': ('taskA_stage4_full_eval_issue14.py', 'Issue #14 [方向A Step 1] Task84 test 全量评估 — beam K=20 六指标产出.  跟 #12 Step 4 taskA_s'),
    'kappa_codebook_sync_issue14': ('taskA_stage4_kappa_codebook_sync_issue14.py', 'Issue #14 [方向A Step 2] κ/codebook 同步性取证 (方向A 特有).  任务: 证明 κ 更新后 codebook / 距离度量同'),
    'root_cause_issue14_step3': ('taskA_stage4_root_cause_issue14_step3.py', 'Issue #14 [方向A Step 3] R@10 < 0.1020 根因诊断.  Issue #14 实测: 全量 R@10 = 0.0724 (-29%')
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
