#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 stub: 调用 common/stage4 主脚本, 硬编码 v15 capmatch SID + HAB Stage 3 ckpt.

实际运行见 v15_pureT5_highLR/stage4.py + 各 task 的 stage4_beam*.py / borda*.py.
本 stub 仅作 R34 4 脚本结构补全, 默认输出 'eval_test.json' 路径.
"""
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")


def main():
    print(f"[stage4] see borda*N*way.py / stage4_beam*.py for actual eval")
    print(f"[stage4] eval uses common/stage4/stage4_eval_pure_t5_v85p_4layer.py with:")
    print(f"  --sid_npy  /fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy")
    print(f"  --expected_sid_sha 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07")
    print(f"  --hyperbolic_attn_bias --enable_residual_hab")


if __name__ == "__main__":
    main()
