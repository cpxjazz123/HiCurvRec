#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 stub: 调用 common eval 脚本, v15 capmatch SID + HAB Stage 3 ckpt."""
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")


def main():
    print(f"[stage4] uses common/stage4/stage4_eval_pure_t5_v85p_4layer.py with:")
    print(f"  --sid_npy  /fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy")
    print(f"  --expected_sid_sha 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07")
    print(f"  --hyperbolic_attn_bias --enable_residual_hab --hab_lambda_max 0.3 (vs v15 0.2)")


if __name__ == "__main__":
    main()
