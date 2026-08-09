#!/usr/bin/env python3
# -*- coding: utf-8
"""vN Stage 4 脚手架 — xxx (写创新).

R34: 本目录含 4 阶段脚本 (stage1.py / stage2.py / stage3.py / stage4.py)
R30: 所有路径硬编码在本脚本, 不读 os.environ.get
R35: 评估强约束 — 单 checkpoint + beam_search=20, 禁 Borda / ensemble / 多 ckpt 融合
R36: Stage 4 不引入新曲率机制, 仅做评估; 曲率创新在 stage2/stage3
R37: 必须从 v85p_SID 基线 (test_R@10=0.105977) 评估判定, 差则回退

xxx (写创新) — 在此填写本版本相对基线的 Stage 4 评估差异 (如有):
  · 评估指标扩展 (例如新增曲率一致性指标)
  · 评估子集切分 (例如分层指标)
  · 严禁 (R35): Borda Rank Fusion / ensemble 多 ckpt / beam≠20 → 立即 raise
  · 严禁 (R36): 通过调评估侧参数掩盖训练问题 → 立即 raise

启动:
    CUDA_VISIBLE_DEVICES=0 python3 -u tasks/vN_template_curvature_innovation/stage4.py
"""
from __future__ import annotations

from pathlib import Path

# ============================================================
# 硬编码路径 (R30)
# ============================================================

REPO = Path("/fs04/ar57/wenyu/GeneRec")
STAGE3_BEST_CKPT = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage3" / "HG_Rec_best.pth"
STAGE2_SID_NPY = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage2" / "sid_output.npy"
OUTPUT_DIR = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage4_eval"

# ============================================================
# 硬编码参数 (R30 禁 env, R35 单 ckpt + beam=20)
# ============================================================

SEED = 42
BEAM_SIZE = 20  # R35: 默认 beam=20, 禁 30/50


def main() -> None:
    """Stage 4 入口 — xxx (写创新)."""
    print(f"[stage4] xxx 写创新 — 此处填写 Stage 4 评估")
    print(f"[stage4] Stage 3 best ckpt: {STAGE3_BEST_CKPT}")
    print(f"[stage4] Stage 2 SID: {STAGE2_SID_NPY}")
    print(f"[stage4] 输出目录: {OUTPUT_DIR}")
    print(f"[stage4] seed={SEED}  beam_size={BEAM_SIZE} (R35: 必须 20)")
    # TODO (xxx 写创新): 实现 Stage 4 评估 + R35 合规检查 + 写 eval_test.json
    raise NotImplementedError("xxx (写创新) — Stage 4 待实现")


if __name__ == "__main__":
    main()