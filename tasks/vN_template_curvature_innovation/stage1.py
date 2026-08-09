#!/usr/bin/env python3
# -*- coding: utf-8
"""vN Stage 1 脚手架 — xxx (写创新).

R34: 本目录含 4 阶段脚本 (stage1.py / stage2.py / stage3.py / stage4.py)
R30: 所有路径硬编码在本脚本, 不读 os.environ.get
R36: Stage 1 仅做数据集 / 输入准备, 不引入新曲率机制; 曲率创新在 stage2/stage3
R37: 必须从上一最优版本 (当前 v85p_SID, test_R@10=0.105977) 复用, 禁在失败品上叠加

xxx (写创新) — 在此填写本版本相对基线的 Stage 1 差异:
  · 输入数据 / 切分 / 编码侧的差异
  · 例如: 多曲率分层输入 / 复用 v85p SID 但改 tokenization / 引入额外几何特征
  · 严禁 (R36): 通过 LR / dropout / label_smoothing / WD sweep 提升指标

启动:
    CUDA_VISIBLE_DEVICES=0 python3 -u tasks/vN_template_curvature_innovation/stage1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# 硬编码路径 (R30)
# ============================================================

REPO = Path("/fs04/ar57/wenyu/GeneRec")
HISTORY_V85P = REPO / "taskA" / "_history" / "taskA_stage2_v15_capmatch_1000ep"  # R37 基线
OUTPUT_DIR = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage1"

# ============================================================
# 硬编码参数 (R30 禁 env, R36 禁 sweep)
# ============================================================

SEED = 42
BEAM_SIZE = 20  # R35: 评估默认 beam=20


def main() -> None:
    """Stage 1 入口 — xxx (写创新)."""
    print(f"[stage1] xxx 写创新 — 此处填写 Stage 1 实际逻辑")
    print(f"[stage1] 输出目录: {OUTPUT_DIR}")
    print(f"[stage1] 复用基线 (R37): {HISTORY_V85P}")
    print(f"[stage1] seed={SEED}  beam_size={BEAM_SIZE}")
    # TODO (xxx 写创新): 实现 Stage 1 数据准备 / 输入编码 / 几何特征构造
    raise NotImplementedError("xxx (写创新) — Stage 1 待实现")


if __name__ == "__main__":
    main()