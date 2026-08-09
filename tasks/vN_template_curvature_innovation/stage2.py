#!/usr/bin/env python3
# -*- coding: utf-8
"""vN Stage 2 脚手架 — xxx (写创新).

R34: 本目录含 4 阶段脚本 (stage1.py / stage2.py / stage3.py / stage4.py)
R30: 所有路径硬编码在本脚本, 不读 os.environ.get
R36: Stage 2 关键创新点 — Stage 2 κ 学习 (K / REC_LAYER_W / capmatch 上限)
R37: 必须从 v85p_SID 基线 (test_R@10=0.105977) 复用 SID 输出, 禁在失败品上叠加

xxx (写创新) — 在此填写本版本相对基线的 Stage 2 曲率机制差异:
  · SID (Semantic ID) 生成侧的曲率改动 — 例如:
      - 换 κ 初值 / κ 正则 (Lipschitz 约束强度)
      - 改 capmatch 容量匹配 (K 序列 / REC_LAYER_W / λ)
      - 引入多曲率分层 SID (Poincaré + Lorentz 双曲率)
      - 改 RQ-VAE 量化层结构 (差分长度 codebook / 残差 codebook)
  · 严禁 (R36): 单纯调 LR / dropout / WD 提升指标 → 立即 raise

启动:
    CUDA_VISIBLE_DEVICES=0 python3 -u tasks/vN_template_curvature_innovation/stage2.py
"""
from __future__ import annotations

from pathlib import Path

# ============================================================
# 硬编码路径 (R30) — R37 复用 v85p_SID 基线 SID 输出
# ============================================================

REPO = Path("/fs04/ar57/wenyu/GeneRec")
BASELINE_SID_NPY = REPO / "taskA" / "_history" / "taskA_stage2_v15_capmatch_1000ep" / "sid_output.npy"
BASELINE_SID_SHA = "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07"  # R37 锚点
OUTPUT_DIR = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage2"

# ============================================================
# 硬编码参数 (R30 禁 env, R36 禁 sweep, R37 复用基线)
# ============================================================

SEED = 42


def main() -> None:
    """Stage 2 入口 — xxx (写创新)."""
    print(f"[stage2] xxx 写创新 — 此处填写 Stage 2 曲率机制")
    print(f"[stage2] 基线 SID (R37 复用): {BASELINE_SID_NPY}")
    print(f"[stage2] 基线 SHA: {BASELINE_SID_SHA}")
    print(f"[stage2] 输出目录: {OUTPUT_DIR}")
    print(f"[stage2] seed={SEED}")
    # TODO (xxx 写创新): 实现 Stage 2 SID / κ 学习 / 量化层改动
    raise NotImplementedError("xxx (写创新) — Stage 2 待实现")


if __name__ == "__main__":
    main()