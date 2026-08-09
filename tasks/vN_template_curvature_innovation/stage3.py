#!/usr/bin/env python3
# -*- coding: utf-8
"""vN Stage 3 脚手架 — xxx (写创新).

R34: 本目录含 4 阶段脚本 (stage1.py / stage2.py / stage3.py / stage4.py)
R30: 所有路径硬编码在本脚本, 不读 os.environ.get
R36: Stage 3 关键创新点 — κ frozen→learnable / 新曲率正则项 / Poincaré / Minkowski / Lorentz 曲率机制变更
R37: 必须从 v85p_SID 基线 (test_R@10=0.105977) 复用, 禁在失败品上叠加

xxx (写创新) — 在此填写本版本相对基线的 Stage 3 曲率机制差异:
  · T5 decoder 训练阶段的曲率改动 — 例如:
      - κ frozen → learnable (per-layer / per-head)
      - 引入新曲率正则项 (曲率平滑 / 曲率稀疏 / 曲率一致性)
      - 换曲率模型 (Poincaré ball → Lorentz / Minkowski 双曲面)
      - per-layer 几何残差 / 分桶曲率 / HAB 距离差调制 (ΔD mode)
  · 严禁 (R36): 单纯调 LR / dropout / label_smoothing / WD 提升指标 → 立即 raise

启动:
    CUDA_VISIBLE_DEVICES=0 python3 -u tasks/vN_template_curvature_innovation/stage3.py
"""
from __future__ import annotations

from pathlib import Path

# ============================================================
# 硬编码路径 (R30)
# ============================================================

REPO = Path("/fs04/ar57/wenyu/GeneRec")
STAGE2_SID = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage2" / "sid_output.npy"
STAGE2_KAPPA_CKPT = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage2" / "hrqvae_kappa_sync.ckpt"
OUTPUT_DIR = REPO / "taskA" / "_history" / "vN_template_curvature_innovation_stage3"

# ============================================================
# 硬编码参数 (R30 禁 env, R36 禁 sweep)
# ============================================================

SEED = 42


def main() -> None:
    """Stage 3 入口 — xxx (写创新)."""
    print(f"[stage3] xxx 写创新 — 此处填写 Stage 3 曲率机制")
    print(f"[stage3] Stage 2 SID 输入: {STAGE2_SID}")
    print(f"[stage3] Stage 2 κ ckpt: {STAGE2_KAPPA_CKPT}")
    print(f"[stage3] 输出目录: {OUTPUT_DIR}")
    print(f"[stage3] seed={SEED}")
    # TODO (xxx 写创新): 实现 Stage 3 T5 decoder 训练 / 曲率机制
    raise NotImplementedError("xxx (写创新) — Stage 3 待实现")


if __name__ == "__main__":
    main()