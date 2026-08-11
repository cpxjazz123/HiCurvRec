#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v35 Stage 1 stub — 复用 v82 baseline item_emb (R34 允许 stub if 产物已存在).

v35 = Stage3 Hyperbolic Contrastive aux Loss (Issue #112).
Stage1 (item embedding) 与 v23 / v22.b / 一切 HG_Rec baseline 完全一致, 不需重跑.

R30 严格: 硬编码路径常量, 不读 os.environ.
R34 合规: 单脚本入口, 不 fork 版本.
"""
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/item_emb_baseline.npy"


def main():
    p = Path(ITEM_EMB_NPY)
    if not p.exists():
        raise FileNotFoundError(f"Stage1 baseline item_emb 不存在: {ITEM_EMB_NPY}")
    print(f"[v35-stage1] reuse Stage1 baseline item_emb from {p} (size={p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()