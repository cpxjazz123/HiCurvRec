#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 Stage 3: skip — 沿用 v18_branch_curvature Stage3 训练产物.

v25 是 Issue #238 "Stage4 Hyperbolic Re-ranking", 不重跑 Stage3.
直接沿用 v18 HG_Rec_best.pth:
  - ckpt: /fs04/ar57/wenyu/GeneRec/taskA/_history/v18_branch_curvature_stage3/HG_Rec_best.pth
  - v18 baseline test_R@10 = 0.1011 (R37 硬约束: v25 test_R@10 必须 > v18)
"""
print("[v25_hyperbolic_rerank_from_v18/stage3] SKIP — 沿用 v18 HG_Rec_best.pth")
print("  ckpt: /fs04/ar57/wenyu/GeneRec/taskA/_history/v18_branch_curvature_stage3/HG_Rec_best.pth")
print("  v18 baseline test_R@10 = 0.1011 (R37 硬约束下限)")