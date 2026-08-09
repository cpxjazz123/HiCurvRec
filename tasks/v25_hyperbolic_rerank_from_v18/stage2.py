#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 Stage 2: skip — 沿用 v15 capmatch 1000ep Stage2 训练产物.

v25 是 Issue #238 "Stage4 Hyperbolic Re-ranking", 不重跑 Stage2.
直接沿用 v15 capmatch Stage2 ckpt (hrqvae_kappa_sync.ckpt) 提供:
  - vq_layers.{0,1,2}.embeddings.weight (3 层 codebook, 切空间)
  - final_cs = [1.3547, 6.0021, 4.3941] (Poincaré 球曲率倒数)
"""
print("[v25_hyperbolic_rerank_from_v18/stage2] SKIP — 沿用 v15 capmatch Stage2 产物")
print("  ckpt: /fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt")
print("  final_cs: [1.3547, 6.0021, 4.3941]")