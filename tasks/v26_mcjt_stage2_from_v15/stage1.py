#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v26_mcjt_stage2_from_v15 Stage 1: skip — 沿用 v15 capmatch preprocessed 数据.

v26 是 Issue #103 "Stage2 Multi-Curvature Joint Training", 只改 Stage2 损失聚合方式 (multi-c + α_c learnable).
Stage1 数据预处理完全沿用 v15 capmatch, 不重跑.
"""
print("[v26_mcjt_stage2_from_v15/stage1] SKIP — 沿用 v15 capmatch 产物")
print("  data: /fs04/ar57/wenyu/GeneRec/datasets/amazon/Musical_Instruments/ (已 preprocess)")
print("  item_emb.parquet SHA: 1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc")
print("  expected_sid_sha (v15 capmatch baseline): 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07")