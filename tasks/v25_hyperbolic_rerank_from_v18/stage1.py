#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 Stage 1: skip — 沿用 v15 capmatch preprocessed 数据.

v25 是 Issue #238 "Stage4 Hyperbolic Re-ranking", 不重跑 Stage1 (preprocess) / Stage2 (HRQ-VAE).
完全沿用 v15 capmatch 的 data + sid_output.npy (sha=5f8331cc).
"""
print("[v25_hyperbolic_rerank_from_v18/stage1] SKIP — 沿用 v15 capmatch 产物")
print("  data: /fs04/ar57/wenyu/GeneRec/datasets/amazon/Musical_Instruments/ (已 preprocess)")
print("  sid_output.npy: /fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy")
print("  expected_sid_sha: 5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07")