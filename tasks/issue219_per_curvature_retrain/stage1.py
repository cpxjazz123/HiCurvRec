#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #119 — Stage1 stub.

R34 严格: 4 脚本必须存在, 但本 issue 仅需 Stage2 重训 + 评估.
Stage1 完全复用 Issue #210 equal128 Stage1 产物, 不重跑.

R31 合规: 单一主脚本, 不 fork 版本.
"""
import os
from pathlib import Path

# Issue #210 equal128 Stage1 产物 (item_emb_baseline.npy + sid_metadata.json)
ISSUE210_STAGE1_OUTPUT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue210_equal_codebook/taskA_stage1_equal128"
print(f"[issue219-stage1] Stage1 完全复用 Issue #210 equal128 → {ISSUE210_STAGE1_OUTPUT}")
print(f"[issue219-stage1] 不重跑 Stage1 (R34 允许: 复用产物)")
