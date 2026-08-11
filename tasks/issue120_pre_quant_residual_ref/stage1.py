#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #120 — Stage1 stub.

R34 严格: 4 脚本必须存在. Issue #120 仅 Stage2 几何诊断, 不重跑 Stage1.
Stage1 完全复用 Issue #210 equal128 Stage1 产物.
"""
import os
from pathlib import Path

# Issue #210 equal128 Stage1 产物 (item_emb_u32.npy + sid_metadata.json)
ISSUE210_STAGE1_OUTPUT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue210_equal_codebook/taskA_stage1_equal128"
ISSUE119_STAGE1_OUTPUT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history_v77_backup/taskA_stage1_hyp_v2/item_emb_u32.npy"
print(f"[issue120-stage1] Stage1 复用 Issue #210 equal128 → {ISSUE210_STAGE1_OUTPUT}")
print(f"[issue120-stage1] item_emb 路径: {ISSUE119_STAGE1_OUTPUT}")
print(f"[issue120-stage1] 不重跑 Stage1 (R34 允许: 复用产物)")
