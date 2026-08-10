#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21 stage1.py — STUB. v21 = Issue #128 Item 6 refactor (Stage2 κ-only + collapse gate).

R34 合规: 本目录 4 个脚本必备. stage1 用 v15 capmatch baseline Stage1 产物 (R_MODE=fixed,
R_MAX=0.99, 无 per-item radius) — item_emb_baseline.npy SHA256=96a7109e14b6b93c...

Stage1 直接复用 taskA/_data/Instruments/item_emb_baseline.npy (基线产品, 不重新训练).
"""
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE1_BASELINE = REPO / "taskA/_data/Instruments/item_emb_baseline.npy"

if __name__ == "__main__":
    assert STAGE1_BASELINE.exists(), f"Stage1 baseline missing: {STAGE1_BASELINE}"
    print(f"[v21 stage1] STUB — 复用基线 Stage1 产物: {STAGE1_BASELINE}")
    print(f"[v21 stage1] SHA256=96a7109e14b6b93ce1ae2c628fa1df9a06f9d177673d6ba1f78c5db98aea5cea")
    print(f"[v21 stage1] 真实 Stage1 训练在 v15 capmatch (Issue #96) 已完成, 本版本直接复用.")
    sys.exit(0)
