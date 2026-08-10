#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23 v2 Stage 1 stub — 复用 v23 Stage1 v82 baseline 产物.

v23 v2 设计: Stage1 输入与 v23 完全一致 (Stage1 Euclidean semantic source),
不重跑 Stage1. 直接复用 canonical item_emb_baseline.npy.
"""
from pathlib import Path

# R30: 硬编码所有路径与常量
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")

# 复用 v23 Stage1 产物 (即 taskA Stage1 v82 baseline)
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/item_emb_baseline.npy"
ITEM_EMB_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/item_emb_baseline_backup.parquet"
ITEM_EMB_EXPECTED_SHA256_NPY = "96a7109e14b6b93ce1ae2c628fa1df9a06f9d177673d6ba1f78c5db98aea5cea"
ITEM_EMB_EXPECTED_SHA256_PARQUET = "fd482f3d224299d9f96ce4aaba6a08d18f9e64d40e2cc008788a6d940481cdf9"

N_ITEMS = 9922
EMB_DIM = 768


def main():
    """v23 v2 Stage1 stub: 验证 canonical Stage1 产物可读, 不重跑."""
    npy_path = Path(ITEM_EMB_NPY)
    assert npy_path.exists(), f"Stage1 npy missing: {npy_path}"
    import hashlib
    sha = hashlib.sha256(npy_path.read_bytes()).hexdigest()
    assert sha == ITEM_EMB_EXPECTED_SHA256_NPY, f"Stage1 npy SHA mismatch: got {sha[:16]}..., expected {ITEM_EMB_EXPECTED_SHA256_NPY[:16]}..."
    print(f"[v23 v2 stage1 stub] npy={npy_path}", flush=True)
    print(f"[v23 v2 stage1 stub] sha256={sha}", flush=True)
    print(f"[v23 v2 stage1 stub] N_ITEMS={N_ITEMS} EMB_DIM={EMB_DIM}", flush=True)
    print(f"[v23 v2 stage1 stub] 复用 v23 Stage1 产物, 不重跑 Stage1 (R34 stub)", flush=True)


if __name__ == "__main__":
    main()