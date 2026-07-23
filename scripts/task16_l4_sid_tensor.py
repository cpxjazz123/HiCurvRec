#!/usr/bin/env python3
"""task6_l4_sid_tensor.py — 把 task57_HHHH s22 (12288, 3) 转 Stage 3 期望的 (4, 11924) 格式

task57_HHHH_s22 输出 (12288, 3): 12288 个 emdedding-aligned item index
Stage 3 TIGER 期望 (4, N): num_hierarchies=4 (3 个 RQ-VAE digit + 1 个 dedup col), N = 11924 catalog items

两个差异:
1. orientation: (N_items, N_hier) → (N_hier, N_items)
2. items 数: 12288 (embedding 范围, 含 padding) → 11924 (catalog items)
3. missing 4th hierarchy: 必须 append dedup col

参考 task388v4_hhhh_s22 (4, 11924) 格式:
  - 4 个 hierarchy 列
  - 每列 unique 范围 3-4 (即 dedup 的 0/1 区分)

Dedup column convention (per task321 ablation + codebase convention):
  - 非重复项: 0
  - 重复项: 1+ (one-hot within dup group)
"""
import sys
import torch
from pathlib import Path

CONCAT_EMB = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt"
INPUT_SID = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task57_HHHH_s22/pickle/merged_predictions_tensor.pt"
OUTPUT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp59/task59_hhhh_l4_sid_tensor.pt"
OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp59")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dedup_col(l3_codes: torch.Tensor):
    """Build dedup column: 0 for unique, 1+ for duplicates within L3 group.

    l3_codes: (N,) — L3 code per item
    Returns: (N,) dedup codes [0, 0, ..., 1, 2, 0, 1, ...]
    """
    n = l3_codes.shape[0]
    dedup_col = torch.zeros(n, dtype=torch.long)
    # Group by l3 code; for each group, assign 0, 1, 2, ... within group
    unique_codes, inverse = torch.unique(l3_codes, return_inverse=True)
    counters = {}
    for i in range(n):
        code = l3_codes[i].item()
        cnt = counters.get(code, 0)
        dedup_col[i] = cnt
        counters[code] = cnt + 1
    return dedup_col


def main():
    print(f"[task59] loading concat embedding from {CONCAT_EMB}")
    embedding = torch.load(CONCAT_EMB, map_location="cpu", weights_only=False)
    print(f"[task59] concat embedding shape: {embedding.shape}")

    print(f"[task59] loading task57 HHHH s22 (12288, 3) from {INPUT_SID}")
    sid3 = torch.load(INPUT_SID, map_location="cpu", weights_only=False).long()
    print(f"[task59] sid3 shape: {sid3.shape}")

    # First 11924 items in concat embedding are valid catalog items
    # task57_inference used concat_embedding[:12288] but actual catalog is 11924
    n_catalog = 11924
    sid3_catalog = sid3[:n_catalog]  # (11924, 3)
    print(f"[task59] sid3_catalog shape: {sid3_catalog.shape}")

    # Build dedup column from L3 codes (col index 2)
    dedup_col = build_dedup_col(sid3_catalog[:, 2])
    print(f"[task59] dedup_col: unique values = {torch.unique(dedup_col).tolist()}, n_items_with_dedup>0 = {(dedup_col > 0).sum().item()}")

    # Stack: (11924, 4) then transpose to (4, 11924)
    sid4 = torch.cat([sid3_catalog, dedup_col.unsqueeze(1)], dim=1)  # (11924, 4)
    sid4 = sid4.t().contiguous()  # (4, 11924)
    print(f"[task59] sid4 (Stage 3 expected shape): {sid4.shape}")

    # Sanity: similar to task388v4_hhhh_s22 format
    for i in range(4):
        u = torch.unique(sid4[i])
        print(f"[task59] hierarchy {i}: unique = {len(u)}, range = {u.min().item()}..{u.max().item()}")

    torch.save(sid4, OUTPUT_PATH)
    print(f"[task59] saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
