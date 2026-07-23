#!/usr/bin/env python3
"""task459_h_subspace_purity.py — H 子空间 vs E 子空间 cluster purity 对比

目的: 验证 P0.3 设计意图 — H 子空间 ([0:32]) 内 cluster purity 显著高于 E 子空间

方法:
  1. 加载 concat_embedding.pt (928-dim, n=12288 商品)
  2. 加载 P0.2 / P0.3 SID tensor (4, n=11924)
  3. 对每个 L1 码字 c:
     - 取 I_c (商品集)
     - 算 H 范围 [0:32] intra-cosine
     - 算 E 范围 [32:928] intra-cosine
  4. 比较 P0.2 vs P0.3 在 H 范围 / E 范围的 cluster purity
"""
import os
import json
import random
from pathlib import Path
import numpy as np
import torch

CONCAT_EMB = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt"
SID_P0_2 = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s22/pickle/merged_predictions_tensor.pt"
SID_P0_3 = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt"

OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp459")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_DIM = 32   # P0.3 H subspace range [0:32]
MAX_PER_CODEWORD = 30
SEED = 42


def load_concat():
    full = torch.load(CONCAT_EMB, map_location="cpu", weights_only=False).float()
    print(f"[H-purity] concat shape: {tuple(full.shape)}")
    return full


def centroid_to_member_cosine(emb_subset: torch.Tensor) -> float:
    n = emb_subset.shape[0]
    if n < 2:
        return float("nan")
    normed = torch.nn.functional.normalize(emb_subset, dim=1)
    centroid = normed.mean(dim=0)
    centroid = torch.nn.functional.normalize(centroid.unsqueeze(0), dim=1)
    cosines = (normed @ centroid.T).squeeze(-1)
    return float(cosines.mean().item())


def compute_purity(sid_tensor, concat_emb, layer, range_start, range_end, max_per_cw, seed):
    """For each L1 codeword at given layer, compute intra-cosine over [range_start, range_end]."""
    layer_idx = layer - 1
    code_per_item = sid_tensor[layer_idx]  # (n_items,)
    unique_codes, inverse = torch.unique(code_per_item, return_inverse=True)
    rng = random.Random(seed)

    purity_values = []
    codeword_sizes = []
    for i, code in enumerate(unique_codes.tolist()):
        member_indices = (inverse == i).nonzero(as_tuple=True)[0]
        if len(member_indices) < 2:
            continue
        member_indices = member_indices.tolist()
        if len(member_indices) > max_per_cw:
            member_indices = rng.sample(member_indices, max_per_cw)
        emb_subset = concat_emb[member_indices, range_start:range_end]
        purity = centroid_to_member_cosine(emb_subset)
        if not np.isnan(purity):
            purity_values.append(purity)
            codeword_sizes.append(len(member_indices))
    return np.array(purity_values), codeword_sizes


def main():
    print(f"[H-purity] Loading data...")
    concat = load_concat()
    sid_p0_2 = torch.load(SID_P0_2, map_location="cpu", weights_only=False).long()
    sid_p0_3 = torch.load(SID_P0_3, map_location="cpu", weights_only=False).long()

    # Define subspaces
    h_range = (0, H_DIM)        # [0:32] — text 前 32-d
    e_range = (H_DIM, 928)     # [32:928] — text 后 736-d + brand + taxonomy + behavior

    # Also test individual factor ranges for reference
    ranges = {
        "H_[0:32]": (0, 32),
        "E_[32:768]": (32, 768),       # text 后 736-d
        "E_brand_[768:800]": (768, 800),
        "E_taxonomy_[800:896]": (800, 896),
        "E_behavior_[896:928]": (896, 928),
        "E_full_[32:928]": (32, 928),
    }

    results = {}
    for layer in (1, 2, 3):
        print(f"\n=== Layer {layer} ===")
        results[layer] = {}
        for range_name, (rs, re) in ranges.items():
            print(f"\n  --- range {range_name} [{rs}:{re}] ---")
            results[layer][range_name] = {}
            for variant_name, sid in [("P0.2", sid_p0_2), ("P0.3", sid_p0_3)]:
                purity_arr, sizes = compute_purity(sid, concat, layer, rs, re, MAX_PER_CODEWORD, SEED)
                mean = float(purity_arr.mean())
                std = float(purity_arr.std())
                results[layer][range_name][variant_name] = {
                    "mean_purity": mean,
                    "std_purity": std,
                    "n_codewords": len(purity_arr),
                    "mean_size": float(np.mean(sizes)),
                }
                print(f"    {variant_name}: purity={mean:.4f} ± {std:.4f} (n={len(purity_arr)}, mean_size={np.mean(sizes):.1f})")

    # Summary table: P0.3 - P0.2 purity diff
    print(f"\n=== P0.3 - P0.2 purity diff ===")
    diff_table = {}
    for layer in (1, 2, 3):
        diff_table[layer] = {}
        for range_name in ranges:
            v_p0_2 = results[layer][range_name]["P0.2"]["mean_purity"]
            v_p0_3 = results[layer][range_name]["P0.3"]["mean_purity"]
            diff = v_p0_3 - v_p0_2
            print(f"  L{layer} {range_name}: P0.2={v_p0_2:.4f} | P0.3={v_p0_3:.4f} | diff={diff:+.4f}")
            diff_table[layer][range_name] = {
                "p0_2": v_p0_2,
                "p0_3": v_p0_3,
                "diff": diff,
            }

    out = {
        "max_per_codeword": MAX_PER_CODEWORD,
        "seed": SEED,
        "h_dim": H_DIM,
        "ranges": {k: list(v) for k, v in ranges.items()},
        "results": {str(k): v for k, v in results.items()},
        "diff_table": {str(k): v for k, v in diff_table.items()},
    }
    out_path = OUT_DIR / "task459_h_purity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[H-purity] Saved to {out_path}")


if __name__ == "__main__":
    main()