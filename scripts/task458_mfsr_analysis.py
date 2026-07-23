#!/usr/bin/env python3
"""task458_mfsr_analysis.py — Phase 2 P5 链 3 多因素支持率 (MFSR)

目的: 验证 P0.3 设计假设 "H 子空间码字内 taxonomy/brand 更一致"

方法:
  1. 对每个 L1/L2/L3 码字 c, 取对应商品集 I_c
  2. 在 I_c 上算 4 因素 (text/brand/taxonomy/behavior) 的 centroid-to-member mean cosine
  3. MFSR_c^factor = mean cosine
  4. avg over all codewords → MFSR_layer^factor

对比 P0.2 (E-Concat) vs P0.3 (Mixed-Curv):
  - P0.3 H 子空间强制 taxonomy+brand 聚类 → 应有 higher MFSR_taxonomy / MFSR_brand
  - P0.3 E 子空间 text+behavior 与 P0.2 一致 → MFSR_text / MFSR_behavior 类似
"""
import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch

SID_P0_2 = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s22/pickle/merged_predictions_tensor.pt"
SID_P0_3 = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt"
MULTI_FACTOR_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/multi_factor.pt"

OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp458")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CODEWORD = 30  # sampling cap per codeword
SEED = 42


def load_data():
    sid_p0_2 = torch.load(SID_P0_2, map_location="cpu", weights_only=False).long()
    sid_p0_3 = torch.load(SID_P0_3, map_location="cpu", weights_only=False).long()
    mf = torch.load(MULTI_FACTOR_PATH, map_location="cpu", weights_only=False)
    item_ids = mf["item_ids"].long()
    return sid_p0_2, sid_p0_3, mf, item_ids


def build_codeword_groups(sid_tensor: torch.Tensor, item_ids: torch.Tensor, layer: int):
    """Group items by their codeword at given layer.

    sid_tensor: (4, n_total_items)
    item_ids: (n_mf,) — multi_factor items, in [0, n_total_items)
    Returns: {code: [item_id_1, item_id_2, ...]} (only mf items)
    """
    code_per_item = sid_tensor[layer - 1, item_ids]  # (n_mf,)
    unique_codes, inverse = torch.unique(code_per_item, return_inverse=True)
    groups = {}
    for i, code in enumerate(unique_codes.tolist()):
        member_indices = (inverse == i).nonzero(as_tuple=True)[0]
        member_item_ids = item_ids[member_indices].tolist()
        if len(member_item_ids) >= 2:
            groups[code] = member_item_ids
    return groups


def centroid_to_member_cosine(embeddings: torch.Tensor, sample_size: int = MAX_PER_CODEWORD, seed: int = SEED):
    """Mean cosine of centroid to each member (excluding self).

    embeddings: (N, D), assume row-normalized or not (we normalize here)
    Returns: float mean cosine
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    if n > sample_size:
        rng = random.Random(seed)
        sample_idx = rng.sample(range(n), sample_size)
        embeddings = embeddings[sample_idx]
        n = sample_size
    normed = torch.nn.functional.normalize(embeddings, dim=1)
    centroid = normed.mean(dim=0)
    centroid = torch.nn.functional.normalize(centroid.unsqueeze(0), dim=1)
    cosines = (normed @ centroid.T).squeeze(-1)  # (n,)
    return float(cosines.mean().item())


def compute_mfsr_for_variant(sid_tensor, mf, item_ids):
    """For each (layer, factor), compute mean MFSR across codewords.

    Returns: dict[layer][factor] = {mfsr_mean, mfsr_std, n_codewords}
    """
    n_items = sid_tensor.shape[1]
    n_mf = len(item_ids)
    print(f"  Total items in SID: {n_items}, multi_factor items: {n_mf}")

    # Filter multi_factor to in-bounds items
    in_bounds = item_ids < n_items
    item_ids_valid = item_ids[in_bounds]
    print(f"  Valid items (in SID bounds): {len(item_ids_valid)}")

    # Pre-extract factor embeddings (aligned to item_ids_valid)
    text_emb = mf["F_text"][in_bounds].float()
    brand_emb = mf["F_brand"][in_bounds].float()
    taxonomy_emb = mf["F_taxonomy"][in_bounds].float()
    behavior_emb = mf["F_behavior"][in_bounds].float()

    factor_data = {
        "text": text_emb,
        "brand": brand_emb,
        "taxonomy": taxonomy_emb,
        "behavior": behavior_emb,
    }

    results = {}
    for layer in (1, 2, 3):
        print(f"\n  --- Layer {layer} ---")
        groups = build_codeword_groups(sid_tensor, item_ids_valid, layer)
        print(f"  Codewords with >=2 items: {len(groups)}")
        results[layer] = {}
        for factor_name, factor_emb in factor_data.items():
            mfsr_values = []
            for code, members in groups.items():
                member_tensor = torch.tensor(members, dtype=torch.long)
                idx = torch.searchsorted(item_ids_valid, member_tensor)
                if (idx >= len(item_ids_valid)).any():
                    idx = torch.clamp(idx, max=len(item_ids_valid) - 1)
                emb_subset = factor_emb[idx]
                mfsr_values.append(centroid_to_member_cosine(emb_subset))
            mfsr_arr = np.array([v for v in mfsr_values if not np.isnan(v)])
            mean = float(mfsr_arr.mean()) if len(mfsr_arr) > 0 else float("nan")
            std = float(mfsr_arr.std()) if len(mfsr_arr) > 0 else float("nan")
            results[layer][factor_name] = {
                "mean": mean,
                "std": std,
                "n_codewords": len(mfsr_arr),
            }
            print(f"    {factor_name}: MFSR={mean:.4f} ± {std:.4f} (n={len(mfsr_arr)})")
    return results


def main():
    print(f"[MFSR] loading data...")
    sid_p0_2, sid_p0_3, mf, item_ids = load_data()
    print(f"[MFSR] sid shape: {tuple(sid_p0_2.shape)}, item_ids: {len(item_ids)}")

    print(f"\n[MFSR] === P0.2 (Euclidean-Concat) ===")
    p0_2_results = compute_mfsr_for_variant(sid_p0_2, mf, item_ids)
    print(f"\n[MFSR] === P0.3 (Mixed-Curvature) ===")
    p0_3_results = compute_mfsr_for_variant(sid_p0_3, mf, item_ids)

    print(f"\n[MFSR] === P0.3 - P0.2 MFSR diff ===")
    diff_table = {}
    for layer in (1, 2, 3):
        diff_table[layer] = {}
        for factor in ("text", "brand", "taxonomy", "behavior"):
            v0_2 = p0_2_results[layer][factor]["mean"]
            v0_3 = p0_3_results[layer][factor]["mean"]
            diff = v0_3 - v0_2
            print(f"  L{layer} {factor}: P0.2={v0_2:.4f} | P0.3={v0_3:.4f} | diff={diff:+.4f}")
            diff_table[layer][factor] = {
                "p0_2": v0_2,
                "p0_3": v0_3,
                "diff": diff,
                "p0_2_n": p0_2_results[layer][factor]["n_codewords"],
                "p0_3_n": p0_3_results[layer][factor]["n_codewords"],
            }

    out = {
        "max_per_codeword": MAX_PER_CODEWORD,
        "seed": SEED,
        "p0_2_results": {str(k): v for k, v in p0_2_results.items()},
        "p0_3_results": {str(k): v for k, v in p0_3_results.items()},
        "diff_table": {str(k): v for k, v in diff_table.items()},
    }
    out_path = OUT_DIR / "task458_mfsr_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[MFSR] Saved to {out_path}")


if __name__ == "__main__":
    main()