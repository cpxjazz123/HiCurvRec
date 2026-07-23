"""task426: L1 分布重塑 — 双消融脚本

设计:
  A. H_E_E_E L1 → 强制重尾 ('flatten'):
     - 取 H_E_E_E SID 的 L1 列
     - Sort codes by count, keep top-N_KEEP codes unchanged
     - Merge bottom-N_MERGE codes into 1 super-code (creating a heavy-tail pattern)
     - This creates 1 large super-code with K_MERGE × avg_freq items,
       mimicking baseline's cov@1 = 5.6% pattern.

  B. baseline E_E_E_E L1 → 强制均匀 ('disperse'):
     - 取 baseline SID 的 L1 列
     - Find top-N_KEEP super-codes (by count)
     - Split each super-code's items into N_SPLIT sub-codes (deterministic hash)
     - This disperses super-codes into smaller chunks, mimicking H_E_E_E's uniform pattern.

设计原则:
  - 修改仅作用于 L1 行 (index 1), 其他 3 行不动
  - 双向对照: A 让 H 接近 baseline's shape, B 让 baseline 接近 H's shape
  - 假如形状本身是 R@10 因果 (而非双曲几何), 应观察到:
      A: R@10 上升 (向 baseline 靠拢)
      B: R@10 下降 (向 H 靠拢)
"""

import os
import sys
import numpy as np
import torch

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/task_artifacts/results/exp388v5/task426_dual_ablation/sid'
os.makedirs(OUT_DIR, exist_ok=True)

# Inputs
SID_H_E_E_E = f'{GRID}/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt'
SID_BASELINE = f'{GRID}/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt'

# Outputs
OUT_H_FLATTEN = f'{OUT_DIR}/sid_H_EEE_L1_flatten.pt'
OUT_BASELINE_DISPERSE = f'{OUT_DIR}/sid_baseline_L1_disperse.pt'


def flatten_l1(sid_path, out_path, n_keep=10, n_merge=100, new_super_code=200):
    """
    Flatten L1 distribution by:
      - Keep top `n_keep` most-used L1 codes unchanged
      - Merge bottom `n_merge` codes (by count) into 1 new super-code at index `new_super_code`

    After flatten: L1 has a new super-code with ~ (n_merge × avg_count) items, mimicking baseline's heavy tail.

    NOTE: For H_E_E_E which uses 253 codes, n_keep=10, n_merge=100 keeps the top 10 + new super from bottom 100.
         For baseline which uses 256 codes, same params work.
    """
    sid = torch.load(sid_path, map_location='cpu', weights_only=False)
    sid_np = sid.numpy().astype(np.int64).copy()
    L1 = sid_np[1]
    counts = np.bincount(L1, minlength=256)

    # sort codes by count desc
    sorted_idx = np.argsort(counts)[::-1]
    top_keep = sorted_idx[:n_keep]
    bottom_merge = sorted_idx[n_keep:n_keep + n_merge]

    print(f"\n[FLATTEN] {sid_path}")
    print(f"  Original: top-5 codes {sorted_idx[:5]}, counts {counts[sorted_idx[:5]]}")
    print(f"  Top {n_keep} kept (unchanged), bottom {n_merge} merged into super-code {new_super_code}")
    print(f"  Bottom merge codes pre-count: {counts[bottom_merge].tolist()[:10]}... total items = {counts[bottom_merge].sum()}")

    # Mask items whose L1 is in bottom_merge; reassign to new_super_code
    mask = np.isin(L1, bottom_merge)
    n_merged = mask.sum()
    print(f"  Items merged: {n_merged}")
    print(f"  New super-code count after merge: {n_merged} ({n_merged / len(L1) * 100:.2f}% of items)")

    sid_np[1][mask] = new_super_code
    new_counts = np.bincount(sid_np[1], minlength=256)
    print(f"  New top-5 codes: {np.argsort(new_counts)[::-1][:5].tolist()}, counts: {new_counts[np.argsort(new_counts)[::-1][:5]].tolist()}")

    out = torch.from_numpy(sid_np)
    torch.save(out, out_path)
    print(f"  ✓ Saved: {out_path}")


def disperse_l1(sid_path, out_path, n_disperse=10, n_split=10):
    """
    Disperse L1 super-codes by:
      - Find top `n_disperse` super-codes (by count)
      - For each super-code's items, stride-split into n_split sub-codes using codes [0, n_disperse*n_split)
        (must stay within T5 vocab_size=256)

    After disperse: top super-codes split, breaking the heavy-tail cov@1 pattern.
    """
    sid = torch.load(sid_path, map_location='cpu', weights_only=False)
    sid_np = sid.numpy().astype(np.int64).copy()
    L1 = sid_np[1]
    n_items = len(L1)
    counts = np.bincount(L1, minlength=256)

    sorted_idx = np.argsort(counts)[::-1]
    top_disperse = sorted_idx[:n_disperse]
    print(f"\n[DISPERSE] {sid_path}")
    print(f"  Top {n_disperse} super-codes to split: {top_disperse.tolist()}")
    print(f"  Pre-split counts: {counts[top_disperse].tolist()}")

    # For each top super-code, reassign items into n_split sub-codes using codes [k*n_split, k*n_split+n_split)
    for k, code in enumerate(top_disperse):
        item_idxs = np.where(L1 == code)[0]
        # Stride split: assign items to sub-codes k*n_split + (i % n_split)
        sub_codes = np.array([k * n_split + (i % n_split) for i in range(len(item_idxs))], dtype=np.int64)
        L1[item_idxs] = sub_codes

    new_counts = np.bincount(L1, minlength=256)
    sorted_new = np.argsort(new_counts)[::-1]
    print(f"  After disperse: top-5 codes {sorted_new[:5].tolist()}, counts {new_counts[sorted_new[:5]].tolist()}")
    print(f"  Original top-1 code was {top_disperse[0]}, now split across {n_split} sub-codes")
    print(f"  Total active codes after: {(new_counts > 0).sum()}")

    out = torch.from_numpy(sid_np)
    torch.save(out, out_path)
    print(f"  ✓ Saved: {out_path}")


def main():
    print("=" * 80)
    print("TASK 426 DUAL ABLATION: SID L1 reshapes")
    print("=" * 80)

    # A: H_E_E_E → flatten L1 (force heavy tail)
    flatten_l1(
        SID_H_E_E_E, OUT_H_FLATTEN,
        n_keep=10, n_merge=100, new_super_code=200,
    )

    # B: baseline → disperse L1 (force uniform-like top)
    disperse_l1(
        SID_BASELINE, OUT_BASELINE_DISPERSE,
        n_disperse=10, n_split=10,
    )

    # Sanity check: load and print stats
    print("\n" + "=" * 80)
    print("VERIFY: load reshaped SID and print L1 stats")
    print("=" * 80)

    for name, path in [('Original H_E_E_E', SID_H_E_E_E),
                       ('Flattened H_E_E_E', OUT_H_FLATTEN),
                       ('Original baseline', SID_BASELINE),
                       ('Dispersed baseline', OUT_BASELINE_DISPERSE)]:
        sid = torch.load(path, map_location='cpu', weights_only=False).numpy().astype(np.int64)
        L1 = sid[1]
        counts = np.bincount(L1, minlength=256)
        n_active = (counts > 0).sum()
        top1 = counts.max()
        top1_frac = top1 / len(L1)
        cov10 = np.sort(counts)[::-1][:10].sum() / len(L1)
        from scipy.stats import linregress
        freq_desc = np.sort(counts)[::-1][np.sort(counts)[::-1] > 0]
        rank = np.arange(1, len(freq_desc) + 1)
        if len(freq_desc) > 1:
            res = linregress(np.log10(rank), np.log10(freq_desc))
            alpha = res.slope
        else:
            alpha = float('nan')
        print(f"  {name:25s}  active={n_active:4d}  cov@1={top1_frac:.3f}  cov@10={cov10:.3f}  α={alpha:+.3f}")


if __name__ == '__main__':
    main()
