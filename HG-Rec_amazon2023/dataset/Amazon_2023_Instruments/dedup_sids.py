"""R53 v3.8: SID npy 4-列 code 去重 (与 gen_codebook.py:163-175 一致逻辑).

输入: dataset/Amazon_2023_Instruments/sids_for_hgrec.npy (24587, 4) int64
      第 4 列全 0 (PAD), 3-列 code 可能冲突 (16349 unique < 24587)
输出: dataset/Amazon_2023_Instruments/sids_for_hgrec_unique.npy (24587, 4) int64
      4-列 code 全部 unique

原因: HG-Rec/data/dataset.py::item2code 严格复用 RecBole TIGER 的 SID 解码，
      要求 (item id, 4-列 code) 一一对应；若 4-列 code 冲突，直接报错而不是静默回退到 PAD。
"""
import numpy as np

DATASET_DIR = "./dataset/Amazon_2023_Instruments"

codes = np.load(f"{DATASET_DIR}/sids_for_hgrec.npy").astype(np.int64)
n_items = codes.shape[0]
assert codes.shape[1] == 4, f"expected (N, 4) SID npy, got {codes.shape}"

n_unique_before = len(set(map(tuple, codes.tolist())))
print(f"[dedup_sids] input: shape={codes.shape}, unique 4-col codes={n_unique_before}/{n_items}")

# 与 gen_codebook.py:166-174 一致: 冲突 item 的第 4 列递增
unique_codes, counts = np.unique(codes, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]
print(f"[dedup_sids] {len(duplicates)} duplicated 4-col codes (total {int(counts.sum() - len(unique_codes))} collisions)")

for duplicate in duplicates:
    duplicate_indices = np.where((codes == duplicate).all(axis=1))[0]
    for i, idx in enumerate(duplicate_indices):
        codes[idx, -1] = i  # 第 4 列递增: 0/1/2/...

# 二次验证
new_unique_codes, new_counts = np.unique(codes, axis=0, return_counts=True)
remaining_dupes = new_unique_codes[new_counts > 1]
if len(remaining_dupes) > 0:
    raise ValueError(
        f"dedup failed: {len(remaining_dupes)} 4-col codes still collide. "
        f"first collision: {remaining_dupes[0]} with count {new_counts[0]}"
    )

n_unique_after = len(set(map(tuple, codes.tolist())))
assert n_unique_after == n_items, f"expected {n_items} unique codes, got {n_unique_after}"

np.save(f"{DATASET_DIR}/sids_for_hgrec_unique.npy", codes)
print(f"[dedup_sids] output: shape={codes.shape}, unique 4-col codes={n_unique_after}/{n_items}")
print(f"[dedup_sids] saved → sids_for_hgrec_unique.npy")
