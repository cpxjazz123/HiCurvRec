#!/usr/bin/env python3
"""
task87_fix_s2_2x_data.py — 修复 Task #87 Stage 2.2 inference 的 2× 数据问题

背景: Stage 2.2 RQ-VAE inference 在 merged_predictions.pkl 写入 23848 rows (2×11924),
Lightning InferenceLoop 配合 `assign_files_by_size=True` + `num_workers=2` + `iterate_per_row=True`
+ `UnboundedSequenceIterable` 把 dataset 跑了 2 遍。row i 和 row i+11924 完全相同
(item_id, cluster_ids 一一对应)。

修复: 取前 11924 rows 重建 tensor,跑 dedup + transpose,保存为 cluster_ids.pt (4, 11924)。
此脚本不动 src/ 上游代码,只后处理输出产物。
"""
import os
import sys
import pickle
import torch

ROOT = "/fs04/ar57/wenyu/GeneRec"
SRC_PKL = f"{ROOT}/logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/merged_predictions.pkl"
OUT_DIR = f"{ROOT}/logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle"
OUT_CLUSTER_IDS = f"{OUT_DIR}/cluster_ids.pt"

# 1. 读取合并预测文件
with open(SRC_PKL, "rb") as f:
    data = pickle.load(f)

print(f"[load] {SRC_PKL}: {len(data)} rows")
assert len(data) == 23848, f"Expected 23848 rows, got {len(data)}"

# 2. 验证前 11924 rows == 后 11924 rows (1:1 重复)
first_half = data[:11924]
second_half = data[11924:]
assert all(
    a["item_id"] == b["item_id"] and a["cluster_ids"] == b["cluster_ids"]
    for a, b in zip(first_half, second_half)
), "First half != Second half (1:1 duplicate)"
print("[verify] First half == Second half (1:1 exact duplicate)")

# 3. 验证 first half 覆盖全部 11924 unique items (id 0..11923)
item_ids = [r["item_id"] for r in first_half]
assert len(set(item_ids)) == 11924
assert min(item_ids) == 0 and max(item_ids) == 11923
print(f"[verify] First half covers all 11924 unique items (id 0..11923)")

# 4. 重建 merged_predictions_tensor.pt (11924, 3)
# 沿用 src/utils/tensor_utils.py 的 merge_list_of_keyed_tensors_to_single_tensor 逻辑
batch_size = len(first_half)
dims = torch.tensor(first_half[0]["cluster_ids"]).size()
output_tensor = torch.zeros((batch_size, *dims))
for row in first_half:
    output_tensor[row["item_id"]] = torch.tensor(row["cluster_ids"])
print(f"[rebuild] merged_predictions_tensor.pt shape: {tuple(output_tensor.shape)}, dtype: {output_tensor.dtype}")

# 5. dedup: 添加 dedup indicator 列 (第 4 列)
unique_rows, inverse_indices, counts = torch.unique(
    output_tensor, dim=0, return_inverse=True, return_counts=True
)
output_indices = torch.zeros_like(inverse_indices)
dup_idx = torch.where(counts > 1)[0]
for i in range(len(dup_idx)):
    n = counts[dup_idx[i]]
    where = torch.where(inverse_indices == dup_idx[i])[0]
    output_indices = output_indices.scatter(0, where, torch.arange(1, n + 1))

dedup_tensor = torch.cat((output_tensor, output_indices.unsqueeze(1)), dim=1).long()
print(f"[dedup] shape after dedup column: {tuple(dedup_tensor.shape)}, dtype: {dedup_tensor.dtype}")
print(f"[dedup] # duplicates (col 4 > 0): {(dedup_tensor[:, 3] > 0).sum().item()}")

# 6. transpose: (N, 4) → (4, N)
cluster_ids = dedup_tensor.transpose(-2, -1).contiguous().long()
print(f"[transpose] cluster_ids.pt shape: {tuple(cluster_ids.shape)}, dtype: {cluster_ids.dtype}")

# 7. 验证最终 cluster_ids
assert cluster_ids.shape == (4, 11924), f"Expected (4, 11924), got {tuple(cluster_ids.shape)}"
assert cluster_ids.dtype == torch.long
assert (cluster_ids[0:3] >= 0).all() and (cluster_ids[0:3] < 256).all(), "Cluster IDs must be in [0, 256)"
print(f"[validate] cluster_ids[0, :5] = {cluster_ids[0, :5].tolist()}")
print(f"[validate] cluster_ids[3, :5] (dedup col) = {cluster_ids[3, :5].tolist()}")

# 8. 保存 cluster_ids.pt
torch.save(cluster_ids, OUT_CLUSTER_IDS)
print(f"[save] {OUT_CLUSTER_IDS}: {os.path.getsize(OUT_CLUSTER_IDS)} bytes")