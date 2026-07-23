#!/usr/bin/env python3
"""
task87_save_sid_dedup.py — Task #87: 保存最终 sid_dedup.pt 给 Stage 3 TIGER

从 cluster_ids.pt (4, 11924) 复制到 products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt
Stage 3 TIGER 用 num_hierarchies=4 加载这个文件作为 semantic_id_path。

验证:
- shape = (4, 11924)
- dtype = torch.long
- col 0..2: cluster IDs in [0, 256)
- col 3: dedup indicator (0 = unique, 1+ = duplicate within L3 group)
"""
import shutil
import os
import torch

SRC = "/fs04/ar57/wenyu/GeneRec/logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt"
DST_DIR = "/fs04/ar57/wenyu/GeneRec/products/task87_tiger_baseline/stage2_rqvae_infer"
DST = f"{DST_DIR}/sid_dedup.pt"

os.makedirs(DST_DIR, exist_ok=True)

# 验证源
src_tensor = torch.load(SRC, map_location="cpu", weights_only=False).long()
print(f"[src] shape={tuple(src_tensor.shape)}, dtype={src_tensor.dtype}")
assert src_tensor.shape == (4, 11924), f"Unexpected src shape {src_tensor.shape}"
assert src_tensor.dtype == torch.long

# 验证 col 0..2 在 [0, 256)
for i in range(3):
    u = torch.unique(src_tensor[i])
    print(f"[validate] hierarchy {i}: unique={len(u)}, range={u.min().item()}..{u.max().item()}")
    assert u.min().item() >= 0 and u.max().item() < 256, "Cluster IDs out of [0, 256)"

# 验证 col 3 dedup
dedup = src_tensor[3]
print(f"[validate] dedup col unique: {torch.unique(dedup).tolist()}")
print(f"[validate] items with dedup>0: {(dedup > 0).sum().item()} / 11924")

# 复制到 products
shutil.copy(SRC, DST)
print(f"[save] {DST}: {os.path.getsize(DST)} bytes")

# 再加载一次确认
loaded = torch.load(DST, map_location="cpu", weights_only=False)
print(f"[verify] loaded shape={tuple(loaded.shape)}, dtype={loaded.dtype}")
assert torch.equal(loaded, src_tensor)
print("[done] sid_dedup.pt ready for Stage 3 TIGER")