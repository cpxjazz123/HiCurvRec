"""task3_collision_rate.py — 测 Stage 2.2 SID tensor 的碰撞率
公式: collision_rate = 1 - n_unique_SIDs / N

输入：merged_predictions_tensor.pt (4, 11924) → transpose 为 (11924, 4)
- 第 4 列是 dedup digit (task.md baseline)
- 前 3 列是 3 层 RKMeans codeword
- "effective SID" 是前 3 列（dedup digit 是后处理的语义层，不参与量化碰撞）
"""
import sys
import torch
import json
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: python3 task3_collision_rate.py <merged_predictions_tensor.pt> [output.json]")
    sys.exit(1)

pt_path = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else pt_path.parent / "collision_rate.json"

sid = torch.load(pt_path, map_location="cpu", weights_only=False)
print(f"[load] {pt_path}")
print(f"  shape={sid.shape}, dtype={sid.dtype}")

# 兼容 (4, 11924) 或 (11924, 4)
if sid.shape[0] < sid.shape[1]:
    sid = sid.transpose(0, 1)  # (11924, 4)
print(f"  transposed: {sid.shape}")

N = sid.shape[0]
# effective SID = 前 3 列（不含 dedup digit）
sid_eff = sid[:, :3] if sid.shape[1] >= 3 else sid
n_unique_eff = torch.unique(sid_eff, dim=0).shape[0]
collision_rate_eff = 1 - n_unique_eff / N

# 全 SID（含 dedup digit）
n_unique_full = torch.unique(sid, dim=0).shape[0]
collision_rate_full = 1 - n_unique_full / N

print(f"\n[3-layer SID (前 3 列)]")
print(f"  N = {N}")
print(f"  n_unique = {n_unique_eff}")
print(f"  collision_rate = {collision_rate_eff:.4f}")

print(f"\n[Full SID (含 dedup digit)]")
print(f"  n_unique = {n_unique_full}")
print(f"  collision_rate = {collision_rate_full:.4f}")

result = {
    "sid_path": str(pt_path),
    "shape": list(sid.shape),
    "N": N,
    "n_unique_3layer": n_unique_eff,
    "collision_rate_3layer": float(collision_rate_eff),
    "n_unique_full": n_unique_full,
    "collision_rate_full": float(collision_rate_full),
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[saved] {out_path}")
