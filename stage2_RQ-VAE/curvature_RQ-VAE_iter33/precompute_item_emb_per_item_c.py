"""Stage 0 precompute 2: item_emb per-item curvature projection.

思路: 对每个 item i, 按 c_per_item[i] 把 baseline item_emb[i] 投影到对应 c 的 Poincaré ball 半径.
近似实现: norm scaling at origin. x_new = x * sqrt(c_to / c_from), 然后投影回 c_to 球.
RQ-VAE 训练时 c 还是 scalar (c_base), 但 input item_emb 已经 per-item curvature-aware.
"""
import numpy as np


# === 硬编码路径与超参 ===
BASELINE_ITEM_EMB = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"
C_PER_ITEM_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter33/dataset/Instruments/c_per_item.npy"
OUTPUT_ITEM_EMB = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter33/dataset/Instruments/item_emb_iter33.npy"

# baseline item_emb 假设的 c_base (在 c_base=1.0 球上的 Euclidean 坐标)
C_BASE = 1.0


def change_curvature_simple(x: np.ndarray, c_from: float, c_to: float, eps: float = 1e-5) -> np.ndarray:
    """Approximate change-curvature at origin via norm scaling. 不投影到 c_to 球,
    让 norm 直接反映 c_per_item: c_to > c_from → norm 增大 (spread), c_to < c_from → norm 减小 (compact).
    RQ-VAE encoder 不假设 input 严格在 Poincaré ball 内, norm modulation 即可生效.
    """
    if abs(c_from - c_to) < 1e-9:
        return x
    # norm scaling 因子 sqrt(c_to/c_from) 来自 logmap0/expmap0 的 1D 等价
    scale = (c_to / c_from) ** 0.5
    return x * scale


# 1. 加载 baseline item_emb
item_emb_baseline = np.load(BASELINE_ITEM_EMB).astype(np.float32)
print(f"baseline item_emb: shape={item_emb_baseline.shape}, dtype={item_emb_baseline.dtype}")
print(f"  norm: min={np.linalg.norm(item_emb_baseline, axis=-1).min():.4f}, "
      f"max={np.linalg.norm(item_emb_baseline, axis=-1).max():.4f}, "
      f"mean={np.linalg.norm(item_emb_baseline, axis=-1).mean():.4f}")

# 2. 加载 c_per_item
c_per_item = np.load(C_PER_ITEM_NPY).astype(np.float32)
print(f"c_per_item: shape={c_per_item.shape}, range=[{c_per_item.min():.4f}, {c_per_item.max():.4f}]")

if c_per_item.shape[0] != item_emb_baseline.shape[0]:
    raise ValueError(
        f"c_per_item length {c_per_item.shape[0]} != item_emb num_items {item_emb_baseline.shape[0]}"
    )

# 3. 对每个 item 应用 per-item curvature projection
item_emb_iter33 = np.zeros_like(item_emb_baseline)
for i in range(item_emb_baseline.shape[0]):
    item_emb_iter33[i] = change_curvature_simple(
        item_emb_baseline[i], c_from=C_BASE, c_to=float(c_per_item[i])
    )

# 4. 校验
norms_new = np.linalg.norm(item_emb_iter33, axis=-1)
print(f"iter33 item_emb: shape={item_emb_iter33.shape}")
print(f"  norm: min={norms_new.min():.4f}, max={norms_new.max():.4f}, mean={norms_new.mean():.4f}")

# 5. 检查是否所有点都在 c_per_item 球内 (radius = 1/sqrt(c))
for c in [0.4, 0.8, 1.2]:
    radius = (1.0 / c) ** 0.5
    in_ball = (norms_new <= radius * (1 - 1e-5)).all()
    print(f"  c={c}, ball_radius={radius:.4f}, all_in_ball={in_ball}")

# 6. 写文件
np.save(OUTPUT_ITEM_EMB, item_emb_iter33)
print(f"written {OUTPUT_ITEM_EMB}: shape={item_emb_iter33.shape}, dtype={item_emb_iter33.dtype}")