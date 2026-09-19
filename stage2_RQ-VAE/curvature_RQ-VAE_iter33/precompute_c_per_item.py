"""Stage 0 precompute: 计算 per-item curvature c_per_item (v2: log-freq quantile binning)

输入:
- train_recbole.parquet (history + target)
- item_emb.npy (确认 item 数量)

输出:
- c_per_item.npy (shape=[num_items], float32)

方案 (v2, 长尾友好):
1. 统计每个 item 在 train 中的 freq (history + target 计数)
2. 对 log(1+freq) 做 10-quantile binning (避免长尾塌缩)
3. 每个 bin 映射到一个 c 值: bin0 (long-tail) → c_low=0.4 (compact), bin9 (popular) → c_high=1.2 (spread)
4. freq=0 item 归入 bin0 (long-tail)

原理: long-tail item 训练样本少, 用 compact c 共享表征; popular item 样本多, 用 spread c 避免 over-smoothing.
论文支撑: 频率感知 representation learning (BBN, Logit Adjustment).
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

# === 硬编码路径与超参 (per CLAUDE.md Rule 1: no CLI args) ===
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/dataset/Amazon_2023_Instruments/train_recbole.parquet"
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"
OUTPUT_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter33/dataset/Instruments/c_per_item.npy"

# === c_per_item 超参 (硬编码) ===
N_BINS = 10  # log-freq 分位数 bin 数
C_LOW = 0.4  # bin 0 (long-tail) 的 c
C_HIGH = 1.2  # bin N-1 (popular) 的 c

# 1. 读 item_emb 确认 num_items
emb = np.load(ITEM_EMB_NPY)
num_items = emb.shape[0]
print(f"num_items (from item_emb.npy): {num_items}")

# 2. 统计 item frequency from train.parquet
df = pd.read_parquet(TRAIN_PARQUET)
print(f"train rows: {df.shape[0]}")

freq_counter = Counter()
for history in df["history"].values:
    if len(history) > 0:
        for item_id in history:
            freq_counter[int(item_id)] += 1
for target in df["target"].values:
    freq_counter[int(target)] += 1

print(f"unique items in train: {len(freq_counter)}")
print(f"max freq: {max(freq_counter.values())}, min freq: {min(freq_counter.values())}")

# 3. 构造 freq_array
freq_array = np.zeros(num_items, dtype=np.float32)
for item_id, freq in freq_counter.items():
    if 0 <= item_id < num_items:
        freq_array[item_id] = float(freq)
    else:
        print(f"WARN: item_id {item_id} out of range, skip")

# 4. log-frequency quantile binning
log_freq = np.log1p(freq_array)  # log(1+freq), handles freq=0
# 用 np.quantile 找 N_BINS 分位点
quantiles = np.linspace(0, 1, N_BINS + 1)
bin_edges = np.quantile(log_freq[log_freq > 0], quantiles)  # 仅在 freq>0 item 上分位
print(f"bin_edges (log_freq): {bin_edges}")

# 把每个 item 分到 bin (0..N_BINS-1), freq=0 归 bin0 (long-tail)
bin_idx = np.zeros(num_items, dtype=np.int64)
for i in range(num_items):
    if freq_array[i] == 0:
        bin_idx[i] = 0  # freq=0 → bin 0 (long-tail, c_low)
    else:
        bin_idx[i] = int(np.searchsorted(bin_edges, log_freq[i], side="right") - 1)
        bin_idx[i] = max(0, min(N_BINS - 1, bin_idx[i]))

# 5. 映射 bin → c
c_per_item = np.zeros(num_items, dtype=np.float32)
for b in range(N_BINS):
    mask = bin_idx == b
    c_value = C_LOW + (C_HIGH - C_LOW) * (b / (N_BINS - 1))
    c_per_item[mask] = c_value

# 6. 校验
if not (c_per_item > 0).all():
    raise RuntimeError(f"c_per_item has non-positive values: min={c_per_item.min()}")
print(f"c_per_item: min={c_per_item.min():.4f}, max={c_per_item.max():.4f}, mean={c_per_item.mean():.4f}")

# 7. bin 分布统计
print("bin → count distribution:")
for b in range(N_BINS):
    mask = bin_idx == b
    n_items = int(mask.sum())
    c_value = C_LOW + (C_HIGH - C_LOW) * (b / (N_BINS - 1))
    print(f"  bin {b}: {n_items} items, c={c_value:.4f}")

# 8. 写文件
np.save(OUTPUT_NPY, c_per_item)
print(f"written {OUTPUT_NPY}: shape={c_per_item.shape}, dtype={c_per_item.dtype}")