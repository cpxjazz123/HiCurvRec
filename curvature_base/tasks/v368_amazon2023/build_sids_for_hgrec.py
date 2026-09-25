"""v83 F3 Spread Loss SID HG-Rec 格式转换: (N, 3) → (N, 4) 第 4 列=PAD=0.
从 curvature_config.py 读 RAW_SIDS_NPY / SIDS_NPY 路径 (R40 自包含 + R53 v3.8 env-var-free).
2026-09-05: 改 n_items=9922 → 动态 src.shape[0] (兼容 2018=9922 / 2023=24587).
"""
import numpy as np
from curvature_config import RAW_SIDS_NPY as SRC, SIDS_NPY as DST

src = np.load(SRC).astype(np.int64)
n_items = src.shape[0]
assert src.ndim == 2 and src.shape[1] == 3, f"unexpected src shape {src.shape}"
assert src.max() < 256, f"src max {src.max()} >= 256 (vocab overflow)"
assert src.min() >= 0, f"src min {src.min()} < 0"
print(f"[build] n_items={n_items} (2026-09-05: 2018=9922 / 2023=24587, dynamic from data)")

# HG-Rec 格式: 第 4 列 PAD=0 (vocab_size=769 = 256*3 + 1)
dst = np.zeros((n_items, 4), dtype=np.int64)
dst[:, :3] = src
dst[:, 3] = 0

np.save(DST, dst)
print(f"[save] {DST}: shape={dst.shape}, dtype={dst.dtype}, max={dst.max()}")
print(f"[check] first 5 rows: {dst[:5].tolist()}")
print(f"[check] PAD col 3 = 0 always: {(dst[:, 3] == 0).all()}")
