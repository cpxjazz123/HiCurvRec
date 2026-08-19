"""v48 SID HG-Rec 格式转换: (9922, 3) → (9922, 4) 第 4 列=PAD=0.
从 v48 Stage 1 RQ-VAE ckpt 推理得到 (9922, 3) SID, 再 pad 一列 0 输出 HG-Rec 格式.
"""
import numpy as np

SRC = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids_v48_lorentz_mix.npy"
DST = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/dataset/Instruments/Instruments_v48_sids_for_hgrec.npy"

src = np.load(SRC).astype(np.int64)
assert src.shape == (9922, 3), f"unexpected src shape {src.shape}"
assert src.max() < 256, f"src max {src.max()} >= 256 (vocab overflow)"
assert src.min() >= 0, f"src min {src.min()} < 0"

# HG-Rec 格式: 第 4 列 PAD=0 (vocab_size=769 = 256*3 + 1)
dst = np.zeros((9922, 4), dtype=np.int64)
dst[:, :3] = src
dst[:, 3] = 0

np.save(DST, dst)
print(f"[save] {DST}: shape={dst.shape}, dtype={dst.dtype}, max={dst.max()}")
print(f"[check] first 5 rows: {dst[:5].tolist()}")
print(f"[check] PAD col 4 = 0 always: {(dst[:, 3] == 0).all()}")