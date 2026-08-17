"""C33: 把 C28 SID 矩阵 (9922, 3) 转成 HG-Rec 可消费的格式.

HG-Rec item2code 公式: offsets[li] = c + sum(codebook_size[0:i]) + 1
  假设 npy 存的是**原始 L0/L1/L2 code** (0..K-1), item2code 内部再加 offset.

C28 codebook_size = [256, 256, 256] (3 层)
  → L0 offset=1, code ∈ [1, 256]
  → L1 offset=257, code ∈ [257, 512]
  → L2 offset=513, code ∈ [513, 768]
  → 第 4 位填 0 (HG-Rec 第 4 层 vocab=1 [只有 0])
  → vocab_size = 769

我们让 build 脚本只输出**原始 L0/L1/L2 code**(0..255), 第 4 位填 0,
然后 train_stage3.py 实例化 GenRecDataset 时传 codebook_size=[256,256,256],
由 HG-Rec item2code 自己加 offset. 这样既符合 HG-Rec 原生代码, 又保证 vocab ≤ 768.
"""
import json
import os
import sys

import numpy as np


SIDS_IN = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/sids_c28_curriculum_m3.npy"
OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/dataset/Instruments"
OUT_NPY = os.path.join(OUT_DIR, "Instruments_c28_sids_for_hgrec.npy")
ITEM_IDS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_ids.json"

N_LAYERS = 3
CODEBOOK_SIZE = [256, 256, 256]
# HG-Rec item2code 会再加 offset, 输出 vocab 上限 = sum(codebook_size)+1 = 769


def main():
    print(f"=== C33 SID 平铺转换 ===")
    sids = np.load(SIDS_IN)
    print(f"[load] sids.shape={sids.shape}, dtype={sids.dtype}")
    print(f"[load] codebook_size={CODEBOOK_SIZE}, n_layers={N_LAYERS}")

    # 我们存**原始 L0/L1/L2 code**(0..255), 第 4 位填 0 (HG-Rec 第 4 层 vocab=1 [只有 0])
    # HG-Rec item2code 会在 GenRecDataset.__init__ 内部调用, 用我们的 codebook_size=[256,256,256]
    # 自动加 offset: L0=c+1 ∈ [1,256], L1=c+257 ∈ [257,512], L2=c+513 ∈ [513,768], L4=0
    if N_LAYERS < 4:
        pad_layer = np.zeros((sids.shape[0], 4 - N_LAYERS), dtype=np.int64)
        out_arr = np.concatenate([sids, pad_layer], axis=1)  # (N, 4)
    else:
        out_arr = sids

    if os.path.exists(ITEM_IDS_JSON):
        with open(ITEM_IDS_JSON) as f:
            item_ids = json.load(f)
        print(f"[check] item_ids.json 有 {len(item_ids)} 个 item, 与 sids 行数 {len(sids)} {'一致' if len(item_ids)==len(sids) else '不一致'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_NPY, out_arr, allow_pickle=False)
    print(f"[save] {OUT_NPY} (array, shape={out_arr.shape}, dtype={out_arr.dtype})")
    print(f"[note] 原始 C28 SID (0..255), 由 HG-Rec item2code 用 codebook_size=[256,256,256] 加 offset")
    print(f"[note] 实际 vocab range: [1, 256] U [257, 512] U [513, 768] U {0}, 总 vocab_size = 769")

    # 验证 lookup 正确性
    for idx in [0, 99, 9921]:
        codes = out_arr[idx]
        item_id = idx + 1
        print(f"[check] out_arr[{idx}] (item_id={item_id}) = {codes.tolist()} (max={codes.max()}, raw ≤{CODEBOOK_SIZE[0]-1})")

    # codebook usage
    print(f"=== codebook usage (C28 SID) ===")
    for li in range(N_LAYERS):
        used = len(np.unique(sids[:, li]))
        print(f"  layer {li}: {used}/{CODEBOOK_SIZE[li]} codes used ({100*used/CODEBOOK_SIZE[li]:.1f}%)")
    for li in range(N_LAYERS):
        lo = sum(CODEBOOK_SIZE[:li]) + 1
        hi = lo + CODEBOOK_SIZE[li]
        used = len(np.unique(offsets[:, li]))
        print(f"  vocab[{lo}, {hi}): {used}/{CODEBOOK_SIZE[li]} tokens used ({100*used/CODEBOOK_SIZE[li]:.1f}%)")


if __name__ == "__main__":
    main()
