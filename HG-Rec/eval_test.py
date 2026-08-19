"""官方 HG-Rec test 评估 (官方代码缺 test stage, 此脚本补齐).

加载 best ckpt (epoch 65, NDCG@20=0.1049) → test.parquet → beam20 评估.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("thr", "train_HG-Rec.py")
thr = importlib.util.module_from_spec(spec); spec.loader.exec_module(thr)
HG_Rec = thr.HG_Rec
GenRecDataset = thr.GenRecDataset
GenRecDataLoader = thr.GenRecDataLoader
evaluate = thr.evaluate

CKPT = "./ckpt/Instruments/Aug-14-2026_20-15-45/HG_Rec_epoch_63.pth"
SID = "./dataset/Instruments/Instruments_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy"

config = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)

def main():
    device = torch.device("cuda:0")
    model = HG_Rec(config).to(device)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {CKPT}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = GenRecDataset(
        dataset_path="./dataset/Instruments/test.parquet",
        code_path=SID, mode="evaluation",
        codebook_size=[64,128,256,1], max_len=20)
    loader = GenRecDataLoader(ds, batch_size=96, shuffle=False)
    print("test samples:", len(ds))

    recalls, ndcgs = evaluate(model, loader, [5,10,20], 20, device)
    print("\n=== TEST (beam20) ===")
    for k in [5,10,20]:
        print(f"  Recall@{k}: {recalls['Recall@'+str(k)]:.4f}  NDCG@{k}: {ndcgs['NDCG@'+str(k)]:.4f}")

if __name__ == "__main__":
    main()
