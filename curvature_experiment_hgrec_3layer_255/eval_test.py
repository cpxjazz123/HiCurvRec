"""官方 HG-Rec test 评估 (官方代码缺 test stage, 此脚本补齐).

加载 best ckpt (epoch 65, NDCG@20=0.1049) → test.parquet → beam20 评估.
R51+: 加 6 确定性约束 + 写 test_final.json (与 RQ-VAE-Recommender 一致).
"""
import os
import sys
import json
import random as _random
import numpy as _np
import torch
import numpy as np
from tqdm import tqdm

# === R51+ 6 确定性约束 (Stage 4 评估, R47 联动) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("thr", "train_HG-Rec.py")
thr = importlib.util.module_from_spec(spec); spec.loader.exec_module(thr)
HG_Rec = thr.HG_Rec
GenRecDataset = thr.GenRecDataset
GenRecDataLoader = thr.GenRecDataLoader
evaluate = thr.evaluate

CKPT = sys.argv[1] if len(sys.argv) > 1 else "./ckpt/Instruments/HG-Rec_3layer_255_seed42/HG_Rec_best.pth"
SID = "./dataset/Instruments/Instruments_t5_rqvae_beta_0.250_codebook_[255,255,255]_4layer_sk_0.000_seed42.npy"
TEST_JSON = sys.argv[2] if len(sys.argv) > 2 else "./ckpt/Instruments/HG-Rec_3layer_255_seed42/test_final.json"

config = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)

def main():
    # R51+: Stage 4 评估端, seed=42 单 seed
    _random.seed(42)
    _np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0")
    model = HG_Rec(config).to(device)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {CKPT}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = GenRecDataset(
        dataset_path="./dataset/Instruments/test.parquet",
        code_path=SID, mode="evaluation",
        codebook_size=[255,255,255,1], max_len=20)
    loader = GenRecDataLoader(ds, batch_size=96, shuffle=False)
    print("test samples:", len(ds))

    recalls, ndcgs = evaluate(model, loader, [5,10,20], 20, device)
    print("\n=== TEST (beam20) ===")
    for k in [5,10,20]:
        print(f"  Recall@{k}: {recalls['Recall@'+str(k)]:.4f}  NDCG@{k}: {ndcgs['NDCG@'+str(k)]:.4f}")

    # R51+: 持久化 test_final.json (与 RQ-VAE-Recommender test_final.json 一致格式)
    os.makedirs(os.path.dirname(TEST_JSON), exist_ok=True)
    test_final = {
        "best_ckpt": os.path.basename(CKPT),
        "test_R@5": float(recalls['Recall@5']),
        "test_R@10": float(recalls['Recall@10']),
        "test_R@20": float(recalls['Recall@20']),
        "test_NDCG@5": float(ndcgs['NDCG@5']),
        "test_NDCG@10": float(ndcgs['NDCG@10']),
        "test_NDCG@20": float(ndcgs['NDCG@20']),
        "n_eval": int(len(ds)),
    }
    with open(TEST_JSON, "w") as f:
        json.dump(test_final, f, indent=2)
    print(f"\nSaved → {TEST_JSON}", flush=True)

if __name__ == "__main__":
    main()
