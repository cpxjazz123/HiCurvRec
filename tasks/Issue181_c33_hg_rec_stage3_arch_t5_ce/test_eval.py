"""C33 standalone test eval — 用 best_ckpt.pt 跑 test eval (beam=20).

独立于 train_stage3.py (避免重新训练). 直接加载 best_ckpt.pt 跑 test 数据集.
"""
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_lib"))

from data.dataset import GenRecDataset  # noqa: E402
from data.dataloader import GenRecDataLoader  # noqa: E402
from model.hg_rec import HG_Rec  # noqa: E402


# === C33 硬编码超参 (与 train_stage3.py 严格一致) ===
SEED = 42
MAX_LEN = 20
CODEBOOK_SIZE = [256, 256, 256]
VOCAB_SIZE = 769
PAD_TOKEN_ID = 0

D_MODEL = 128
D_FF = 1024
NUM_HEADS = 6
D_KV = 64
NUM_LAYERS = 6
NUM_DECODER_LAYERS = 4
DROPOUT_RATE = 0.1

INFER_SIZE = 96
BEAM_SIZE = 20
TOPK_LIST = [5, 10, 20]

DATASET_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/dataset/Instruments/Instruments_c28_sids_for_hgrec.npy"
CKPT_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c33_ckpt/best_ckpt.pt"


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    config = {
        "num_layers": NUM_LAYERS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "num_heads": NUM_HEADS,
        "d_kv": D_KV,
        "dropout_rate": DROPOUT_RATE,
        "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": PAD_TOKEN_ID,
        "decoder_start_token_id": PAD_TOKEN_ID,
        "feed_forward_proj": "relu",
    }
    model = HG_Rec(config).to(device)
    print(f"[load] best ckpt from {CKPT_PATH}", flush=True)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=False))
    model.eval()
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    # test 数据集
    test_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "test.parquet"),
        code_path=CODE_PATH,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN_ID,
    )
    test_loader = GenRecDataLoader(test_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=2)
    print(f"[data] test={len(test_ds)}", flush=True)

    recalls = {f"Recall@{k}": [] for k in TOPK_LIST}
    ndcgs = {f"NDCG@{k}": [] for k in TOPK_LIST}
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=BEAM_SIZE,
            )
            preds = preds[:, 1:]  # 去 decoder_start_token_id (=PAD)
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOPK_LIST:
                recalls[f"Recall@{k}"].append(recall_at_k(pos_index, k))
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k))

    avg_recalls = {k: torch.cat(v).mean().item() for k, v in recalls.items()}
    avg_ndcgs = {k: torch.cat(v).mean().item() for k, v in ndcgs.items()}
    elapsed = time.time() - t0
    print(f"\n=== TEST FINAL (beam={BEAM_SIZE}, single ckpt, R35) ===", flush=True)
    print(f"  test time: {elapsed:.1f}s", flush=True)
    for k, v in avg_recalls.items():
        print(f"  {k}: {v:.4f}", flush=True)
    for k, v in avg_ndcgs.items():
        print(f"  {k}: {v:.4f}", flush=True)

    # 保存 test result json
    import json
    result = {
        "ckpt": CKPT_PATH,
        "best_valid_ndcg20": 0.1065,  # epoch 58
        "test_recall@5": avg_recalls["Recall@5"],
        "test_recall@10": avg_recalls["Recall@10"],
        "test_recall@20": avg_recalls["Recall@20"],
        "test_ndcg@5": avg_ndcgs["NDCG@5"],
        "test_ndcg@10": avg_ndcgs["NDCG@10"],
        "test_ndcg@20": avg_ndcgs["NDCG@20"],
    }
    out_json = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c33_train/test_final.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[save] {out_json}", flush=True)


if __name__ == "__main__":
    main()