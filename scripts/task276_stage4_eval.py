"""
Task #276 Stage 4 — A2 curriculum R@10 evaluation (test set)
2026-07-29

输入: products/task276/stage3/<run_id>/HG_Rec_best.pth
       (Stage 3 已训练 + early stop, NDCG@20 best)
       A2 code_path: products/task276/stage2/A2_t5_hrqvae_poincare.npy
输出: products/task276/stage4/eval_metrics.json
       {Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20}

逻辑:
  1. Load HG_Rec model with same Stage 3 config
  2. Load best ckpt weights
  3. Create test_dataloader (test.parquet)
  4. evaluate() → metrics @ topk_list [5, 10, 20], beam_size=20
  5. Save JSON

R11.3: 跟 task84_hgrec_stage3_train.py evaluate() 函数一致 (line 80-100)
R5: 跟 HG-Rec baseline (Task #84) R@10=0.1020 比较 — GO if > 0.1020
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, os.path.join(REPO, "HG-Rec"))

import random
import numpy as np

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.HG_Rec import HG_Rec


def set_seed(seed):
    """Local copy — same as task84_hgrec_stage3_train.py"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_index(preds, labels, maxk=20):
    """跟 task84_hgrec_stage3_train.py line 23-49 一致 (本地 copy)"""
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

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


def evaluate(model, eval_loader, topk_list, beam_size, device, max_len):
    """跟 task84_hgrec_stage3_train.py evaluate() 一致 (line 80-104)

    注意: HG_Rec.generate() 内部硬编码 max_length=5, 不能再传 max_length kwarg (会冲突)
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            # 跟 task84 evaluate() 一致: 不传 max_length, HG_Rec.generate 内部默认 5
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)

            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)

    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--code_path", default=None,
                        help="Path to A2_t5_hrqvae_poincare.npy")
    parser.add_argument("--dataset_name", default="Instruments")
    parser.add_argument("--dataset_path", default=f"{REPO}/HG-Rec/dataset/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
    # HG-Rec 模型 config (跟 stage3 一致)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=1025)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--eos_token_id", type=int, default=0)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.ckpt_path), "..", "stage4_eval_metrics.json")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"[Task #276 Stage 4] ckpt_path={args.ckpt_path}")
    print(f"[Task #276 Stage 4] code_path={args.code_path}")
    print(f"[Task #276 Stage 4] dataset={args.dataset_name}")
    print(f"[Task #276 Stage 4] device={args.device}")

    # 1. Model config (跟 stage3 一致)
    config = {
        "num_layers": args.num_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "d_kv": args.d_kv,
        "dropout_rate": args.dropout_rate,
        "vocab_size": args.vocab_size,
        "pad_token_id": args.pad_token_id,
        "eos_token_id": args.eos_token_id,
        "feed_forward_proj": args.feed_forward_proj,
        "max_len": args.max_len,
    }

    # 2. Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HG_Rec(config).to(device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[Task #276 Stage 4] model loaded: {model.n_parameters} params")

    # 3. Build test_dataloader
    code_path = args.code_path
    if code_path is None:
        # 默认从 ckpt 同目录推断
        code_path = os.path.join(args.dataset_path, args.dataset_name,
                                 f"{args.dataset_name}_t5_hrqvae_poincare.npy")

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, args.dataset_name, "test.parquet"),
        code_path=code_path,
        mode="evaluation",
        codebook_size=[64, 128, 256, 1],
        max_len=args.max_len,
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[Task #276 Stage 4] test_dataloader: {len(test_dataset)} examples, {len(test_dataloader)} batches")

    # 4. Evaluate
    set_seed(args.seed)
    avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, args.topk_list, args.beam_size, device, args.max_len)
    print(f"[Task #276 Stage 4] Test Recall: {avg_recalls}")
    print(f"[Task #276 Stage 4] Test NDCG:   {avg_ndcgs}")

    # 5. Save metrics
    metrics = {
        "ckpt_path": args.ckpt_path,
        "code_path": code_path,
        "n_test_examples": len(test_dataset),
        "beam_size": args.beam_size,
        **avg_recalls,
        **avg_ndcgs,
    }
    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Task #276 Stage 4] metrics saved to {args.output_path}")

    # 6. GO/NO-GO check
    r10 = avg_recalls.get("Recall@10", 0)
    HG_REC_BASELINE = 0.1020
    if r10 > HG_REC_BASELINE:
        print(f"[Task #276 Stage 4] ✅ GO: R@10={r10:.4f} > HG-Rec baseline {HG_REC_BASELINE}")
    else:
        print(f"[Task #276 Stage 4] ❌ NO-GO: R@10={r10:.4f} <= HG-Rec baseline {HG_REC_BASELINE}")


if __name__ == "__main__":
    main()