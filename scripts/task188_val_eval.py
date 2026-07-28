#!/usr/bin/env python3
"""Task #188 后续 — 12 ckpt val set 评估 (补救丢失的 val metrics).

直接 fork task84_hgrec_stage3_train.py 的 evaluate 函数, 加载 12 个 HG_Rec_best.pth,
在 valid.parquet 上 evaluate.

排队 — 等 Task #194 Stage 3 (4 GPU 全占) 完成后跑.

输出:
- logs/task188/val_metrics_per_run.json
"""
import os, sys, json, glob, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{REPO}/HG-Rec")
os.chdir(f"{REPO}/HG-Rec")

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.HG_Rec import HG_Rec
from model.utils import set_color
# Inline copy of evaluate helpers from train_HG-Rec.py:23-60
# (avoiding importlib for hyphenated filename + broken model.hg_rec lowercase import)


def calculate_pos_index(preds, labels, maxk=20):
    """Match predictions vs labels (one-hot at first match position, then break)."""
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f'preds.shape[1] = {preds.shape[1]} != {maxk}'
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


def evaluate_val(model, eval_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    with torch.no_grad():
        for batch in tqdm(eval_loader, ncols=80, desc=set_color("Evaluating", "pink")):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
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
    config = {
        "dataset_path": f"{REPO}/HG-Rec/dataset/",
        "dataset_name": "Instruments",
        "topk_list": [5, 10, 20],
        "beam_size": 20,
        "infer_size": 96,
    }
    TIERS = ["t1_8pct", "t2_10pct", "t3_12pct", "t4_13pct"]
    SEEDS = ["42", "123", "2024"]

    device = torch.device("cuda:0")
    results = {}

    for tier in TIERS:
        results[tier] = {}
        code_path_name = f"_t5_rqvae_{tier}.npy"
        # Build validation dataset
        try:
            valid_ds = GenRecDataset(
                dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'valid.parquet'),
                code_path=os.path.join(config['dataset_path'], config['dataset_name'],
                                       config['dataset_name'] + code_path_name),
                mode='evaluation',
                codebook_size=[64, 128, 256, 1],
                max_len=20,
            )
        except Exception as e:
            print(f"❌ {tier}: dataset load failed: {e}")
            continue

        valid_loader = GenRecDataLoader(valid_ds, batch_size=config['infer_size'], shuffle=False)

        for seed in SEEDS:
            ckpt_dir = f"{REPO}/products/task188/t5small_{tier}/seed{seed}/Instruments"
            ckpts = sorted(glob.glob(f"{ckpt_dir}/*/HG_Rec_best.pth"))
            if not ckpts:
                print(f"⚠️ no ckpt for {tier}/seed{seed}")
                continue
            ckpt_path = ckpts[-1]
            print(f"\n=== {tier}/seed{seed} → {ckpt_path} ===")

            # Build model via config dict (mirror task84_hgrec_stage3_train.py args)
            model_config = {
                "num_layers": 6,
                "num_decoder_layers": 4,
                "d_model": 128,
                "d_ff": 1024,
                "num_heads": 6,
                "d_kv": 64,
                "dropout_rate": 0.1,
                "vocab_size": 1025,
                "pad_token_id": 0,
                "eos_token_id": 0,
                "feed_forward_proj": "relu",
            }
            model = HG_Rec(model_config).to(device)

            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
            except Exception as e:
                print(f"⚠️ load failed: {e}")
                del model
                torch.cuda.empty_cache()
                continue

            try:
                avg_recalls, avg_ndcgs = evaluate_val(model, valid_loader, config['topk_list'],
                                                     config['beam_size'], device)
                results[tier][seed] = {
                    "R@5": avg_recalls["Recall@5"],
                    "R@10": avg_recalls["Recall@10"],
                    "R@20": avg_recalls["Recall@20"],
                    "N@5": avg_ndcgs["NDCG@5"],
                    "N@10": avg_ndcgs["NDCG@10"],
                    "N@20": avg_ndcgs["NDCG@20"],
                    "ckpt": ckpt_path,
                }
                print(f"  R@10={avg_recalls['Recall@10']:.4f}, N@10={avg_ndcgs['NDCG@10']:.4f}")
            except Exception as e:
                print(f"⚠️ eval failed: {e}")
                import traceback
                traceback.print_exc()

            del model
            torch.cuda.empty_cache()

    # Save
    out_path = f"{REPO}/logs/task188/val_metrics_per_run.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out_path}")


if __name__ == "__main__":
    main()