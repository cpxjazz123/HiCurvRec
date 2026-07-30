"""
Task #338 / Issue #51 Stage 4 — Test eval (beam=20) on best_ckpt.pth.

task84 script 的 test eval 被注释了. 我们手写一个最小化的 test eval:
加载 best_ckpt → 在 test.parquet 上 evaluate → 写 verdict JSON.

用法:
  python3 scripts/task338_issue51_stage4_test_eval.py \\
    --ckpt_path products/task338/t5_out/Instruments/Jul-30-2026_18-40-03/HG_Rec_best.pth \\
    --code_path _t5_rqvae_issue51_combined.npy \\
    --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

BASE = Path('/home/wlia0047/ar57/wenyu/GeneRec')
sys.path.insert(0, str(BASE / 'HG-Rec'))
sys.path.insert(0, str(BASE / 'scripts'))

os.environ['PYTHONPATH'] = str(BASE / 'HG-Rec') + ':' + os.environ.get('PYTHONPATH', '')
os.environ['HF_HOME'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'
os.environ['TRITON_CACHE_DIR'] = '/home/wlia0047/.triton/cache_task338_stage4_test'

import torch
import numpy as np
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.HG_Rec import HG_Rec

# Inline metric functions (from HG-Rec/train_HG-Rec.py)
def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True)
    p.add_argument('--dataset_path', default=str(BASE / 'HG-Rec/dataset'))
    p.add_argument('--dataset_name', default='Instruments')
    p.add_argument('--code_path', required=True,
                   help='Suffix only — task84 script convention: <dataset_name><code_path>')
    p.add_argument('--codebook_size', type=int, nargs='+', default=[64, 128, 256, 1])
    p.add_argument('--max_len', type=int, default=20)
    p.add_argument('--beam_size', type=int, default=20)
    p.add_argument('--infer_size', type=int, default=96)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_json', default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, args.dataset_name, 'test.parquet'),
        code_path=os.path.join(args.dataset_path, args.dataset_name, args.dataset_name + args.code_path),
        mode='evaluation',
        codebook_size=args.codebook_size,
        max_len=args.max_len,
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=args.infer_size, shuffle=False)

    # Build model (T5-mini)
    config = {
        'vocab_size': 1025,
        'd_model': 128,
        'd_ff': 1024,
        'num_layers': 6,
        'num_decoder_layers': 4,
        'num_heads': 6,
        'd_kv': 64,
        'max_len': args.max_len,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'dropout_rate': 0.1,
        'feed_forward_proj': 'relu',
    }
    model = HG_Rec(config)

    # Load best ckpt
    sd = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Loaded ckpt: {args.ckpt_path}")

    # Evaluate
    topk_list = [5, 10, 20]
    from tqdm import tqdm
    def evaluate_fn(model, eval_loader, topk_list, beam_size, device):
        model.eval()
        recalls = {'Recall@' + str(k): [] for k in topk_list}
        ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
        with torch.no_grad():
            for batch in tqdm(eval_loader, ncols=100, desc="Evaluating"):
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
    avg_recalls, avg_ndcgs = evaluate_fn(model, test_dataloader, topk_list, args.beam_size, device)
    print(f"\nTest Dataset:")
    print(f"  Recall@5  = {avg_recalls['Recall@5']:.4f}")
    print(f"  Recall@10 = {avg_recalls['Recall@10']:.4f}")
    print(f"  Recall@20 = {avg_recalls['Recall@20']:.4f}")
    print(f"  NDCG@5    = {avg_ndcgs['NDCG@5']:.4f}")
    print(f"  NDCG@10   = {avg_ndcgs['NDCG@10']:.4f}")
    print(f"  NDCG@20   = {avg_ndcgs['NDCG@20']:.4f}")

    # Save JSON
    if args.output_json is None:
        # default: verdicts/task<N>_stage4_beam<beam>.json
        task_id = Path(args.ckpt_path).parts[-3]  # 'task338'
        args.output_json = BASE / 'verdicts' / f'{task_id}_stage4_beam{args.beam_size}.json'
    else:
        args.output_json = Path(args.output_json)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'task': 'Issue #51 (Task #338)',
        'ckpt_path': str(args.ckpt_path),
        'code_path': args.code_path,
        'beam_size': args.beam_size,
        'topk_list': topk_list,
        'test_recalls': avg_recalls,
        'test_ndcgs': avg_ndcgs,
    }
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Verdict JSON: {args.output_json}")


if __name__ == '__main__':
    main()