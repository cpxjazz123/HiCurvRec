"""v9 test eval — DDP 4-card, 加载 v9 best ckpt, 计算 test R@10.

唯一改动 vs eval_test.py:
  - 用 HG_Rec_FFN_Curv 替代 HG_Rec (因为 v9 ckpt 含 curvature_ffn 参数)
  - DDP 4-card 评估 (R35b 同口径)
  - 评估完整 test 集
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.hg_rec_ffn_curv import HG_Rec_FFN_Curv
from model.utils import *


def setup_ddp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu(); labels = labels.detach().cpu()
    assert preds.shape[1] == maxk
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True; break
    return pos_index

def recall_at_k(pos_index, k): return pos_index[:, :k].sum(dim=1).cpu().float()
def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def all_reduce_sum(value, device):
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def genrec_collate(batch, pad_token=0):
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    flattened_histories = torch.stack(
        [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
    )
    flattened_targets = torch.stack([torch.tensor(target, dtype=torch.int64) for target in targets])
    attention_masks = torch.stack(
        [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
    )
    return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--infer_size', type=int, default=96)
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--topk_list', type=int, nargs='+', default=[5, 10, 20])
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    config = {
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1, 'vocab_size': 1025, 'pad_token_id': 0,
        'eos_token_id': 0, 'feed_forward_proj': 'relu',
        'dataset_name': 'Instruments', 'dataset_path': './dataset/',
        'codebook_size': [64, 128, 256, 1],
        'code_path': '_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy',
        'max_len': 20,
    }
    model = HG_Rec_FFN_Curv(config, c_init=1.0, lambda_c=1e-4).to(device)
    if rank == 0:
        print(f"[v9-test] loading {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.eval()

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation', codebook_size=config['codebook_size'], max_len=config['max_len'],
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.infer_size, sampler=test_sampler, num_workers=2, collate_fn=genrec_collate)

    local_recalls = {f'Recall@{k}': 0.0 for k in args.topk_list}
    local_ndcgs = {f'NDCG@{k}': 0.0 for k in args.topk_list}
    local_count = 0

    if rank == 0:
        print(f"[v9-test] evaluating test ({len(test_dataset)} samples, 4-card DDP)")
    with torch.no_grad():
        for batch in tqdm(test_loader, ncols=100) if rank == 0 else test_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            preds = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=args.beam_size)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], args.beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=args.beam_size)
            local_count += input_ids.shape[0]
            for k in args.topk_list:
                local_recalls[f'Recall@{k}'] += recall_at_k(pos_index, k).sum().item()
                local_ndcgs[f'NDCG@{k}'] += ndcg_at_k(pos_index, k).sum().item()

    for k in args.topk_list:
        local_recalls[f'Recall@{k}'] = all_reduce_sum(local_recalls[f'Recall@{k}'], device)
        local_ndcgs[f'NDCG@{k}'] = all_reduce_sum(local_ndcgs[f'NDCG@{k}'], device)
    total_count = all_reduce_sum(local_count, device)
    avg_recalls = {k: v / max(total_count, 1) for k, v in local_recalls.items()}
    avg_ndcgs = {k: v / max(total_count, 1) for k, v in local_ndcgs.items()}

    if rank == 0:
        result = {
            'n_test': int(total_count),
            'Recall@5': avg_recalls['Recall@5'],
            'Recall@10': avg_recalls['Recall@10'],
            'Recall@20': avg_recalls['Recall@20'],
            'NDCG@5': avg_ndcgs['NDCG@5'],
            'NDCG@10': avg_ndcgs['NDCG@10'],
            'NDCG@20': avg_ndcgs['NDCG@20'],
            'curvatures': model.module.get_curvatures(),
        }
        print(f"[v9-test] {json.dumps(result, indent=2)}")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"[v9-test] saved to {args.output}")

    cleanup_ddp()
    os._exit(0)


if __name__ == "__main__":
    main()