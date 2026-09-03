import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random as _random
import pandas as pd
from tqdm import tqdm
import logging

# === R51+ 6 确定性约束 (Stage 3 T5 HG-Rec 训练, R47 联动) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.hg_rec import HG_Rec
from model.utils import *

def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
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
    # Assume only one ground truth item per example
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, ncols=100, desc=set_color(f"Training Epoch {epoch}", "pink")):
        
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list, beam_size, device):
    # R51+: DDP-wrapped 模型无 generate, 必须用 module
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    with torch.no_grad():
        for batch in tqdm(eval_loader, ncols=100, desc=set_color("Evaluating", "pink")):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = raw_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            # print(f"pos_index shape: {pos_index.shape}, pos_index: {pos_index}")
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    # Calculate average recalls and ndcgs
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    """Set random seed for reproducibility."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HG_Rec configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_name', type=str, default='Instruments', help='The name of dataset (R51+ fixed to Instruments)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/', help='Path to the dataset')
    parser.add_argument('--codebook_size', type=int, nargs='+', default=[255,255,255,1], help='emb num of every vq (v3.12 R36 强约束: 三层完全一样 255)')
    parser.add_argument('--code_path', type=str, default='_t5_rqvae_beta_0.250_codebook_[255,255,255]_4layer_sk_0.000_seed42.npy', help='Path to the item-to-code mapping file (4 layer hidden_dim [512,256,128,64])')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (R51+ 统一 seed=42)')
    parser.add_argument('--save_path', type=str, default='./ckpt/', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam size for generation')
    config = vars(parser.parse_args())
    # Set up logging

    cur_time = "{}".format(get_local_time())
    log_path = os.path.join(config['log_path'], config['dataset_name'], cur_time) 
    ckpt_path = os.path.join(config['save_path'], config['dataset_name'], cur_time)
    ensure_dir(log_path)
    ensure_dir(ckpt_path)

    logging.basicConfig(
        filename=os.path.join(log_path,'HG_Rec.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")

    # === R51+ DDP 4 卡初始化 (由 torchrun 调度) ===
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # R51+ per-rank seed 偏移
        seed_offset = rank * 1000
        _random.seed(config['seed'] + seed_offset)
        np.random.seed(config['seed'] + seed_offset)
        torch.manual_seed(config['seed'] + seed_offset)
        torch.cuda.manual_seed_all(config['seed'] + seed_offset)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility (单卡或 rank 0 主路径)
    set_seed(config['seed'])

    model = HG_Rec(config)
    if rank == 0:
        print(model.n_parameters)
        logging.info(model.n_parameters)

    train_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'train.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'],config['dataset_name'] + config['code_path']),
        mode='train',
        codebook_size = config['codebook_size'],
        max_len=config['max_len']
    )
    validation_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'],'valid.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'],config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size = config['codebook_size'],
        max_len=config['max_len']
    )

    # R51+: DistributedSampler for DDP 4 卡分片 (只 train 需要)
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config['seed'], drop_last=False)
    else:
        train_sampler = None

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler)
    # R35b 联动: rank 0 全量 valid 评估 (避免 4 卡切片再 all_reduce 引入非确定性)
    # 只在 rank 0 创建 validation_dataloader, 其他 rank 用 None
    if rank == 0:
        validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    else:
        validation_dataloader = None

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    model.to(device)
    # R51+: DDP 包装模型
    if is_ddp:
        # find_unused_parameters=True (T5 conditional forward 可能某些 step 不参与)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    best_ndcg = 0.0
    early_stop_counter = 0

    for epoch in range(config['num_epochs']):
        if is_ddp:
            train_sampler.set_epoch(epoch)  # R51+: 每 epoch 不同 seed 切分
        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device, epoch)
        if rank == 0:
            logging.info(f"Training loss: {train_loss}")
        # R35b: 只 rank 0 全量评估, 其他 rank barrier 等待
        if is_ddp:
            dist.barrier()
        if rank == 0:
            avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, config['topk_list'], config['beam_size'], device)
            logging.info(f"Validation Dataset: {avg_recalls}")
            logging.info(f"Validation Dataset: {avg_ndcgs}")

            if avg_ndcgs['NDCG@20'] > best_ndcg:
                best_ndcg = avg_ndcgs['NDCG@20']
                early_stop_counter = 0  # Reset early stop counter
                logging.info(f"Best NDCG@20: {best_ndcg}")
                # Save the best model (rank 0 only, save unwrapped state_dict)
                _ckpt_path = os.path.join(ckpt_path,"HG_Rec_epoch_%d.pth"%(epoch))
                raw_model = model.module if is_ddp else model
                torch.save(raw_model.state_dict(), _ckpt_path)
                logging.info(f"Best model saved to {ckpt_path}")
                early_stop_signal = False
            else:
                early_stop_counter += 1
                logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
                if early_stop_counter >= config['early_stop']:
                    logging.info("Early stopping triggered.")
                    early_stop_signal = True
                else:
                    early_stop_signal = False
        else:
            early_stop_signal = False

        # R51+: 广播早停信号到所有 rank
        if is_ddp:
            sig = torch.tensor([1 if early_stop_signal else 0], dtype=torch.uint8, device=device)
            dist.broadcast(sig, src=0)
            if sig.item() == 1:
                break

