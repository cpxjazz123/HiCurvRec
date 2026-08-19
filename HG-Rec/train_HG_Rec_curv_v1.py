"""v1 training script: HG_Rec_Curv with learnable κ via Poincaré Embedding.

修改 vs train_HG-Rec.py:
- model = HG_Rec_Curv(config) 替换 HG_Rec(config)
- 输出每 5 epoch 当前 κ 值 (监控 κ 是否在学习)
- 输出 lr/kappa history 到独立 log 便于画图
- 复用相同的 optimizer / 数据 / 训练循环结构
"""
import torch
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
import random
import pandas as pd
from tqdm import tqdm
import logging

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.hg_rec_curv import HG_Rec_Curv
from model.utils import *


def calculate_pos_index(preds, labels, maxk=20):
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
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    with torch.no_grad():
        for batch in tqdm(eval_loader, ncols=100, desc=set_color("Evaluating", "pink")):
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HG_Rec_Curv v1 configuration")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--infer_size', type=int, default=96)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_kv', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=1025)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--eos_token_id', type=int, default=0)
    parser.add_argument('--feed_forward_proj', type=str, default='relu')
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default='Instruments')
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--codebook_size', type=int, nargs='+', default=[64, 128, 256, 1])
    parser.add_argument('--code_path', type=str,
                        default='_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'])
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--topk_list', type=list, default=[5, 10, 20])
    parser.add_argument('--beam_size', type=int, default=20)
    config = vars(parser.parse_args())

    cur_time = "{}".format(get_local_time())
    log_path = os.path.join(config['log_path'], config['dataset_name'], cur_time)
    ckpt_path = os.path.join(config['save_path'], config['dataset_name'], cur_time)
    ensure_dir(log_path)
    ensure_dir(ckpt_path)

    logging.basicConfig(
        filename=os.path.join(log_path, 'HG_Rec_curv.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Configuration: {config}")

    model = HG_Rec_Curv(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    train_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'train.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='train',
        codebook_size=config['codebook_size'],
        max_len=config['max_len']
    )
    validation_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'valid.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size=config['codebook_size'],
        max_len=config['max_len']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device, epoch)
        logging.info(f"Training loss: {train_loss}")
        # 监控 κ 是否在学习
        cur_kappa = model.model.shared.kappa.item()
        cur_log_kappa = model.model.shared.log_kappa.item()
        logging.info(f"[κ_monitor] κ={cur_kappa:.6f} log_κ={cur_log_kappa:.6f}")

        avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, config['topk_list'], config['beam_size'], device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")

        cur_ndcg = avg_ndcgs['NDCG@20']
        if cur_ndcg > best_ndcg:
            best_ndcg = cur_ndcg
            early_stop_counter = 0
            best_epoch = epoch
            logging.info(f"Best NDCG@20: {best_ndcg}")
            _ckpt_path = os.path.join(ckpt_path, "HG_Rec_curv_epoch_%d.pth" % epoch)
            torch.save(model.state_dict(), _ckpt_path)
            logging.info(f"Best model saved to {_ckpt_path}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break

    logging.info(f"Training finished. Best epoch={best_epoch}, best_ndcg@20={best_ndcg}")
    logging.info(f"Final κ={model.model.shared.kappa.item():.6f}")