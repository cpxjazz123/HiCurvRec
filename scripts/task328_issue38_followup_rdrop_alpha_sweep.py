#!/usr/bin/env python3
"""
Task #328 — Issue #38 Layer 2 Follow-up: R-Drop alpha sweep (4-arm)
R-Drop alpha tuning on Stage 3 protocol layer (Issue #38 follow-up)
Fork of scripts/task320_issue38_5arm_stage3_train.py with --rdrop_alpha CLI arg

Layer 1 (task320): R-Drop α=1.0 GO (test_R@10=0.1034 +1.4% baseline)
Layer 2 (本 task328): R-Drop alpha sweep α ∈ {0.5, 1.0, 2.0, 4.0}

Stage 1/2 config: Issue #30 GO endpoint (r_l=[0.1,1,10]+s_l=[2,2,2])
Stage 3: T5-mini 5.5M + R-Drop + Adam lr=1e-4 + constant LR + dropout 0.1
Stage 4: K=100 beam_size (Issue #30 ε ceiling)

R12: best ckpt save + delete old ckpt (no checkpoint history)
"""

import sys, os, json, time, argparse, logging, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')


def get_local_time():
    return time.strftime("%b-%d-%Y_%H-%M-%S", time.localtime())


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def rdrop_loss(logits1, logits2, labels, alpha=1.0):
    """R-Drop loss: 0.5*(CE1+CE2) + alpha * symmetric_KL(logits1, logits2)
    CE: standard cross-entropy on both forward passes
    KL: symmetric KL divergence between two output distributions
    """
    ce1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), labels.view(-1), ignore_index=0)
    ce2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), labels.view(-1), ignore_index=0)

    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()

    mask = (labels != 0).float().unsqueeze(-1)  # (B, S, 1)
    kl_1_2 = (p1 * (log_p1 - log_p2)).sum(dim=-1)  # (B, S)
    kl_2_1 = (p2 * (log_p2 - log_p1)).sum(dim=-1)
    kl = (kl_1_2 + kl_2_1) * mask.squeeze(-1)
    kl = kl.sum() / mask.sum().clamp(min=1)

    return 0.5 * (ce1 + ce2) + alpha * kl


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, alpha):
    model.train()
    total_loss = 0.0
    n_batches = 0

    from tqdm import tqdm
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f"R-Drop-α{alpha} Ep{epoch+1}")
    for batch_idx, batch in pbar:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()

        # R-Drop: 2 forward passes with same input, compute symmetric KL
        loss1_pre, logits1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss2_pre, logits2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = rdrop_loss(logits1, logits2, labels, alpha=alpha)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rdrop_alpha', type=float, required=True, help='R-Drop alpha coefficient (e.g., 0.5, 1.0, 2.0, 4.0)')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_id', type=int, default=328)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        'batch_size': args.batch_size,
        'infer_size': 96,
        'lr': 1e-4,
        'device': f'cuda:{args.gpu}',
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
        'feed_forward_proj': 'relu', 'max_len': 20,
        'dataset_name': 'Instruments',
        'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
        'codebook_size': [64, 128, 256, 1],
        # Issue #30 GO endpoint
        'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',
        'topk_list': [5, 10, 20],
        'beam_size': 20,
        'mode': 'train',
        'log_path': f'/home/wlia0047/ar57/wenyu/GeneRec/logs/task{args.task_id}_rdrop_alpha_{args.rdrop_alpha}/',
        'save_path': f'/home/wlia0047/ar57/wenyu/GeneRec/products/task{args.task_id}_rdrop_alpha_sweep/alpha_{args.rdrop_alpha}/',
        'early_stop': 999,
        'disable_early_stop': True,
        'resume_from': None,
        'seed': args.seed,
    }

    cur_time = get_local_time()
    log_path = os.path.join(config['log_path'], config['dataset_name'], cur_time)
    ckpt_path = os.path.join(config['save_path'], config['dataset_name'], cur_time)
    ensure_dir(log_path)
    ensure_dir(ckpt_path)

    logging.basicConfig(
        filename=os.path.join(log_path, 'HG_Rec.log'),
        level=logging.INFO,
        format='%(asctime)s - %(asctime)s - %(message)s'
    )

    print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] Launched at {cur_time}')
    print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] ckpt_path = {ckpt_path}')
    logging.info(f'R-Drop alpha={args.rdrop_alpha}')

    model = HG_Rec(config)
    device = torch.device(f'cuda:{args.gpu}')
    model.to(device)

    train_ds = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'train.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='train',
        codebook_size=config['codebook_size'],
        max_len=config['max_len'],
    )
    valid_ds = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'valid.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size=config['codebook_size'],
        max_len=config['max_len'],
    )

    from transformers import T5Config
    tokenizer = T5Config(
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        d_kv=config['d_kv'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dropout_rate=config['dropout_rate'],
    )
    train_ds.tokenizer = tokenizer
    valid_ds.tokenizer = tokenizer

    train_dl = GenRecDataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    valid_dl = GenRecDataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    total_steps = args.num_epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] Adam lr={config["lr"]} constant, total_steps={total_steps}')

    from task84_hgrec_stage3_train import evaluate

    best_ndcg = 0.0
    best_ckpt = os.path.join(ckpt_path, 'HG_Rec_best.pth')
    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)  # R12: delete old

    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, epoch, args.rdrop_alpha)
        avg_recalls, avg_ndcgs = evaluate(model, valid_dl, config['topk_list'], config['beam_size'], device)
        ndcg20 = avg_ndcgs['NDCG@20']
        print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] Epoch {epoch+1}/{args.num_epochs} loss={train_loss:.4f} val_NDCG@20={ndcg20:.4f} val_R@10={avg_recalls["Recall@10"]:.4f}')

        if ndcg20 > best_ndcg:
            best_ndcg = ndcg20
            torch.save(model.state_dict(), best_ckpt)
            print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] saved best ckpt @ epoch {epoch+1} val_NDCG@20={ndcg20:.4f}')

    print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] Training done. best_NDCG@20={best_ndcg:.4f}')
    print(f'[TASK{args.task_id}/alpha={args.rdrop_alpha}] Best ckpt: {best_ckpt}')

    with open(os.path.join(ckpt_path, 'arm_config.json'), 'w') as f:
        json.dump({'rdrop_alpha': args.rdrop_alpha, 'best_ndcg20': best_ndcg, 'task_id': args.task_id}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
