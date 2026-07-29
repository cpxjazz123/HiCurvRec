#!/usr/bin/env python3
"""
Task #320 — Issue #38 5-arm Stage 3 retraining
Fork of scripts/task84_hgrec_stage3_train.py with 5-arm protocol change:
  Arm A: Optimizer 改造 (AdamW + cosine + warmup 5000 + dropout 0.05)
  Arm B: LR schedule (inverse sqrt + warmup 10000 + Adam)
  Arm C: Regularization (R-Drop alpha=1.0)
  Arm D: Mixed precision (BF16 autocast)
  Arm E: control (Adam + constant LR + dropout 0.1 + FP32)

Stage 1/2 config: Issue #30 GO endpoint (r_l=[0.1,1,10]+s_l=[2,2,2])
Stage 3: T5-mini 5.5M, 200 epoch full sweep, seed=42, R12 best_ckpt save
Stage 4: K=100 beam_size (Issue #30 ε ceiling)

R11.5: 5 arms 复用单脚本, --arm 参数选择 (避免 5 个独立 fork)
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

# Suppress noisy logging from torch/transformers
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')


def get_local_time():
    return time.strftime("%b-%d-%Y_%H-%M-%S", time.localtime())


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ===== Arm-specific hyperparameter builders =====
def get_arm_config(arm):
    """Return arm-specific (optimizer_name, lr, scheduler_name, dropout_rate, use_bf16, use_rdrop, rdrop_alpha)."""
    if arm == 'A':
        # Optimizer 改造: AdamW + higher lr + cosine + warmup + lower dropout
        return {
            'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 0.05,
            'scheduler': 'cosine', 'warmup_steps': 5000,
            'dropout_rate': 0.05, 'use_bf16': False, 'use_rdrop': False, 'rdrop_alpha': 0.0,
        }
    elif arm == 'B':
        # LR schedule: inverse sqrt (Noam-style) + warmup 2000 + Adam (R11.5 fix v2: d_model=128 scale gives formula peak LR ~ 1.98e-3 with base lr=1e-4, but early steps still tiny. Use Noam-style with base lr = target_peak and lambda = 1.0 at warmup)
        # Use base lr = 1e-4 (matches Arm E baseline), lambda scaled so peak at warmup = 1e-4
        # Noam: lr = d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)
        # We want peak at warmup = base_lr → multiply by base_lr / (d_model^-0.5 * warmup^-0.5) = base_lr / 0.00198 = 50.5 for d_model=128, warmup=2000
        return {
            'optimizer': 'adam', 'lr': 1e-4, 'weight_decay': 0.0,
            'scheduler': 'inv_sqrt', 'warmup_steps': 2000,
            'dropout_rate': 0.1, 'use_bf16': False, 'use_rdrop': False, 'rdrop_alpha': 0.0,
        }
    elif arm == 'C':
        # Regularization: R-Drop alpha=1.0 + baseline Adam
        return {
            'optimizer': 'adam', 'lr': 1e-4, 'weight_decay': 0.0,
            'scheduler': 'constant', 'warmup_steps': 0,
            'dropout_rate': 0.1, 'use_bf16': False, 'use_rdrop': True, 'rdrop_alpha': 1.0,
        }
    elif arm == 'D':
        # Mixed precision: BF16 autocast + baseline Adam
        return {
            'optimizer': 'adam', 'lr': 1e-4, 'weight_decay': 0.0,
            'scheduler': 'constant', 'warmup_steps': 0,
            'dropout_rate': 0.1, 'use_bf16': True, 'use_rdrop': False, 'rdrop_alpha': 0.0,
        }
    elif arm == 'E':
        # control: baseline (Adam + constant + dropout 0.1 + FP32)
        return {
            'optimizer': 'adam', 'lr': 1e-4, 'weight_decay': 0.0,
            'scheduler': 'constant', 'warmup_steps': 0,
            'dropout_rate': 0.1, 'use_bf16': False, 'use_rdrop': False, 'rdrop_alpha': 0.0,
        }
    else:
        raise ValueError(f"Unknown arm: {arm}")


def build_optimizer(model, arm_cfg):
    name = arm_cfg['optimizer']
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=arm_cfg['lr'])
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=arm_cfg['lr'], weight_decay=arm_cfg['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, arm_cfg, total_steps):
    sched_name = arm_cfg['scheduler']
    warmup = arm_cfg['warmup_steps']
    if sched_name == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif sched_name == 'cosine':
        # cosine decay with warmup (linear warmup then cosine)
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)
    elif sched_name == 'inv_sqrt':
        # inverse square root (Noam-style): lr = base_lr * scale * (d_model^-0.5) * min(step^-0.5, step*warmup^-1.5)
        # We want actual_lr at warmup end = base_lr (e.g., 1e-4)
        # At warmup end: noam_factor = d_model^-0.5 * warmup^-0.5 = 1.9764e-3 for d_model=128, warmup=2000
        # So scale = 1 / noam_factor_at_warmup = 506 (so base_lr * 506 * 1.9764e-3 = base_lr)
        from torch.optim.lr_scheduler import LambdaLR
        d_model = 128  # T5-mini d_model
        # Compute scale factor (so peak actual lr = base_lr at warmup end)
        peak_noam = (d_model ** -0.5) * (warmup ** -0.5) if warmup > 0 else 1.0
        scale = 1.0 / peak_noam if peak_noam > 0 else 1.0
        def lr_lambda(step):
            step = max(1, step)
            noam_factor = (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
            return noam_factor * scale
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def rdrop_loss(logits1, logits2, labels, alpha=1.0):
    """R-Drop loss: CE(logits1, labels) + CE(logits2, labels) + alpha * KL(logits1 || logits2) symmetric.
    logits1, logits2: (B, S, V) from 2 forward passes with same input
    labels: (B, S) ground truth
    """
    # Standard CE on both passes
    ce1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), labels.view(-1), ignore_index=0)
    ce2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), labels.view(-1), ignore_index=0)

    # Symmetric KL
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()

    # KL(p1 || p2) = sum p1 * log(p1/p2) = sum p1 * (log_p1 - log_p2)
    # mask out padding
    mask = (labels != 0).float().unsqueeze(-1)  # (B, S, 1)
    kl_1_2 = (p1 * (log_p1 - log_p2)).sum(dim=-1)  # (B, S)
    kl_2_1 = (p2 * (log_p2 - log_p1)).sum(dim=-1)
    kl = (kl_1_2 + kl_2_1) * mask.squeeze(-1)  # (B, S)
    kl = kl.sum() / mask.sum().clamp(min=1)

    return 0.5 * (ce1 + ce2) + alpha * kl


def train_arm(model, train_loader, optimizer, scheduler, device, epoch, arm_cfg, scaler=None):
    """Train 1 epoch with arm-specific protocol."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    use_bf16 = arm_cfg['use_bf16']
    use_rdrop = arm_cfg['use_rdrop']

    from tqdm import tqdm
    from task84_hgrec_stage3_train import calculate_pos_index  # for metric helpers (unused here)

    def set_color(s, c): return s
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=set_color(f"Arm-{arm_cfg['_arm_name']} Ep{epoch+1}", "cyan"))
    for batch_idx, batch in pbar:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()

        if use_rdrop:
            # R-Drop: 2 forward passes
            with torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext():
                out1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Note: model() with labels returns .logits + pre-computed loss (CE); we need raw logits for R-Drop
            # If model's forward with labels returns .loss, we need to manually compute CE
            logits1 = out1.logits if hasattr(out1, 'logits') else out1
            logits2 = out2.logits if hasattr(out2, 'logits') else out2
            loss = rdrop_loss(logits1, logits2, labels, alpha=arm_cfg['rdrop_alpha'])
        else:
            if use_bf16:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss if hasattr(out, 'loss') else out[0]
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss if hasattr(out, 'loss') else out[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # per-step scheduler (cosine/inv_sqrt)

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

    return total_loss / max(1, n_batches)


from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, required=True, choices=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_id', type=int, default=320)
    args = parser.parse_args()

    arm_cfg = get_arm_config(args.arm)
    arm_cfg['_arm_name'] = args.arm

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        'batch_size': args.batch_size,
        'infer_size': 96,
        'lr': arm_cfg['lr'],
        'device': f'cuda:{args.gpu}',
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': arm_cfg['dropout_rate'],
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
        'log_path': f'/home/wlia0047/ar57/wenyu/GeneRec/logs/task{args.task_id}/',
        'save_path': f'/home/wlia0047/ar57/wenyu/GeneRec/products/task{args.task_id}/arm{args.arm}/',
        'early_stop': 999,  # disable early stop for fair 5-arm comparison
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print(f'[TASK{args.task_id}/Arm{args.arm}] Launched at {cur_time}')
    print(f'[TASK{args.task_id}/Arm{args.arm}] arm_cfg = {arm_cfg}')
    print(f'[TASK{args.task_id}/Arm{args.arm}] ckpt_path = {ckpt_path}')
    logging.info(f'Arm config: {arm_cfg}')
    logging.info(f'Full config: {config}')

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

    optimizer = build_optimizer(model, arm_cfg)
    total_steps = args.num_epochs * len(train_dl)
    scheduler = build_scheduler(optimizer, arm_cfg, total_steps)
    print(f'[TASK{args.task_id}/Arm{args.arm}] optimizer={arm_cfg["optimizer"]} scheduler={arm_cfg["scheduler"]} total_steps={total_steps}')

    from task84_hgrec_stage3_train import evaluate

    best_ndcg = 0.0
    best_ckpt = os.path.join(ckpt_path, 'HG_Rec_best.pth')
    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)  # R12: delete old

    for epoch in range(args.num_epochs):
        train_loss = train_arm(model, train_dl, optimizer, scheduler, device, epoch, arm_cfg)
        avg_recalls, avg_ndcgs = evaluate(model, valid_dl, config['topk_list'], config['beam_size'], device)
        ndcg20 = avg_ndcgs['NDCG@20']
        print(f'[TASK{args.task_id}/Arm{args.arm}] Epoch {epoch+1}/{args.num_epochs} loss={train_loss:.4f} val_NDCG@20={ndcg20:.4f} val_R@10={avg_recalls["Recall@10"]:.4f}')

        if ndcg20 > best_ndcg:
            best_ndcg = ndcg20
            torch.save(model.state_dict(), best_ckpt)
            print(f'[TASK{args.task_id}/Arm{args.arm}] saved best ckpt @ epoch {epoch+1} val_NDCG@20={ndcg20:.4f}')

    print(f'[TASK{args.task_id}/Arm{args.arm}] Training done. best_NDCG@20={best_ndcg:.4f}')
    print(f'[TASK{args.task_id}/Arm{args.arm}] Best ckpt: {best_ckpt}')

    # Save final arm config to disk for downstream Stage 4 eval
    with open(os.path.join(ckpt_path, 'arm_config.json'), 'w') as f:
        json.dump({'arm': args.arm, 'arm_cfg': arm_cfg, 'best_ndcg20': best_ndcg, 'task_id': args.task_id}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
