#!/usr/bin/env python3
"""
Task #318 — Issue #38 Arm 1: Stage 3 optimizer ablation on Issue #30 GO endpoint
- Fork of scripts/task84_hgrec_stage3_train.py with parameterized optimizer
- Same Stage 1/2 config as task301 (Issue #30 r_l=[0.1,1,10]+s_l=[2,2,2])
- 4-arm optimizer sweep: Adam (control) / AdamW (wd=0.01) / Adafactor / SGD
- Stage 4 eval at K=100 (Issue #30 ceiling)
- R11.5: Decision = 4-arm parallel on 4 L40S GPUs (~ 1.5h each, ~ 2h total wall time)
"""

import sys, os, json, time, argparse, logging, importlib
import torch
import torch.optim as optim
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader


def get_local_time():
    import time
    return time.strftime("%b-%d-%Y_%H-%M-%S", time.localtime())


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def build_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == 'adafactor':
        # Adafactor: lr typically higher than Adam
        from transformers import Adafactor
        return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, required=True, choices=['adam', 'adamw', 'adafactor', 'sgd'])
    parser.add_argument('--lr', type=float, default=None, help='Override lr (default: per-optimizer default)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_id', type=int, default=318)
    args = parser.parse_args()

    # Per-optimizer default lr (R11.5: Adafactor/SGD need different scales)
    OPTIMIZER_LR = {'adam': 1e-4, 'adamw': 1e-4, 'adafactor': 1e-3, 'sgd': 0.01}
    lr = args.lr if args.lr is not None else OPTIMIZER_LR[args.optimizer]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        'batch_size': args.batch_size,
        'infer_size': 96,
        'lr': lr,
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
        'log_path': f'/home/wlia0047/ar57/wenyu/GeneRec/logs/task{args.task_id}/',
        'save_path': f'/home/wlia0047/ar57/wenyu/GeneRec/products/task{args.task_id}/{args.optimizer}/',
        'early_stop': 20,
        'disable_early_stop': False,
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

    print(f'[TASK{args.task_id}/{args.optimizer}/lr={lr}] Launched at {cur_time}')
    print(f'[TASK{args.task_id}/{args.optimizer}] log_path = {log_path}')
    print(f'[TASK{args.task_id}/{args.optimizer}] ckpt_path = {ckpt_path}')
    logging.info(f'Configuration: {config}')

    model = HG_Rec(config)
    device = torch.device(f'cuda:{args.gpu}')
    model.to(device)

    # Build train/val datasets
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

    # T5 tokenizer config
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

    # Build optimizer (R11.5: Stage 3 protocol change)
    optimizer = build_optimizer(model, args.optimizer, lr)

    # Import train/evaluate from Stage 3 fork
    from task84_hgrec_stage3_train import train, evaluate, calculate_pos_index, recall_at_k, ndcg_at_k

    best_ndcg = 0.0
    early_stop_counter = 0
    best_ckpt = os.path.join(ckpt_path, 'HG_Rec_best.pth')
    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)  # R12: delete old

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_dl, optimizer, device, epoch)
        avg_recalls, avg_ndcgs = evaluate(model, valid_dl, config['topk_list'], config['beam_size'], device)
        ndcg20 = avg_ndcgs['NDCG@20']
        print(f'[TASK{args.task_id}/{args.optimizer}] Epoch {epoch+1}/{args.num_epochs} loss={train_loss:.4f} val_NDCG@20={ndcg20:.4f} val_R@10={avg_recalls["Recall@10"]:.4f}')

        if ndcg20 > best_ndcg:
            best_ndcg = ndcg20
            torch.save(model.state_dict(), best_ckpt)  # R12: save latest
            print(f'[TASK{args.task_id}/{args.optimizer}] saved best ckpt @ epoch {epoch+1} val_NDCG@20={ndcg20:.4f}')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['early_stop']:
                print(f'[TASK{args.task_id}/{args.optimizer}] Early stop @ epoch {epoch+1}')
                break

    print(f'[TASK{args.task_id}/{args.optimizer}] Training done. best_NDCG@20={best_ndcg:.4f}')
    print(f'[TASK{args.task_id}/{args.optimizer}] Best ckpt: {best_ckpt}')


if __name__ == '__main__':
    main()