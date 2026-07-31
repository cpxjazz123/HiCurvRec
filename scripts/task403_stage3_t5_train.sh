#!/bin/bash
# Task #403 / Stage 3 T5-mini 30 epoch 短训
# v2: 复用 task84 训练循环 (train() 函数), 加 num_epochs=30 + 早期 ckpt saving
# Per task402 verdict §4 关键发现 #4: 验证"训练时长 ≤ 30 epoch"杠杆 Stage 3 是否成立

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task403_s3
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task403_stage3_train_${TS}.log

CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task403_issue109_stage3_short_train
mkdir -p $CKPT_DIR/ckpt/Instruments/$TS

mkdir -p $LOG_DIR
echo "===== [Task #403 Stage 3 v2] task401 ckpt + 30 epoch 短训 launched at $(date) =====" | tee $LOG_FILE

# task402 SID file (复用)
SID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task402.npy
if [ ! -f "$SID_FILE" ]; then
    echo "❌ $SID_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi
echo "SID file: $SID_FILE (task402 复用)" | tee -a $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s3_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s3_mod = _ilu.module_from_spec(_s3_spec)
_s3_spec.loader.exec_module(_s3_mod)
train = _s3_mod.train
evaluate = _s3_mod.evaluate

# task403 Stage 3 config: 跟 task402 完全一致, 仅 num_epochs=30 (vs task402=200)
config = {
    'batch_size': 256,
    'infer_size': 96,
    'lr': 1e-4,
    'device': 'cuda:0',
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'num_epochs': 30,  # task403 关键: 30 epoch 短训 (vs task402=200)
    'early_stop': 5,   # 30 epoch 短训, 早停阈值相应缩短 (vs task402=20)
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_rqvae_task402.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
model.to(device)  # R11.5 FIX: task84 main 自动 model.to(device), fork 后必须手动 to
print(f'[Task #403 Stage 3 v2] Model loaded + to(device), num_epochs={config["num_epochs"]}, early_stop={config["early_stop"]}', flush=True)

train_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'train.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='train',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
val_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'valid.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='train',  # task84 兼容: 'validation' 不被支持, 用 'train' 加载 val parquet
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)

train_loader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = GenRecDataLoader(val_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #403 Stage 3 v2] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}', flush=True)

# Optimizer + R12 ckpt 强制落盘
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

train_state = {
    'best_val_metric': -float('inf'),
    'best_epoch': -1,
    'epochs_no_improve': 0,
    'history': [],
}

save_dir = f'$CKPT_DIR/ckpt/Instruments/$TS'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'HG_Rec_best.pth')

for epoch in range(config['num_epochs']):
    # 训练 (复用 task84 train 函数)
    train_loss = train(model, train_loader, optimizer, device, epoch)
    print(f'[Epoch {epoch+1}/{config["num_epochs"]}] train_loss={train_loss:.4f}', flush=True)

    # 验证 (复用 task84 evaluate 函数)
    val_recalls, val_ndcgs = evaluate(model, val_loader, config['topk_list'], config['beam_size'], device)
    print(f'[Epoch {epoch+1}/{config["num_epochs"]}] val_recalls={val_recalls} val_ndcgs={val_ndcgs}', flush=True)

    # R12: 每 epoch save best ckpt (delete old)
    if os.path.exists(save_path):
        os.remove(save_path)  # R12: 删旧
    torch.save(model.state_dict(), save_path)
    print(f'[Epoch {epoch+1}/{config["num_epochs"]}] ckpt saved (overwrite): {save_path}', flush=True)

    # 简化: 用 R@10 改进判断 (recall 越高越好)
    if isinstance(val_recalls, dict):
        val_r10 = float(val_recalls.get('Recall@10', 0.0))
    else:
        val_r10 = float(val_recalls[1]) if len(val_recalls) > 1 else 0.0  # topk_list[1] = 10

    improved = val_r10 > train_state['best_val_metric']
    if improved:
        train_state['best_val_metric'] = val_r10
        train_state['best_epoch'] = epoch
        train_state['epochs_no_improve'] = 0
    else:
        train_state['epochs_no_improve'] += 1

    print(f'[Epoch {epoch+1}/{config["num_epochs"]}] val_R@10={val_r10:.4f} improved={improved} best_epoch={train_state["best_epoch"]+1} best_R@10={train_state["best_val_metric"]:.4f} patience={train_state["epochs_no_improve"]}/{config["early_stop"]}', flush=True)

    # Early stop
    if train_state['epochs_no_improve'] >= config['early_stop']:
        print(f'[Task #403 Stage 3 v2] Early stop at epoch {epoch+1} (no improvement for {config["early_stop"]} epochs)', flush=True)
        break

print(f'[Task #403 Stage 3 v2] Training done. best_epoch={train_state["best_epoch"]+1} best_val_R@10={train_state["best_val_metric"]:.4f}', flush=True)
print(f'[Task #403 Stage 3 v2] best ckpt saved at: {save_path}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #403 Stage 3 v2] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

echo "✅ Task #403 Stage 3 完成" | tee -a $LOG_FILE
