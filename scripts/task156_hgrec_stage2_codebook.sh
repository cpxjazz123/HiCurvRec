#!/bin/bash
# Task #156 Stage 2 (R8 recovery) — direct SID codebook generation from best_loss ckpt
# 2026-07-24 (Stage 1 完成了但 main pipeline 在 Stage 2 heredoc 静默失败)
# 直接从 products/task156/ckpt/.../best_loss_model.pth 推 SID codebook
# 复用 Task #84 stage 2 写法 (mirror scripts/task84_hgrec_stage2_codebook.py)
# R7: GPU 3 (其他任务没占用)

set -eo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task156

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task156
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage2_codebook_${TS}.log

mkdir -p $LOG_DIR

BEST_LOSS_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt/Jul-24-2026_20-36-39_beta_0.250_codebook_[32,64,256]_sk_0.500/best_loss_model.pth
OUTPUT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy

if [ ! -f "$BEST_LOSS_CKPT" ]; then
    echo "❌ $BEST_LOSS_CKPT NOT FOUND" | tee "$LOG_FILE"
    exit 1
fi

echo "===== [Task #156 Stage 2] R8 Recovery — SID codebook launched at $(date) =====" | tee "$LOG_FILE"
echo "Recipe: code-default β=0.25, [32,64,256], sk=0.5, 1869 ep" | tee -a $LOG_FILE
echo "Best_loss ckpt: $BEST_LOSS_CKPT" | tee -a $LOG_FILE
echo "Output: $OUTPUT_PATH" | tee -a $LOG_FILE

python3 -u <<'PYTHON_EOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os, collections
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch, numpy as np
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from model.hrqvae import HRQVAE
from tqdm import tqdm

# Stage 1 args (code-default)
ckpt_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt/Jul-24-2026_20-36-39_beta_0.250_codebook_[32,64,256]_sk_0.500/best_loss_model.pth'
data_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'
output_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy'

ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
args = ckpt['args']
state_dict = ckpt['state_dict']

print(f"[Stage 2] beta={args.beta}, num_emb_list={args.num_emb_list}, sk_epsilons={args.sk_epsilons}, epoch={ckpt.get('epoch', -1)}, best_loss={ckpt.get('best_loss', -1)}", flush=True)

data = EmbDataset(data_path)
print(f"[Stage 2] Dataset: {len(data)} items, dim={data.dim}", flush=True)

model = HRQVAE(
    in_dim=data.dim,
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=0.0, bn=False,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    beta=args.beta,
    kmeans_init=args.kmeans_init,
    kmeans_iters=args.kmeans_iters,
    sk_eps=args.sk_epsilons, sk_iters=args.sk_iters,
)
model.load_state_dict(state_dict)
device = torch.device('cuda:0')
model = model.to(device).eval()
print(f"[Stage 2] Model loaded on {device}", flush=True)

loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=False, pin_memory=True)
all_indices = []
all_indices_str = []
prefix = ['<a_{}>', '<b_{}>', '<c_{}>', '<d_{}>']
for d in tqdm(loader, desc='Stage 2 forward'):
    d = d.to(device)
    with torch.no_grad():
        indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_indices.append(code)
        all_indices_str.append(str(code))

print(f"[Stage 2] PRE-resolve: items={len(all_indices_str)}, unique={len(set(all_indices_str))}", flush=True)

# Collision resolve (从 fork 复制: use_sk=True re-run on collisions)
tt = 0
while tt < 30 and len(all_indices_str) != len(set(all_indices_str)):
    idx2ids = collections.defaultdict(list)
    for i, s in enumerate(all_indices_str):
        idx2ids[s].append(i)
    for s, ids in idx2ids.items():
        if len(ids) > 1:
            d = data[ids].to(device)
            with torch.no_grad():
                indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(ids, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
    tt += 1
print(f"[Stage 2] POST-resolve iterations: {tt}", flush=True)

# 转 int array + dedup digit
codes = []
for value in all_indices:
    codes.append([int(t.split('_')[1].strip('>')) for t in value])
codes_array = np.array(codes, dtype=int)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]
for duplicate in duplicates:
    dup_indices = np.where((codes_array == duplicate).all(axis=1))[0]
    for i, idx in enumerate(dup_indices):
        codes_array[idx, -1] = i

print(f"[Stage 2] Final shape: {codes_array.shape}, unique SID: {len(np.unique(codes_array, axis=0))}", flush=True)
np.save(output_path, codes_array)
print(f"[Stage 2] Saved: {output_path}", flush=True)
PYTHON_EOF

STAGE2_EXIT=$?
echo "===== [Task #156 Stage 2] exit code: $STAGE2_EXIT at $(date) =====" | tee -a $LOG_FILE

if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage 2 FAILED" | tee -a $LOG_FILE
    exit 1
fi

if [ ! -f "$OUTPUT_PATH" ]; then
    echo "❌ Stage 2 SID codebook NOT created: $OUTPUT_PATH" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Task #156 Stage 2 SID codebook 生成成功" | tee -a $LOG_FILE
ls -la $OUTPUT_PATH | tee -a $LOG_FILE
