#!/bin/bash
# Task #153 — HG-Rec β=0.5 ablation (paper Table 6 Instruments 行)
# 2026-07-24 (用户 2026-07-24 提议: 验证 paper β=0.5 是否解释 R@10 -22.4% 差距)
#
# 唯一改动 vs Task #84 baseline: --beta 1.0 → --beta 0.5
# 其他 recipe 完全一致 (R11.3 决策: 单一变量原则)
#
# R7 GPU: GPU 1 (空闲 — Task #151 P5-CID 占 GPU 0)
# 预计: Stage 1 ~25 min + Stage 2 ~3 min + Stage 3 ~70 min + Stage 4 ~30 sec = ~1.7h

set -eo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task153

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task153
mkdir -p "$LOG_DIR"

CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task153/ckpt
SID_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task153/ckpt_hgrec
mkdir -p "$CKPT_DIR" "$SID_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task153

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/beta05_stage1_${TS}.log"

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

echo "===== [Task #153] HG-Rec β=0.5 Stage 1 launched at $(date) =====" | tee "$LOG_FILE"
echo "唯一改动: --beta 1.0 → 0.5 (paper-faithful)" | tee -a "$LOG_FILE"
echo "GPU: 1 (R7 空闲)" | tee -a "$LOG_FILE"
echo "Ckpt dir: $CKPT_DIR" | tee -a "$LOG_FILE"

# Stage 1: HRQ-VAE training (β=0.5)
ITEM_EMB=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet
if [ ! -f "$ITEM_EMB" ]; then
    echo "❌ $ITEM_EMB NOT FOUND" | tee -a "$LOG_FILE"
    exit 1
fi

python3 train_hrqvae.py \
    --data_path "$ITEM_EMB" \
    --ckpt_dir "$CKPT_DIR" \
    --loss_type poincare \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --sk_epsilons 0.0 0.0 0.000 \
    --layers 512 256 128 64 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --eval_step 5 \
    --num_workers 4 \
    --beta 0.5 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG_FILE"

STAGE1_EXIT=$?
echo "===== [Task #153] Stage 1 exit code: $STAGE1_EXIT at $(date) =====" | tee -a "$LOG_FILE"

if [ $STAGE1_EXIT -ne 0 ]; then
    echo "❌ Stage 1 FAILED" | tee -a "$LOG_FILE"
    exit 1
fi

# Stage 2: SID codebook (类似 Task #84 stage 2)
BEST_LOSS_CKPT=$(ls -t "$CKPT_DIR"/Instruments/*/best_loss_model.pth 2>/dev/null | head -1)
echo "Stage 1 best_loss ckpt: $BEST_LOSS_CKPT" | tee -a "$LOG_FILE"

# Stage 2 用 fork script 写死 Task #153 paths (类似 task84_hgrec_stage2_codebook.py)
# 简化: 直接调用 task84_hgrec_stage2_codebook.py 但 override ckpt/output
python3 -c "
import sys, os
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
import torch, numpy as np
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from model.hrqvae import HRQVAE

ckpt_path = '$BEST_LOSS_CKPT'
output_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare_beta05.npy'

ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
args = ckpt['args']
state_dict = ckpt['state_dict']

print(f'[Stage 2] β={args[\"beta\"]}, num_emb_list={args[\"num_emb_list\"]}, e_dim={args[\"e_dim\"]}, epoch={ckpt.get(\"epoch\", -1)}')

data = EmbDataset(args['data_path'])
model = HRQVAE(
    in_dim=data.dim,
    num_emb_list=args['num_emb_list'],
    e_dim=args['e_dim'],
    layers=args['layers'],
    dropout_prob=0.0, bn=False,
    loss_type=args['loss_type'],
    quant_loss_weight=args['quant_loss_weight'],
    beta=args['beta'],
    kmeans_init=args['kmeans_init'],
    kmeans_iters=args['kmeans_iters'],
    sk_eps=args['sk_epsilons'], sk_iters=args['sk_iters'],
)
model.load_state_dict(state_dict)
device = torch.device('cuda:0')
model = model.to(device).eval()

loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=True, pin_memory=True)
all_indices = []
all_indices_str = []
prefix = ['<a_{}>', '<b_{}>', '<c_{}>', '<d_{}>']
from tqdm import tqdm
for d in tqdm(loader, desc='Stage 2 forward'):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)
print(f'[Stage 2] PRE-resolve: items={len(all_indices_str)}, unique={len(set(all_indices_str))}')

# 解析 collision
tt = 0
while tt < 30 and len(all_indices_str) != len(set(all_indices_str)):
    import collections
    idx2ids = collections.defaultdict(list)
    for i, s in enumerate(all_indices_str):
        idx2ids[s].append(i)
    for s, ids in idx2ids.items():
        if len(ids) > 1:
            d = data[ids].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(ids, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
    tt += 1

# dedup 4th digit
codes = []
for value in all_indices:
    codes.append([int(t.split('_')[1].strip('>')) for t in value])
codes_array = np.array(codes)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]
for duplicate in duplicates:
    dup_indices = np.where((codes_array == duplicate).all(axis=1))[0]
    for i, idx in enumerate(dup_indices):
        codes_array[idx, -1] = i

print(f'[Stage 2] POST-resolve: shape={codes_array.shape}, unique={len(np.unique(codes_array, axis=0))}')
np.save(output_path, codes_array)
print(f'[Stage 2] Saved: {output_path}')
" 2>&1 | tee -a "$LOG_FILE"

STAGE2_EXIT=$?
echo "===== [Task #153] Stage 2 exit code: $STAGE2_EXIT at $(date) =====" | tee -a "$LOG_FILE"

if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage 2 FAILED" | tee -a "$LOG_FILE"
    exit 1
fi

# Stage 3 + Stage 4 留给 task153_hgrec_stage3_train.sh + task153_hgrec_stage4_eval.sh
echo "✅ Task #153 Stage 1+2 完成. Stage 3+4 待运行 (见 task153_hgrec_stage3_train.sh)" | tee -a "$LOG_FILE"