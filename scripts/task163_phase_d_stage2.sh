#!/bin/bash
# Task #163 Phase D Stage 2 — 真 ORC θ_init RQ-VAE SID inference
# Fork of task163_orc_init_stage2.sh, with early-exit fix (避免 Phase C 撞 collision loop 卡死)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task163
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/phase_d_stage2_${TS}.log

# Stage 1 ckpt 来自 Phase D 训出来的 (跟 Phase C 是不同的 _TRAINING_PID 时间戳目录)
STAGE1_CKPT=$(ls -td /home/wlia0047/ar57/wenyu/GeneRec/products/task163/phase_d_rqvae_real_orc/*/best_loss_model.pth 2>/dev/null | head -1)
if [ -z "$STAGE1_CKPT" ] || [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ Phase D Stage 1 ckpt MISSING (run task163_phase_d_stage1.sh first)" | tee $LOG_FILE
    exit 1
fi
echo "Stage 1 ckpt: $STAGE1_CKPT" | tee $LOG_FILE

OUTPUT=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_orc_init.npy
echo "Output: $OUTPUT" | tee $LOG_FILE
echo "===== [Task #163 Phase D Stage 2] ORC-init SID inference launched at $(date) =====" | tee $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a $LOG_FILE
import os, sys, json, numpy as np, torch
import pandas as pd

# Reuse Phase C Stage 2 imports (model/ lives under HG-Rec/, not GeneRec root)
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.utils import EmbDataset

CKPT_PATH = "$STAGE1_CKPT"
OUTPUT_PATH = "$OUTPUT"

ckpt = torch.load(CKPT_PATH, map_location='cpu')
print(f"[Stage 2] ckpt: best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}", flush=True)
args = ckpt['args']
print(f"[Stage 2] args: M={args.get('M','?')}, kappa_max={args.get('kappa_max','?')}, "
      f"num_emb_list={args.get('num_emb_list','?')}, beta={args.get('beta','?')}", flush=True)

model = FreeCurvHRQVAE(
    in_dim=768,
    M=args.get('M', 1),
    e_dim=args.get('e_dim', 32),
    layers=args.get('layers', [512, 256, 128]),
    num_emb_list=args.get('num_emb_list', [32, 64, 256]),
    loss_type=args.get('loss_type', 'poincare'),
    beta=args.get('beta', 0.5),
    quant_loss_weight=args.get('quant_loss_weight', 1.0),
    sk_eps=args.get('sk_epsilons', [0, 0, 0]),
    sk_iters=args.get('sk_iters', 50),
    kappa_max=args.get('kappa_max', 2.0),
)
model.load_state_dict(ckpt['state_dict'])
model = model.to('cuda:0').eval()

# Print learned kappas per layer
for li, vq in enumerate(model.hrq.vq_layers):
    kappas = (model.kappa_max * torch.tanh(vq.theta_m)).detach().cpu().numpy()
    print(f"[Stage 2] Layer {li} κ_m = [{', '.join(f'{k:+.4f}' for k in kappas)}]", flush=True)

DATA_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

dataset = EmbDataset(DATA_PATH)
data = torch.tensor(np.stack(dataset.embeddings), dtype=torch.float32)
N = data.shape[0]
print(f"[Stage 2] Loaded {N} items, emb_dim={data.shape[1]}", flush=True)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=False)

# Initial code assignment (no sinkhorn)
all_indices = []
all_indices_str = []
num_emb_list = args.get('num_emb_list', [32, 64, 256])

with torch.no_grad():
    for batch in loader:
        batch = batch.to('cuda:0')
        indices = model.get_indices(batch, use_sk=False)
        # indices shape: (B, num_layers)
        for b in range(batch.shape[0]):
            code = [f"L{i}_{int(indices[b, i])}>" for i in range(len(num_emb_list))]
            all_indices.append(code)
            all_indices_str.append(str(code))

print(f"[Stage 2] Initial codes: {len(all_indices_str)} items", flush=True)

# Collision resolution with EARLY-EXIT FIX (跟 Phase C 区别)
def check_collision(s_list):
    return len(s_list) == len(set(s_list))

def get_collision_items(s_list):
    d = {}
    for i, x in enumerate(s_list):
        d.setdefault(x, []).append(i)
    return [v for v in d.values() if len(v) > 1]

MAX_ITER = 30
EARLY_EXIT_NO_CHANGE = 3
tt = 0
prev_n_groups = None
no_change_count = 0

while tt < MAX_ITER:
    if check_collision(all_indices_str):
        print(f"[Stage 2] collision resolved at iter {tt} (no collisions)", flush=True)
        break
    cg = get_collision_items(all_indices_str)
    n_groups = len(cg)
    print(f"  collision iter {tt}: {n_groups} groups (early-exit {no_change_count}/{EARLY_EXIT_NO_CHANGE})", flush=True)

    if n_groups == prev_n_groups:
        no_change_count += 1
        if no_change_count >= EARLY_EXIT_NO_CHANGE:
            print(f"[Stage 2] EARLY-EXIT: n_groups stuck at {n_groups} for {EARLY_EXIT_NO_CHANGE} iters, breaking", flush=True)
            break
    else:
        no_change_count = 0
    prev_n_groups = n_groups

    for items in cg:
        d_batch = data[items].to('cuda:0')
        idx = model.get_indices(d_batch, use_sk=True)
        # idx shape: (n_items_in_group, num_layers)
        for row_i, it in enumerate(items):
            code = [f"L{i}_{int(idx[row_i, i])}>" for i in range(len(num_emb_list))]
            all_indices[it] = code
            all_indices_str[it] = str(code)
    tt += 1

# Convert to int array + append dedup column
codes = []
for v in all_indices:
    codes.append([int(x.split('_')[1].strip('>')) for x in v])
codes_array = np.array(codes, dtype=int)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

# Append cumulative dedup digit
seen = {}
final_codes = codes_array.copy()
for i, row in enumerate(codes_array):
    key = tuple(row)
    if key not in seen:
        seen[key] = 0
    else:
        seen[key] += 1
        final_codes[i, -1] = seen[key]
codes_array = final_codes

print(f"[Stage 2] {len(codes_array)} items, max token id = {codes_array.max()}", flush=True)
print(f"[Stage 2] Required vocab_size >= {codes_array.max() + 1}", flush=True)

np.save(OUTPUT_PATH, codes_array)
print(f"[Stage 2] Saved: {OUTPUT_PATH}", flush=True)
print(f"[Stage 2] DONE.", flush=True)
PYTHON_EOF

echo "===== [Task #163 Phase D Stage 2] SID inference completed at $(date) =====" | tee -a $LOG_FILE