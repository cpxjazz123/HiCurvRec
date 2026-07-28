#!/bin/bash
# Task #207 — Stage 2+3+4 full pipeline launcher
# Run via sbatch for dependency chain: Stage 2 → Stage 3 → Stage 4
#
# Usage:
#   sbatch scripts/task207_stage234_pipeline.sh          # 4 SIDs (hyp + euc)
#
# Output SID files:
#   HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task207_hyp.npy
#   HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task207_euc.npy
#
# Stage 3 ckpt:
#   products/task207/stage3_hyp/
#   products/task207/stage3_euc/

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec

# ============================================================
# Stage 2: SID codebook inference (both arms, parallel)
# ============================================================
echo "===== [Stage 2] SID inference for both arms ====="

for ARM in hyp euc; do
    case $ARM in
        hyp) CKPT=$REPO/products/task207/arm_hyp/Jul-26-2026_16-16-57_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth ;;
        euc) CKPT=$REPO/products/task207/arm_euc/Jul-26-2026_16-16-57_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth ;;
    esac
    OUT_NPY=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task207_${ARM}.npy
    LOG=$REPO/logs/task207/stage2_${ARM}.log

    bash <<'STAGE2_SCRIPT' 2>&1 | tee "$LOG"
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH

python3 -u <<'PYEOF'
import sys, collections
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
import torch, numpy as np
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from model.hrqvae import HRQVAE
from tqdm import tqdm

arm = '$ARM'
ckpt_path = '$CKPT'
output_path = '$OUT_NPY'
data_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'

ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
args = ckpt['args']
state_dict = ckpt['state_dict']
print(f"[Stage 2 {arm}] beta={args.beta}, num_emb_list={args.num_emb_list}, loss_type={args.loss_type}")

data = EmbDataset(data_path)
# Rebuild model with same config
model = HRQVAE(in_dim=data.dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
    layers=args.layers, dropout_prob=0.0, bn=False, loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight, beta=args.beta,
    kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
    sk_eps=args.sk_epsilons, sk_iters=args.sk_iters,
    curvature_list=args.curvatures if hasattr(args, 'curvatures') and args.curvatures is not None else None,
    euclidean_qloss=args.euclidean_qloss if hasattr(args, 'euclidean_qloss') else False)
model.load_state_dict(state_dict)
device = torch.device('cuda:0')
model = model.to(device).eval()

loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=False, pin_memory=True)
all_indices = []
all_indices_str = []
prefix = ['<a_{}>', '<b_{}>', '<c_{}>', '<d_{}>']
for d in tqdm(loader, desc=f'Stage 2 {arm}'):
    d = d.to(device)
    with torch.no_grad():
        indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_indices.append(code)
        all_indices_str.append(str(code))

print(f"[Stage 2 {arm}] PRE-resolve: items={len(all_indices_str)}, unique={len(set(all_indices_str))}")

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
print(f"[Stage 2 {arm}] POST-resolve iterations: {tt}")

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

print(f"[Stage 2 {arm}] Final shape: {codes_array.shape}, unique SID: {len(np.unique(codes_array, axis=0))}")
np.save(output_path, codes_array)
print(f"[Stage 2 {arm}] Saved: {output_path}")
PYEOF
STAGE2_SCRIPT

    echo "[Stage 2 ${ARM}] done, exit=$?"
done

# ============================================================
# Stage 3: T5-mini 9.18M training (both arms, parallel)
# ============================================================
echo "===== [Stage 3] T5-mini training ====="
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

for ARM in hyp euc; do
    CODE_PATH=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task207_${ARM}.npy
    SAVE_DIR=$REPO/products/task207/stage3_${ARM}
    LOG=$REPO/logs/task207/stage3_${ARM}.log
    mkdir -p $SAVE_DIR

    echo "[Stage 3 ${ARM}] Starting T5-mini 9.18M training..."
    CUDA_VISIBLE_DEVICES=${1:-0} TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task207_s3_${ARM} \
    python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --dataset_path $REPO/HG-Rec/dataset/ \
        --code_path $CODE_PATH \
        --codebook_size 64 128 256 1 \
        --num_epochs 200 \
        --batch_size 256 \
        --lr 1e-4 \
        --early_stop 20 \
        --save_path $SAVE_DIR \
        --log_path $REPO/logs/task207/ \
        --mode train \
        > $LOG 2>&1

    echo "[Stage 3 ${ARM}] done, exit=$?"
done

# ============================================================
# Stage 4: Evaluation
# ============================================================
echo "===== [Stage 4] Evaluation ====="
for ARM in hyp euc; do
    SAVE_DIR=$REPO/products/task207/stage3_${ARM}
    BEST_CKPT=$(ls -t $SAVE_DIR/*/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "[Stage 4 ${ARM}] ⚠️ No best ckpt found, trying staging dir..."
        BEST_CKPT=$(find $SAVE_DIR -name "HG_Rec_best.pth" -type f 2>/dev/null | head -1)
    fi

    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        echo "[Stage 4 ${ARM}] ❌ No best ckpt found, skipping eval"
        continue
    fi

    LOG=$REPO/logs/task207/stage4_${ARM}.log
    echo "[Stage 4 ${ARM}] Evaluating $BEST_CKPT ..."

    python3 -u <<EVALEOF 2>&1 | tee "$LOG"
import sys, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util
spec = importlib.util.spec_from_file_location('s3_mod',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
evaluate = mod.evaluate

arm = '$ARM'
ckpt_path = '$BEST_CKPT'
config = {
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'code_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task207_${ARM}.npy',
    'codebook_size': [64, 128, 256, 1],
    'batch_size': 256, 'infer_size': 96,
    'num_epochs': 200, 'lr': 1e-4, 'device': 'cuda',
    'num_layers': 6, 'num_decoder_layers': 4,
    'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
    'dropout_rate': 0.1, 'vocab_size': 1025,
    'pad_token_id': 0, 'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20, 'seed': 2025, 'topk_list': [5, 10, 20],
}

model = HG_Rec(config)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.to('cuda').eval()

eval_dataset = GenRecDataset(config['dataset_name'], config['dataset_path'],
    config['code_path'], config['codebook_size'], config['max_len'], mode='evaluation')
eval_loader = GenRecDataLoader(eval_dataset, config['batch_size'], config['infer_size'])

topk_list = [5, 10, 20]
metrics = evaluate(model, eval_loader, topk_list, beam_size=20, device='cuda')
result = {}
for k, v in zip(topk_list * 2, metrics):
    pass  # evaluate returns (recalls, ndcgs) as flat list
# Actually evaluate returns (recalls, ndcgs) — recall first then ndcg for each k
num_k = len(topk_list)
recalls = metrics[:num_k]
ndcgs = metrics[num_k:]
result = {f'R@{k}': float(r) for k, r in zip(topk_list, recalls)}
result.update({f'N@{k}': float(n) for k, n in zip(topk_list, ndcgs)})
result['arm'] = arm
result['ckpt'] = ckpt_path

print(f"[Stage 4 {arm}] Results: {json.dumps(result, indent=2)}")
with open('/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task207_{arm}_recall_eval.json', 'w') as f:
    json.dump(result, f, indent=2)
EVALEOF

    echo "[Stage 4 ${ARM}] done, exit=$?"
done

echo "===== [Task #207] Full pipeline complete at $(date) ====="
echo "Results: verdicts/task207_{hyp,euc}_recall_eval.json"
