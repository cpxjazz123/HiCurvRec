#!/bin/bash
# Task #194 / Issue #40 Gate 1 — protocol-matched K0=64 control
# 目的: 排除 Issue #40 owner 质疑的 protocol confound, 验证 task194 K0=64 +2.0% 是 K0 effect 还是
#       Stage 1 (batch_size + epochs) + Stage 2 (Sinkhorn) protocol leak.
# 实施: 跑 #84 baseline recipe @ K0=64 = 4-element confusion fixed → 应该 reproduce baseline 0.1020,
#       不是 task194 K0=64 的 0.1041.
# R11.5 决策 (Issue #40 owner 0 评论, R14 强制处理, 2026-07-30 12:50 owner explicit OK):
#   - Stage 1: num_emb_list=64 128 256 (3 elements, MIRROR #84 baseline)
#              batch_size=1024 (NOT task194 256)
#              epochs=1000 (NOT task194 500)
#              loss_type=poincare, beta=0.5, sk_epsilons=0.0 0.0 0.0 (Stage 1 train, no Sinkhorn)
#   - Stage 2: task84_hgrec_stage2_codebook.py (NOT task194 fork)
#              sk_eps override = NONE (走 ckpt args sk_eps=0.0 → argmin only)
#   - Stage 3: task84_hgrec_stage3_train.py with code_path=_t5_hrqvae_poincare.npy
#              codebook_size=64 128 256 1 (MIRROR #84 baseline)
#   - Stage 4: standard test eval, n_test=24772, beam_size=20
# GPU 2 (R7 检查: GPU 0=task320 Arm C, GPU 1=task327, GPU 2/3=FREE)
# Time budget: Stage 1 ~3-4h + Stage 2 5min + Stage 3 ~1.5h + Stage 4 5min = ~5h

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task194_gate1
PRODUCTS_DIR=$REPO/products/task194/protocol_match_k064
TASK_ID=task194_gate1_k064
mkdir -p $LOG_DIR $PRODUCTS_DIR

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task194_gate1_k064
mkdir -p $TRITON_CACHE_DIR

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# ---------------------------------------------------------------------------
# Stage 1 — HRQ-VAE training (MIRROR #84 baseline recipe, K0=64)
# ---------------------------------------------------------------------------
STAGE1_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE1_LOG=$LOG_DIR/stage1_${STAGE1_TS}.log

# Verify item_emb.parquet exists (required Stage 1 prep)
ITEM_EMB=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
if [ ! -f "$ITEM_EMB" ]; then
    echo "❌ $ITEM_EMB NOT FOUND" | tee -a "$STAGE1_LOG"
    exit 1
fi

# R12 R7: GPU 2 (FREE, R7 检查 2026-07-30 12:50)
export CUDA_VISIBLE_DEVICES=2

echo "===== [Issue #40 Gate 1 / K=64] Stage 1 launched at $(date) =====" | tee "$STAGE1_LOG"
echo "Recipe: #84 baseline mirror (num_emb_list=[64,128,256], batch_size=1024, epochs=1000, sk_eps=0.0)" | tee -a "$STAGE1_LOG"
echo "GPU 2 (R7 free: task327=GPU 1, task320 Arm C=GPU 0)" | tee -a "$STAGE1_LOG"

cd $REPO/HG-Rec
python3 train_hrqvae.py \
    --data_path "$ITEM_EMB" \
    --ckpt_dir $PRODUCTS_DIR \
    --loss_type poincare \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --sk_epsilons 0.0 0.0 0.0 \
    --layers 512 256 128 64 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --eval_step 5 \
    --num_workers 4 \
    --device cuda:0 \
    > $STAGE1_LOG 2>&1 &
STAGE1_PID=$!
echo "$STAGE1_PID" > $PRODUCTS_DIR/_STAGE1_PID
echo "Stage 1 PID: $STAGE1_PID" | tee -a "$STAGE1_LOG"

wait $STAGE1_PID
STAGE1_EXIT=$?
echo "===== Stage 1 exit: $STAGE1_EXIT at $(date) =====" | tee -a "$STAGE1_LOG"
if [ $STAGE1_EXIT -ne 0 ]; then
    echo "❌ Stage 1 failed, exit=$STAGE1_EXIT" | tee -a "$STAGE1_LOG"
    exit 1
fi

# R12: 落盘 best_collision_model.pth
STAGE1_BEST=$(ls -t $PRODUCTS_DIR/Instruments/*/best_collision_model.pth 2>/dev/null | head -1)
if [ -z "$STAGE1_BEST" ]; then
    echo "❌ Stage 1 best_collision_model.pth NOT FOUND" | tee -a "$STAGE1_LOG"
    exit 1
fi
echo "✅ Stage 1 best ckpt: $STAGE1_BEST" | tee -a "$STAGE1_LOG"

rm -f $PRODUCTS_DIR/_STAGE1_PID

# ---------------------------------------------------------------------------
# Stage 2 — Codebook inference (upstream fork, NO sk_eps_override)
# ---------------------------------------------------------------------------
STAGE2_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE2_LOG=$LOG_DIR/stage2_${STAGE2_TS}.log

OUTPUT_NPY=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_protocol_match_k064.npy

echo "===== [Issue #40 Gate 1 / K=64] Stage 2 launched at $(date) =====" | tee "$STAGE2_LOG"
echo "Driver: task194_stage2 fork (CLI args, but --sk_eps_override 0.0 → behavior = #84 baseline argmin only)" | tee -a "$STAGE2_LOG"
echo "Best ckpt: $STAGE1_BEST" | tee -a "$STAGE2_LOG"
echo "Output: $OUTPUT_NPY" | tee -a "$STAGE2_LOG"

# R11.3 决策: task84_hgrec_stage2_codebook.py upstream fork 是 hardcoded #84 baseline paths.
# 用 task194_stage2 fork (CLI args), 但 --sk_eps_override 0.0 = 行为等同于 upstream argmin only,
# 不是 task194 K0=64 那次的 Sinkhorn 强制.
# scripts/ fork 不属于 src/, 不触发 R11.4 critical decision.
python3 $REPO/scripts/task194_stage2_codebook.py \
    --ckpt_path "$STAGE1_BEST" \
    --output_path "$OUTPUT_NPY" \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    --sk_eps_override 0.0 \
    >> $STAGE2_LOG 2>&1

STAGE2_EXIT=$?
echo "===== Stage 2 exit: $STAGE2_EXIT at $(date) =====" | tee -a "$STAGE2_LOG"
if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage 2 failed" | tee -a "$STAGE2_LOG"
    exit 1
fi

if [ ! -f "$OUTPUT_NPY" ]; then
    echo "❌ Stage 2 output $OUTPUT_NPY NOT FOUND" | tee -a "$STAGE2_LOG"
    exit 1
fi
echo "✅ Stage 2 output: $OUTPUT_NPY ($(stat -c%s $OUTPUT_NPY) bytes)" | tee -a "$STAGE2_LOG"

# ---------------------------------------------------------------------------
# Stage 3 — T5-mini training (MIRROR #84 baseline)
# ---------------------------------------------------------------------------
STAGE3_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE3_LOG=$LOG_DIR/stage3_${STAGE3_TS}.log

# Switch to genrec_env for Stage 3 (uses T5-mini fork)
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# Stage 3 reuses GPU 2 (FREE — task327 still on GPU 1)
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

STAGE3_SAVE=$PRODUCTS_DIR/t5mini_k064

echo "===== [Issue #40 Gate 1 / K=64] Stage 3 launched at $(date) =====" | tee "$STAGE3_LOG"
echo "Recipe: #84 baseline mirror (code_path=_t5_hrqvae_protocol_match_k064.npy)" | tee -a "$STAGE3_LOG"

python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_protocol_match_k064.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path $STAGE3_SAVE \
    --log_path $LOG_DIR/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    >> $STAGE3_LOG 2>&1 &
STAGE3_PID=$!
echo "$STAGE3_PID" > $PRODUCTS_DIR/_STAGE3_PID
echo "Stage 3 PID: $STAGE3_PID" | tee -a "$STAGE3_LOG"

wait $STAGE3_PID
STAGE3_EXIT=$?
echo "===== Stage 3 exit: $STAGE3_EXIT at $(date) =====" | tee -a "$STAGE3_LOG"
if [ $STAGE3_EXIT -ne 0 ]; then
    echo "❌ Stage 3 failed" | tee -a "$STAGE3_LOG"
    exit 1
fi

STAGE3_BEST=$(ls -t $STAGE3_SAVE/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$STAGE3_BEST" ]; then
    echo "❌ Stage 3 best ckpt NOT FOUND" | tee -a "$STAGE3_LOG"
    exit 1
fi
echo "✅ Stage 3 best ckpt: $STAGE3_BEST" | tee -a "$STAGE3_LOG"

rm -f $PRODUCTS_DIR/_STAGE3_PID

# ---------------------------------------------------------------------------
# Stage 4 — Test eval (Test size 24772, beam 20, topk [5,10,20])
# ---------------------------------------------------------------------------
STAGE4_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE4_LOG=$LOG_DIR/stage4_${STAGE4_TS}.log

echo "===== [Issue #40 Gate 1 / K=64] Stage 4 launched at $(date) =====" | tee "$STAGE4_LOG"

python3 - <<PYEOF 2>&1 | tee -a "$STAGE4_LOG"
import sys, os, json, time
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/scripts')
import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '$REPO/scripts/task84_hgrec_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 6, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
    'num_heads': 6, 'd_kv': 64, 'vocab_size': 1025, 'max_len': 20,
    'pad_token_id': 0, 'eos_token_id': 0,
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_hrqvae_protocol_match_k064.npy',
    'dataset_name': 'Instruments',
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'mode': 'evaluation', 'beam_size': 20,
    'seed': 42,
}

data_loader = GenRecDataLoader(config)
test_data = data_loader.load_data()
test_ds = GenRecDataset(config, test_data)

model = HG_Rec(config['codebook_size'], config['num_layers'], config['num_decoder_layers'],
              config['d_model'], config['num_heads'], config['d_ff'], 1024,
              config['vocab_size'], config['max_len'], 0.1)
model = model.to(config['device'])

sd = torch.load('$STAGE3_BEST', map_location='cpu', weights_only=False)
if 'model_state_dict' in sd:
    sd = sd['model_state_dict']
model.load_state_dict(sd, strict=False)
model.eval()

t0 = time.time()
test_dl = GenRecDataLoader(test_ds, batch_size=config['infer_size'], shuffle=False)
metrics = evaluate(model, test_dl, [5, 10, 20], config['beam_size'], config['device'])
elapsed = time.time() - t0

result = {
    'task': 'task194_issue40_gate1_protocol_matched_k064',
    'recipe_mirror': '#84 baseline (4-element confusion-fixed, batch_size 1024, epochs 1000, NO Sinkhorn)',
    'ckpt': '$STAGE3_BEST',
    'beam_size': 20,
    'metrics': metrics,
    'elapsed_sec': round(elapsed, 2),
}
print(json.dumps(result, indent=2))
out_path = '$LOG_DIR/stage4_metrics.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved to {out_path}")
PYEOF

STAGE4_EXIT=$?
echo "===== Stage 4 exit: $STAGE4_EXIT at $(date) =====" | tee -a "$STAGE4_LOG"
echo "✅ Issue #40 Gate 1 / K=64 protocol-matched control DONE" | tee -a "$STAGE4_LOG"
echo "Read result: cat $LOG_DIR/stage4_metrics.json" | tee -a "$STAGE4_LOG"
