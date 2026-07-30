#!/bin/bash
# Task #194 / Issue #40 Gate 1 — Stage 2/3/4 continuation
# Stage 1 already done (12:56-13:02, 6 min). best_collision_model.pth at known path.
# Continuation: Stage 2 (5min) + Stage 3 (~1.5h) + Stage 4 (~5min) = ~2h
# Recipe: #84 baseline mirror, K0=64, sk_eps=0 (NO Sinkhorn, argmin only)
# GPU: Prefer GPU 1 (task327 will finish ~16:35 AEST). Stage 2 can share GPU.

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task194_gate1
PRODUCTS_DIR=$REPO/products/task194/protocol_match_k064

# Stage 1 ckpt (confirmed exists after Stage 1 silent "failure" was actually just launcher glob path bug)
STAGE1_BEST=$(ls -t $PRODUCTS_DIR/*/best_collision_model.pth 2>/dev/null | head -1)
if [ -z "$STAGE1_BEST" ]; then
    echo "❌ Stage 1 best_collision_model.pth NOT FOUND in $PRODUCTS_DIR/*/" | tee -a $LOG_DIR/main.log
    exit 1
fi
echo "✅ Stage 1 best ckpt: $STAGE1_BEST" | tee -a $LOG_DIR/main.log

mkdir -p $LOG_DIR

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task194_gate1_k064
mkdir -p $TRITON_CACHE_DIR

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# R11.5: Use GPU 1 (task327 Stage 3 Ep121/200 ~16:35 AEST finishes, freeing GPU 1).
# Or fall back to GPU 0 (task328 α=0.5 only at Ep7/200, ~5h remaining, ~13GB used / 46GB free).
# For Stage 2 (1 min, light) → use GPU 1 even with task327 (low impact, 1 min)
export CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------------------
# Stage 2 — Codebook inference (NO Sinkhorn, argmin only)
# ---------------------------------------------------------------------------
STAGE2_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE2_LOG=$LOG_DIR/stage2_${STAGE2_TS}.log
OUTPUT_NPY=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_protocol_match_k064.npy

echo "===== [Issue #40 Gate 1 / K=64 continuation] Stage 2 launched at $(date) =====" | tee "$STAGE2_LOG"
echo "Driver: task194_stage2 fork with --sk_eps_override 0.0 (= #84 baseline argmin only)" | tee -a "$STAGE2_LOG"
echo "Best ckpt: $STAGE1_BEST" | tee -a "$STAGE2_LOG"
echo "Output: $OUTPUT_NPY" | tee -a "$STAGE2_LOG"

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
STAGE3_SAVE=$PRODUCTS_DIR/t5mini_k064

echo "===== [Issue #40 Gate 1 / K=64 continuation] Stage 3 launched at $(date) =====" | tee "$STAGE3_LOG"
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

echo "===== [Issue #40 Gate 1 / K=64 continuation] Stage 4 launched at $(date) =====" | tee "$STAGE4_LOG"

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
    json.dump(result, indent=2)
print(f"Saved to {out_path}")
PYEOF

STAGE4_EXIT=$?
echo "===== Stage 4 exit: $STAGE4_EXIT at $(date) =====" | tee -a "$STAGE4_LOG"
echo "✅ Issue #40 Gate 1 / K=64 protocol-matched control DONE" | tee -a "$STAGE4_LOG"
echo "Read result: cat $LOG_DIR/stage4_metrics.json" | tee -a "$STAGE4_LOG"