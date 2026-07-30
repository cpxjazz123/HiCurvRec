#!/usr/bin/env bash
# Task #327 / K=256 + Issue #30 — Stage 4 beam_size ablation eval
# 目的: 验证 K=256 anchor + Issue #30 r_l+s_l synergy 是否突破 R@10=0.1053 (task194 anchor)
# R11.5 决策: 复用 task307 beam ablation 模板, K=50 amplifier 验证
# 关联: task194 K=256 anchor + task301 Issue #30 marginal + task307 K=50 amplifier

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

# Find ckpt (either real ckpt after Stage 3 completes, or use task301 #30 ckpt as proxy)
CKPT_PATH=$(ls -td $REPO/products/task327/t5mini_k256_issue30/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: task327 Stage 3 ckpt not found at products/task327/t5mini_k256_issue30/"
    echo "Wait for Stage 3 to complete (PID 1531589 on GPU 1)"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"

LOG_DIR=$REPO/logs/task327
mkdir -p $LOG_DIR

run_one_beam() {
    local BEAM=$1
    local TS=$(date +%Y%m%d_%H%M%S)
    local LOG="$LOG_DIR/stage4_beam${BEAM}_${TS}.log"
    echo "===== [BEAM=$BEAM] launching at $TS =====" | tee "$LOG"

    CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" - <<PYEOF 2>&1 | tee -a "$LOG"
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
    'codebook_size': [256, 128, 256, 1],
    'code_path': '_t5_rqvae_k256_issue30.npy',
    'dataset_name': 'Instruments',
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'mode': 'evaluation', 'beam_size': $BEAM,
    'seed': 42,
}

# R11.3 决策: 直接用 task84 fork 的 evaluate 函数, 不写新 standalone
# 加载 dataset + model
data_loader = GenRecDataLoader(config)
test_data = data_loader.load_data()
test_ds = GenRecDataset(config, test_data)

model = HG_Rec(config['codebook_size'], config['num_layers'], config['num_decoder_layers'],
              config['d_model'], config['num_heads'], config['d_ff'], 1024,
              config['vocab_size'], config['max_len'], 0.1)
model = model.to(config['device'])

# Load ckpt
sd = torch.load('$CKPT_PATH', map_location='cpu', weights_only=False)
if 'model_state_dict' in sd:
    sd = sd['model_state_dict']
model.load_state_dict(sd, strict=False)
model.eval()

# Evaluate on test set
t0 = time.time()
test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
metrics = evaluate(model, test_dl, [5, 10, 20], config['beam_size'], config['device'])
elapsed = time.time() - t0

result = {
    'task': 'task327_k256_issue30_synergy',
    'ckpt': '$CKPT_PATH',
    'beam_size': $BEAM,
    'metrics': metrics,
    'elapsed_sec': round(elapsed, 2),
}
print(json.dumps(result, indent=2))

# 保存 JSON
out_path = '$LOG_DIR/stage4_beam${BEAM}_metrics.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved to {out_path}")
PYEOF
}

# 跑 K=20 (Issue #30 default) + K=50 (Issue #30 K=50 amplifier) + K=100
run_one_beam 20
run_one_beam 50
run_one_beam 100

echo "===== Task #327 Stage 4 beam ablation done ====="