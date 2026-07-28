#!/bin/bash
# Task #233 Stage 4 — c=10 ep1 SID → T5-mini 9.18M test eval
# 目的: 测 c=10 ep1 SID (51.41% collision) → T5 R@10. 预期 < 0.07 (vs baseline 0.1020).
# baseline 对照: HG-Rec Task #84 R@10=0.1020 (T5-mini + 64/128/256/dedup codebook).
# Stage 3 ckpt: products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/stage3_t5mini/Instruments/Jul-27-2026_17-20-49/HG_Rec_best.pth (22 MB, R12 forced)
# SID: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task233_c10_ep1.npy (9922, 4)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
# GPU 0 优先 (Task #233 Stage 3 用 GPU 0, TRITON cache 已建)
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/wenyu/scratch/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task233_s4
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task233
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/c10_ep1_stage4_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task233_c10_ep1_metrics.json

echo "===== [Task #233 Stage 4] c=10 ep1 + T5-mini 9.18M test eval launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: c=10 ep1 RQ-VAE SID (51.41% collision) + T5-mini 9.18M (4+4 layers, d_model=256, d_ff=1024)" | tee -a $LOG_FILE
echo "GPU 0" | tee -a $LOG_FILE
echo "Output JSON: $RESULT_JSON" | tee -a $LOG_FILE

export RESULT_JSON

BEST_CKPT="/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/stage3_t5mini/Instruments/Jul-27-2026_17-20-49/HG_Rec_best.pth"
if [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING: $BEST_CKPT" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE
export BEST_CKPT

python3 -u <<'PYTHON_EOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

# Task #233 Stage 3 训练时用的是 [64, 128, 256, 1] codebook (跟 baseline 一致),
# 但 c=10 ep1 SID inference output 也是 (9922, 4). 所以 eval codebook_size = [64, 128, 256, 1].
config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 4, 'num_decoder_layers': 4,
    'd_model': 256, 'd_ff': 1024, 'num_heads': 4, 'd_kv': 64,
    'dropout_rate': 0.1, 'vocab_size': 1025,
    'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_rqvae_task233_c10_ep1.npy',
    'topk_list': [5, 10, 20], 'beam_size': 20,
}

import torch
device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Stage 4] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load(os.environ['BEST_CKPT'], map_location='cpu'))
model.to(device)
print(f'[Stage 4] Model loaded on {device}', flush=True)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation', codebook_size=config['codebook_size'], max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task233_c10_ep1',
    'recipe': 'c=10 ep1 RQ-VAE SID (51.41% collision) + T5-mini 9.18M (4+4 layers, d_model=256, d_ff=1024)',
    'direction': 'M-arm 方向 I c=10 (wdiv=100, beta=0.5, 50 epoch train, ep1 ckpt)',
    'best_ckpt': os.environ['BEST_CKPT'],
    'sid_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task233_c10_ep1.npy',
    'sid_collision_rate_pre_resolve': 0.6141,
    'sid_unique_count': 9922,
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname(os.environ['RESULT_JSON']), exist_ok=True)
with open(os.environ['RESULT_JSON'], 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4] Saved: {os.environ["RESULT_JSON"]}', flush=True)
print(f'[Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #233 Stage 4] Test eval completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE