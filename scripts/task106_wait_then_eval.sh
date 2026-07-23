#!/bin/bash
# Task #106 / #107 — wait for HG-Rec seed=43/123 Stage 3 to finish + launch Stage 4 test eval
# 修复 (2026-07-23 22:09): PID 文件不可靠 (subshell 立即退出), 改用 ckpt mtime 稳定性判断
# 训练停止 = HG_Rec_best.pth 在过去 5 分钟内 mtime 未变化
# Stage 4 eval 启动前 sleep 60s 给 ckpt flush

set -e

TASK_ID=${1:-106}
CKPT_PATTERN="/home/wlia0047/ar57/wenyu/GeneRec/products/task${TASK_ID}/ckpt_hgrec/Instruments/*/HG_Rec_best.pth"
LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
STABLE_THRESHOLD_SEC=300  # 5 分钟 (300 sec) 内 mtime 未变化即视为稳定

echo "[$(date +%H:%M:%S)] Task #${TASK_ID} Daemon started. Polling ckpt: $CKPT_PATTERN"
echo "[$(date +%H:%M:%S)] Stable threshold: ${STABLE_THRESHOLD_SEC}s (5 min)"

while true; do
    # Find latest ckpt
    BEST_CKPT=$(ls -t $CKPT_PATTERN 2>/dev/null | head -1)

    if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
        # No ckpt yet — keep waiting
        sleep 60
        continue
    fi

    # Check ckpt mtime
    MTIME=$(stat -c %Y "$BEST_CKPT" 2>/dev/null)
    NOW=$(date +%s)
    AGE=$((NOW - MTIME))

    # Check if training process (python3 ... task84_hgrec_stage3_train.py ... --save_path ...task${TASK_ID}) still alive
    TRAIN_PID=$(pgrep -f "task84_hgrec_stage3_train.py.*task${TASK_ID}" | head -1)

    echo "[$(date +%H:%M:%S)] Task #${TASK_ID} ckpt age: ${AGE}s, training PID: ${TRAIN_PID:-none}"

    # Trigger conditions: (a) no training process AND ckpt older than threshold, OR (b) ckpt older than 2× threshold (failsafe)
    if [ -z "$TRAIN_PID" ] && [ "$AGE" -gt "$STABLE_THRESHOLD_SEC" ]; then
        echo "[$(date +%H:%M:%S)] Task #${TASK_ID} ✓ training stopped + ckpt stable ${AGE}s > ${STABLE_THRESHOLD_SEC}s"
        echo "[$(date +%H:%M:%S)] Task #${TASK_ID} Sleeping 60s for ckpt flush..."
        sleep 60
        break
    fi

    sleep 60
done

# Verify best ckpt still exists
BEST_CKPT=$(ls -t $CKPT_PATTERN 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Task #${TASK_ID} HG_Rec_best.pth NOT FOUND, exiting"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Task #${TASK_ID} Found best ckpt: $BEST_CKPT"
echo "[$(date +%H:%M:%S)] Task #${TASK_ID} Launching Stage 4 test eval..."

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task${TASK_ID}_hgrec_stage4_eval_${TS}.log

python3 -c "
import sys, os, glob, json, torch
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
import importlib.util as _ilu
_s_spec = _ilu.spec_from_file_location('s3_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s_mod = _ilu.module_from_spec(_s_spec)
_s_spec.loader.exec_module(_s_mod)
evaluate = _s_mod.evaluate
del _ilu, _s_spec, _s_mod

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 6, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
    'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
    'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
    'feed_forward_proj': 'relu', 'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_hrqvae_poincare.npy',
    'topk_list': [5, 10, 20], 'beam_size': 20,
}

BEST_CKPT = glob.glob('/home/wlia0047/ar57/wenyu/GeneRec/products/task${TASK_ID}/ckpt_hgrec/Instruments/*/HG_Rec_best.pth')[0]
print(f'[Stage 4 Task #${TASK_ID}] Loading best ckpt: {BEST_CKPT}')
device = torch.device('cuda:0')
model = HG_Rec(config)
model.load_state_dict(torch.load(BEST_CKPT, map_location='cpu'))
model.to(device)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation', codebook_size=config['codebook_size'], max_len=config['max_len'],
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Stage 4 Task #${TASK_ID}] Test dataset size: {len(test_dataset)}')

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
result = {'task_id': ${TASK_ID}, 'best_ckpt': BEST_CKPT, 'test_recalls': avg_recalls, 'test_ndcgs': avg_ndcgs}
result_json = f'/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task${TASK_ID}_hgrec_seed_test_metrics.json'
os.makedirs(os.path.dirname(result_json), exist_ok=True)
with open(result_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4 Task #${TASK_ID}] Test metrics saved: {result_json}')
print(f'[Stage 4 Task #${TASK_ID}] Test metrics:\\n{avg_recalls}\\n{avg_ndcgs}')
" 2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] Task #${TASK_ID} Stage 4 eval completed, exit code: $EXIT_CODE"