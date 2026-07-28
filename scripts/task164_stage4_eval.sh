#!/bin/bash
# Task #164 Phase B κ-decouple → Stage 4: Test eval on T5-small 60M best ckpt
# Fork task157_hgrec_stage4_eval.sh, code_path=phase_b_kappa_decouple

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task164_p1_s4

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task164
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage4_eval_${TS}.log
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task164_phase_b_kappa_decouple_metrics.json

mkdir -p $LOG_DIR

echo "===== [Task #164 Stage 4] Phase B + T5-small 60M test eval launched at $(date) =====" | tee $LOG_FILE
echo "GPU 0" | tee -a $LOG_FILE
echo "Output JSON: $RESULT_JSON" | tee -a $LOG_FILE

BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task164/phase_b_kappa_decouple/*/stage3_60m/*/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi
echo "Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
import sys, os, glob, torch, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

# T5-small 60M config (mirror Task #160 / #164 Stage 3)
config = {
    'batch_size': 256,
    'infer_size': 96,
    'lr': 1e-4,
    'device': 'cuda:0',
    'num_layers': 6,
    'num_decoder_layers': 6,
    'd_model': 512,
    'd_ff': 2048,
    'num_heads': 8,
    'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [32, 64, 256, 1],
    'code_path': '_t5_rqvae_phase_b_kappa_decouple.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Stage 4] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load('$BEST_CKPT', map_location='cpu'))
model.to(device)
print(f'[Stage 4] Model loaded on {device}', flush=True)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task164_phase_b_kappa_decouple',
    'recipe': 'κ-Stereographic + Phase A 100ep frozen + Phase B 100ep lr_theta=1e-5 + T5-small 60M',
    'final_kappa_m': [-0.0908, -0.0938, -0.0963],
    't5_config': '6 enc + 6 dec, d_model=512, d_ff=2048, 8 heads d_kv=64 (~44.6M params)',
    'best_ckpt': '$BEST_CKPT',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4] Saved: $RESULT_JSON', flush=True)
print(f'[Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #164 Stage 4] Test eval completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Stage 4 FAILED" | tee -a $LOG_FILE
    exit 1
fi

echo "===== Final test recalls =====" | tee -a $LOG_FILE
python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
print(f'R@5:   {d[\"test_recalls\"][0]:.4f}')
print(f'R@10:  {d[\"test_recalls\"][1]:.4f}')
print(f'R@20:  {d[\"test_recalls\"][2]:.4f}')
print(f'NDCG@5:  {d[\"test_ndcgs\"][0]:.4f}')
print(f'NDCG@10: {d[\"test_ndcgs\"][1]:.4f}')
print(f'NDCG@20: {d[\"test_ndcgs\"][2]:.4f}')
print('---')
print('Baseline 对比:')
print('  Task #84 HG-Rec baseline c=[1,1,1] R@10=0.1020')
print('  Task #88 HG-Rec c=[0.5,0.5,0.5] R@10=0.1051')
print('  Phonism (vanilla+Sinkhorn) R@10=0.1058')
" | tee -a $LOG_FILE
