#!/bin/bash
# Task #243 Stage 4 — eval both epoch=200 + epoch=400 ckpts
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task243 $REPO/verdicts

for EPOCH in 200 400; do
    CKPTS=$(find $REPO/products/task243/t5mini_epoch${EPOCH}/Instruments -name "HG_Rec_best.pth" 2>/dev/null | sort)
    if [ -z "$CKPTS" ]; then
        echo "❌ No HG_Rec_best.pth in products/task243/t5mini_epoch${EPOCH}/Instruments/"
        exit 1
    fi
    BEST_CKPT=$(echo "$CKPTS" | tail -1)
    LOG_FILE=$REPO/logs/task243/stage4_epoch${EPOCH}_eval.out
    RESULT_JSON=$REPO/verdicts/task243_epoch${EPOCH}_test_metrics.json

    echo "===== [Task #243 Stage 4] epoch=$EPOCH eval launched at $(date) =====" | tee $LOG_FILE
    echo "ckpt: $BEST_CKPT" | tee -a $LOG_FILE

    CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task243_s4_e${EPOCH} \
    python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

from transformers import T5ForConditionalGeneration
import importlib.util as _ilu

_s3_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s3 = _ilu.module_from_spec(_s3_spec)
_s3_spec.loader.exec_module(_s3)

ckpt_path = '$BEST_CKPT'
config = _s3.get_config()
config['mode'] = 'evaluation'

# Override config to load ckpt
class Args:
    pass
args = Args()
args.dataset_name = config['dataset_name']
args.dataset_path = config['dataset_path']
args.code_path = '_t5_rqvae_code_default.npy'
args.codebook_size = config['codebook_size']
args.num_epochs = config['num_epochs']
args.batch_size = config['batch_size']
args.lr = config['lr']
args.num_layers = config['num_layers']
args.num_decoder_layers = config['num_decoder_layers']
args.d_model = config['d_model']
args.d_ff = config['d_ff']
args.num_heads = config['num_heads']
args.d_kv = config['d_kv']
args.vocab_size = config['vocab_size']
args.pad_token_id = config['pad_token_id']
args.eos_token_id = config['eos_token_id']
args.feed_forward_proj = config['feed_forward_proj']
args.max_len = config['max_len']
args.dropout_rate = 0.0
args.device = 'cuda:0'
args.mode = 'evaluation'
args.log_path = '/home/wlia0047/ar57/wenyu/GeneRec/logs/task243/'
args.seed = 42
args.early_stop = 20
args.beam_size = 20
args.infer_size = 96
args.feed_forward_proj = 'relu'
args.save_path = '/tmp/task243_eval_${EPOCH}'

# Load model from ckpt
model = HG_Rec(args).to('cuda:0')
sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()

# Test eval
test_dataset = GenRecDataset(
    dataset_path=os.path.join(args.dataset_path, args.dataset_name, 'test.parquet'),
    code_path=os.path.join(args.dataset_path, args.dataset_name, args.dataset_name + args.code_path),
    mode='evaluation',
    codebook_size=args.codebook_size,
    max_len=args.max_len,
)
test_loader = GenRecDataLoader(test_dataset, batch_size=128, shuffle=False).get_loader()

from model.HG_Rec import calculate_pos_index
results = {}
ks = [5, 10, 20]
preds_all = []
labels_all = []
for batch in test_loader:
    input_ids = batch['input_ids'].to('cuda:0')
    attention_mask = batch['attention_mask'].to('cuda:0')
    labels = batch['labels'].to('cuda:0')
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4, num_beams=20, num_return_sequences=20)
        # Reshape: (B, beam, seq_len)
        outputs = outputs.view(input_ids.size(0), 20, -1)
        preds_all.append(outputs.cpu())
        labels_all.append(labels.cpu())

preds_all = torch.cat(preds_all, dim=0)
labels_all = torch.cat(labels_all, dim=0)

for k in ks:
    pos_idx = calculate_pos_index(preds_all[:, :k], labels_all, maxk=k)
    recall = pos_idx.any(dim=1).float().mean().item()
    results[f'Recall@{k}'] = recall

# NDCG
import numpy as np
def ndcg_at_k(preds, labels, k):
    pos_idx = calculate_pos_index(preds, labels, maxk=k)
    # Find first hit position
    n = preds.shape[0]
    ndcg_list = []
    for i in range(n):
        hits = pos_idx[i]
        if not hits.any():
            ndcg_list.append(0.0)
            continue
        first_hit = hits.nonzero()[0].item()
        ndcg_list.append(1.0 / np.log2(first_hit + 2))
    return float(np.mean(ndcg_list))

for k in [5, 10, 20]:
    ndcg = ndcg_at_k(preds_all[:, :k], labels_all, k)
    results[f'NDCG@{k}'] = ndcg

print(f"epoch=$EPOCH results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

with open('$RESULT_JSON', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved to $RESULT_JSON")
PYTHON_EOF
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #243 Stage 4 eval done ==="
