#!/bin/bash
# Task #243 Stage 4 — eval both epoch=200 + epoch=400 ckpts
# R7 fix: bash vars (EPOCH/BEST_CKPT/RESULT_JSON) via env vars to heredoc
# R-fix2: HG_Rec.generate 硬编码 num_return_sequences=num_beams, 不能重复传
# R-fix3: calculate_pos_index 内联 (在 train_HG-Rec.py, 不能 from model.HG_Rec 导入)
# R-fix4: GenRecDataLoader 直接是 iterator, 没有 get_loader()
# R-fix5: batch keys = history/target/attention_mask (不是 input_ids/labels)
# R-fix6: preds shape 是 (B, beam, 5) (max_length=5), 截断到 (B, beam, 4) 对齐 labels
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

    export TASK243_CKPT="$BEST_CKPT"
    export TASK243_RESULT="$RESULT_JSON"
    export TASK243_EPOCH="$EPOCH"

    CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task243_s4_e${EPOCH} \
    python3 -u <<'PYTHON_EOF' 2>&1 | tee -a "$LOG_FILE"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import numpy as np
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    matches = (preds == labels.unsqueeze(1)).all(dim=2)
    return matches

ckpt_path = os.environ['TASK243_CKPT']
result_path = os.environ['TASK243_RESULT']
epoch_label = os.environ['TASK243_EPOCH']

config = {
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'code_path': '_t5_rqvae_code_default.npy',
    'codebook_size': [64, 128, 256, 1],
    'num_epochs': int(epoch_label),
    'batch_size': 256,
    'lr': 1e-4,
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dropout_rate': 0.0,
    'device': 'cuda:0',
    'mode': 'evaluation',
    'log_path': '/home/wlia0047/ar57/wenyu/GeneRec/logs/task243/',
    'seed': 42,
    'early_stop': 20,
    'beam_size': 20,
    'infer_size': 96,
    'save_path': '/tmp/task243_eval_e' + epoch_label,
}

print(f"[eval] loading model from {ckpt_path}")
model = HG_Rec(config).to('cuda:0')
sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()
print("[eval] model loaded OK")

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len'],
)
test_loader = GenRecDataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"[eval] test_loader OK, {len(test_dataset)} samples")

preds_all = []
labels_all = []
batch_count = 0
for batch in test_loader:
    batch_count += 1
    input_ids = batch['history'].to('cuda:0')
    attention_mask = batch['attention_mask'].to('cuda:0')
    labels = batch['target'].to('cuda:0')
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4, num_beams=20)
        outputs = outputs.view(input_ids.size(0), 20, -1)[:, :, :4]
        preds_all.append(outputs.cpu())
        labels_all.append(labels.cpu())
    if batch_count % 20 == 0:
        print(f"[eval] batch {batch_count}/~{len(test_dataset)//128+1}")

preds_all = torch.cat(preds_all, dim=0)
labels_all = torch.cat(labels_all, dim=0)
print(f"[eval] all preds: {preds_all.shape}, labels: {labels_all.shape}")

results = {}
for k in [5, 10, 20]:
    pos_idx = calculate_pos_index(preds_all[:, :k], labels_all, maxk=k)
    recall = pos_idx.any(dim=1).float().mean().item()
    results[f'Recall@{k}'] = recall

def ndcg_at_k(preds, labels, k):
    pos_idx = calculate_pos_index(preds, labels, maxk=k)
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

print(f"epoch={epoch_label} results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved to {result_path}")
PYTHON_EOF
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #243 Stage 4 eval done ==="
