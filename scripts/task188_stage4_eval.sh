#!/bin/bash
# Task #188 Phase 4 — Stage 4 eval × 12 runs (4 codebooks × 3 seeds)
# 12 evaluations, each ~30s. 4 GPU parallel, 3 rounds total (4 evals per round).
# Output: verdicts/task188_paper_table7_metrics.json per-run, then aggregate.

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task188_stage4
mkdir -p $REPO/logs/task188 $REPO/verdicts

# Codebook path map
declare -A CB_SUFFIX
CB_SUFFIX[t1_8pct]="_t5_rqvae_t1_8pct.npy"
CB_SUFFIX[t2_10pct]="_t5_rqvae_t2_10pct.npy"
CB_SUFFIX[t3_12pct]="_t5_rqvae_t3_12pct.npy"
CB_SUFFIX[t4_13pct]="_t5_rqvae_t4_13pct.npy"

# Tier collision rate metadata (from Phase 2 codebook generation)
declare -A COLLISION
COLLISION[t1_8pct]="0.0862"   # 8.62%
COLLISION[t2_10pct]="0.1009"  # 10.09%
COLLISION[t3_12pct]="0.1117"  # 11.17%
COLLISION[t4_13pct]="0.1276"  # 12.76%

SEEDS=(42 123 2024)
TIERS=(t1_8pct t2_10pct t3_12pct t4_13pct)

run_stage4_eval() {
    local tier=$1
    local gpu=$2
    local seed=$3
    local ckpt_dir=$REPO/products/task188/t5small_${tier}/seed${seed}/Instruments
    local best_ckpt=$(ls -t $ckpt_dir/*/HG_Rec_best.pth 2>/dev/null | head -1)
    local result_json=$REPO/verdicts/task188_${tier}_seed${seed}_metrics.json
    local log_dir=$REPO/logs/task188/stage4_${tier}_seed${seed}
    mkdir -p $log_dir

    if [ -z "$best_ckpt" ] || [ ! -f "$best_ckpt" ]; then
        echo "❌ Best ckpt MISSING: ${tier}/seed${seed} (looked in $ckpt_dir)"
        return 1
    fi

    local code_path=${CB_SUFFIX[$tier]}
    local collision=${COLLISION[$tier]}

    echo "[$(date)] stage4 tier=$tier gpu=$gpu seed=$seed ckpt=$best_ckpt"

    cd $REPO/HG-Rec

    python3 -u <<PYTHON_EOF 2>&1 | tee "$log_dir/stage4_eval.log"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import torch
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

config = {
    'batch_size': 256,
    'infer_size': 96,
    'lr': 1e-4,
    'device': 'cuda:0',
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '$code_path',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:$gpu')
model = HG_Rec(config)
print(f'[Task #188 Stage 4] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load('$best_ckpt', map_location='cpu'))
model.to(device)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #188 Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task188_paper_table7_multiseed',
    'tier': '$tier',
    'seed': $seed,
    'collision_rate': $collision,
    'recipe': 'T5-small 5.5M + paper-aligned Poincaré distance',
    'best_ckpt': '$best_ckpt',
    'selected_by': 'best_val_NDCG@20',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$result_json'), exist_ok=True)
with open('$result_json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #188 Stage 4] Saved: $result_json', flush=True)
print(f'[Task #188 Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #188 Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF
}

# 3 rounds × 4 tiers, 4 GPU parallel
for round in 0 1 2; do
    seed=${SEEDS[$round]}
    echo "===== Round $((round+1)): seed=$seed =====" | tee -a $REPO/logs/task188/phase4_launcher.out
    declare -A PIDS_R
    for i in 0 1 2 3; do
        tier=${TIERS[$i]}
        gpu=$i
        (
            run_stage4_eval $tier $gpu $seed
            echo "[$(date)] DONE tier=$tier gpu=$gpu seed=$seed" | tee -a $REPO/logs/task188/phase4_launcher.out
        ) &
        PIDS_R[$tier]=$!
        disown
    done
    echo "Round $((round+1)) PIDs: $(for k in "${!PIDS_R[@]}"; do echo -n "$k=${PIDS_R[$k]} "; done)"
    wait "${PIDS_R[@]}"
done

# Aggregate variance matrix after all 12 done
python3 -u <<PYTHON_EOF
import json, os
import statistics as stats

tiers = ['t1_8pct', 't2_10pct', 't3_12pct', 't4_13pct']
seeds = [42, 123, 2024]

# Load all 12 per-run results
results = {}
for t in tiers:
    results[t] = {}
    for s in seeds:
        fp = f'/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task188_{t}_seed{s}_metrics.json'
        if not os.path.exists(fp):
            print(f"⚠️ Missing: {fp}")
            continue
        with open(fp) as f:
            results[t][s] = json.load(f)

# Aggregate mean ± std per tier
agg = {}
print(f"{'tier':<12} {'metric':<8} {'seed42':<10} {'seed123':<10} {'seed2024':<10} {'mean':<10} {'std':<10}")
for t in tiers:
    agg[t] = {}
    for metric_set, suffix in [('test_recalls', 'Recall'), ('test_ndcgs', 'NDCG')]:
        agg[t][suffix] = {}
        for k in ['5', '10', '20']:
            key = f'{suffix}@{k}'
            values = []
            for s in seeds:
                r = results[t].get(s, None)
                if r is None: continue
                v = r[metric_set].get(key, None)
                if v is not None:
                    values.append((s, v))
            if not values:
                continue
            vals = [v for _, v in values]
            mean = stats.mean(vals)
            std = stats.stdev(vals) if len(vals) > 1 else 0.0
            per_seed_str = ' '.join(f"s{s}={v:.4f}" for s, v in values)
            print(f"{t:<12} {key:<8} {per_seed_str:<30} {mean:<10.4f} {std:<10.4f}")
            agg[t][suffix][k] = {'per_seed': dict(values), 'mean': mean, 'std': std, 'count': len(vals)}

# Save aggregated file
out_path = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task188_paper_table7_multiseed_metrics.json'
with open(out_path, 'w') as f:
    json.dump({
        'task': 'task188_paper_table7_multiseed_aggregate',
        'tier_metrics': agg,
        'seeds': seeds,
        'tiers': tiers,
        'hgrec_baseline_R@10': 0.1020,
    }, f, indent=2)
print(f"\nSaved aggregate: {out_path}")

# Hypotheses R1 (R@10 monotonic ↓ with collision) and R2 (std < 5% of mean)
print("\n=== Hypothesis R1: R@10 monotonic ↓ with collision ===")
r10_means = [(t, agg[t]['Recall']['10']['mean']) for t in tiers if t in agg and 'Recall' in agg[t]]
print("Tiers ordered by collision: t1=8.62% < t2=10.09% < t3=11.17% < t4=12.76%")
for t, m in r10_means:
    print(f"  {t}: R@10 = {m:.4f}")
is_monotonic = all(r10_means[i][1] >= r10_means[i+1][1] for i in range(len(r10_means)-1))
print(f"Monotonic decreasing: {is_monotonic}")

print("\n=== Hypothesis R2: std < 5% of mean (across seeds) ===")
for t, _ in r10_means:
    m = agg[t]['Recall']['10']['mean']
    s = agg[t]['Recall']['10']['std']
    if m > 0:
        cv = s / m
        verdict = "PASS" if cv < 0.05 else "FAIL"
        print(f"  {t}: R@10 mean={m:.4f} std={s:.4f} CV={cv:.2%} -> {verdict}")
PYTHON_EOF

echo "===== ALL 12 Stage 4 evals + variance matrix COMPLETE at $(date) =====" | tee -a $REPO/logs/task188/phase4_launcher.out
