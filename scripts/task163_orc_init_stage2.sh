#!/bin/bash
# Task #163 Stage 2 — ORC-based θ_init free-curv SID inference (CPU)
# Fork of task162 paper-faithful stage 2 logic, different ckpt + output paths
# Stage 1 ckpt: products/task163/rqvae_orc_init/<ts>/best_loss_model.pth
# 输出: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_orc_init.npy (N, 4)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task163
STAGE1_DIR=$(ls -td /home/wlia0047/ar57/wenyu/GeneRec/products/task163/rqvae_orc_init/*/ | head -1)
STAGE1_CKPT="${STAGE1_DIR}best_loss_model.pth"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ Stage 1 ckpt missing: $STAGE1_CKPT" | tee -a $LOG_DIR/stage2_error.log
    exit 1
fi

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/orc_stage2_inference_${TS}.log

OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments
mkdir -p $LOG_DIR $OUT_DIR

OUT_NPY=$OUT_DIR/Instruments_t5_rqvae_orc_init.npy

echo "===== [Task #163 Stage 2] ORC-init free-curv SID inference launched at $(date) =====" | tee $LOG_FILE
echo "Stage 1 ckpt: $STAGE1_CKPT" | tee -a $LOG_FILE
echo "Output: $OUT_NPY" | tee -a $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
import sys, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.utils import EmbDataset

ckpt_path = "$STAGE1_CKPT"
output_path = "$OUT_NPY"
device = "cpu"

ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
args = ckpt['args']
state_dict = ckpt['state_dict']
print(f"[Stage 2] ckpt: best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}", flush=True)
print(f"[Stage 2] args: M={args.get('M', '?')}, kappa_max={args.get('kappa_max', '?')}, "
      f"num_emb_list={args.get('num_emb_list', '?')}, beta={args.get('beta', '?')}", flush=True)

data = EmbDataset(args['data_path'])
model = FreeCurvHRQVAE(
    in_dim=data.dim, num_emb_list=args['num_emb_list'], e_dim=args['e_dim'],
    M=args['M'], kappa_max=args['kappa_max'], layers=args['layers'],
    dropout_prob=0.0, bn=False, loss_type=args['loss_type'],
    quant_loss_weight=args['quant_loss_weight'], beta=args['beta'],
    kmeans_init=args['kmeans_init'], kmeans_iters=args['kmeans_iters'],
    sk_eps=args['sk_epsilons'], sk_iters=args['sk_iters'],
)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

for li, vq in enumerate(model.hrq.vq_layers):
    kappas = vq.kappa_m().detach().cpu().tolist()
    print(f"[Stage 2] Layer {li} κ_m = [{', '.join(f'{k:+.4f}' for k in kappas)}]", flush=True)

loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=False, pin_memory=False)
prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

all_indices, all_indices_str = [], []
for d in loader:
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_indices.append(code)
        all_indices_str.append(str(code))

def check_collision(s):
    return len(s) == len(set(s.tolist()))
def get_collision_item(s):
    d = {}
    for i, x in enumerate(s):
        d.setdefault(x, []).append(i)
    return [v for v in d.values() if len(v) > 1]

tt = 0
while True:
    if tt >= 30 or check_collision(np.array(all_indices_str)):
        break
    cg = get_collision_item(np.array(all_indices_str))
    print(f"  collision iter {tt}: {len(cg)} groups", flush=True)
    for items in cg:
        d = data[items].to(device)
        idx = model.get_indices(d, use_sk=True).view(-1, model.get_indices(d, use_sk=True).shape[-1]).cpu().numpy()
        for it, ix in zip(items, idx):
            c = [prefix[i].format(int(j)) for i, j in enumerate(ix)]
            all_indices[it] = c
            all_indices_str[it] = str(c)
    tt += 1

codes = []
for v in all_indices:
    codes.append([int(x.split('_')[1].strip('>')) for x in v])
codes_array = np.array(codes)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

uniq, counts = np.unique(codes_array, axis=0, return_counts=True)
dup = uniq[counts > 1]
if len(dup) > 0:
    for dup_row in dup:
        ix = np.where((codes_array == dup_row).all(axis=1))[0]
        for i, j in enumerate(ix):
            codes_array[j, -1] = i

final_uniq, final_counts = np.unique(codes_array, axis=0, return_counts=True)
new_dup = final_uniq[final_counts > 1]
if len(new_dup) > 0:
    raise ValueError(f"Failed to resolve SID duplicates: {new_dup}")

print(f"[Stage 2] {len(codes_array)} items, {len(np.unique(codes_array, axis=0))} unique SIDs, "
      f"shape={codes_array.shape}", flush=True)
np.save(output_path, codes_array)
print(f"[Stage 2] Saved: {output_path}", flush=True)

meta = {
    'task': 'Task #163 ORC-init free-curv Stage 2',
    'ckpt_path': ckpt_path,
    'output_path': output_path,
    'M': args['M'],
    'kappa_max': args['kappa_max'],
    'final_kappa_per_layer': [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers],
    'n_items': int(len(codes_array)),
    'n_unique': int(len(np.unique(codes_array, axis=0))),
    'shape': list(codes_array.shape),
}
with open(output_path.replace('.npy', '_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
PYTHON_EOF

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #163 Stage 2] ORC-init free-curv SID inference completed at $(date), exit=$EXIT_CODE =====" | tee -a $LOG_FILE
