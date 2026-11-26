"""Read-only mixed-vs-Euclidean selection probe on the frozen #263 checkpoint.

No optimizer, no GNN forward/backward, no test split. The same frozen token cache
and checkpoint are used; only the selection cost is toggled.
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
ARCH=ROOT/'stage2_RQ-VAE/_archive/TIGER_frozen_topology_gnn_rq_issue263_final'
sys.path.insert(0,str(ARCH))
from train_config import load_train_config
from model import RQVAE
from types import SimpleNamespace
CFG=load_train_config(ARCH/'config/train_config.json')
ck=torch.load(ROOT/'results/stage2_RQ-VAE/_runs/stage2_ae668bc04dc44d52a8bed5f9408e8e74/TIGER_frozen_topology_gnn_rq/result/rqvae_best.pth',map_location='cpu',weights_only=False)
state=ck.get('state_dict',ck)
emb=np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy')
mcfg=SimpleNamespace(hidden_sizes=CFG.HIDDEN_SIZES,codebook_num=CFG.CODEBOOK_NUM,codebook_size=tuple(CFG.CODEBOOK_SIZE if len(CFG.CODEBOOK_SIZE)>1 else list(CFG.CODEBOOK_SIZE)*CFG.CODEBOOK_NUM),codebook_dim=CFG.CODEBOOK_DIM,dropout=0.0,beta=CFG.BETA,vq_type=CFG.VQ_TYPE,ema_decay=CFG.EMA_DECAY,fix_code_embs=False,sk_epsilon=CFG.SK_EPSILON,sk_iters=CFG.SK_ITERS)
model=RQVAE(mcfg,in_dim=emb.shape[1],distance=None)
missing,unexpected=model.load_state_dict(state,strict=False)
model=model.cuda().eval()
cat=torch.as_tensor(emb,dtype=torch.float32,device='cuda')
with torch.no_grad():
    model.curvature_on=True
    mixed=model.get_indices(cat,infer_use_sk=False).cpu().numpy().astype(np.int64)
    model.curvature_on=False
    euclid=model.get_indices(cat,infer_use_sk=False).cpu().numpy().astype(np.int64)
same=np.all(mixed==euclid,axis=1)
diff=int((~same).sum())
out={'schema':'genrec.issue265.selection_diagnostic/1','checkpoint':str(ROOT/'results/stage2_RQ-VAE/_runs/stage2_ae668bc04dc44d52a8bed5f9408e8e74/TIGER_frozen_topology_gnn_rq/result/rqvae_best.pth'),'state_missing':list(missing),'state_unexpected':list(unexpected),'items':len(mixed),'different_items':diff,'different_fraction':diff/len(mixed),'mixed_sha256':hashlib.sha256(mixed.tobytes()).hexdigest(),'euclid_sha256':hashlib.sha256(euclid.tobytes()).hexdigest(),'scope':'frozen #263 checkpoint; token cache; no test split'}
Path('results/research/issue265/DIAGNOSIS_SELECTION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
