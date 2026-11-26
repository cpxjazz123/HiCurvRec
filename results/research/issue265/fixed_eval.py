"""Fixed equal-weight curvature probe on the final issue265 candidate."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]; TREE=ROOT/'stage2_RQ-VAE/TIGER_euclid_local_curve_supervision_rq'
sys.path.insert(0,str(TREE))
from train_config import load_train_config
from model import RQVAE
from model.geometry import fixed_squared
from model.estimator import sampled_values
CFG=load_train_config(TREE/'config/train_config.json')
ck=torch.load(ROOT/'results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/result/rqvae_best.pth',map_location='cpu',weights_only=False)
state=ck.get('state_dict',ck); emb=np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy')
from types import SimpleNamespace
mcfg=SimpleNamespace(hidden_sizes=CFG.HIDDEN_SIZES,codebook_num=CFG.CODEBOOK_NUM,codebook_size=tuple(CFG.CODEBOOK_SIZE if len(CFG.CODEBOOK_SIZE)>1 else list(CFG.CODEBOOK_SIZE)*CFG.CODEBOOK_NUM),codebook_dim=CFG.CODEBOOK_DIM,dropout=0.0,beta=CFG.BETA,vq_type=CFG.VQ_TYPE,ema_decay=CFG.EMA_DECAY,fix_code_embs=False,sk_epsilon=CFG.SK_EPSILON,sk_iters=CFG.SK_ITERS)
model=RQVAE(mcfg,in_dim=emb.shape[1],distance=None).cuda().eval(); missing,unexpected=model.load_state_dict(state,strict=False)
cat=torch.as_tensor(emb,dtype=torch.float32,device='cuda')
sample=model.structural_sampler().draw(0,0,1,track=False)
ids=sample['ids']; item_ids=torch.as_tensor(ids,dtype=torch.long,device='cuda')
with torch.no_grad():
 _,weights=model.structure.gates_for_items(model.store,ids,'cuda',behavior_on=True)
 q,_,_,_=model.quantize(model.encoder(cat[item_ids]),weights,balanced=False)
 target=torch.as_tensor(sample['target'],dtype=torch.float32,device='cuda')
 fixed=sampled_values(q,sample['triples'],sample['references'],lambda x,y:fixed_squared(x,y,scale=model._tangent_scale_value),paired_squared=lambda x,y:fixed_squared(x[:,None,:],y[:,None,:],scale=model._tangent_scale_value).reshape(-1))
 fixed_loss=float(F.huber_loss(fixed/model._target_scale_value,target/model._target_scale_value,reduction='mean'))
 # same q/gates through the candidate's local objective
 local=model.sampled_curvature(cat,sample,weights=weights)
 print(json.dumps({'fixed_equal_weight_loss':fixed_loss,'local_trainable_loss':float(local.detach()),'items':len(ids),'triples':len(sample['triples']),'centre':int(sample['centre']),'missing':list(missing),'unexpected':list(unexpected)},indent=2))
 Path('results/research/issue265/FIXED_EVAL.json').write_text(json.dumps({'fixed_equal_weight_loss':fixed_loss,'local_trainable_loss':float(local.detach()),'items':len(ids),'triples':len(sample['triples']),'centre':int(sample['centre']),'scope':'one frozen training sample; fixed Euclidean equal-expert evaluation, not full all-triple'},indent=2)+'\n')
