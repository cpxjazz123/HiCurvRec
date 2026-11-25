"""Switch the R06 topology-token residual off on the trained checkpoint.

Read-only evaluation: level 1 and level 3 are the controls, level 2 is the
target of the token residual.  No training and no SID rewrite.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
TREE=Path(__file__).resolve().parents[3]/'stage2_RQ-VAE'/'TIGER_topology_residual_l2_rq'
sys.path.insert(0,str(TREE))
from model.mechanism import RQVAE
ROOT=TREE.parents[1]
ROLE=json.loads((TREE/'role_table.json').read_text())
ROLES=np.asarray(ROLE['table'],dtype=np.int64); NAMES={int(v):k for k,v in ROLE['roles'].items()}
def main():
 ckpt=Path(sys.argv[1]); out=Path(sys.argv[2])
 torch.set_num_threads(2)
 emb=torch.from_numpy(np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy').astype('float32'))
 cfg=SimpleNamespace(hidden_sizes=[512,256,128],codebook_dim=32,dropout=0.,codebook_num=3,
  codebook_size=[256,256,256],vq_type='vq',beta=.25,sk_epsilon=.003,sk_iters=50)
 m=RQVAE(cfg,in_dim=768); payload=torch.load(ckpt,map_location='cpu',weights_only=False)
 m.load_state_dict(payload['state_dict'],strict=False); m.freeze_tangent_scale(emb); m.eval()
 def codes():
  with torch.no_grad():
   m.catalogue_q(emb,infer_use_sk=False)
   return m.last_tokens[:,:3].cpu().numpy().copy()
 m.token_residual_on=True; on=codes(); shift=m.last_token_shift.detach().norm(dim=1).numpy().copy()
 m.token_residual_on=False; off=codes()
 diff=on!=off
 per={}
 for role in sorted(set(ROLES.tolist())):
  mask=ROLES==role; name=NAMES.get(int(role),'NO_ROLE')
  per[name]={'role':int(role),'items':int(mask.sum()),
   'changed':[int(diff[mask,i].sum()) for i in range(3)],
   'fraction':[float(diff[mask,i].mean()) for i in range(3)],
   'shift_norm_mean':float(shift[mask].mean())}
 report={'measure':'same R06 checkpoint, token residual switch off',
  'checkpoint':str(ckpt),'items':len(emb),'levels':['L1','L2','L3'],
  'overall':{'changed':[int(diff[:,i].sum()) for i in range(3)],
   'fraction':[float(diff[:,i].mean()) for i in range(3)],
   'shift_norm_mean':float(shift.mean()),'shift_norm_max':float(shift.max())},
  'roles':per}
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__': main()
