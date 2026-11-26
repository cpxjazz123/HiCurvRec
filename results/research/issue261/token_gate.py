"""Issue #261: read-only GNN-token and gate statistics from frozen checkpoints."""
from __future__ import annotations
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
REPO=Path(__file__).resolve().parents[3]; OUT=REPO/'results/research/issue261'; sys.path.insert(0,str(REPO))
import torch
from tools.issue252 import archive_audit

ARMS={100:'TIGER_gnn_attention_sectional_rq',101:'TIGER_three_expert_attention_rq',102:'TIGER_module_budget_prefix_rq',103:'TIGER_periodic_budget_prefix_rq',104:'TIGER_role_conditioned_l1_rq',105:'TIGER_role_conditioned_l3_rq',106:'TIGER_role_conditioned_l2_rq',107:'TIGER_topology_residual_l2_rq'}

def purge():
 for name in list(sys.modules):
  if name.split('.')[0] in ('model','losses','distances','train_config','train_rqvae','assignment_floor'):
   del sys.modules[name]

def role_table():
 return np.asarray(json.load(open(REPO/'stage2_RQ-VAE/TIGER_topology_residual_l2_rq/role_table.json'))['table'],dtype=np.int64)

def eff_rank(x):
 x=np.asarray(x,dtype=np.float64); x=x-x.mean(0); vals=np.linalg.eigvalsh(x.T@x/len(x)).clip(1e-12); return float(vals.sum()**2/(vals**2).sum())

def main():
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 emb=torch.from_numpy(np.load(REPO/'stage1_GeneEmbedding/output/sentence_t5.npy').astype('float32')).to(device)
 roles=role_table(); edges=None
 import pyarrow.parquet as pq
 e=pq.read_table(REPO/'results/research/behavior_graph/edges.parquet',columns=['source','target']).to_pydict()
 deg=np.zeros(len(emb),dtype=np.int64)
 for u,v in zip(e['source'],e['target']): deg[int(u)]+=1; deg[int(v)]+=1
 out={}
 for n,arm in ARMS.items():
  try:
   inv=json.load(open(OUT/'INVENTORY.json'))['arms'][str(n)]; tree=Path(inv['source_tree']); ckpt=Path(inv['checkpoint'])
   purge(); sys.path.insert(0,str(tree))
   import importlib.util
   spec=importlib.util.spec_from_file_location('tg_trainer_'+str(n),tree/'train_rqvae.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
   payload=torch.load(ckpt,map_location='cpu',weights_only=False); state=payload.get('state_dict',payload)
   model=mod.RQVAE(mod._tokenizer_config(),in_dim=768,distance=mod.DISTANCE).to(device)
   model.load_state_dict(state,strict=False); model.eval()
   ids=np.arange(len(emb),dtype=np.int64)
   with torch.no_grad():
    tokens,weights=model.structure.gates_for_items(model.store,ids,device,chunk=1024,behavior_on=True)
    tokens=tokens.detach().cpu().numpy().astype(np.float64); weights=weights.detach().cpu().numpy().astype(np.float64)
   # token shape N,T,D; mask unavailable here: all 4 real tokens returned by this API for current trees
   tn=np.linalg.norm(tokens,axis=2)
   tvar=tokens.var(axis=1).mean(axis=1)
   # item-level token covariance across the four type tokens
   tc=tokens.reshape(tokens.shape[0],-1)
   er=eff_rank(tc)
   role_stats={}
   for role in sorted(set(roles.tolist())):
    m=roles==role
    if m.sum()<2: continue
    # cosine between item token means within role (sample deterministic first 200)
    idx=np.flatnonzero(m)[:200]; means=tokens[idx].mean(1); means=means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12)
    cos=means@means.T; iu=np.triu_indices(len(means),1)
    role_stats[str(int(role))]={'n':int(m.sum()),'token_norm_mean':float(tn[m].mean()),'token_var_mean':float(tvar[m].mean()),'within_cosine_mean':float(cos[iu].mean()) if len(idx)>1 else None}
   # gate weights: accept [N,3], [N,5], [N,2,3], [N,2,5]
   w=np.asarray(weights,dtype=np.float64)
   if w.ndim==3 and w.shape[0]==len(emb): w=w.reshape(len(emb),-1)
   if w.ndim==1 and len(w)==len(emb): w=w[:,None]
   if w.ndim!=2 or len(w)!=len(emb): raise ValueError('unexpected gate shape '+str(weights.shape))
   if w.shape[1] in (3,5):
    n_exp=w.shape[1]; p=(w/np.maximum(w.sum(1,keepdims=True),1e-12))[:,None,:]; groups=1
   elif w.shape[1]%3==0:
    n_exp=3; ww=w.reshape(len(w),-1,3); p=ww/np.maximum(ww.sum(2,keepdims=True),1e-12); groups=ww.shape[1]
   elif w.shape[1]%5==0:
    n_exp=5; ww=w.reshape(len(w),-1,5); p=ww/np.maximum(ww.sum(2,keepdims=True),1e-12); groups=ww.shape[1]
   else: raise ValueError('unexpected gate width '+str(w.shape))
   ent=-(p*np.log(np.maximum(p,1e-12))).sum(2)
   gate={'shape':list(weights.shape),'rows':int(len(w)),'groups':int(groups),'experts':int(n_exp),'normalized_entropy_mean':float((ent/np.log(n_exp)).mean()),'max_weight_mean':float(p.max(2).mean()),'group_mean':p.mean(0).tolist(),'expert_frequency':(p.reshape(-1,n_exp).mean(0)).tolist()}
   # role conditional gate means
   rg={}
   for role in sorted(set(roles.tolist())):
    m=roles==role
    if m.sum(): rg[str(int(role))]=p[m].reshape(-1,n_exp).mean(0).tolist()
   gate['role_expert_weights']=rg
   # degree association
   def spear(a,b):
    from scipy.stats import spearmanr
    return float(spearmanr(a,b).correlation)
   try: gate['token_norm_degree_spearman']=spear(tn.mean(1),deg)
   except Exception: gate['token_norm_degree_spearman']=None
   out[str(n)]={'arm':arm,'tokens':tokens.shape,'token_norm_mean':float(tn.mean()),'token_norm_std':float(tn.std()),'token_variance_mean':float(tvar.mean()),'token_effective_rank':er,'roles':role_stats,'gate':gate}
   print(n,arm,'done',tokens.shape,weights.shape,flush=True)
  except Exception as e:
   out[str(n)]={'arm':arm,'status':'NA','reason':repr(e)}
   print(n,arm,'FAIL',repr(e),flush=True)
 (OUT/'TOKEN_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:(v.get('status','PASS')) for k,v in out.items()},indent=2))
if __name__=='__main__': main()
