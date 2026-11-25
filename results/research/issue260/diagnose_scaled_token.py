#!/usr/bin/env python3.9
"""Scale the topology token offset to the L2 codebook resolution (no training)."""
from __future__ import annotations
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[3]
ARM=REPO/'stage2_RQ-VAE'/'TIGER_topology_residual_l2_rq'
RUN=REPO/'results/stage2_RQ-VAE/_runs/issue256_r05_formal/TIGER_role_conditioned_l2_rq/result'
OUT=REPO/'results/research/issue260/diagnosis'
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(ARM))
from train_config import load_train_config
from distances import build_distance
from model import RQVAE
from model.selection import select

def quantize(model,encoded,weights,role_rows,shift=None):
 residual=encoded; out=[]
 for level,layer in enumerate(model.rq.vq_layers):
  target=residual+shift if (level==1 and shift is not None) else residual
  lw=model.level_weights(weights,level,role_rows)
  cost=model.assignment_cost(target,layer.embed.weight,lw)
  selected=select(cost,layer.embed.weight,balanced=False,epsilon=layer.sk_epsilon,
   iterations=layer.sk_iters,training=False,hard_ids=layer._indices(cost,False))
  out.append(selected['ids']); residual=residual-selected['q']
 return torch.stack(out,dim=1).detach().cpu().numpy()

def rates(c):
 p=np.load(REPO/'results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz')
 a,pos,neg=p['anchors'],p['positives'],p['negatives']
 return {l:{'positive':float((c[a,:l]==c[pos,:l]).all(1).mean()),
            'negative':float((c[a,None,:l]==c[neg,:l]).all(2).mean())} for l in (1,2,3)}

def main():
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 cfg=load_train_config(ARM/'config/train_config.json')
 sizes=list(cfg.CODEBOOK_SIZE)*(cfg.CODEBOOK_NUM if len(cfg.CODEBOOK_SIZE)==1 else 1)
 tokenizer=SimpleNamespace(hidden_sizes=cfg.HIDDEN_SIZES,codebook_num=cfg.CODEBOOK_NUM,
  codebook_size=tuple(sizes),codebook_dim=cfg.CODEBOOK_DIM,dropout=0.,beta=cfg.BETA,
  vq_type=cfg.VQ_TYPE,ema_decay=cfg.EMA_DECAY,fix_code_embs=False,
  sk_epsilon=cfg.SK_EPSILON,sk_iters=cfg.SK_ITERS)
 model=RQVAE(tokenizer,in_dim=768,distance=build_distance(cfg)).to(device)
 model.load_state_dict(torch.load(RUN/'rqvae_best.pth',map_location='cpu')['state_dict'],strict=False)
 model.eval(); emb=torch.from_numpy(np.load(REPO/'stage1_GeneEmbedding/output/sentence_t5.npy')).float().to(device)
 ids=np.arange(len(emb),dtype=np.int64); role_rows=model.role_mix.rows_for(ids,device)
 l2_scale=model.rq.vq_layers[1].embed.weight.norm(dim=1).mean().detach()
 with torch.no_grad():
  encoded=model.encoder(emb)
  tokens,weights=model.structure.gates_for_items(model.store,ids,device,chunk=1024,behavior_on=True)
  mask=model.structure.last_token_mask[:,:-1]
  token_mean=(tokens*mask[...,None]).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
  base=quantize(model,encoded,weights,role_rows)
  shifts={'unit_l2scale':F.normalize(token_mean,dim=-1)*l2_scale,
          'unit_half_l2scale':F.normalize(token_mean,dim=-1)*l2_scale*0.5,
          'unit_0.01':F.normalize(token_mean,dim=-1)*0.01}
  out={'schema':'genrec.issue260.scaled_token_probe/1','l2_codebook_mean_norm':float(l2_scale),
       'token_norm_mean':float(token_mean.norm(dim=1).mean()),
       'base':{'rates':rates(base),'prefix2':len({tuple(v[:2]) for v in base}),
               'triples':len({tuple(v) for v in base})},'variants':{}}
  for name,shift in shifts.items():
   c=quantize(model,encoded,weights,role_rows,shift=shift)
   out['variants'][name]={'shift_norm_mean':float(shift.norm(dim=1).mean()),
    'rates':rates(c),'prefix2':len({tuple(v[:2]) for v in c}),
    'triples':len({tuple(v) for v in c}),
    'changed':{str(i+1):float((base[:,i]!=c[:,i]).mean()) for i in range(3)}}
 Path(OUT/'token_scale_probe.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
