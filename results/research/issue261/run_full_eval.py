"""Issue #261 read-only full evaluation of one frozen R2 arm.

This only orchestrates public evaluators; it never trains or rewrites a SID.
For 100–102 the archive receipt's external-infra hashes predate later public
infra commits; candidate source hashes are still checked, and the mismatch is
recorded in INVENTORY.json rather than hidden.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from tools.issue252.final_evaluation import run_final_evaluation, load_sid, _coverage, _training_module, _GeometryCrosscheck
from tools.issue252 import archive_audit
from tools.issue251.evaluation import evaluate, evaluate_heldout, evaluate_continuous_heldout, evaluate_reconstruction
from tools.issue251.full_evaluation import full_evaluate
from tools.issue253.evaluation_reference import fixed_three_squared_numpy, fixed_three_paired_numpy
from tools.graph_infra.heldout_eval import HeldOutEvaluation
from tools.graph_infra import fidelity
import numpy as np, torch
from tools.issue252.final_evaluation import _record, _frozen_inputs, _compact

ASSETS=REPO/'assets'; EMB=REPO/'stage1_GeneEmbedding/output/sentence_t5.npy'; EXPECTED_ITEMS=24587; CODE=(256,256,256)
FINAL=REPO/'results/research/issue261/full_eval'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('arm'); ap.add_argument('--allow-infra-drift',action='store_true'); ap.add_argument('--out',default='')
 a=ap.parse_args(); arm=a.arm
 finals=sorted((REPO/'results/stage3_T5Train'/arm).glob('**/test_final.json'))
 if len(finals)!=1: raise SystemExit('test_final count')
 fd=json.load(open(finals[0])); sid=Path(fd['sid_path']); ckpt=Path(fd['stage2_checkpoint']); bind=Path(fd['archive_binding']); exp=Path(fd['stage2_export_manifest'])
 receipt=json.load(open(bind)); tree=Path(receipt['source_tree'])
 # candidate source must still match; infra mismatch is recorded, not treated as candidate drift.
 actual=archive_audit.source_hashes(tree)
 if actual!=receipt['candidate_sources']: raise SystemExit('candidate source drift')
 if not a.allow_infra_drift:
  archive_audit.validate_binding(bind,sid,ckpt,tree)
 payload=torch.load(ckpt,map_location='cpu',weights_only=False)
 selections=[payload[k] for k in ('selection','checkpoint_selection') if k in payload]
 if payload.get('epoch')!=3000 or not selections or any(v!='final_epoch' for v in selections): raise SystemExit('not final epoch')
 state=payload['state_dict']; books=[]
 for level,cap in enumerate(CODE):
  b=state.get(f'rq.vq_layers.{level}.embed.weight')
  if b is None or tuple(b.shape)!=(cap,32): raise SystemExit('codebook missing')
  books.append(b.detach().cpu().numpy())
 if not all(np.isfinite(b).all() for b in books): raise SystemExit('nonfinite book')
 rows=load_sid(sid,EXPECTED_ITEMS,list(CODE)); labels,coverage=_coverage(ASSETS,len(rows))
 target=np.load(EMB,mmap_mode='r',allow_pickle=False)
 if target.shape!=(EXPECTED_ITEMS,768): raise SystemExit('embedding shape')
 # purge top-level candidate packages
 for name in list(sys.modules):
  if name.split('.')[0] in ('model','losses','distances','train_config','train_rqvae','assignment_floor'):
   del sys.modules[name]
 sys.path.insert(0,str(tree))
 training=_training_module(tree)
 model=training.RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE).cpu()
 model.load_state_dict(state,strict=True); model.eval()
 calibration=[float(getattr(model,k)) for k in ('tangent_scale','curvature_weight','target_scale')]
 if not np.isfinite(calibration).all() or min(calibration)<=0: raise SystemExit('bad calibration')
 scale=calibration[0]; q=fidelity.decode_hard_sid(rows,books)
 z=np.empty((len(rows),32),dtype=np.float32); rec=np.empty_like(target); c_rec=np.empty_like(target)
 torch.set_num_threads(1)
 with torch.inference_mode():
  for start in range(0,len(rows),1024):
   end=min(start+1024,len(rows)); enc=model.encoder(torch.from_numpy(np.array(target[start:end],copy=True)))
   z[start:end]=enc.cpu().numpy(); c_rec[start:end]=model.decoder(enc).cpu().numpy()
   hard=torch.from_numpy(q[start:end]).to(dtype=enc.dtype); rec[start:end]=model.decoder(hard).cpu().numpy()
 fixed=fixed_three_squared_numpy
 hard_report=evaluate(rows,books,np.empty((0,3),dtype=np.int64),np.empty(0),{},component_labels=labels,scale=scale,fixed_squared=fixed,embeddings=target,reconstruction=rec)
 cont={'representation':'continuous_encoder_z','reconstruction':evaluate_reconstruction(target,c_rec,representation='continuous-z')}
 for name,coord,report in (('hard_q',q,hard_report),('continuous_z',z,cont)):
  full=full_evaluate(coord,ASSETS,scale=scale,metric=fixed_three_squared_numpy,paired_metric=fixed_three_paired_numpy)
  report.update(full)
 held=HeldOutEvaluation(REPO,ASSETS)
 if not held.strict_banks(): raise SystemExit('heldout missing')
 def three_nd(x,y,scale=1.):
  xa=np.asarray(x); ya=np.asarray(y)
  if xa.ndim>2 and xa.shape[:-1]==ya.shape[:-1]:
   return fixed_three_paired_numpy(xa.reshape(-1,xa.shape[-1]),ya.reshape(-1,ya.shape[-1]),scale).reshape(xa.shape[:-1])
  return fixed_three_squared_numpy(xa.reshape(-1,xa.shape[-1]),ya.reshape(-1,ya.shape[-1]),scale)
 hard_report['strict_heldout']=evaluate_heldout(rows,books,held,scale=scale,fixed_squared=three_nd)
 cont['strict_heldout']=evaluate_continuous_heldout(z,held,scale=scale,fixed_squared=three_nd)
 # unified endpoint diagnostics, independent of a candidate's native metric
 zq=np.asarray(z,dtype=np.float64); qq=np.asarray(q,dtype=np.float64)
 def eff(a):
  a=a-a.mean(0); vals=np.linalg.eigvalsh(a.T@a/len(a)).clip(1e-12); return float(vals.sum()**2/(vals**2).sum())
 code_stats=[]
 for l,b in enumerate(books):
  d=np.linalg.norm(b[:,None,:]-b[None,:,:],axis=-1); iu=np.triu_indices(len(b),1); code_stats.append({'level':l,'spacing_mean':float(d[iu].mean()),'norm_mean':float(np.linalg.norm(b,axis=1).mean())})
 quant={'quant_error':float(np.linalg.norm(zq-qq,axis=1).mean()),'effective_rank_z':eff(zq),'effective_rank_q':eff(qq),'codebooks':code_stats}
 out={'schema':'genrec.issue261.full_eval/1','arm':arm,'run_id':fd['stage2_run_id'],'sid':_record(sid),'checkpoint':_record(ckpt),'binding':_record(bind),'source_tree':str(tree),'coverage':coverage,'quant':quant,'hard_q':hard_report,'continuous_z':cont,'calibration':dict(zip(('tangent_scale','curvature_weight','target_scale'),calibration)),'candidate_sources_ok':True,'infra_binding_validated':not a.allow_infra_drift}
 p=Path(a.out) if a.out else FINAL/f'{arm}.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n'); print(json.dumps({'arm':arm,'out':str(p),'coverage':coverage,'hard_curve':hard_report.get('curvature',{}),'hard_recon':hard_report.get('reconstruction',{})}))
if __name__=='__main__': main()
