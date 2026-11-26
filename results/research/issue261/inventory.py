"""Issue #261: read-only inventory and identity check of R2 bundles 100-107."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'results/research/issue261'
sys.path.insert(0,str(REPO))
from tools.issue252 import archive_audit

ARMS={
 100:'TIGER_gnn_attention_sectional_rq',
 101:'TIGER_three_expert_attention_rq',
 102:'TIGER_module_budget_prefix_rq',
 103:'TIGER_periodic_budget_prefix_rq',
 104:'TIGER_role_conditioned_l1_rq',
 105:'TIGER_role_conditioned_l3_rq',
 106:'TIGER_role_conditioned_l2_rq',
 107:'TIGER_topology_residual_l2_rq',
}
EXPECTED_R10={
 100:0.0492870697609638,101:0.046431866850049616,102:0.05273420498267727,
 103:0.05266456588728913,104:0.0490085133794112,105:0.05151552081338463,
 106:0.05034906596563311,107:0.049600445690210486,
}

def sha(p):
 p=Path(p); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def find_final(arm):
 root=REPO/'results/stage3_T5Train'/arm
 fs=sorted(root.glob('**/test_final.json'))
 if len(fs)!=1: raise RuntimeError(f'{arm}: expected one test_final, got {fs}')
 return fs[0]

def main():
 ledger=json.load(open(REPO/'results/research/issue139/r2_ledger.json'))
 out={'schema':'genrec.issue261.inventory/1','arms':{},'data':{},'training_count':0}
 for n,arm in ARMS.items():
  final=find_final(arm); fd=json.load(open(final))
  sid=Path(fd['sid_path']); ckpt=Path(fd['stage2_checkpoint']); binding=Path(fd['archive_binding'])
  export=Path(fd['stage2_export_manifest'])
  run=fd['stage2_run_id']; rec=Path(fd['predictions']).parent
  stage2=Path(fd['sid_path']).parent
  metrics=sorted((REPO/'results/stage2_RQ-VAE'/arm/'stage2_metrics').glob('*.json'))
  layer=metrics[0] if len(metrics)==1 else None
  src=Path(json.load(open(binding))['source_tree'])
  train=stage2/'logs/training_metrics.jsonl'
  train_rows=[json.loads(x) for x in open(train)] if train.is_file() else []
  train_events=[x for x in train_rows if x.get('event')=='train']
  train_end=[x for x in train_rows if x.get('event')=='train_end']
  preds=sorted(rec.glob('rank*.jsonl'))
  pred_rows=sum(sum(1 for line in open(p) if line.strip()) for p in preds)
  ledger_entry=ledger.get(str(n),{})
  files=ledger_entry.get('files',{})
  def cmp(rel,path):
   want=files.get(rel); got=sha(path) if path.is_file() else None
   return {'path':str(path),'sha256':got,'ledger_sha256':want,'status':'PASS' if want==got else 'FAIL' if want else 'NA'}
  checks={
   'source_tree':{'path':str(src),'exists':src.is_dir()},
   'checkpoint':cmp('rqvae_best.pth',ckpt),
   'sid':cmp('item_sids.json',sid),
   'archive_binding':cmp(f'{arm}/archive_binding.json',binding),
   'export_manifest':cmp(f'{arm}/export_manifest.json',export),
  }
  recall_path=OUT/'_r2'/str(n)/'recall.json'
  checks['recall_json']=cmp('recall.json',recall_path)
  try:
   receipt,sources=archive_audit.validate_binding(binding,sid,ckpt,src)
   binding_status='PASS'; binding_error=None
  except Exception as e:
   receipt=None;sources={};binding_status='FAIL';binding_error=repr(e)
  checks['archive_binding_validation']={'status':binding_status,'error':binding_error}
  receipt=receipt if receipt is not None else json.load(open(binding))
  try:
   actual_sources=archive_audit.source_hashes(src); cand_ok=actual_sources==receipt.get('candidate_sources')
   actual_infra=archive_audit.imported_repo_sources(src,Path(receipt['repository'])) if receipt else {}
   exp_infra=receipt.get('infrastructure_sources',{}) if receipt else {}
   drift=sorted(k for k in set(exp_infra)|set(actual_infra) if exp_infra.get(k)!=actual_infra.get(k))
  except Exception as e:
   cand_ok=False; drift=[]; binding_error=str(e)
  checks['candidate_sources']={'status':'PASS' if cand_ok else 'FAIL','files':len(actual_sources) if 'actual_sources' in locals() else None}
  checks['infra_sources']={'status':'PASS' if not drift else 'DRIFT','drift_files':drift}
  recall=json.load(open(recall_path)) if recall_path.is_file() else {}
  recall_keys=sorted(recall)
  test_keys=sorted(fd)
  checks['recall_identity']={
   'status':'PASS' if (recall.get('test_recall@10')==fd.get('test_recall@10') and recall.get('sid_sha256')==fd.get('sid_sha256') and recall.get('stage2_run_id')==run) else 'FAIL',
   'test_recall@10':recall.get('test_recall@10'),'sid_sha256':recall.get('sid_sha256'),'run':recall.get('stage2_run_id'),
   'recall_key_count':len(recall_keys),'test_final_key_count':len(test_keys)}
  checks['stage2_training']={'metrics':str(train),'events':len(train_events),'epochs_complete':bool(train_events and train_events[0].get('epoch')==1 and train_events[-1].get('epoch')==3000 and len(train_events)==3000),'train_end':bool(train_end),'train_end_epoch':train_end[-1].get('epoch') if train_end else None}
  checks['stage3']={'final':str(final),'n_eval':fd.get('n_eval'),'test_recall@10':fd.get('test_recall@10'),'expected_r10':EXPECTED_R10[n],'prediction_shards':len(preds),'prediction_rows':pred_rows,'rows_match':pred_rows==fd.get('n_eval'),'protocol':fd.get('stage3_input_config',{}).get('num_epochs')==150}
  checks['layer_record']={'path':str(layer) if layer else None,'sha256':sha(layer) if layer else None,'sid_sha_match':bool(layer and json.load(open(layer)).get('sid',{}).get('sha256')==sha(sid))}
  out['arms'][str(n)]={'arm':arm,'run_id':run,'final':str(final),'sid':str(sid),'checkpoint':str(ckpt),'source_tree':str(src),'archive_binding':str(binding),'checks':checks}
  out['training_count']+=0
 # frozen data
 for p in ['results/stage0_build_parquet/train.parquet','results/stage0_build_parquet/valid.parquet','stage1_GeneEmbedding/output/sentence_t5.npy','results/research/behavior_graph/edges.parquet','assets/v8_runtime_layout/targets.npy','assets/v8_runtime_layout/target_offsets.npy','assets/manifest.json']:
  path=REPO/p; out['data'][p]={'exists':path.is_file(),'sha256':sha(path) if path.is_file() else None}
 (OUT/'INVENTORY.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
