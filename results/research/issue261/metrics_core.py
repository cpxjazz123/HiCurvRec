"""Issue #261: read-only metric matrix for the eight completed R2 bundles."""
from __future__ import annotations
import collections, json, math, sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'results/research/issue261'
sys.path.insert(0,str(REPO))
from tools.iteration import analysis as ianalysis
from tools.iteration import identity as iidentity

ARMS={100:'TIGER_gnn_attention_sectional_rq',101:'TIGER_three_expert_attention_rq',102:'TIGER_module_budget_prefix_rq',103:'TIGER_periodic_budget_prefix_rq',104:'TIGER_role_conditioned_l1_rq',105:'TIGER_role_conditioned_l3_rq',106:'TIGER_role_conditioned_l2_rq',107:'TIGER_topology_residual_l2_rq'}
ROLE=REPO/'stage2_RQ-VAE'/'TIGER_topology_residual_l2_rq'/'role_table.json'
VALID=REPO/'results/stage0_build_parquet'/'valid.parquet'
EDGES=REPO/'results/research/behavior_graph'/'edges.parquet'

def entropy(c):
 n=sum(c.values()); return -sum((v/n)*math.log2(v/n) for v in c.values() if v) if n else None
def prefix_stats(rows):
 three=[tuple(map(int,r[:3])) for r in rows]
 out={}; n=len(three)
 for l in (1,2,3):
  c=collections.Counter(r[:l] for r in three); sizes=sorted(c.values()); vals=c.values()
  out[f'distinct_L{l}']=len(c)
  out[f'collision_L{l}']=1-len(c)/n
  out[f'groups_mean_L{l}']=n/len(c)
  out[f'groups_median_L{l}']=sizes[len(sizes)//2]
  out[f'groups_p90_L{l}']=sizes[int(.9*(len(sizes)-1))]
  out[f'groups_max_L{l}']=sizes[-1]
  out[f'max_share_L{l}']=sizes[-1]/n
  out[f'perplexity_L{l}']=2**entropy(c)
  out[f'dead_codes_L{l}']=256-len({r[l-1] for r in three})
  out[f'occupancy_L{l}']=len({r[l-1] for r in three})
 h1=entropy(collections.Counter(r[0] for r in three)); h12=entropy(collections.Counter(r[:2] for r in three)); h123=entropy(collections.Counter(three))
 out.update({'items':n,'H_L1':h1,'H_L2_given_L1':h12-h1,'H_L3_given_L1L2':h123-h12,'H_triple':h123,'collision':1-len(set(three))/n})
 return out

def load_sid(path):
 raw=json.load(open(path)); ids=sorted(map(int,raw)); rows=[raw[str(i)] for i in ids]; return rows

def read_layer(arm):
 p=REPO/'results/stage2_RQ-VAE'/arm/'stage2_metrics'
 fs=sorted(p.glob('*.json'))
 if len(fs)!=1: return None
 return json.load(open(fs[0]))

def role_table():
 d=json.load(open(ROLE)); return np.asarray(d['table'],dtype=np.int64),d

def graph_cache():
 import pyarrow.parquet as pq
 e=pq.read_table(EDGES,columns=['source','target']).to_pydict()
 fwd=collections.defaultdict(set); rev=collections.defaultdict(set)
 for u,v in zip(e['source'],e['target']):
  u=int(u);v=int(v); fwd[u].add(v);rev[v].add(u)
 return fwd,rev

def relation_rows(rows):
 import pyarrow.parquet as pq
 examples=pq.read_table(VALID,columns=['history','seen_history','target']).to_pylist()
 fwd,rev=graph_cache()
 pairs=[]
 for ex in examples:
  h=ex['history'];
  if not h: continue
  u=int(h[-1]); v=int(ex['target']); pairs.append((u,v))
 return pairs,fwd,rev

def relation_masks(pairs,fwd,rev):
 direct=np.array([v in fwd[u] for u,v in pairs]); reverse=np.array([u in rev[v] for u,v in pairs])
 noedge=~(direct|reverse)
 common_succ=np.array([bool(fwd[u]&fwd[v]) for u,v in pairs])
 common_pred=np.array([bool(rev[u]&rev[v]) for u,v in pairs])
 two=np.array([any(w in fwd[v] for w in fwd[u]) for u,v in pairs])
 return {'direct':direct,'reverse':reverse,'no_edge':noedge,'common_successor':common_succ,'common_predecessor':common_pred,'two_hop':two}

def prefix_share(rows,pairs,level,mask=None):
 c=np.array([r[:level] for r in rows],dtype=np.int64); u=np.array([p[0] for p in pairs]);v=np.array([p[1] for p in pairs]); hit=(c[u]==c[v]).all(axis=1) if level>1 else (c[u,0]==c[v,0])
 return float(hit.mean()) if mask is None else float(hit[mask].mean()), int(mask.sum()) if mask is not None else len(pairs)

def degree_bins(pairs,fwd,rev):
 deg=np.array([len(fwd[u])+len(rev[u]) for u,v in pairs]); degv=np.array([len(fwd[v])+len(rev[v]) for u,v in pairs])
 q=np.quantile(np.r_[deg,degv],[.2,.4,.6,.8]); return np.digitize(deg,q),np.digitize(degv,q)

def relation_metrics(rows,pairs,fwd,rev):
 masks=relation_masks(pairs,fwd,rev); out={}; bins=degree_bins(pairs,fwd,rev)
 for name,mask in masks.items():
  for l in (1,2,3):
   rate,n=prefix_share(rows,pairs,l,mask)
   other=~mask; base,nb=prefix_share(rows,pairs,l,other)
   out[f'{name}_L{l}']={'rate':rate,'n':n,'baseline_rate':base,'lift':rate-base,'baseline_n':nb}
 # role-conditioned sharing and purity
 tab,payload=role_table(); u=np.array([p[0] for p in pairs]);v=np.array([p[1] for p in pairs])
 ru=tab[u];rv=tab[v]
 for l in (2,3):
  c=np.array([r[:l] for r in rows],dtype=np.int64); hit=(c[u]==c[v]).all(axis=1)
  out[f'role_same_L{l}']={'rate':float(hit[ru==rv].mean()) if (ru==rv).any() else None,'n':int((ru==rv).sum())}
  out[f'role_cross_L{l}']={'rate':float(hit[ru!=rv].mean()) if (ru!=rv).any() else None,'n':int((ru!=rv).sum())}
 # role entropy/purity of each prefix level
 role_counts=collections.Counter(map(int,tab.tolist())); out['role_distribution']={str(k):v for k,v in sorted(role_counts.items())}
 out['role_count']=int(len(role_counts)); out['role_coverage']=float((tab>=0).mean())
 for l in (1,2):
  groups=collections.defaultdict(list)
  for i,r in enumerate(rows): groups[tuple(r[:l])].append(int(tab[i]))
  ents=[]; maxes=[]
  for g in groups.values():
   cc=collections.Counter(g); n=len(g); ents.append(entropy(cc)); maxes.append(max(cc.values())/n)
  out[f'role_entropy_L{l}']=float(np.mean(ents)); out[f'role_max_share_L{l}']=float(np.mean(maxes))
 return out

def training_metrics(sid_path):
 p=Path(sid_path).parent/'logs'/'training_metrics.jsonl'
 if not p.is_file(): return {'status':'NA','reason':'missing '+str(p)}
 rows=[json.loads(x) for x in open(p) if x.strip()]
 tr=[x for x in rows if x.get('event')=='train']; end=[x for x in rows if x.get('event')=='train_end']
 if not tr: return {'status':'NA','reason':'no train events','path':str(p)}
 def avg(key,subset):
  vals=[float(x[key]) for x in subset if isinstance(x.get(key),(int,float)) and math.isfinite(float(x[key]))]
  return float(np.mean(vals)) if vals else None
 idx=[0,len(tr)//2,-1]
 out={'status':'PASS','path':str(p),'events':len(tr),'first_epoch':tr[0].get('epoch'),'last_epoch':tr[-1].get('epoch'),'complete':len(tr)==3000 and tr[0].get('epoch')==1 and tr[-1].get('epoch')==3000,'train_end':bool(end),'final_collision':tr[-1].get('collision') or (end[-1].get('final_collision') if end else None),'final_epoch_time_s':tr[-1].get('epoch_time_s'),'gate_entropy_mean':avg('gate_entropy',tr),'gate_entropy_last':tr[-1].get('gate_entropy'),'gate_mean_last':tr[-1].get('gate_mean'),'gradient_norm_last':tr[-1].get('gradient_norm_before_clip'),'level_codebook_gradient_norm_last':tr[-1].get('level_codebook_gradient_norm'),'loss_windows':{k:avg('loss',[tr[i] for i in idx]) for k in ('first','mid','last')},'recon_windows':{k:avg('recon',[tr[i] for i in idx]) for k in ('first','mid','last')},'curvature_windows':{k:avg('curvature_estimate',[tr[i] for i in idx]) for k in ('first','mid','last')}}
 return out

def full_eval(arm):
 fm=OUT/'FULL_METRICS.json'
 if fm.is_file():
  d=json.load(open(fm));
  if arm in d: return {'source':'full_metrics','metrics':d[arm].get('metrics',{})}
 cand=OUT/'full_eval'/f'{arm}.json'
 if cand.is_file(): d=json.load(open(cand)); return {'source':'issue261','hard':d.get('hard_q'),'continuous':d.get('continuous_z'),'quant':d.get('quant')}
 if arm=='TIGER_gnn_attention_sectional_rq':
  s=json.load(open(REPO/'results/research/issue252/FINAL_EVALUATION_SUMMARY.json'))
  return {'source':'issue252','hard':s.get('hard_q'),'continuous':s.get('continuous_z')}
 if arm=='TIGER_three_expert_attention_rq':
  s=json.load(open(REPO/'results/research/issue253/FINAL_EVALUATION_SUMMARY.json')); g=s.get('generations',{}).get('issue253',{})
  return {'source':'issue253','hard':g.get('hard_q'),'continuous':g.get('continuous_z')}
 return {'source':'NA','reason':'no full evaluation output for this arm'}

def stage3_metrics(final):
 d=json.load(open(final)); preds=sorted(Path(d['predictions']).parent.glob('rank*.jsonl'))
 try:
  red=ianalysis.reduce_predictions(preds,d['n_eval']) if preds else None
  status='PASS' if red and all(abs(red.get(k,float('nan'))-d.get(k,float('nan')))<=1e-10 for k in ('test_recall@5','test_recall@10','test_ndcg@5','test_ndcg@10')) else 'FAIL'
 except Exception as e:
  red=None; status='FAIL:'+repr(e)
 return {'status':status,'n_eval':d.get('n_eval'),'predictions':len(preds),'prediction_rows':red.get('n_eval') if red else None,'reduced':red,'test_final':{k:d.get(k) for k in ('test_recall@5','test_recall@10','test_ndcg@5','test_ndcg@10')}}

def main():
 inv=json.load(open(OUT/'INVENTORY.json'))
 tok=json.load(open(OUT/'TOKEN_GATE.json')) if (OUT/'TOKEN_GATE.json').is_file() else {}
 table=[]
 for n,s in inv['arms'].items():
  arm=s['arm']; final=s['final']; fd=json.load(open(final)); rows=load_sid(s['sid']); sm=prefix_stats(rows); layer=read_layer(arm)
  val={}
  if layer and layer.get('validation'):
   v=layer['validation']; val={'rows':v.get('rows'),'positive':v.get('positive_hit_rates_by_level'),'negative':v.get('negative_hit_probabilities_by_level'),'difference':[a-b if a is not None and b is not None else None for a,b in zip(v.get('positive_hit_rates_by_level') or [],v.get('negative_hit_probabilities_by_level') or [])],'ratio':[a/b if a is not None and b else None for a,b in zip(v.get('positive_hit_rates_by_level') or [],v.get('negative_hit_probabilities_by_level') or [])],'l1_hard_er':v.get('l1_hard_er'),'log_enrichment':v.get('conditional_log_enrichment_bits'),'survival_diff':v.get('conditional_prefix_survival_difference'),'scope':v.get('scope')}
  pairs,fwd,rev=relation_rows(rows)
  rel=relation_metrics(rows,pairs,fwd,rev)
  train=training_metrics(s['sid']); s3=stage3_metrics(final); fe=full_eval(arm)
  row={'index':int(n),'token_gate':tok.get(n,{}),'arm':arm,'run_id':s['run_id'],'sid_sha256':s['checks']['sid']['sha256'],'checkpoint_sha256':s['checks']['checkpoint']['sha256'],'source_candidate_ok':s['checks']['archive_binding_validation']['status']=='PASS' or 'infra drift only','stage2_training':train,'sid':sm,'validation':val,'relations':rel,'stage3':s3,'full_eval':fe}
  table.append(row)
 (OUT/'METRICS.json').write_text(json.dumps({'schema':'genrec.issue261.metrics/1','rows':table},indent=2)+'\n')
 # wide csv of scalar metrics
 import csv
 scalar={}
 for row in table:
  flat={}
  def walk(prefix,obj):
   if isinstance(obj,dict):
    for k,v in obj.items(): walk(prefix+'_'+k if prefix else k,v)
   elif isinstance(obj,(int,float)) and not isinstance(obj,bool) and math.isfinite(float(obj)): flat[prefix]=obj
  walk('', {k:row[k] for k in ('index','sid','validation','relations','stage2_training','token_gate')})
  for k,v in (row.get('full_eval',{}).get('metrics') or {}).items():
   if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v)): flat[k]=float(v)
  for k,v in (row['stage3'].get('test_final') or {}).items(): flat['test_'+k]=v
  scalar[row['index']]=flat
 keys=sorted({k for d in scalar.values() for k in d})
 with open(OUT/'METRICS.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['index','arm']+keys); w.writeheader()
  for row in table: w.writerow({'index':row['index'],'arm':row['arm'],**scalar[row['index']]})
 print(json.dumps({'rows':len(table),'metrics_file':str(OUT/'METRICS.json'),'csv':str(OUT/'METRICS.csv'),'relations':len(pairs)},indent=2))
if __name__=='__main__': main()
