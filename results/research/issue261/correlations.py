"""Issue #261: flatten the metric matrix and compute exploratory associations."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'results/research/issue261'
sys.path.insert(0,str(OUT))
from stats import pearson,spearman,fisher_ci,p_value_pearson,leave_one_out

TESTS=['test_recall@5','test_recall@10','test_ndcg@5','test_ndcg@10']

def scalar_leaves(prefix,obj,out):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if k in ('sid_path','predictions','path','source','reason','scope','representation','metric','curvature_definition','calibration','coverage','shards','strict_heldout','strict_prefix','hard_q','continuous_z','full_eval','stage3','validation','relations','stage2_training'): continue
   scalar_leaves(f'{prefix}.{k}' if prefix else k,v,out)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): scalar_leaves(f'{prefix}.{i}' if prefix else str(i),v,out)
 elif isinstance(obj,(int,float)) and not isinstance(obj,bool):
  if obj is not None and math.isfinite(float(obj)): out[prefix]=float(obj)

def normalized_full(arm,fe):
 if not isinstance(fe,dict): return {}
 out={}
 if fe.get('source')=='full_metrics' and isinstance(fe.get('metrics'),dict):
  return dict(fe['metrics'])
 if fe.get('source')=='issue261':
  h=fe.get('hard') or {}; c=fe.get('continuous') or {}
 elif fe.get('source')=='issue252':
  h=fe.get('hard_q') or {}; c=fe.get('continuous_z') or {}
 elif fe.get('source')=='issue253':
  g=fe.get('hard_q') or {}; h=(g.get('cells') or {}).get('common_fixed3') or {}; c=(fe.get('continuous_z') or {}).get('cells',{}).get('common_fixed3') or {}
 else:
  h={}; c={}
 for label,block in (('hard',h),('continuous',c)):
  if not isinstance(block,dict): continue
  for key in ('curvature','distance_distortion','reconstruction','strict_heldout'):
   if key in block:
    scalar_leaves(f'full.{label}.{key}',block[key],out)
  if 'curvature' in block:
   cur=block['curvature']
   for k in ('macro_mae','micro_mae','count','excluded'):
    if isinstance(cur.get(k),(int,float)): out[f'full.{label}.curvature.{k}']=float(cur[k])
   for sign,v in (cur.get('strata') or {}).items():
    if isinstance(v,dict) and isinstance(v.get('macro_mae'),(int,float)): out[f'full.{label}.curvature.strata_{sign}_macro_mae']=float(v['macro_mae'])
   sign=cur.get('sign') or {}
   if isinstance(sign.get('accuracy'),(int,float)): out[f'full.{label}.curvature.sign_accuracy']=float(sign['accuracy'])
  if isinstance(block.get('reconstruction'),dict):
   for k in ('mean_squared_error','mean_l2_error'):
    if isinstance(block['reconstruction'].get(k),(int,float)): out[f'full.{label}.reconstruction_{k}']=float(block['reconstruction'][k])
 if 'quant' in fe and isinstance(fe['quant'],dict):
  scalar_leaves('quant',fe['quant'],out)
 return out

def matrix():
 d=json.load(open(OUT/'METRICS.json'))
 rows=[]
 for r in d['rows']:
  flat={}
  scalar_leaves('sid',r['sid'],flat); scalar_leaves('validation',r['validation'],flat); scalar_leaves('relations',r['relations'],flat); scalar_leaves('training',r['stage2_training'],flat); scalar_leaves('token_gate',r.get('token_gate',{}),flat)
  # explicit test targets
  for k,v in (r['stage3'].get('test_final') or {}).items():
   if isinstance(v,(int,float)) and math.isfinite(float(v)): flat[k]=float(v)
  flat.update(normalized_full(r['arm'],r.get('full_eval')))
  rows.append({'index':r['index'],'arm':r['arm'],'values':flat})
 return rows

def corr(x,y):
 return {'n':sum(1 for a,b in zip(x,y) if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))),'pearson':pearson(x,y),'spearman':spearman(x,y),'ci':fisher_ci(pearson(x,y),sum(1 for a,b in zip(x,y) if a is not None and b is not None)),'p':p_value_pearson(pearson(x,y),sum(1 for a,b in zip(x,y) if a is not None and b is not None))}

def pairs(rows,key,target):
 return [(r['values'].get(key),r['values'].get(target)) for r in rows if key in r['values'] and target in r['values']]

def main():
 rows=matrix(); predictors=sorted({k for r in rows for k in r['values'] if k not in TESTS})
 corr_rows=[]; detail={}; sens=[]
 for pred in predictors:
  for target in TESTS:
   ps=pairs(rows,pred,target); x=[a for a,b in ps]; y=[b for a,b in ps]
   c=corr(x,y)
   corr_rows.append({'predictor':pred,'target':target,**{k:c[k] for k in ('n','pearson','spearman','p')},'ci_low':c['ci'][0],'ci_high':c['ci'][1]})
   detail[f'{pred}::{target}']={'series':[{'index':r['index'],'arm':r['arm'],'x':r['values'].get(pred),'y':r['values'].get(target)} for r in rows]}
   for variant,sel in (('all',lambda r:True),('exclude_107',lambda r:r['index']!=107),('exclude_100',lambda r:r['index']!=100)):
    ps2=[(r['values'].get(pred),r['values'].get(target)) for r in rows if sel(r) and pred in r['values'] and target in r['values']]
    cc=corr([a for a,b in ps2],[b for a,b in ps2]); loo=leave_one_out([a for a,b in ps2],[b for a,b in ps2])
    sens.append({'predictor':pred,'target':target,'variant':variant,'n':cc['n'],'pearson':cc['pearson'],'spearman':cc['spearman'],'loo_low':loo[0] if loo else None,'loo_high':loo[1] if loo else None})
 (OUT/'CORRELATIONS.json').write_text(json.dumps({'schema':'genrec.issue261.correlations/1','tests':TESTS,'correlations':corr_rows,'series_note':'raw arm-level predictor/target values are in METRICS.csv and METRICS.json; full series omitted to keep this evidence compact'},separators=(',',':'))+'\n')
 with open(OUT/'CORRELATIONS.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['predictor','target','n','pearson','spearman','p','ci_low','ci_high']); w.writeheader(); w.writerows(corr_rows)
 with open(OUT/'SENSITIVITY.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['predictor','target','variant','n','pearson','spearman','loo_low','loo_high']); w.writeheader(); w.writerows(sens)
 print(json.dumps({'predictors':len(predictors),'correlations':len(corr_rows),'rows':len(rows)},indent=2))
if __name__=='__main__': main()
