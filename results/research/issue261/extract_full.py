"""Reduce the large issue-specific full evaluations to numeric evidence."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'results/research/issue261'
SKIP={'per_item','per_item_l2','decoded_q','shards','prefix','prefix_sharing','codebook_occupancy','strict_prefix'}
def walk(prefix,obj,out):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if k in SKIP: continue
   walk(f'{prefix}.{k}' if prefix else k,v,out)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): walk(f'{prefix}.{i}',v,out)
 elif isinstance(obj,(int,float)) and not isinstance(obj,bool) and math.isfinite(float(obj)): out[prefix]=float(obj)
def main():
 out={}
 for p in sorted((OUT/'full_eval').glob('*.json')):
  arm=p.stem; d=json.load(open(p)); flat={}
  for key in ('coverage','quant','hard_q','continuous_z','calibration'):
   if key in d: walk(key,d[key],flat)
  out[arm]={'source':'issue261','file':str(p),'metrics':flat,'run_id':d.get('run_id'),'sid':d.get('sid'),'checkpoint':d.get('checkpoint')}
 (OUT/'FULL_METRICS.json').write_text(json.dumps(out,indent=2)+'\n')
 # csv
 keys=sorted({k for v in out.values() for k in v['metrics']})
 with open(OUT/'FULL_METRICS.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['arm']+keys); w.writeheader()
  for arm,v in out.items(): w.writerow({'arm':arm,**v['metrics']})
 print(json.dumps({'arms':list(out),'metrics':len(keys)},indent=2))
if __name__=='__main__': main()
