"""Append one Stage3 progress sample (read-only) to analysis/STAGE3_PROGRESS.json."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
run=Path(sys.argv[1]) if len(sys.argv)>1 else None
if run is None: raise SystemExit('usage: stage3_progress.py <stage3 run dir>')
rows=[json.loads(l) for l in (run/'training_metrics.jsonl').open()]
tr=[r for r in rows if r.get('event')=='train']
out=ROOT/'results/research/issue269/analysis/STAGE3_PROGRESS.json'
data=json.loads(out.read_text()) if out.is_file() else {'schema':'genrec.issue269.stage3_progress/1','samples':[]}
data['samples'].append({'epoch':tr[-1]['epoch'],'wall_time_s':tr[-1]['wall_time_s'],
    'epoch_train_time_s':tr[-1]['epoch_train_time_s'],'loss':tr[-1]['loss'],
    'lr':tr[-1]['lr']})
data['last']=data['samples'][-1]
out.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
print(json.dumps(data['last']))
