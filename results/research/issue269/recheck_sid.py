"""Independent recomputation of the exported SID numbers (issue C13 recheck)."""
import hashlib, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
run=Path(sys.argv[1]).resolve()
sid=json.loads((run/'item_sids.json').read_text())
trainer=run/'item_sids.trainer.json'
record=ROOT/'results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/stage2_metrics'/(
    hashlib.sha256((run/'item_sids.json').read_bytes()).hexdigest()+'.json')
def chain(payload):
    rows=[tuple(int(v) for v in payload[str(i)][:3]) for i in range(24587)]
    total=len(rows)
    def H(c):
        return -sum(k/total*math.log2(k/total) for k in c.values())
    return {'items':total,'distinct_l1':len({r[0] for r in rows}),'distinct_l2':len({r[:2] for r in rows}),
            'distinct_triples':len(set(rows)),'collision':1-len(set(rows))/total,
            'H_L1':H(__import__('collections').Counter(r[0] for r in rows)),
            'H_triple':H(__import__('collections').Counter(rows))}
out={'schema':'genrec.issue269.sid_recheck/1','sid':str(run/'item_sids.json'),
     'sid_sha256':hashlib.sha256((run/'item_sids.json').read_bytes()).hexdigest(),
     'recomputed':chain(sid)}
if trainer.is_file():
    out['trainer_sid_sha256']=hashlib.sha256(trainer.read_bytes()).hexdigest()
    out['trainer_recomputed']=chain(json.loads(trainer.read_text()))
if record.is_file():
    rec=json.loads(record.read_text())
    out['stored']=rec['catalogue']; out['stored_sid_sha256']=rec['sid']['sha256']
    pairs=(('collision','l3_collision'),('distinct_l2','distinct_l2_prefixes'),
           ('distinct_triples','distinct_l3_triples'))
    out['matches_stored']=all(abs(out['recomputed'][a]-rec['catalogue'][b])<1e-12 for a,b in pairs)
Path(ROOT/'results/research/issue269/analysis/SID_RECHECK.json').write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=1))
