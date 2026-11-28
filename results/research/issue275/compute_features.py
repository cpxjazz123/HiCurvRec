"""Issue #275 — recompute the frozen SID layer features for every bound SID.

Reuses the frozen evaluator functions (`tools.issue252.layer_metrics.catalogue_metrics` and
`.validation_metrics`) and the frozen SID loader (`tools.issue252.final_evaluation.load_sid`);
writes one JSON per SID under results/research/issue275/features/layer/<sid_sha>.json.
No training, no checkpoint forward, no Stage3 score is read by the evaluator.
"""
import hashlib, json, sys
from pathlib import Path
import pyarrow.parquet as pq

REPO = Path('.').resolve()
sys.path.insert(0, str(REPO))
from tools.issue252 import layer_metrics as lm
from tools.issue252.final_evaluation import load_sid

OUT = REPO/'results/research/issue275/features/layer'
OUT.mkdir(parents=True, exist_ok=True)
inv = json.load(open('results/research/issue275/RUN_INVENTORY.json'))
valid = REPO/'results/stage0_build_parquet'/'valid.parquet'
train = REPO/'results/stage0_build_parquet'/'train.parquet'

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

examples = pq.read_table(valid, columns=['history','seen_history','target']).to_pylist()
seen = set()
rows_out = []
for u in inv['units']:
    sid_sha = u.get('sid_sha256')
    if not sid_sha or sid_sha in seen: continue
    seen.add(sid_sha)
    sid_paths = [r['sid'] for r in u['records'] if r.get('sid')]
    sid_path = next((p for p in sid_paths if p and Path(p).is_file()), None)
    if sid_path is None:
        rows_out.append({'sid_sha256': sid_sha, 'status': 'no_sid_file'}); continue
    out = OUT/f'{sid_sha}.json'
    if out.exists():
        rows_out.append({'sid_sha256': sid_sha, 'status': 'reused', 'path': str(out)}); continue
    try:
        rows = load_sid(Path(sid_path), 24587, (256,256,256))
        rec = {'schema':'genrec.issue275.layer_features/1',
               'sid':{'path':str(sid_path),'sha256':sha(sid_path)},
               'catalogue': lm.catalogue_metrics(rows),
               'validation': lm.validation_metrics(rows, examples),
               'evaluator':{'path':str(Path(lm.__file__)),'sha256':sha(lm.__file__)},
               'inputs':{'valid':{'path':str(valid),'sha256':sha(valid)},
                         'train':{'path':str(train),'sha256':sha(train)}}}
        out.write_text(json.dumps(rec, indent=1, allow_nan=False)+'\n')
        rows_out.append({'sid_sha256': sid_sha, 'status': 'computed', 'path': str(out),
                         'l1_pos': rec['validation'].get('valid_l1_positive_hit_rate'),
                         'l2': rec['catalogue'].get('distinct_l2_prefixes')})
    except Exception as exc:
        rows_out.append({'sid_sha256': sid_sha, 'status': f'error: {exc!r}'[:200]})
    if len(rows_out) % 10 == 0: print(len(rows_out), 'done', flush=True)
json.dump({'schema':'genrec.issue275.layer_features_run/1','rows':rows_out},
          open('results/research/issue275/features/layer_run.json','w'), indent=1)
from collections import Counter
print(Counter(r['status'].split(':')[0] for r in rows_out))
