"""Issue #275 — RUN_INVENTORY builder (analysis-only, FROZEN).

Assembles every Stage3 record, binds it to the Stage2 SID/checkpoint identity from the record's
own fields, joins the existing #270 cohort (family/binding), the 41 layer records, the MCP stage3
state files (quantisation error / effective rank / collision) and the #271 geometry assets.

Frozen rules (pre-registered in FEATURE_CONTRACT.md):
* A label unit is one SID sha256. All Stage3 records of that SID are kept.
* Primary label = mean test_recall@10 over the records at the SID's majority world_size;
  other-protocol records are retained as sensitivity. Noise = max-min over the used records.
* No value is fabricated: a missing feature stays null and is reported by the queue audit.
"""
import glob, json, hashlib, os
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue275'
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_RULES = [
    ('TIGER_RQ-VAE', 'tiger'), ('TIGER_BEHAVIOR', 'behavior'),
    ('TIGER_Uniform_KNN', 'knn'), ('TIGER_drq', 'drq'), ('TIGER_g4_closure', 'g4_closure'),
    ('TIGER_graphlet', 'graphlet'), ('TIGER_i234', 'i234'), ('TIGER_role_', 'role'),
    ('TIGER_sid_graphlet', 'sid_graphlet'), ('TIGER_af3', 'af3'), ('TIGER_mixed_rq', 'mixed'),
    ('TIGER_pairrank', 'pairrank'), ('TIGER_samecentre', 'samecentre'),
    ('TIGER_sinkhorn', 'sinkhorn'), ('TIGER_curv', 'curvature'), ('TIGER_euclid', 'euclid'),
    ('TIGER_calibrated_kappa', 'kappa'), ('TIGER_fixed_triplet_kappa', 'kappa'),
    ('TIGER_prefix_curvature', 'curvature'), ('TIGER_global_j', 'j_budget'),
    ('TIGER_balanced_j', 'j_budget'), ('TIGER_active_j', 'j_budget'),
    ('TIGER_module_budget', 'j_budget'), ('TIGER_periodic_budget', 'j_budget'),
    ('TIGER_codebook_budget', 'j_budget'), ('TIGER_role_geom', 'role_geom'),
    ('TIGER_behavior_edge', 'behavior_edge'), ('TIGER_trq', 'structural'),
    ('TIGER_p4_topoq', 'structural'), ('TIGER_export_matched', 'graphlet'),
    ('TIGER_same', 'samecentre'),
]

def family_of(arm: str) -> str:
    for pref, fam in FAMILY_RULES:
        if arm.startswith(pref):
            return fam
    return 'other'

def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()

# ---- Stage3 records
records = []
for f in glob.glob('results/stage3_T5Train/*/logs/Amazon_2023_Instruments/*/test_final.json'):
    try:
        d = json.load(open(f))
    except Exception as exc:
        print('bad', f, exc); continue
    parts = Path(f).parts
    arm = parts[2]
    sid_path = d.get('sid_path')
    if not sid_path:
        cand = Path('results/stage2_RQ-VAE')/arm/'item_sids.json'
        if cand.is_file():
            sid_path = str(cand)
        else:
            hits = sorted(glob.glob(f'results/stage2_RQ-VAE/_runs/*/{arm}/result/item_sids.json'))
            sid_path = hits[-1] if hits else None
    sid_sha = d.get('sid_sha256')
    if not sid_sha and sid_path and Path(sid_path).is_file():
        sid_sha = sha_file(sid_path)
    ckpt = d.get('stage2_checkpoint')
    ck_sha = d.get('stage2_checkpoint_sha256')
    if not ckpt and sid_path:
        c2 = Path(sid_path).parent/'rqvae_best.pth'
        if c2.is_file():
            ckpt = str(c2); ck_sha = ck_sha or sha_file(c2)
    elif ckpt and not ck_sha and Path(ckpt).is_file():
        ck_sha = sha_file(ckpt)
    records.append({
        'arm': arm, 'test_final': f,
        'test_final_sha256': sha_file(f),
        'recall@10': d.get('test_recall@10'), 'recall@5': d.get('test_recall@5'),
        'ndcg@10': d.get('test_ndcg@10'), 'n_eval': d.get('n_eval'),
        'sid': sid_path, 'sid_sha256': sid_sha,
        'checkpoint': ckpt, 'checkpoint_sha256': ck_sha,
        'stage2_run_id': d.get('stage2_run_id'), 'stage2_source_tree': d.get('stage2_source_tree'),
        'export_manifest': d.get('stage2_export_manifest'),
        'export_manifest_sha256': d.get('stage2_export_manifest_sha256'),
        'world_size': d.get('stage3_world_size'), 'epochs': d.get('stage3_completed_epochs'),
        'stage3_run': d.get('stage3_run_id'), 'trainer_sha256': d.get('stage3_trainer_sha256'),
        'archive_binding': d.get('archive_binding'),
        'record_verified': bool(sid_sha and ck_sha),
    })

by_sid = defaultdict(list)
for r in records:
    key = r['sid_sha256'] or f"UNBOUND::{r['arm']}::{r['test_final']}"
    by_sid[key].append(r)

units = []
for sid_sha, recs in by_sid.items():
    ws = Counter([r['world_size'] for r in recs if r['world_size'] is not None])
    major_ws = ws.most_common(1)[0][0] if ws else None
    used = [r for r in recs if major_ws is None or r['world_size'] == major_ws]
    vals = [r['recall@10'] for r in used if r['recall@10'] is not None]
    units.append({
        'sid_sha256': sid_sha if not sid_sha.startswith('UNBOUND') else None,
        'arms': sorted({r['arm'] for r in recs}),
        'family': family_of(recs[0]['arm']),
        'n_runs': len(recs), 'world_size_major': major_ws,
        'world_sizes': dict(ws),
        'label_recall@10': sum(vals) / len(vals) if vals else None,
        'label_ndcg@10': (lambda v: sum(v) / len(v) if v else None)(
            [r['ndcg@10'] for r in used if r['ndcg@10'] is not None]),
        'noise_range': (max(vals) - min(vals)) if len(vals) > 1 else 0.0,
        'records': recs,
    })

# ---- R2 ledger (name-based binding channel: variant/stage2_rel -> sid sha + r2 id)
ledger = {}
try:
    raw = json.load(open('results/research/issue139/r2_ledger.json'))
    for rid, e in raw.items():
        pair = e.get('pair') or {}
        for nm in {pair.get('variant'), pair.get('stage2_rel')}:
            if nm:
                ledger.setdefault(nm, {'r2_id': rid, 'sid_sha256': pair.get('sid_sha256'),
                                       'sid_file_sha256': (e.get('files') or {}).get('item_sids.json'),
                                       'commit_tier': pair.get('commit_tier'),
                                       'target': e.get('target')})
except Exception as exc:
    print('ledger unavailable', exc)

# ---- #270 cohort (family/binding hints)
cohort = {}
try:
    c = json.load(open('results/research/issue270/COHORT.json'))
    for row in c.get('rows', []):
        if row.get('sid_sha256'):
            cohort[row['sid_sha256']] = row
except Exception as exc:
    print('cohort unavailable', exc)

# ---- 41 layer records
layer = {}
for f in glob.glob('results/stage2_RQ-VAE/**/stage2_metrics/*.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    sha = os.path.basename(f)[:-5]
    if sha not in layer:
        layer[sha] = {'path': f, 'sha256': sha_file(f), 'metrics': d.get('metrics', {})}

# ---- MCP stage3 state (sid_quality: collision, quant_error, effective_rank)
state = {}
for f in glob.glob(str(Path.home()/'.cache/genrec-mcp/stage3__*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('sid'):
        state[d['sid']] = {'collision': d.get('collision'),
                           'collision_gate_waived': d.get('collision_gate_waived'),
                           'quant_error': (d.get('sid_quality') or {}).get('quant_error'),
                           'effective_rank': (d.get('sid_quality') or {}).get('effective_rank'),
                           'sid_quality_reason': (d.get('sid_quality') or {}).get('reason')}
state_by_sha = {}
for sid_path, v in state.items():
    if Path(sid_path).is_file():
        state_by_sha[sha_file(sid_path)] = v

# ---- geometry assets (#271)
geo = {}
for f in glob.glob('results/research/issue271/assets/geometry/*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('sid_sha256'):
        geo[d['sid_sha256']] = {
            'path': f, 'sha256': sha_file(f),
            'mean_abs_error': d.get('mean_abs_error'), 'mean_error': d.get('mean_error'),
            'spearman': d.get('spearman'), 'sign_accuracy': d.get('sign_accuracy'),
            'abs_better_than_zero': d.get('abs_better_than_zero'),
            'binding': d.get('binding'), 'record_verified': d.get('record_verified'),
        }

# ---- Stage2 final recon from the run metrics (by stage2_run_id/arm)
def final_recon(run_id, arm):
    pats = []
    if run_id:
        pats.append(f'results/stage2_RQ-VAE/_runs/{run_id}/{arm}/result/logs/training_metrics.jsonl')
    pats.append(f'results/stage2_RQ-VAE/{arm}/logs/*/training_metrics.jsonl')
    pats.append(f'results/stage2_RQ-VAE/{arm}/logs/training_metrics.jsonl')
    pats.extend(sorted(glob.glob(f'results/stage2_RQ-VAE/_runs/*/{arm}/result/logs/training_metrics.jsonl')))
    for p in pats:
        for f in glob.glob(p):
            last = None
            try:
                for line in open(f):
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get('event') == 'train' and e.get('recon') is not None:
                        last = e
            except Exception:
                continue
            if last:
                return {'recon': last.get('recon'), 'epoch': last.get('epoch'),
                        'metrics_path': f, 'metrics_sha256': sha_file(f)}
    return None

for u in units:
    sha = u['sid_sha256']
    r0 = u['records'][0]
    if not sha:
        # name-based R2 binding for legacy records
        for nm in {r0['arm']}:
            if nm in ledger:
                l = ledger[nm]
                u['sid_sha256'] = l['sid_sha256']; u['binding_channel'] = 'r2_ledger'
                u['r2'] = l
                sha = l['sid_sha256']
                for r in u['records']:
                    if not r['sid_sha256']:
                        r['sid_sha256'] = l['sid_sha256']; r['binding_channel'] = 'r2_ledger'
                break
        if not sha:
            u['binding_channel'] = 'missing'
    else:
        u['binding_channel'] = 'stage3_identity'
    if 'r2' not in u and r0['arm'] in ledger:
        u['r2'] = ledger[r0['arm']]
    u['layer'] = layer.get(sha)
    u['stage3_state'] = state_by_sha.get(sha)
    u['geometry'] = geo.get(sha)
    u['cohort'] = cohort.get(sha)
    u['stage2_recon'] = final_recon(r0.get('stage2_run_id'), r0['arm'])

# ---- coverage audit
def cov(key):
    return sum(1 for u in units if u.get(key))

audit = {
    'n_stage3_records': len(records),
    'n_label_units': len(units),
    'unique_sids': sum(1 for u in units if u['sid_sha256']),
    'unbound_units': sum(1 for u in units if not u['sid_sha256']),
    'families': dict(Counter(u['family'] for u in units)),
    'world_sizes': dict(Counter(str(u['world_size_major']) for u in units)),
    'coverage': {
        'layer_record': cov('layer'),
        'stage3_state_quant_rank': cov('stage3_state'),
        'geometry_271': cov('geometry'),
        'stage2_final_recon': cov('stage2_recon'),
    },
    'binding_channels': dict(Counter(u.get('binding_channel','missing') for u in units)),
    'r2_bound_units': sum(1 for u in units if u.get('r2')),
    'records_verified_identity': sum(1 for r in records if r['record_verified']),
    'records_missing_identity': sum(1 for r in records if not r['record_verified']),
}
inventory = {'schema': 'genrec.issue275.inventory/1', 'audit': audit,
             'units': units, 'records': records}
(OUT / 'RUN_INVENTORY.json').write_text(json.dumps(inventory, indent=1))
print(json.dumps(audit, indent=1))
