"""Issue #275 — joint feature/label table (analysis-only).

Joins, per SID sha256: Stage3 label (mean recall@10 at the majority world_size), SID layer features
(frozen evaluator), quantisation error / effective rank / collision (frozen sid_quality cache),
Stage2 final reconstruction (run metrics), geometry (#271 assets). Missing stays null.
"""
import glob, json, os, re
from pathlib import Path
from collections import Counter

REPO = Path('.').resolve()
P = REPO/'results/research/issue275'
inv = json.load(open(P/'RUN_INVENTORY.json'))

def family_of(arm):
    rules = [('TIGER_RQ-VAE','tiger'),('TIGER_BEHAVIOR','behavior'),('TIGER_Uniform_KNN','knn'),
             ('TIGER_drq','drq'),('TIGER_g4_closure','g4_closure'),('TIGER_graphlet','graphlet'),
             ('TIGER_i234','i234'),('TIGER_role_geom','role_geom'),('TIGER_role_','role'),
             ('TIGER_sid_graphlet','sid_graphlet'),('TIGER_af3','af3'),('TIGER_mixed_rq','mixed'),
             ('TIGER_pairrank','pairrank'),('TIGER_samecentre','samecentre'),('TIGER_sinkhorn','sinkhorn'),
             ('TIGER_curv','curvature'),('TIGER_euclid','euclid'),('TIGER_calibrated_kappa','kappa'),
             ('TIGER_fixed_triplet_kappa','kappa'),('TIGER_global_j','j_budget'),('TIGER_balanced_j','j_budget'),
             ('TIGER_active_j','j_budget'),('TIGER_module_budget','j_budget'),('TIGER_periodic_budget','j_budget'),
             ('TIGER_codebook_budget','j_budget'),('TIGER_behavior_edge','behavior_edge'),('TIGER_trq','structural'),
             ('TIGER_p4_topoq','structural'),('TIGER_export_matched','graphlet')]
    for pref, fam in rules:
        if arm.startswith(pref): return fam
    return 'other'

def date_of(rec):
    s = rec.get('stage3_run') or rec.get('test_final') or ''
    m = re.search(r'([A-Z][a-z]{2})-(\d{2})-(\d{4})', s)
    if m:
        mon = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}[m.group(1)]
        return f'{m.group(3)}-{mon:02d}'
    return None

layer = {}
for f in glob.glob(str(P/'features/layer/*.json')):
    d = json.load(open(f)); layer[os.path.basename(f)[:-5]] = d
state = {}
for f in glob.glob(str(Path.home()/'.cache/genrec-mcp/stage3__*.json')):
    d = json.load(open(f)); sid = d.get('sid')
    if sid and Path(sid).is_file():
        import hashlib
        h = hashlib.sha256(Path(sid).read_bytes()).hexdigest()
        state[h] = {'collision': d.get('collision'),
                    'quant_error': (d.get('sid_quality') or {}).get('quant_error'),
                    'effective_rank': (d.get('sid_quality') or {}).get('effective_rank')}
geo = {}
for f in glob.glob('results/research/issue271/assets/geometry/*.json'):
    d = json.load(open(f))
    if d.get('sid_sha256'):
        geo[d['sid_sha256']] = {'mean_abs_error': d.get('mean_abs_error'), 'spearman': d.get('spearman'),
                                'sign_accuracy': d.get('sign_accuracy'), 'path': f}

rows = []
for u in inv['units']:
    sha = u.get('sid_sha256')
    if not sha: continue
    arm = u['arms'][0]
    rec = u['records'][0]
    lay = (layer.get(sha) or {})
    cat = lay.get('catalogue') or {}
    val = lay.get('validation') or {}
    st = state.get(sha) or {}
    g = geo.get(sha) or {}
    rows.append({
        'sid_sha256': sha, 'arm': arm, 'family': family_of(arm),
        'label_recall@10': u.get('label_recall@10'), 'label_ndcg@10': u.get('label_ndcg@10'),
        'noise_range': u.get('noise_range'), 'n_runs': u.get('n_runs'),
        'world_size_major': u.get('world_size_major'), 'binding_channel': u.get('binding_channel'),
        'r2_id': (u.get('r2') or {}).get('r2_id'), 'date': date_of(rec),
        # SID core features (frozen evaluator)
        'collision3': cat.get('l3_collision'),
        'l3_collision': cat.get('l3_collision'),
        'l2_distinct': cat.get('distinct_l2_prefixes'),
        'l3_distinct': cat.get('distinct_l3_triples'),
        'l3_refine_bits': cat.get('l3_refinement_gain_bits'),
        'occupancy': cat.get('codebook_occupancy'),
        'l1_pos_hit': val.get('valid_l1_positive_hit_rate') if val.get('valid_l1_positive_hit_rate') is not None else val.get('l1_positive_hit_rate'),
        'l1_neg_hit': val.get('valid_l1_negative_hit_rate') if val.get('valid_l1_negative_hit_rate') is not None else val.get('l1_negative_hit_probability'),
        'l1_hard_er': val.get('valid_l1_hard_er') if val.get('valid_l1_hard_er') is not None else val.get('l1_hard_er'),
        # vector features (frozen sid_quality cache)
        'collision_gate': st.get('collision'), 'quant_error': st.get('quant_error'),
        'effective_rank': st.get('effective_rank'),
        # reconstruction (run metrics)
        'recon': (u.get('stage2_recon') or {}).get('recon'),
        # geometry (#271)
        'geo_mae': g.get('mean_abs_error'), 'geo_spearman': g.get('spearman'),
        'geo_sign_acc': g.get('sign_accuracy'),
    })
# dedupe by sid (keep first)
seen=set(); out=[]
for r in rows:
    if r['sid_sha256'] in seen: continue
    seen.add(r['sid_sha256']); out.append(r)
audit = {'rows': len(out),
         'with_label': sum(1 for r in out if r['label_recall@10'] is not None),
         'with_layer': sum(1 for r in out if r['l1_pos_hit'] is not None),
         'with_quant_rank': sum(1 for r in out if r['quant_error'] is not None),
         'with_recon': sum(1 for r in out if r['recon'] is not None),
         'with_geometry': sum(1 for r in out if r['geo_mae'] is not None),
         'families': dict(Counter(r['family'] for r in out)),
         'protocols': dict(Counter(str(r['world_size_major']) for r in out)),
         'dates': dict(Counter(str(r['date']) for r in out)),
         'label_stats': {'min': min(r['label_recall@10'] for r in out if r['label_recall@10'] is not None),
                         'max': max(r['label_recall@10'] for r in out if r['label_recall@10'] is not None)}}
joint = {'schema':'genrec.issue275.joint/1','audit':audit,'rows':out}
(P/'JOINT_TABLE.json').write_text(json.dumps(joint, indent=1))
print(json.dumps(audit, indent=1))
