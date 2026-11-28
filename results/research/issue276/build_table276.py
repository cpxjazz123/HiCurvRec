"""Issue #276 — build the unified SID table (SID-only, vector, geometry) from stored evidence.

Joins, per SID content sha256 / canonical mapping:
  features  : results/research/issue275/features/layer/*.json and issue276/features/layer/*.json
  labels    : #275 RUN_INVENTORY units and records, git recovery binds, stored reports
              (`sid_utilization/vector_stats_all.json`, `sid_utilization/metrics_corrected.json`)
  vector    : stored pr / quant_err / sep / distinct_ratio (provenance = stored report) and
              metrics_corrected quant_error / effective_rank / collision by arm
  geometry  : results/research/issue271/assets/geometry/*.json keyed by sid_sha256

Every join keeps its provenance path; nothing is imputed. Outputs SID_TABLE.json + TABLE_MANIFEST.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

P = Path('results/research/issue276')
F_LAYER_DIRS = [Path('results/research/issue275/features/layer'), P / 'features/layer']


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def arm_key(s: str) -> str:
    return str(s).lower().replace('tiger_', '').replace('/', '__')


def arm_match(a: str, b: str) -> bool:
    ka, kb = arm_key(a), arm_key(b)
    return ka == kb or ka.endswith('__' + kb) or kb.endswith('__' + ka)


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    # ---- features ----
    feats: dict[str, dict] = {}
    for d in F_LAYER_DIRS:
        for f in sorted(d.glob('*.json')):
            rec = load(f)
            if not rec:
                continue
            sha = rec.get('sid', {}).get('sha256') or f.stem
            rec['_feature_path'] = str(f)
            feats.setdefault(sha, rec)

    # ---- labels: inventory units (per SID) ----
    inv = load('results/research/issue275/RUN_INVENTORY.json') or {}
    unit_by_sha = {}
    for u in inv.get('units', []):
        if u.get('sid_sha256'):
            unit_by_sha[u['sid_sha256']] = u

    # ---- labels/binds from git recovery ----
    git = load(P / 'recovery/git_sids.json') or {}
    bind_by_sha: dict[str, list] = {}
    for arm in git.get('arms', []):
        for b in arm.get('binds', []):
            if not b.get('sid'):
                continue
            bind_by_sha.setdefault(b['sid'], []).append({
                'arm': arm['arm'], 'recall': b['record'].get('recall'),
                'date': b['record'].get('date'), 'test_final': b['record'].get('test_final'),
                'identity': b.get('identity'), 'source': 'git_bind'})

    # ---- stored reports ----
    vstats = load('results/research/sid_utilization/vector_stats_all.json') or []
    v_by_sha = {}
    for r in vstats:
        sp = r.get('sid_path')
        if sp and Path(sp).is_file():
            v_by_sha[sha_file(Path(sp))] = {**r, '_source': 'results/research/sid_utilization/vector_stats_all.json'}
    corrected = {r['arm']: r for r in (load('results/research/sid_utilization/metrics_corrected.json') or [])}

    # ---- ledger (L1 identity) and corrected metrics (L2 identity) ----
    ledger = load('results/research/issue139/r2_ledger.json') or {}
    ledger_sids = {}
    for _num, e in ledger.items():
        if not isinstance(e, dict):
            continue
        files = e.get('files') or {}
        arm_l = e.get('arm') or e.get('key') or ''
        sid_sha = files.get('item_sids.json') if isinstance(files, dict) else None
        if arm_l and sid_sha:
            ledger_sids[str(arm_l)] = sid_sha

    def identity_of(arm: str, sha: str, calc_collision):
        for la, ls in ledger_sids.items():
            if arm_match(arm, la) and ls == sha:
                return 'L1_ledger'
        for ca, ce in corrected.items():
            if arm_match(arm, ca) and isinstance(ce.get('collision'), (int, float)) \
                    and calc_collision is not None and abs(ce['collision'] - calc_collision) < 1e-12:
                return 'L2_collision'
        return 'L3_era_name'

    # ---- corrected-forward vector series ----
    vf_raw = load(P / 'vector_forward.json') or {}
    vf = {r['sid_sha256']: r for r in vf_raw.get('rows', [])
          if r.get('status') == 'ok' and r.get('sid_sha256')}

    # ---- geometry ----
    geo = {}
    for f in glob.glob('results/research/issue271/assets/geometry/*.json'):
        d = load(f)
        if d and d.get('sid_sha256'):
            geo[d['sid_sha256']] = {'mean_abs_error': d.get('mean_abs_error'),
                                    'spearman': d.get('spearman'),
                                    'sign_accuracy': d.get('sign_accuracy'), '_source': f}

    rows = []
    for sha, rec in feats.items():
        u = unit_by_sha.get(sha)
        binds = bind_by_sha.get(sha, [])
        vs = v_by_sha.get(sha)
        cat = rec.get('catalogue') or {}
        val = rec.get('validation') or {}
        arms = []
        if u:
            arms = [{'arm': a, 'source': 'inventory_unit', 'identity': u.get('binding_channel')}
                    for a in u.get('arms', [])]
        calc_coll = (rec.get('catalogue') or {}).get('l3_collision')
        for b in binds:
            ident = identity_of(b['arm'], sha, calc_coll)
            b = {**b, 'identity': ident}
            arms.append({'arm': b['arm'], 'source': 'git_bind', 'identity': ident})
        if vs:
            arms.append({'arm': vs.get('arm'), 'source': 'vector_stats_all', 'identity': 'sid_path_sha'})
        # label resolution: inventory unit first, then git bind, then stored report
        label, label_sources, strength = None, [], None
        if u and isinstance(u.get('label_recall@10'), (int, float)):
            label = u['label_recall@10']
            strength = 'primary'
            label_sources.append({'source': 'inventory_unit', 'recall': label,
                                  'n_runs': u.get('n_runs'), 'spread': u.get('noise_range')})
        binds_eff = [{**b, 'identity': identity_of(b['arm'], sha, (rec.get('catalogue') or {}).get('l3_collision'))}
                     for b in binds]
        primary = [b for b in binds_eff if b.get('identity') in ('L1_ledger', 'L2_collision')
                   and isinstance(b.get('recall'), (int, float))]
        for b in binds_eff:
            if isinstance(b.get('recall'), (int, float)):
                label_sources.append({'source': 'git_bind/' + str(b['arm']), 'recall': b['recall'],
                                      'date': b.get('date'), 'identity': b.get('identity')})
        if label is None and primary:
            label = float(sum(b['recall'] for b in primary) / len(primary))
            strength = 'primary'
            label_sources.append({'source': 'git_bind_mean_L1L2', 'recall': label,
                                  'n': len(primary),
                                  'spread': max(b['recall'] for b in primary)
                                  - min(b['recall'] for b in primary)})
        if vs and isinstance(vs.get('recall'), (int, float)):
            label_sources.append({'source': 'vector_stats_all', 'recall': float(vs['recall'])})
            if label is None:
                label = float(vs['recall'])
                strength = 'secondary'
        # corrected-metrics row whose stored collision equals this SID's computed collision:
        # a content-verified (L2) arm -> label binding is primary, not name-based.
        corr, corr_arm, coll_verified = None, None, False
        for cand_arm in [a['arm'] for a in arms]:
            for ca, ce in corrected.items():
                if arm_match(cand_arm, ca):
                    corr, corr_arm = ce, ca
                    if isinstance(ce.get('collision'), (int, float)) and calc_coll is not None \
                            and abs(ce['collision'] - calc_coll) < 1e-12:
                        coll_verified = True
                    break
            if corr is not None:
                break
        if corr is not None and coll_verified:
            arms.append({'arm': corr_arm, 'source': 'metrics_corrected', 'identity': 'L2_collision'})
        if corr and isinstance(corr.get('recall'), (int, float)):
            label_sources.append({'source': 'metrics_corrected', 'recall': float(corr['recall']),
                                  'arm': corr_arm, 'collision': corr.get('collision'),
                                  'collision_verified': coll_verified})
            if label is None:
                label = float(corr['recall'])
                strength = 'primary' if coll_verified else 'secondary'
            elif coll_verified and strength != 'primary':
                strength = 'primary'
        rows.append({
            'sid_sha256': sha,
            'mapping_sha256': rec.get('mapping_sha256'),
            'arms': arms,
            'label_recall@10': label,
            'label_strength': strength,
            'label_sources': label_sources,
            'features': {
                'l1_pos_hit': val.get('l1_positive_hit_rate'),
                'l1_neg_hit': val.get('l1_negative_hit_probability'),
                'l1_hard_er': val.get('l1_hard_er'),
                'l2_pos_hit': (val.get('positive_hit_rates_by_level') or [None, None])[1],
                'l2_neg_hit': (val.get('negative_hit_probabilities_by_level') or [None, None])[1],
                'l2_distinct': cat.get('distinct_l2_prefixes'),
                'l3_collision': cat.get('l3_collision'),
                'l3_refine_bits': cat.get('l3_refinement_gain_bits'),
                'distinct_l3_ratio': (cat.get('distinct_l3_triples') or 0) / cat.get('items')
                                     if cat.get('items') else None,
                'prefix_entropy_bits': cat.get('prefix_entropy_bits'),
                'codebook_occupancy': cat.get('codebook_occupancy'),
                '_source': rec.get('_feature_path'),
                '_format_normalized': rec.get('format_normalized_from'),
            },
            'vector_forward': vf.get(sha),
            'vector_stored': ({'pr_z': vs.get('pr'), 'quant_err': vs.get('quant_err'),
                        'sep': vs.get('sep'), 'distinct_ratio': vs.get('distinct_ratio'),
                        'between': vs.get('between'), 'within': vs.get('within'),
                        '_source': vs.get('_source'), '_ckpt': vs.get('ckpt')}
                       if vs else None),
            'vector_arm': ({'quant_error': corr.get('quant_error'),
                            'effective_rank': corr.get('effective_rank'),
                            'collision': corr.get('collision'),
                            '_source': 'results/research/sid_utilization/metrics_corrected.json'}
                           if corr else None),
            'geometry': geo.get(sha),
        })

    # cross-check: stored collision == computed l3_collision (L2 identity evidence)
    for r in rows:
        c = (r['vector_arm'] or {}).get('collision')
        calc = r['features'].get('l3_collision')
        r['collision_check'] = (None if c is None or calc is None else abs(c - calc) < 1e-12)

    # group by canonical mapping when known (near-duplicate merge), else by content sha
    groups: dict[str, list] = {}
    for r in rows:
        key = r['mapping_sha256'] or r['sid_sha256']
        groups.setdefault(key, []).append(r)

    SID = {
        'schema': 'genrec.issue276.sid_table/1',
        'feature_definition': {
            'l1_pos_hit': 'tools.issue252.layer_metrics.validation_metrics -> l1_positive_hit_rate',
            'l1_neg_hit': '-> l1_negative_hit_probability',
            'l1_hard_er': '-> l1_hard_er',
            'l2_pos_hit': '-> positive_hit_rates_by_level[1]',
            'l2_neg_hit': '-> negative_hit_probabilities_by_level[1]',
            'l2_distinct': 'catalogue_metrics -> distinct_l2_prefixes',
            'l3_collision': 'catalogue_metrics -> l3_collision (1 - distinct triples / items)',
            'l3_refine_bits': 'catalogue_metrics -> l3_refinement_gain_bits',
            'pr_z': 'vector_stats_all: participation ratio of the covariance of encoder outputs z',
            'quant_err': 'vector_stats_all: mean ||z - sum(selected codewords)||',
            'sep': 'vector_stats_all: mean between-item distance / mean within-SID distance',
            'distinct_ratio': 'vector_stats_all: distinct 3-level SIDs / items',
            'geometry': 'issue271 assets: curvature MAE / Spearman / sign accuracy (fixed scale)',
        },
        'groups': [{'group': k, 'rows': v} for k, v in sorted(groups.items())],
        'n_rows': len(rows),
    }
    P.mkdir(parents=True, exist_ok=True)
    (P / 'SID_TABLE.json').write_text(json.dumps(SID, indent=1) + '\n')

    lab = [r for r in rows if isinstance(r['label_recall@10'], (int, float))]
    hi = [r for r in lab if r['label_recall@10'] > 0.06]
    manifest = {
        'schema': 'genrec.issue276.table_manifest/1',
        'n_sid_rows': len(rows),
        'n_with_label': len(lab),
        'n_high_recall': len(hi),
        'n_label_primary': sum(1 for r in rows if r.get('label_strength') == 'primary'),
        'n_label_secondary': sum(1 for r in rows if r.get('label_strength') == 'secondary'),
        'n_high_recall_primary': sum(1 for r in rows
                                     if isinstance(r['label_recall@10'], (int, float))
                                     and r['label_recall@10'] > 0.06
                                     and r.get('label_strength') == 'primary'),
        'n_with_vector': sum(1 for r in rows if r['vector_forward']),
        'n_with_vector_stored': sum(1 for r in rows if r['vector_stored']),
        'n_with_vector_arm': sum(1 for r in rows if r['vector_arm']),
        'n_with_geometry': sum(1 for r in rows if r['geometry']),
        'n_groups': len(groups),
        'collision_check_true': sum(1 for r in rows if r['collision_check'] is True),
        'collision_check_false': sum(1 for r in rows if r['collision_check'] is False),
        'label_source_counts': {s: sum(1 for r in rows
                                       for x in r['label_sources'] if x['source'].split('/')[0] == s)
                                for s in ('inventory_unit', 'git_bind', 'vector_stats_all',
                                          'metrics_corrected')},
        'inputs': {
            'features_275': 'results/research/issue275/features/layer/',
            'features_276': 'results/research/issue276/features/layer/',
            'inventory': 'results/research/issue275/RUN_INVENTORY.json',
            'git_recovery': 'results/research/issue276/recovery/git_sids.json',
            'vector_stats_all': 'results/research/sid_utilization/vector_stats_all.json',
            'metrics_corrected': 'results/research/sid_utilization/metrics_corrected.json',
            'geometry': 'results/research/issue271/assets/geometry/',
        },
    }
    (P / 'TABLE_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
