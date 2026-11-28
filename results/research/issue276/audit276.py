"""Issue #276 — recovery audit (C1): per-arm identity evidence for the 12 named high-recall arms and
the full high-recall record set. Reads stored evidence only; writes recovery/AUDIT.json + AUDIT.md.
"""
from __future__ import annotations

import json
from pathlib import Path

P = Path('results/research/issue276')
NAMED = ['r48k2', 'r48k1', 'r48k2u', 'BEHAVIOR_R48', 'behavior_residual_geometry_transport',
         'crowd_couser_w3', 'crowd_posattract_w3', 'resample', 'resample_fn_mask',
         'resample_l1only', 'resample_s43', 'resample_s44']


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main() -> None:
    table = load(P / 'SID_TABLE.json') or {'groups': []}
    rows = [r for g in table['groups'] for r in g['rows']]
    git = load(P / 'recovery/git_sids.json') or {'arms': []}
    rec = load(P / 'recovery/RECOVERY.json') or {}
    corrected = {r['arm']: r for r in (load('results/research/sid_utilization/metrics_corrected.json') or [])}
    vstats = load('results/research/sid_utilization/vector_stats_all.json') or []

    def rows_for(pattern: str):
        return [r for r in rows if any(pattern in (a.get('arm') or '') for a in r.get('arms', []))]

    def git_for(pattern: str):
        return [a for a in git['arms'] if pattern in (a.get('arm') or '')]

    arms_out = []
    for name in NAMED:
        rs = rows_for(name)
        gs = git_for(name)
        versions = [{k: v.get(k) for k in ('sha256', 'collision', 'n_items', 'deleted', 'commit', 'path')}
                    for a in gs for v in a.get('versions', []) if v.get('sha256')]
        corr = {k: v for k, v in corrected.items() if name in k}
        vs = [v for v in vstats if name in (v.get('arm') or '')]
        binds = [b for a in gs for b in a.get('binds', [])]
        arms_out.append({
            'query': name,
            'table_rows': len(rs),
            'labels': [r['label_recall@10'] for r in rs],
            'identity': [sorted({a.get('identity') for a in r.get('arms', [])}) for r in rs],
            'has_features': [all((r['features'] or {}).get(k) is not None
                                 for k in ('l1_pos_hit', 'l3_collision')) for r in rs],
            'has_vector': [bool(r.get('vector_forward')) for r in rs],
            'sids_sha256': [r['sid_sha256'] for r in rs],
            'git_versions': len(versions),
            'git_binds': len(binds),
            'metrics_corrected': bool(corr),
            'vector_stats': bool(vs),
            'status': ('RECOVERED_PRIMARY' if rs and any('L1_ledger' in (i or []) or 'L2_collision' in (i or [])
                                                         for i in [sorted({a.get('identity') for a in r.get('arms', [])}) for r in rs])
                       else 'RECOVERED_WEAK' if rs else 'NOT_RECOVERED'),
        })

    hist = load('results/research/issue275/RUN_INVENTORY.json') or {}
    coh = load('results/research/issue270/COHORT.json') or {}
    hi_records = []
    for r in hist.get('records', []):
        if isinstance(r.get('recall@10'), (int, float)) and r['recall@10'] > 0.06:
            hi_records.append({'channel': 'inventory', 'arm': r['arm'], 'recall': r['recall@10'],
                               'sid_sha256': r.get('sid_sha256')})
    for r in coh.get('rows', []):
        if isinstance(r.get('recall@10'), (int, float)) and r['recall@10'] > 0.06:
            hi_records.append({'channel': 'cohort', 'arm': r['arm'], 'recall': r['recall@10'],
                               'sid_sha256': r.get('sid_sha256')})
    all_sids = {r['sid_sha256'] for r in rows}
    # high-recall records whose arm now has a content-bound SID in the table
    bound_arms = {a.get('arm') for r in rows for a in r.get('arms', [])}
    hi_bound = [h for h in hi_records if h['arm'] in bound_arms]

    out = {
        'schema': 'genrec.issue276.audit/1',
        'n_table_rows': len(rows),
        'n_high_recall_records': len(hi_records),
        'n_high_recall_records_with_bound_arm': len(hi_bound),
        'n_high_recall_table_rows': sum(1 for r in rows
                                        if isinstance(r['label_recall@10'], (int, float))
                                        and r['label_recall@10'] > 0.06),
        'named_arms': arms_out,
        'recovery_first_pass': {k: rec.get(k) for k in ('records', 'high_recall_records',
                                                        'high_recall_bound', 'sid_files_indexed',
                                                        'ledger_orphans', 'r2_cache_sids') if k in rec},
    }
    (P / 'recovery/AUDIT.json').write_text(json.dumps(out, indent=1) + '\n')

    lines = ['# Issue #276 — recovery audit (C1)', '',
             'Identity channels: L1 = R2-ledger sha256 of `item_sids.json`; L2 = computed L3 collision equals',
             'the stored `metrics_corrected[arm].collision`; L3 = git era + directory name only.',
             '', '## The 12 arms named by the issue', '',
             '| arm | table rows | label | identity | features | vector | git versions | status |',
             '|---|---|---|---|---|---|---|---|']
    for a in arms_out:
        lab = ', '.join(f'{x:.4f}' for x in a['labels'] if isinstance(x, (int, float))) or '—'
        ident = ';'.join('/'.join(i) for i in a['identity']) or '—'
        lines.append(f"| {a['query']} | {a['table_rows']} | {lab} | {ident} | "
                     f"{sum(a['has_features'])} | {sum(a['has_vector'])} | {a['git_versions']} | {a['status']} |")
    lines += ['', '## High-recall records', '',
              f"* records >0.06 (inventory ∪ cohort): **{out['n_high_recall_records']}**",
              f"* of them with an arm that now has a content-bound SID row: **{out['n_high_recall_records_with_bound_arm']}**",
              f"* table rows with recall >0.06: **{out['n_high_recall_table_rows']}**",
              '', '## First-pass recovery (before the git channel)', '',
              '```json', json.dumps(out['recovery_first_pass'], indent=1), '```']
    (P / 'recovery/AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: out[k] for k in ('n_table_rows', 'n_high_recall_records',
                                          'n_high_recall_records_with_bound_arm',
                                          'n_high_recall_table_rows')}, indent=1))
    for a in arms_out:
        print(f"{a['query']:38s} rows={a['table_rows']} labels={a['labels'][:2]} {a['status']}")


if __name__ == '__main__':
    main()
