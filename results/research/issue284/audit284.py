"""Issue #284: audit of #283 artifacts and the identity/export-version landscape (C1)."""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')

P = Path('results/research/issue284')
P283 = Path('results/research/issue283')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    out = {'schema': 'genrec.issue284.audit/1'}
    # 1. #283 artifact hashes
    man = json.load(open(P283 / 'STATS_MANIFEST.json'))
    checks = {}
    for name, e in man['files'].items():
        p = Path('results/research/issue283') / name
        ok = p.is_file() and sha(p) == e['sha256']
        checks[name] = {'exists': p.is_file(), 'sha_ok': bool(ok), 'sha256': e['sha256']}
    out['p283_files'] = checks
    out['p283_commit'] = '801048ca8153fd765e38b045a4c0578cb6343a0c'
    # 2. #283 headline numbers copied for traceability
    rows_manifest = json.load(open(P283 / 'ROWS_MANIFEST.json'))
    summary = json.load(open(P283 / 'JOIN_SUMMARY.json'))
    model = json.load(open(P283 / 'MODEL_COMPARISON.json'))
    matched = json.load(open(P283 / 'MATCHED.json'))
    out['p283_rows'] = {'total_rows': rows_manifest['total_rows'],
                        'family_counts': rows_manifest['family_counts'],
                        'class_counts_nonzero': len(rows_manifest['class_counts'])}
    out['p283_chain'] = {'k1': summary['levels']['1']['chain'], 'k2': summary['levels']['2']['chain'],
                         'k3': summary['levels']['3']['chain']}
    out['p283_model'] = {'baseline': model['baseline_A']['oof'], 'new': model['new_model']['oof'],
                         'new_ci': model['new_model']['ci95'],
                         'fixed_all1_chain': model['fixed_arms']['all1_chain']}
    out['p283_matched'] = {'marginal_chain': matched['marginal_matched_chain'],
                           'baseline': matched['baseline_A']}
    # 3. labels / folds / prospective
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    pros = json.load(open('results/research/issue277/PROSPECTIVE.json'))
    rows = []
    for g in table['groups']:
        for r in g['rows']:
            rr = dict(r); rr['group'] = g['group']; rows.append(rr)
    lab = [r for r in rows if isinstance(r['label_recall@10'], (int, float))]
    strong = 0
    high = 0
    high_strong = 0
    for r in lab:
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [s.get('source') for s in r.get('label_sources', [])]
        s = ('L1_ledger' in ident) or any(str(x).startswith('inventory_unit') for x in srcs)
        strong += int(s); high += int(r['label_recall@10'] > 0.06)
        high_strong += int(s and r['label_recall@10'] > 0.06)
    out['labels'] = {'n_table': len(rows), 'n_labelled': len(lab), 'n_strong': strong,
                     'n_weak': len(lab) - strong, 'n_high': high, 'n_high_strong': high_strong,
                     'fold_sizes': {str(k): int(sum(1 for r in lab if folds['assignment'][r['group']] == k))
                                    for k in range(5)},
                     'fold_rule': folds['rule'], 'fold_seed': folds['seed']}
    out['prospective'] = {'n': len(pros['rows']), 'all_in_table': all(
        any(r['sid_sha256'] == p['sid_sha256'] for r in rows) for p in pros['rows']),
        'labelled': sum(1 for p in pros['rows'] if any(
            r['sid_sha256'] == p['sid_sha256'] and isinstance(r['label_recall@10'], (int, float))
            for r in rows))}
    # 4. new time-postordered labels (R2 114/115) with exact SID + run identity
    new_labels = []
    for sid_pref, recall_path, r2 in (
            ('9f6bf6731e88', 'results/stage3_T5Train/TIGER_pairrank_enc_rq/logs/Amazon_2023_Instruments/'
                             'Nov-28-2026_13-00-13/test_final.json', '114'),
            ('61f898498c19', 'results/stage3_T5Train/TIGER_codebook_budget_j_rq/logs/Amazon_2023_Instruments/'
                             'Nov-28-2026_14-29-03/test_final.json', '115')):
        found = []
        for p in glob.glob('results/stage2_RQ-VAE/**/item_sids.json', recursive=True):
            s = sha(p)
            if s.startswith(sid_pref):
                found.append({'path': p, 'sha256': s})
        tf = json.load(open(recall_path))
        new_labels.append({'r2': r2, 'sid_prefix': sid_pref, 'sid_files': found,
                           'test_final': recall_path, 'test_final_sha256': sha(recall_path),
                           'test_recall@10': tf['test_recall@10'], 'test_recall@5': tf['test_recall@5'],
                           'in_table': any(r['sid_sha256'].startswith(sid_pref) for r in rows),
                           'table_label': next((r['label_recall@10'] for r in rows
                                                if r['sid_sha256'].startswith(sid_pref)), None)})
    out['new_labels'] = new_labels
    # 5. #274 R2/R3 export versions (trainer vs checkpoint), never merged
    exports = []
    for run, arm, r2 in (
            ('stage2_818150011991495cbeebd225909b1f91', 'TIGER_pairrank_enc_rq', 'R2'),
            ('stage2_6fe2faf7fc204330b129dfa4900ae641', 'TIGER_pairrank_cover_rq', 'R3')):
        d = f'results/stage2_RQ-VAE/_runs/{run}/{arm}/result'
        m = json.load(open(f'{d}/export_manifest.json'))
        files = {n: {'path': f'{d}/{n}', 'sha256': sha(f'{d}/{n}') if Path(f'{d}/{n}').is_file() else None}
                 for n in ('item_sids.trainer.json', 'item_sids.json')}
        exports.append({'label': r2, 'run_id': run, 'arm': arm,
                        'original_sid_sha256': m.get('original_sid_sha256'),
                        'checkpoint_sid_sha256': m.get('sid_sha256'),
                        'checkpoint_sha256': m.get('checkpoint_sha256'),
                        'differing_items_at_repair': m.get('original_differing_items'),
                        'differing_items_after': m.get('differing_items'),
                        'same_l1': m.get('same_l1'), 'same_l3': m.get('same_l3'),
                        'files': files,
                        'table_membership': {s: any(r['sid_sha256'] == s for r in rows)
                                             for s in (m.get('original_sid_sha256'), m.get('sid_sha256'))},
                        'labels': {s: next((r['label_recall@10'] for r in rows if r['sid_sha256'] == s), None)
                                   for s in (m.get('original_sid_sha256'), m.get('sid_sha256'))}})
    out['p274_exports'] = exports
    # 6. #272 recent archived runs and their SID shas
    runs = ['stage2_0a2be6e306e145fe91b3d7be08a37c69', 'stage2_4891fa8df3e843839de770e52e156892',
            'stage2_4dffd8b4528440b3aca334b380749f01', 'stage2_70feb0bb3b174adb8e046df6df09aed3',
            'stage2_bba78cf7c160467da1c8ae7de9b6e26f', 'stage2_cc9a31a9325b439f98c8b1779aaa32aa']
    p272 = []
    for run in runs:
        hits = glob.glob(f'results/stage2_RQ-VAE/_runs/{run}/*/result/item_sids*.json')
        p272.append({'run_id': run, 'sids': [
            {'path': h, 'sha256': sha(h), 'in_table': any(r['sid_sha256'] == sha(h) for r in rows),
             'label': next((r['label_recall@10'] for r in rows if r['sid_sha256'] == sha(h)), None)}
            for h in sorted(hits)]})
    out['p272_runs'] = p272
    # 7. JOIN_TABLE chain counts availability for the 2 new-label SIDs
    join = json.load(open(P283 / 'JOIN_TABLE.json'))
    out['join_table'] = {'n_entries': len(join['entries']),
                         'new_labels_present': [any(k.startswith(nl['sid_prefix']) for k in join['entries'])
                                                for nl in new_labels]}
    (P / 'AUDIT_284.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('labels', 'prospective', 'p272_runs')}, indent=1)[:2500])
    print('new labels:', [(n['r2'], n['test_recall@10'], n['in_table'], n['table_label'])
                          for n in new_labels])


if __name__ == '__main__':
    main()
