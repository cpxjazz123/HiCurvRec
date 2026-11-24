"""Read-only parent audit; old executed sources are not replaced by live helpers."""
from __future__ import annotations

import json
from pathlib import Path

from tools.issue252 import archive_audit
from tools.issue255 import historical
from tools.issue255.paired_comparison import validate_completed

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
OUTPUT = ROOT / 'results/research/issue256'
PARENTS = (
    (252, 'issue252_formal_r2', 'TIGER_gnn_attention_sectional_rq', '100'),
    (253, 'stage2_787483d6ef9842d1a7c4593a61ae925c', 'TIGER_three_expert_attention_rq', '101'),
    (255, 'issue255_formal_r1', 'TIGER_prefix_curvature_gate_rq', None),
)


def load(path):
    return json.loads(Path(path).read_text())


def describe(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': archive_audit.sha256(path)}


def main():
    ledger = load(ROOT / 'results/research/issue139/r2_ledger.json')
    remote = load(OUTPUT / 'PARENT_R2_AUDIT.json')
    protocol_final = ROOT / ('results/stage3_T5Train/TIGER_three_expert_attention_rq/logs/'
                             'Amazon_2023_Instruments/Nov-24-2026_21-16-28/test_final.json')
    protocol = load(protocol_final)['stage3_input_config']
    rows = []
    for issue, run_id, arm, number in PARENTS:
        run = ROOT / 'results/stage2_RQ-VAE/_runs' / run_id / arm
        source, result = run / 'src', run / 'result'
        binding_path = result / 'archive_binding.json'
        binding = load(binding_path)
        provenance = binding['checkpoint_provenance']
        assert binding['run_id'] == run_id and Path(binding['source_tree']).resolve() == source.resolve()
        candidate = archive_audit.source_hashes(source)
        assert candidate == binding['candidate_sources'] == provenance['candidate_sources']
        expected = binding['infrastructure_sources']
        assert expected == provenance['infrastructure_sources']
        live = archive_audit.imported_repo_sources(source, ROOT)
        differences = []
        for name in sorted(set(expected) | set(live)):
            if expected.get(name) == live.get(name):
                continue
            # Historical helper evolution is not a license to execute an old
            # model with changed helpers. Its original bytes must be archived.
            if number is None or ledger[number]['files'].get(arm + '/' + name) != expected.get(name):
                raise ValueError('unproven historical infrastructure source: ' + name)
            assert remote[number]['all_remote_sha256_match']
            differences.append({'path': name, 'executed_sha256': expected[name],
                                'current_sha256': live.get(name),
                                'original_bytes_verified_at': 'r2:genrec/' + number + '/' + arm + '/' + name})
        assets_checked = 0
        for section in ('assets', 'training_inputs'):
            for name, expected_hash in provenance.get(section, {}).items():
                if section == 'assets' and Path(name).suffix == '.json':
                    continue
                path = Path(name)
                path = path if path.is_absolute() else ROOT / path
                assert archive_audit.sha256(path) == expected_hash, name
                assets_checked += 1
        config_proofs = {}
        for name, expected_hash in binding['repository_configs'].items():
            _, config_proofs[name] = historical._config(ROOT, name, expected_hash)
        artifacts = {}
        for key, filename in (('sid', 'item_sids.json'), ('checkpoint', 'rqvae_best.pth'),
                              ('export', 'export_manifest.json'), ('binding', 'archive_binding.json')):
            artifacts[key] = describe(result / filename)
        assert artifacts['sid'] == binding['sid']
        assert artifacts['checkpoint'] == binding['checkpoint']
        assert artifacts['export'] == binding['export_manifest']
        archive_audit._check_export(load(result / 'export_manifest.json'), source,
                                   result / 'rqvae_best.pth', result / 'item_sids.json', ROOT)
        stage2 = {'run_id': run_id, 'source_tree': str(source), 'artifacts': artifacts}
        completed = []
        for final_path in sorted((ROOT / 'results/stage3_T5Train' / arm).glob('logs/*/*/test_final.json')):
            final = load(final_path)
            if final.get('sid_sha256') == artifacts['sid']['sha256']:
                receipt = validate_completed(final_path.parent, stage2, protocol, 57439,
                    '54ca6bfbcd55a42e718660dd0004370a41a5e1c2c5038a86bbfc84f32b963690')
                completed.append({'receipt': receipt, 'metrics': {
                    key: final[key] for key in ('test_recall@5', 'test_recall@10',
                                                'test_ndcg@5', 'test_ndcg@10')}})
        assert len(completed) == (0 if issue == 255 else 1)
        if number:
            archive = ledger[number]
            assert archive['files']['item_sids.json'] == artifacts['sid']['sha256']
            assert archive['files']['rqvae_best.pth'] == artifacts['checkpoint']['sha256']
            assert remote[number]['sid_sha256'] == artifacts['sid']['sha256']
            assert remote[number]['checkpoint_sha256'] == artifacts['checkpoint']['sha256']
        docs = {}
        for name in ('REPORT.md', 'VERDICT.md', 'CONTRACT.md', 'RECONCILIATION.json', 'MODULE_CHECKS.json'):
            path = ROOT / 'results/research' / ('issue%d' % issue) / name
            if path.is_file():
                docs[name] = describe(path)
        rows.append({'issue': issue, 'stage2': stage2,
                     'candidate_source_files': len(candidate),
                     'infrastructure_source_files': len(expected),
                     'historical_infrastructure_differences': differences,
                     'runtime_leaves_checked': assets_checked,
                     'historical_config_proofs': config_proofs,
                     'stage3': completed, 'r2_archive': number,
                     'documents': docs, 'source_identity_pass': True})
        print('issue%d source/assets/paired-run PASS; Stage3=%d; archive=%s' %
              (issue, len(completed), number), flush=True)
    path = ROOT / 'results/research/issue254'
    rows.append({'issue': 254, 'kind': 'read-only analysis, no new model or SID',
                 'documents': {name: describe(path / name) for name in
                               ('VERDICT.md', 'MANIFEST.json', 'EVIDENCE.json',
                                'RECONCILIATION.json', 'TEST_RESULTS.json')},
                 'test_error_item_tables_read_for_selection': False})
    report = {'schema': 'genrec.issue256.parent_audit/1', 'parents': rows,
              'r2_audit': describe(OUTPUT / 'PARENT_R2_AUDIT.json'),
              'r2_ledger': describe(ROOT / 'results/research/issue139/r2_ledger.json'),
              'audit_source': describe(Path(__file__)), 'exit_code': 0,
              'command': 'python3 -m tools.issue256.audit_parents',
              'qualification': 'Original executed helper bytes for Issue252 differ from current helpers; '
                               'both original objects were independently SHA256-verified in R2. '
                               'No old model was rerun against different current helpers.'}
    (OUTPUT / 'PARENT_SOURCE_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
