"""A growing manifest must preserve old config bytes and every old input leaf."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.issue252 import archive_audit
from tools.issue255 import historical


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


@pytest.fixture
def old_generation(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'model').mkdir(parents=True)
    (source / 'train_rqvae.py').write_text('VALUE = 1\n')
    (source / 'model/__init__.py').write_text('')
    leaf = dump(tmp_path / 'assets/leaf.json', {'frozen': 1})
    old_manifest = json.dumps({'assets': [{'path': 'assets/leaf.json', 'sha256': leaf['sha256']}]}).encode()
    dump(tmp_path / 'assets/manifest.json', {'assets': [{'path': 'new_asset'}]})
    sid = dump(tmp_path / 'result/item_sids.json', {'0': [0, 0, 0, 768]})
    checkpoint = tmp_path / 'result/rqvae_best.pth'
    checkpoint.write_bytes(b'checkpoint identity')
    ckpt = {'path': str(checkpoint), 'sha256': digest(checkpoint.read_bytes())}
    export = dump(tmp_path / 'result/export_manifest.json', {
        'schema': 'genrec.stage2.export_manifest/1', 'tokens_match': True,
        'same_l3': 1.0, 'differing_items': 0,
        'tree': str(source), 'sid': sid['path'], 'sid_sha256': sid['sha256'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': ckpt['sha256']})
    candidate = archive_audit.source_hashes(source)
    provenance = {'candidate_sources': candidate, 'infrastructure_sources': {},
                  'asset_manifest_sha256': digest(old_manifest),
                  'assets': {'assets/leaf.json': leaf['sha256']}, 'training_inputs': {}}
    audit = Path(archive_audit.__file__).resolve()
    binding = dump(tmp_path / 'result/archive_binding.json', {
        'schema': archive_audit.SCHEMA, 'run_id': 'old_run', 'repository': str(tmp_path),
        'source_tree': str(source), 'candidate_sources': candidate,
        'infrastructure_sources': {}, 'checkpoint_provenance': provenance,
        'repository_configs': {'assets/manifest.json': digest(old_manifest), 'assets/leaf.json': leaf['sha256']},
        'sid': sid, 'checkpoint': ckpt, 'export_manifest': export,
        'audit': {'path': str(audit), 'sha256': digest(audit.read_bytes())}})
    monkeypatch.setattr(historical, '_git_config', lambda repo, name: old_manifest)
    return binding, sid, ckpt, source, tmp_path, old_manifest


def test_append_only_registry_uses_exact_prior_config_bytes(old_generation):
    binding, sid, ckpt, source, _, _ = old_generation
    receipt, proof = historical.validate_binding(binding['path'], sid['path'], ckpt['path'], source)
    assert receipt['run_id'] == 'old_run'
    assert proof['assets/manifest.json']['origin'].startswith('git:')
    assert proof['assets/leaf.json']['origin'] == 'current_exact_bytes'


def test_an_old_runtime_leaf_cannot_change(old_generation):
    binding, sid, ckpt, source, repo, _ = old_generation
    (repo / 'assets/leaf.json').write_text('{"frozen": 2}')
    with pytest.raises(ValueError, match='leaf'):
        historical.validate_binding(binding['path'], sid['path'], ckpt['path'], source)


def test_wrong_historical_config_is_not_a_fallback(old_generation, monkeypatch):
    binding, sid, ckpt, source, _, _ = old_generation
    monkeypatch.setattr(historical, '_git_config', lambda repo, name: b'wrong archived bytes')
    with pytest.raises(ValueError, match='config'):
        historical.validate_binding(binding['path'], sid['path'], ckpt['path'], source)


def test_historical_source_drift_is_rejected(old_generation):
    binding, sid, ckpt, source, _, _ = old_generation
    (source / 'train_rqvae.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='source'):
        historical.validate_binding(binding['path'], sid['path'], ckpt['path'], source)
