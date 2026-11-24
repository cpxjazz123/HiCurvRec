"""Validate an old generation against its exact configs, not a newer registry.

The comparison's manifest bytes are retained in an immutable Git revision.
Runtime asset leaves, executed code, checkpoints and SIDs must still match
exactly; an append to the registry never licenses a changed old input. No
asset is rebuilt, overwritten or copied into the training tree.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tools.issue252 import archive_audit

COMPARISON_CONFIG_COMMIT = '6d96efe092547277173906b152f2d5b3e9172ced'


def _git_config(repo, name):
    process = subprocess.run(['git', 'show', COMPARISON_CONFIG_COMMIT + ':' + name],
                             cwd=str(repo), capture_output=True)
    if process.returncode:
        raise ValueError('historical config unavailable at frozen revision: ' + name)
    return process.stdout


def _config(repo, name, expected):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('invalid historical config path: ' + name)
    path = repo / relative
    if path.is_file() and archive_audit.sha256(path) == expected:
        return path.read_bytes(), {'sha256': expected, 'origin': 'current_exact_bytes'}
    data = _git_config(repo, name)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('historical config hash mismatch: ' + name)
    return data, {'sha256': expected, 'origin': 'git:' + COMPARISON_CONFIG_COMMIT + ':' + name}


def validate_binding(path, sid, checkpoint, tree=None):
    """Stdlib-only historical audit; returns original receipt and config proofs."""
    path, sid, checkpoint = (Path(p).resolve() for p in (path, sid, checkpoint))
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != archive_audit.SCHEMA
            or any(not isinstance(receipt.get(k), str) or not receipt[k]
                   for k in ('run_id', 'repository', 'source_tree'))):
        raise ValueError('historical binding schema/run identity is missing')
    repo, source = Path(receipt['repository']).resolve(), Path(receipt['source_tree']).resolve()
    if tree is not None and Path(tree).resolve() != source:
        raise ValueError('historical source identity differs')
    candidate = archive_audit.source_hashes(source)
    infrastructure = archive_audit.imported_repo_sources(source, repo)
    provenance = receipt.get('checkpoint_provenance', {})
    if any(block.get('candidate_sources') != candidate or block.get('infrastructure_sources') != infrastructure
           for block in (receipt, provenance)):
        raise ValueError('historical source/config hashes differ')
    for field, expected in (('sid', sid), ('checkpoint', checkpoint),
                            ('audit', Path(archive_audit.__file__).resolve())):
        item = receipt.get(field, {})
        if item.get('path') != str(expected) or item.get('sha256') != archive_audit.sha256(expected):
            raise ValueError('historical artifact binding differs: ' + field)
    export_record = receipt['export_manifest']
    export = Path(export_record['path']).resolve()
    if export_record['sha256'] != archive_audit.sha256(export):
        raise ValueError('historical export manifest differs')
    archive_audit._check_export(json.loads(export.read_text()), source, checkpoint, sid, repo)
    # Frozen old leaves are actual runtime inputs. A Git copy is NOT a license
    # to evaluate old code against a changed current normalization/graph/table.
    for section in ('assets', 'training_inputs'):
        for name, expected in provenance.get(section, {}).items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid historical runtime leaf path: ' + name)
            leaf = repo / relative
            if not leaf.is_file() or archive_audit.sha256(leaf) != expected:
                raise ValueError('historical runtime leaf changed: ' + name)
    configs = {name: value for name, value in provenance.get('assets', {}).items()
               if Path(name).suffix == '.json'}
    if 'asset_manifest_sha256' in provenance:
        configs['assets/manifest.json'] = provenance['asset_manifest_sha256']
    if 'mathematical_contract_sha256' in provenance:
        configs[archive_audit._contract_path(provenance, repo)] = provenance['mathematical_contract_sha256']
    if receipt.get('repository_configs') != dict(sorted(configs.items())):
        raise ValueError('historical config inventory differs from checkpoint provenance')
    contents, proofs = {}, {}
    for name, expected in configs.items():
        contents[name], proofs[name] = _config(repo, name, expected)
    if 'assets/manifest.json' in contents:
        manifest = json.loads(contents['assets/manifest.json'])
        frozen_assets = {entry['path']: entry['sha256'] for entry in manifest['assets']}
        if frozen_assets != provenance.get('assets', {}):
            raise ValueError('historical manifest leaf identity differs from checkpoint')
    return receipt, proofs


def load_bound_checkpoint(*, checkpoint, sid, tree, binding, export_manifest):
    import torch
    from tools.issue252.final_evaluation import validate_final_checkpoint
    checkpoint, sid, tree, export_manifest = (Path(p).resolve() for p in
                                             (checkpoint, sid, tree, export_manifest))
    receipt, _ = validate_binding(binding, sid, checkpoint, tree)
    if receipt['export_manifest']['path'] != str(export_manifest):
        raise ValueError('historical explicit export manifest differs')
    payload = validate_final_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
    source = Path(payload['source_tree'])
    if not source.is_absolute():
        source = Path(receipt['repository']) / source
    if (payload['run_id'] != receipt['run_id'] or source.resolve() != tree
            or payload['provenance'] != receipt['checkpoint_provenance']
            or archive_audit.sha256(checkpoint) != receipt['checkpoint']['sha256']):
        raise ValueError('historical checkpoint/source provenance differs')
    return payload, receipt
