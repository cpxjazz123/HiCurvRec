"""Verify (never upload) the executed Stage2 source/checkpoint/SID generation.

Run with the Stage2 interpreter after the existing exact export check::

    from tools.issue252.archive_audit import create_binding
    create_binding(tree, checkpoint, sid, sid.parent / 'export_manifest.json',
                   output=sid.parent / 'archive_binding.json')

The checkpoint must contain ``run_id``, ``source_tree`` and its existing
``provenance.candidate_sources`` / ``provenance.infrastructure_sources`` captured
before training and checked at save. The latter must cover imported_repo_sources.
No receipt is produced for an unbound historical checkpoint. This module does not
re-export, rewrite SID/checkpoint, launch GPU work, or access R2.

Receipt schema genrec.stage2.archive_binding/1 records repository, run_id,
source_tree, candidate_sources, infrastructure_sources and repository_configs
(relative path -> SHA), SID/checkpoint/export_manifest path and SHA descriptors,
checkpoint provenance,
and this audit implementation's SHA. All receipt inputs are rehashed by staging.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
SCHEMA = 'genrec.stage2.archive_binding/1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(tree):
    """The candidate's existing executed-source inventory, including JSON config."""
    tree = Path(tree).resolve()
    files = {}
    for path in sorted(tree.rglob('*')):
        relative = path.relative_to(tree)
        if (not path.is_file() or path.suffix not in ('.py', '.json')
                or {'__pycache__', 'logs'} & set(relative.parts)):
            continue
        if path.is_symlink():
            raise ValueError('source symlink is not an immutable source file: ' + str(path))
        files[relative.as_posix()] = sha256(path)
    if 'train_rqvae.py' not in files or not any(p.startswith('model/') for p in files):
        raise ValueError('missing necessary Stage2 source train_rqvae.py/model')
    return files


def imported_repo_sources(tree, repo=REPO):
    """Hash transitive, statically imported repository Python modules.

    Candidate modules resolve before repository modules, just as the trainer's
    import path does. Site packages/stdlib are not vendored. Package __init__
    files and ``from package import submodule`` dependencies are included.
    """
    tree, repo = Path(tree).resolve(), Path(repo).resolve()
    queue = [(tree / 'train_rqvae.py', 'train_rqvae')]
    execution_entry = tree / 'execution_entry.py'
    if execution_entry.is_file():
        queue.append((execution_entry, 'execution_entry'))
    seen, external = set(), {}

    def enqueue(module, required=False):
        parts = module.split('.')
        if not module or any(not part.isidentifier() for part in parts):
            return
        resolved = False
        for size in range(1, len(parts) + 1):
            name = '.'.join(parts[:size])
            for root in (tree, repo):
                stem = root.joinpath(*parts[:size])
                candidates = (stem.with_suffix('.py'), stem / '__init__.py')
                found = next((p for p in candidates if p.is_file()), None)
                if found is not None:
                    if found.is_symlink():
                        raise ValueError('imported source is a symlink: ' + str(found))
                    queue.append((found, name))
                    if size == len(parts):
                        resolved = True
                    break
                if size == len(parts) and stem.is_dir():
                    resolved = True
        local_root = any((root / parts[0]).exists() for root in (tree, repo))
        if required and local_root and not resolved:
            raise ValueError('missing required repository source import: ' + module)

    while queue:
        path, module = queue.pop()
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_relative_to(tree):
            if not path.is_relative_to(repo):
                raise ValueError('imported source escapes repository: ' + str(path))
            external[path.relative_to(repo).as_posix()] = sha256(path)
        syntax = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        package = module.split('.') if path.name == '__init__.py' else module.split('.')[:-1]
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue(alias.name, required=True)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError('invalid relative source import: ' + str(path))
                    prefix = package[:len(package) - node.level + 1]
                    base = '.'.join(prefix + ([node.module] if node.module else []))
                else:
                    base = node.module or ''
                enqueue(base, required=True)
                for alias in node.names:
                    if alias.name != '*':
                        enqueue(base + '.' + alias.name if base else alias.name)
    return dict(sorted(external.items()))


def _existing(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('missing necessary ' + label + ': ' + str(path))
    return path


def _record(path):
    return {'path': str(path), 'sha256': sha256(path)}


def _manifest_path(value, repo):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _repository_configs(provenance, repo):
    """Small runtime configs already bound by the checkpoint (never asset binaries)."""
    hashes = {}
    if 'asset_manifest_sha256' in provenance:
        hashes['assets/manifest.json'] = provenance['asset_manifest_sha256']
    if 'mathematical_contract_sha256' in provenance:
        hashes['results/research/issue252/CONTRACT.md'] = provenance['mathematical_contract_sha256']
    for name, digest in provenance.get('assets', {}).items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid checkpoint asset configuration path: ' + name)
        if relative.suffix == '.json':
            hashes[name] = digest
    for name, digest in hashes.items():
        path = _existing(repo / name, 'repository config')
        if sha256(path) != digest:
            raise ValueError('checkpoint repository config hash mismatch: ' + name)
    return dict(sorted(hashes.items()))


def _check_export(manifest, tree, checkpoint, sid, repo):
    if manifest.get('schema') != 'genrec.stage2.export_manifest/1':
        raise ValueError('missing exact export manifest schema')
    if (manifest.get('tokens_match') is not True or manifest.get('same_l3') != 1.0
            or manifest.get('differing_items') != 0):
        raise ValueError('SID export does not match every checkpoint token')
    for field, path, label in (('sid', sid, 'SID'), ('checkpoint', checkpoint, 'checkpoint')):
        if (not manifest.get(field) or _manifest_path(manifest[field], repo) != path
                or manifest.get(field + '_sha256') != sha256(path)):
            raise ValueError(label + ' export identity mismatch')
    if not manifest.get('tree') or _manifest_path(manifest['tree'], repo) != tree:
        raise ValueError('source tree export identity mismatch')


def create_binding(tree, checkpoint, sid, export_manifest, *, output=None, repo=REPO):
    """CPU audit of real checkpoint provenance and exact existing export evidence.

    Return the receipt; atomically write it only when ``output`` is supplied.
    Call before Stage3 so its final record can capture the receipt SHA.
    """
    import torch

    repo, tree = Path(repo).resolve(), Path(tree).resolve()
    checkpoint = _existing(checkpoint, 'Stage2 checkpoint')
    sid = _existing(sid, 'SID')
    export_manifest = _existing(export_manifest, 'export manifest')
    checkpoint_sha = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('checkpoint is missing executed run identity')
    if not payload.get('source_tree') or _manifest_path(payload['source_tree'], repo) != tree:
        raise ValueError('checkpoint source tree identity mismatch')
    provenance = payload.get('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('checkpoint is missing source provenance')
    candidate = source_hashes(tree)
    infrastructure = imported_repo_sources(tree, repo)
    if provenance.get('candidate_sources') != candidate:
        raise ValueError('checkpoint candidate source/config hashes mismatch')
    if provenance.get('infrastructure_sources') != infrastructure:
        raise ValueError('checkpoint imported repository source hashes mismatch')
    manifest = json.loads(export_manifest.read_text(encoding='utf-8'))
    _check_export(manifest, tree, checkpoint, sid, repo)
    configs = _repository_configs(provenance, repo)
    if sha256(checkpoint) != checkpoint_sha:
        raise ValueError('checkpoint changed while auditing its source binding')
    receipt = {
        'schema': SCHEMA, 'repository': str(repo), 'run_id': payload['run_id'],
        'source_tree': str(tree), 'candidate_sources': candidate,
        'infrastructure_sources': infrastructure, 'checkpoint_provenance': provenance,
        'repository_configs': configs,
        'checkpoint': {'path': str(checkpoint), 'sha256': checkpoint_sha}, 'sid': _record(sid),
        'export_manifest': _record(export_manifest),
        'audit': _record(Path(__file__).resolve()),
    }
    if output is not None:
        output = Path(output)
        if output.resolve() in (sid, checkpoint, export_manifest) or output.resolve().is_relative_to(tree):
            raise ValueError('archive binding must not overwrite its inputs/source')
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.partial')
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        temporary.replace(output)
    return receipt


def validate_binding(path, sid, checkpoint, tree=None):
    """Recheck a receipt using stdlib only; return (receipt, archive source map)."""
    path = _existing(path, 'archive binding')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA
            or any(not isinstance(receipt.get(key), str) or not receipt[key]
                   for key in ('run_id', 'repository', 'source_tree'))):
        raise ValueError('missing archive binding schema/run identity')
    repo = Path(receipt['repository']).resolve()
    source = Path(receipt['source_tree']).resolve()
    if tree is not None and Path(tree).resolve() != source:
        raise ValueError('explicit source tree differs from executed source tree')
    candidate = source_hashes(source)
    infrastructure = imported_repo_sources(source, repo)
    if (receipt.get('candidate_sources') != candidate
            or receipt.get('infrastructure_sources') != infrastructure):
        raise ValueError('executed source/config hashes mismatch')
    provenance = receipt.get('checkpoint_provenance', {})
    if (provenance.get('candidate_sources') != candidate
            or provenance.get('infrastructure_sources') != infrastructure):
        raise ValueError('checkpoint source binding mismatch')
    for field, expected, label in (('sid', Path(sid).resolve(), 'SID'),
                                   ('checkpoint', Path(checkpoint).resolve(), 'checkpoint'),
                                   ('audit', Path(__file__).resolve(), 'audit source')):
        descriptor = receipt.get(field, {})
        if descriptor.get('path') != str(expected):
            raise ValueError(label + ' binding path mismatch')
        _existing(expected, label)
        if descriptor.get('sha256') != sha256(expected):
            raise ValueError(label + ' binding hash mismatch')
    descriptor = receipt.get('export_manifest', {})
    export = _existing(descriptor.get('path', ''), 'export manifest')
    if descriptor.get('sha256') != sha256(export):
        raise ValueError('export manifest binding hash mismatch')
    _check_export(json.loads(export.read_text(encoding='utf-8')), source,
                  Path(checkpoint).resolve(), Path(sid).resolve(), repo)
    files = {name: source / name for name in candidate}
    if {'archive_binding.json', 'export_manifest.json'} & files.keys():
        raise ValueError('source/config collides with reserved archive evidence paths')
    configs = _repository_configs(provenance, repo)
    if receipt.get('repository_configs') != configs:
        raise ValueError('checkpoint repository config binding mismatch')
    for name in set(infrastructure) | set(configs):
        if name in files:
            raise ValueError('imported source archive path collision: ' + name)
        files[name] = repo / name
    files['archive_binding.json'] = path
    files['export_manifest.json'] = export
    return receipt, files
