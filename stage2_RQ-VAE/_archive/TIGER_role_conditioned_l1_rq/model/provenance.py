"""Bind candidate checkpoints to immutable assets and the executed source tree."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):
            value.update(block)
    return value.hexdigest()


def binding(repo, tree):
    repo,tree=Path(repo),Path(tree)
    manifest=repo/'assets/manifest.json'
    metadata=json.loads(manifest.read_text())
    entries=metadata['assets']
    train=repo/'results/stage0_build_parquet/train.parquet'
    train_digest=sha256(train)
    if train_digest!=metadata['inputs']['train']:
        raise ValueError('training input hash mismatch: '+str(train.relative_to(repo)))
    assets={}
    for entry in entries:
        actual=sha256(repo/entry['path'])
        if actual!=entry['sha256']:
            raise ValueError('immutable asset hash mismatch: '+entry['path'])
        assets[entry['path']]=actual
    sources={str(p.relative_to(tree)):sha256(p) for p in sorted(tree.rglob('*'))
             if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts
             and 'logs' not in p.parts}
    return {'schema':1,'asset_manifest_sha256':sha256(manifest),
            'assets':assets,'candidate_sources':sources,
            'training_inputs':{str(train.relative_to(repo)):train_digest}}
