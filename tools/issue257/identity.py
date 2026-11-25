"""Resolve one immutable run from its own artefacts; no issue or arm is hardcoded.

Every downstream tool (validation plan, analysis, report, archive, replay)
starts from here, so a run is identified by the hashes of what it produced, not
by the directory name that happens to hold it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stage3_final(path):
    """The test_final.json that belongs to the given path, if it has one."""
    path = Path(path).resolve()
    if path.is_file() and path.name == 'test_final.json':
        return path
    for candidate in (path / 'test_final.json',
                      *(path.glob('logs/*/*/test_final.json'))):
        if candidate.is_file():
            return candidate.resolve()
    return None


def identity(path):
    """Run identity from a SID, a result directory, a run directory or a test_final.

    Returns the artefact hashes, the arm, the run id and, when the run carries a
    Stage3 final record, the Stage3 side as well.  Missing optional artefacts are
    reported as ``None`` and never guessed from neighbouring files.
    """
    path = Path(path).resolve()
    final = _stage3_final(path)
    if final is not None:
        block = _read(final)
        sid = Path(block['sid_path']).resolve()
        stage2 = block['stage2_run_id']
        arm = sid.parent.parent.name if sid.parent.name == 'result' else None
        if arm is None:
            arm = block.get('arm') or block.get('variant')
        out = {
            'schema': 'genrec.pipeline.identity/1',
            'kind': 'stage3_final',
            'arm': arm,
            'stage2_run_id': stage2,
            'stage3_run_id': block.get('stage3_run_id'),
            'test_final': record(final),
            'sid': _with_recorded(record(sid), block.get('sid_sha256')),
            'stage2_checkpoint': _with_recorded(record(Path(block['stage2_checkpoint'])),
                                                block.get('stage2_checkpoint_sha256')),
            'stage2_export_manifest': _with_recorded(record(Path(block['stage2_export_manifest'])),
                                                     block.get('stage2_export_manifest_sha256')),
            'archive_binding': _with_recorded(record(Path(block['archive_binding'])),
                                              block.get('archive_binding_sha256')),
            'predictions': _shards(block.get('predictions')),
            'metrics': {'n_eval': block.get('n_eval'),
                        'test_recall@5': block.get('test_recall@5'),
                        'test_recall@10': block.get('test_recall@10'),
                        'test_ndcg@5': block.get('test_ndcg@5'),
                        'test_ndcg@10': block.get('test_ndcg@10')},
        }
        return out
    sid = path / 'item_sids.json' if path.is_dir() else path
    if not sid.is_file():
        raise ValueError('no SID or test_final at ' + str(path))
    result = sid.parent
    out = {
        'schema': 'genrec.pipeline.identity/1',
        'kind': 'stage2_result',
        'arm': result.parent.name,
        'stage2_run_id': result.parent.parent.name,
        'stage3_run_id': None,
        'test_final': None,
        'sid': record(sid),
        'stage2_checkpoint': record(result / 'rqvae_best.pth') if (result / 'rqvae_best.pth').is_file() else None,
        'stage2_export_manifest': record(result / 'export_manifest.json') if (result / 'export_manifest.json').is_file() else None,
        'archive_binding': record(result / 'archive_binding.json') if (result / 'archive_binding.json').is_file() else None,
        'metrics': None,
        'predictions': None,
    }
    return out


def _shards(predictions):
    if not predictions:
        return []
    first = Path(predictions)
    shards = sorted(first.parent.glob('rank*.jsonl'))
    return [str(path) for path in shards] if shards else [str(first)]


def _with_recorded(item, recorded):
    if recorded:
        item['recorded_sha256'] = str(recorded)
    return item


def source_tree(identity_block):
    """The executed source tree named by the run's own binding, if it exists."""
    binding = identity_block.get('archive_binding')
    if not binding:
        return None
    payload = _read(binding['path'])
    tree = Path(payload.get('source_tree', ''))
    return str(tree.resolve()) if tree.is_dir() else None


def layer_metrics_path(identity_block, repo=REPO):
    """Where the shared layer-metrics record for this SID lives."""
    arm = identity_block.get('arm')
    digest = identity_block['sid']['sha256']
    return Path(repo) / 'results' / 'stage2_RQ-VAE' / str(arm) / 'stage2_metrics' / (digest + '.json')
