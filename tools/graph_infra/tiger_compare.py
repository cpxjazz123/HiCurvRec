"""The frozen TIGER comparison interface (issue #249, CONTRACT.md §7).

The SID table is the one Stage3 actually read; it is never re-trained and never regenerated.

Pure SID metrics are recomputed directly from that file. The geometry metrics need a codebook and a
``q`` that are **provably from the same SID generation**; the frozen checkpoint is not (the #246
audit measured its re-export as differing in every one of the 24,587 rows), so those metrics are
reported as NA with the reason instead of being faked with the other checkpoint.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.graph_infra.fidelity import (prefix_collision, prefix_sharing, codebook_occupancy,
                                        decode_hard_sid)

STAGE3_SID = Path('results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json')
CHECKPOINT = Path('results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_stage3_sid(repo: Path) -> tuple[np.ndarray, str]:
    """The SID table Stage3 read, with its hash."""
    path = Path(repo) / STAGE3_SID
    table = json.loads(path.read_text())
    keys = sorted(table, key=int)
    rows = np.array([table[key] for key in keys], dtype=np.int64)
    return rows, sha256(path)


def pure_sid_metrics(rows: np.ndarray) -> dict:
    """Everything that can be recomputed from the SID table alone."""
    return {
        'collision_prefix3': prefix_collision(rows, 3),
        'duplicate_prefix3': prefix_collision(rows, 3)['duplicate_prefix_share'],
        'collision_full': prefix_collision(rows, rows.shape[1]),
        'prefix_sharing': prefix_sharing(rows),
        'codebook_occupancy': codebook_occupancy(rows),
        'source': 'recomputed directly from the frozen item_sids.json',
    }


def load_codebooks(checkpoint_path: Path) -> list[np.ndarray]:
    """The three level codebooks of a checkpoint, if it carries them."""
    import torch
    blob = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(blob, dict):
        for key in ('codebooks', 'rq_codebooks', 'vq_codebooks'):
            if key in blob:
                return [np.asarray(level, dtype=np.float64) for level in blob[key]]
    raise KeyError('the checkpoint carries no codebook list')


def compare(repo: Path, checkpoint: Path | None = None) -> dict:
    """The frozen TIGER comparison, with NA wherever the provenance does not hold.

    ``checkpoint`` is the candidate checkpoint whose codebook would decode the SID; it is only
    accepted when it is the one that produced the table. The default answer is NA.
    """
    repo = Path(repo)
    rows, sid_hash = load_stage3_sid(repo)
    report = {'sid_path': str(STAGE3_SID), 'sid_sha256': sid_hash,
              'sid_items': int(len(rows)),
              'pure_sid_metrics': pure_sid_metrics(rows)}
    geometry = {'status': 'NA',
                'reason': ('no codebook and checkpoint provably from the same SID generation as '
                            'the Stage3 table: the frozen rqvae_best.pth belongs to a different '
                            'generation (#246 measured its re-export as differing in all 24,587 '
                            'rows), so the decoded q and the reconstruction error are NA')}
    if checkpoint is not None:
        path = Path(checkpoint)
        try:
            codebooks = load_codebooks(path)
        except Exception as error:  # noqa: BLE001 - the reason is reported, not raised
            geometry['reason'] = f'the checkpoint carries no usable codebook: {error}'
        else:
            geometry = {'status': 'NA',
                        'reason': 'the candidate checkpoint is not proven to belong to the same '
                                   'SID generation as the Stage3 table',
                        'candidate_checkpoint': str(path),
                        'candidate_checkpoint_sha256': sha256(path),
                        'codebook_levels': [int(level.shape[0]) for level in codebooks]}
    report['geometry_metrics'] = geometry
    return report


def decode_candidate(repo: Path, checkpoint: Path, sid_path: Path) -> np.ndarray:
    """Decode a SID table with a candidate checkpoint's codebooks (used for the separate,
    checkpoint-consistent reference measurement, never for the Stage3 table)."""
    rows = json.loads(Path(sid_path).read_text())
    keys = sorted(rows, key=int)
    table = np.array([rows[key] for key in keys], dtype=np.int64)
    return decode_hard_sid(table, load_codebooks(Path(checkpoint)))
