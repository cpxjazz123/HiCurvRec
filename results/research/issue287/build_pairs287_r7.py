"""Issue #287 round 7: the bucket-size column of the hard-negative index.

Round 6 measured that separating individual mined false positives does not lower N: the Sinkhorn
balanced assignment keeps the bucket sizes, so an item removed from the anchor's bucket is replaced by
another.  N = sum_k n_k^2 / N^2 is dominated by the *large* buckets, so round 7 weights every mined
false positive by the size of the bucket it would leave - the separation then targets the imbalance
instead of individual items.

This script derives the frozen inputs from round 6's index: the initial bucket sizes are the sizes of
the anchors' L1 code buckets in the frozen TIGER export (the same export the initial negatives come
from).  During training the trainer re-mines both the negatives and the bucket sizes from the current
partition with the same rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'
NODES = 24587


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base = np.load(OUT / 'pairs_hardneg.npz')
    anchor = base['anchor'].astype(np.int64)
    raw = json.loads((REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json').read_text())
    codes = np.zeros(NODES, dtype=np.int64)
    for item, value in raw.items():
        codes[int(item)] = int(value[0])
    counts = np.bincount(codes, minlength=256).astype(np.int64)
    bucket_size = counts[codes[anchor]].astype(np.int64)
    path = OUT / 'pairs_bucketsize.npz'
    np.savez(path, anchor=anchor, pos=base['pos'], neg=base['neg'], is_holdout=base['is_holdout'],
             nbr_ptr=base['nbr_ptr'], nbr_items=base['nbr_items'], bucket_size=bucket_size)
    manifest = {
        'schema': 'genrec.issue287.bucketsize/1',
        'path': str(path), 'sha256': sha256(path),
        'rules': {
            'positive': "round 2's frozen curvature-screened positive",
            'negative': "round 6's mined false positive (an item sharing the anchor's L1 code with no "
                        "observed relation), refreshed from the current partition every 500 epochs",
            'weight': "the size of the anchor's L1 code bucket in the frozen TIGER export; refreshed "
                      "with the negatives from the current partition",
            'curvature_role': 'offline selection of the positive only',
        },
        'counts': {'groups': int(len(anchor)), 'anchors': int(len(np.unique(anchor))),
                   'bucket_size_mean': round(float(bucket_size.mean()), 3),
                   'bucket_size_median': float(np.median(bucket_size)),
                   'bucket_size_max': int(bucket_size.max()),
                   'bucket_size_min': int(bucket_size.min())},
        'inputs': {'base_index_sha256': sha256(OUT / 'pairs_hardneg.npz'),
                   'tiger_sid_sha256': sha256(REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json')},
    }
    (OUT / 'pairs_bucketsize.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main()
