"""Issue #287 round 6: the hard-negative index.

The measured obstacle of rounds 1-5 is a monotone trade-off: every mechanism that pulls the
behaviourally related items together also lets the unrelated items into the same L1 bucket, because
the ranking never pushes the items that *already* share a code away.  Round 6 therefore changes the
**negative construction**: the negative of an anchor is an item that **already shares the anchor's L1
code** and has no observed training-graph relation with it - a measured false positive, the item
population that the metric N counts.

The index is built offline from the frozen assets:

* the **positive** column is round 2's frozen, curvature-screened positive (the anchor's
  curvature-selected direct partner);
* the **anchor's direct neighbours** are emitted as a CSR so that the miner can exclude every
  behaviourally related item from the negative pool;
* the **initial negative** column is mined once from the frozen TIGER checkpoint export
  (`results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json`): the candidates are the items that share the
  anchor's L1 code in that export and are not the anchor's direct neighbours; the pick is a seeded
  uniform choice over the candidates sorted by item id (deterministic, unbiased).

During training the trainer refreshes the negative column from the **current** assignment (the same
rule, the current partition), so the negatives stay the items that wrongly share the code now.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'
NODES = 24587
PICK_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def edge_lists() -> tuple[np.ndarray, np.ndarray]:
    """The directed edge lists of the visible training graph."""
    out = []
    for name in ('visible_forward.npz', 'visible_reverse.npz'):
        blob = np.load(REPO / 'assets/train' / name)
        indptr = blob['indptr']; indices = blob['indices']
        nodes = len(indptr) - 1
        rows = np.repeat(np.arange(nodes), np.diff(indptr))
        out.append(np.stack([rows.astype(np.int64), indices.astype(np.int64)], axis=1))
    return out[0], out[1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = np.load(OUT / 'pairs_same_anchor.npz')
    anchor = base['anchor'].astype(np.int64)
    pos = base['pos'].astype(np.int64)
    holdout = base['is_holdout'].astype(bool)

    forward, reverse = edge_lists()
    edges = np.concatenate([forward, reverse], axis=0)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    unique_rows, row_starts = np.unique(edges[:, 0], return_index=True)
    nbr_ptr = np.zeros(NODES + 1, dtype=np.int64)
    counts = np.bincount(edges[:, 0], minlength=NODES)
    np.cumsum(counts, out=nbr_ptr[1:])
    nbr_items = edges[:, 1].astype(np.int64)

    raw = json.loads((REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json').read_text())
    codes = np.zeros(NODES, dtype=np.int64)
    for item, value in raw.items():
        codes[int(item)] = int(value[0])
    by_code: dict[int, list[int]] = {}
    for item, code in enumerate(codes):
        by_code.setdefault(int(code), []).append(item)

    rng = np.random.default_rng(PICK_SEED)
    negatives = np.full(len(anchor), -1, dtype=np.int64)
    pool_sizes = []
    for row, (a, p) in enumerate(zip(anchor.tolist(), pos.tolist())):
        neighbours = set(nbr_items[nbr_ptr[a]:nbr_ptr[a + 1]].tolist())
        neighbours.add(int(p))
        candidates = [item for item in by_code.get(int(codes[a]), []) if item not in neighbours]
        pool_sizes.append(len(candidates))
        if candidates:
            negatives[row] = int(rng.choice(np.asarray(sorted(candidates), dtype=np.int64)))
    usable = negatives >= 0
    rows = np.nonzero(usable)[0]
    anchor = anchor[rows]; pos = pos[rows]; negatives = negatives[rows]; holdout = holdout[rows]

    path = OUT / 'pairs_hardneg.npz'
    np.savez(path, anchor=anchor, pos=pos, neg=negatives, is_holdout=holdout,
             nbr_ptr=nbr_ptr, nbr_items=nbr_items)
    manifest = {
        'schema': 'genrec.issue287.hardneg/1',
        'path': str(path), 'sha256': sha256(path),
        'rules': {
            'positive': "round 2's frozen curvature-screened positive (the anchor's direct partner)",
            'negative': 'an item that already shares the anchor L1 code in the frozen TIGER export '
                        'and has no observed training-graph relation with the anchor; a seeded '
                        'uniform pick over the candidates sorted by item id',
            'negative_refresh': 'the trainer re-mines the same rule from the current assignment every '
                                'HARDNEG_REFRESH_EPOCHS epochs (the rule is frozen; only the partition '
                                'it reads changes)',
            'curvature_role': 'offline selection of the positive inside the comparability group only; '
                              'the curvature never enters the loss, the margin or a weight',
            'false_negative_note': 'the negative is a measured false positive in the frozen export, '
                                    'not a proof of behavioural irrelevance',
        },
        'counts': {'groups': int(len(anchor)),
                   'anchors': int(len(np.unique(anchor))),
                   'train_groups': int((~holdout).sum()),
                   'holdout_groups': int(holdout.sum()),
                   'mean_false_positive_pool': round(float(np.mean(pool_sizes)), 3),
                   'median_false_positive_pool': float(np.median(pool_sizes)),
                   'items_touched': int(len(np.unique(np.concatenate([anchor, pos, negatives]))))},
        'inputs': {'base_index_sha256': sha256(OUT / 'pairs_same_anchor.npz'),
                   'tiger_sid_sha256': sha256(REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json')},
    }
    (OUT / 'pairs_hardneg.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main()
