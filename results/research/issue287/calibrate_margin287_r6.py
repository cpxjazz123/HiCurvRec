"""Issue #287 round 6: the frozen margin for the hard-negative ranking.

The margin is calibrated **once** on the frozen TIGER checkpoint, over the frozen hard-negative index
(the training-graph pair set and the mined false positives).  No value of the candidate is used and
there is no scan over margins: the rule is fixed as the median of ``d_negative - d_positive`` over the
frozen pairs, exactly the rule the earlier rounds used - only the distance space changed (the item
representation instead of the selected L1 codeword).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path('.').resolve()
TREE = REPO / 'stage2_RQ-VAE' / 'TIGER_hardneg_pairrank_rq'
sys.path.insert(0, str(TREE))

from model import RQVAE            # noqa: E402  (the tree's own model)
import train_rqvae as trainer      # noqa: E402  (the tree's frozen constants)

from hard_negative_loss import HardNegativeIndex, encoder_distances  # noqa: E402

OUT = REPO / 'results/research/issue287'
INDEX = OUT / 'pairs_hardneg.npz'
SID = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json'
CKPT = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    index = HardNegativeIndex.load(INDEX)
    embeddings = np.asarray(np.load(trainer.EMBEDDING_FILE), dtype='float32')
    config = trainer._tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE)
    payload = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any('encoder' in key for key in missing):
        raise SystemExit('the encoder weights are missing from the frozen checkpoint: '
                         + ', '.join(sorted(k for k in missing if 'encoder' in k)))
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(embeddings)).numpy().astype(np.float64)

    anchor, pos, neg = index.anchor, index.pos, index.neg
    d_pos = np.linalg.norm(z[pos] - z[anchor], axis=1)
    d_neg = np.linalg.norm(z[neg] - z[anchor], axis=1)
    gap = d_neg - d_pos
    median_gap = float(np.median(gap))
    # The earlier rounds' rule is the median of (d_negative - d_positive).  Here the mined false
    # positives are *already closer* than the behavioural positives (that is the mechanism's
    # premise), so the median gap is negative; the margin is its magnitude, which keeps the hinge
    # inactive for the median pair exactly as in the earlier rounds.
    margin = abs(median_gap)
    out = {'schema': 'genrec.issue287.margin_r6/1',
           'rule': 'margin = |median(d_negative - d_positive)| over the frozen TIGER checkpoint\'s item '
                   'representation and the frozen hard-negative index (computed once, before any '
                   'candidate training); the magnitude is used because the mined false positives are '
                   'already closer than the behavioural positives, so the median gap is negative',
           'margin': margin,
           'distance_space': 'the encoder bottleneck (the item representation)',
           'd_positive_mean': round(float(d_pos.mean()), 6),
           'd_positive_median': round(float(np.median(d_pos)), 6),
           'd_negative_mean': round(float(d_neg.mean()), 6),
           'd_negative_median': round(float(np.median(d_neg)), 6),
           'gap_median': round(float(np.median(gap)), 6),
           'gap_q25': round(float(np.quantile(gap, 0.25)), 6),
           'gap_q75': round(float(np.quantile(gap, 0.75)), 6),
           'share_positive_closer': round(float((gap > 0).mean()), 6),
           'pairs': int(len(anchor)),
           'missing_encoder_keys': sorted(missing),
           'unexpected_keys': sorted(unexpected),
           'inputs': {'index_sha256': sha256(INDEX),
                      'sid_sha256': sha256(SID),
                      'checkpoint_sha256': sha256(CKPT)}}
    (OUT / 'MARGIN_287_R6.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
