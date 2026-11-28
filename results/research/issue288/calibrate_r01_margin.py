"""Issue #288 R01: the frozen margin for the same-anchor ranking on the FINAL hard q.

The frozen rule is #287's: the margin is the median of (d_negative - d_positive) over the frozen
TIGER checkpoint's codewords and the frozen same-anchor index, computed once before the run.  R01
changes only the supervised representation: the final hard q = the sum of the three levels' hard
codewords (the #274 form) instead of the L1 codeword alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue288'
INDEX = REPO / 'results/research/issue287/pairs_same_anchor.npz'
SID = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json'
CKPT = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    data = np.load(INDEX)
    anchor, pos, neg = data['anchor'], data['pos'], data['neg']
    codes = np.zeros((24587, 3), dtype=np.int64)
    raw = json.loads(SID.read_text())
    for item, value in raw.items():
        codes[int(item)] = [int(v) for v in value[:3]]
    payload = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    codebooks = [state[f'rq.vq_layers.{i}.embed.weight'].float().numpy() for i in range(3)]
    q = np.zeros((24587, codebooks[0].shape[1]), dtype=np.float64)
    for level, cb in enumerate(codebooks):
        q += cb[codes[:, level]]
    d_pos = np.linalg.norm(q[pos] - q[anchor], axis=1)
    d_neg = np.linalg.norm(q[neg] - q[anchor], axis=1)
    gap = d_neg - d_pos
    margin = float(np.median(gap))
    out = {'schema': 'genrec.issue288.margin_r01/1',
           'rule': 'margin = median(d_negative - d_positive) over the frozen TIGER final hard q and '
                   'the frozen same-anchor index (computed once, before the R01 run)',
           'supervised_representation': 'the final hard q = sum of the three levels\' hard codewords',
           'margin': margin,
           'd_positive_mean': round(float(d_pos.mean()), 6),
           'd_positive_median': round(float(np.median(d_pos)), 6),
           'd_negative_mean': round(float(d_neg.mean()), 6),
           'd_negative_median': round(float(np.median(d_neg)), 6),
           'gap_median': round(float(np.median(gap)), 6),
           'gap_q25': round(float(np.quantile(gap, 0.25)), 6),
           'gap_q75': round(float(np.quantile(gap, 0.75)), 6),
           'share_positive_closer': round(float((gap > 0).mean()), 6),
           'pairs': int(len(anchor)),
           'inputs': {'index_sha256': sha(INDEX), 'sid_sha256': sha(SID),
                      'checkpoint_sha256': sha(CKPT)}}
    (OUT / 'MARGIN_R01.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
