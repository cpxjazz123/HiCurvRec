"""Issue #287 round 2: the frozen margin for the same-anchor L1 ranking.

The margin is calibrated **once** on the frozen TIGER checkpoint's L1 codebook, over the frozen
same-anchor index (the training-graph pair set).  No value of the candidate is used, no scan over
margins: the rule is fixed as the median of (d_negative - d_positive) over the frozen pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'
INDEX = OUT / 'pairs_same_anchor_r5.npz'
SID = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json'
CKPT = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'


def main() -> None:
    data = np.load(INDEX)
    anchor, pos, neg = data['anchor'], data['pos'], data['neg']
    codes = np.zeros((24587, 3), dtype=np.int64)
    raw = json.loads(SID.read_text())
    for item, value in raw.items():
        codes[int(item)] = [int(v) for v in value[:3]]
    payload = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    codebook = state['rq.vq_layers.0.embed.weight'].float().numpy()
    c_l1 = codebook[codes[:, 0]]
    d_pos = np.linalg.norm(c_l1[anchor] - c_l1[pos], axis=1)
    d_neg = np.linalg.norm(c_l1[anchor] - c_l1[neg], axis=1)
    gap = d_neg - d_pos
    margin = float(np.median(gap))
    out = {'schema': 'genrec.issue287.margin/1',
           'rule': 'margin = median(d_negative - d_positive) over the frozen TIGER L1 codebook and the '
                   'frozen same-anchor index (computed once, before any candidate training)',
           'margin': margin,
           'd_positive_mean': round(float(d_pos.mean()), 6), 'd_positive_median': round(float(np.median(d_pos)), 6),
           'd_negative_mean': round(float(d_neg.mean()), 6), 'd_negative_median': round(float(np.median(d_neg)), 6),
           'gap_median': round(float(np.median(gap)), 6),
           'gap_q25': round(float(np.quantile(gap, 0.25)), 6), 'gap_q75': round(float(np.quantile(gap, 0.75)), 6),
           'share_positive_closer': round(float((gap > 0).mean()), 6),
           'pairs': int(len(anchor)),
           'inputs': {'index_sha256': __import__('hashlib').sha256(INDEX.read_bytes()).hexdigest(),
                      'sid_sha256': __import__('hashlib').sha256(SID.read_bytes()).hexdigest()}}
    (OUT / 'MARGIN_287_R5.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
