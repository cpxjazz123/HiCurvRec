"""Issue #287 round 4: the L1-scale margin for the frozen #274 pair index.

Round 1 used Issue #274's margin (0.2529), which was calibrated on the **final-q** distance scale.
The L1 codeword scale is 0.19-0.25, so the hinge was saturated (margin satisfaction 0.113) and the
term behaved like an unhinged contraction of the L1 codebook.  This script calibrates the same
frozen rule (the median of d_negative - d_positive) on the frozen TIGER's **L1** codebook over the
frozen #274 index, once, before any candidate training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'
INDEX = REPO / 'results/research/issue274/pairs_index.npz'
SID = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json'
CKPT = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'


def main() -> None:
    data = np.load(INDEX)
    pos_b, pos_c, neg_d, neg_e = data['pos_b'], data['pos_c'], data['neg_d'], data['neg_e']
    codes = np.zeros((24587, 3), dtype=np.int64)
    raw = json.loads(SID.read_text())
    for item, value in raw.items():
        codes[int(item)] = [int(v) for v in value[:3]]
    payload = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    codebook = state['rq.vq_layers.0.embed.weight'].float().numpy()
    c_l1 = codebook[codes[:, 0]]
    d_pos = np.linalg.norm(c_l1[pos_b] - c_l1[pos_c], axis=1)
    d_neg = np.linalg.norm(c_l1[neg_d] - c_l1[neg_e], axis=1)
    gap = d_neg - d_pos
    out = {'schema': 'genrec.issue287.margin274_l1/1',
           'rule': 'margin = median(d_negative - d_positive) over the frozen TIGER L1 codebook and the '
                   'frozen #274 pair index (computed once, before any candidate training)',
           'margin': float(np.median(gap)),
           'd_positive_mean': round(float(d_pos.mean()), 6),
           'd_negative_mean': round(float(d_neg.mean()), 6),
           'gap_median': round(float(np.median(gap)), 6),
           'gap_q25': round(float(np.quantile(gap, 0.25)), 6),
           'gap_q75': round(float(np.quantile(gap, 0.75)), 6),
           'share_positive_closer': round(float((gap > 0).mean()), 6),
           'pairs': int(len(gap)),
           'inputs': {'index': str(INDEX.relative_to(REPO)),
                      'sid_sha256': __import__('hashlib').sha256(SID.read_bytes()).hexdigest()}}
    (OUT / 'MARGIN_274_L1.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
