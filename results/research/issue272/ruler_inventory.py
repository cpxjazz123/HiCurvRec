"""Issue #272: one common-ruler inventory of the frozen reference arms."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / 'stage2_RQ-VAE/TIGER_global_j_rq'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(TREE))
from model import global_constraint as gc

OUT = REPO / 'results/research/issue272'
ARMS = {
    'TIGER_RQ-VAE': (REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json',
                     REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'),
    'TIGER_euclid_local_curve_supervision_rq': (
        REPO / 'results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/result/item_sids.json',
        REPO / 'results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/result/rqvae_best.pth'),
    'TIGER_calibrated_kappa_rq': (
        REPO / 'results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/item_sids.json',
        REPO / 'results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/rqvae_best.pth'),
    'TIGER_global_j_rq': (
        REPO / 'results/stage2_RQ-VAE/_runs/stage2_cc9a31a9325b439f98c8b1779aaa32aa/TIGER_global_j_rq/result/item_sids.json',
        REPO / 'results/stage2_RQ-VAE/_runs/stage2_cc9a31a9325b439f98c8b1779aaa32aa/TIGER_global_j_rq/result/rqvae_best.pth'),
}


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    graph = gc.load_graph()
    out = {'schema': 'genrec.issue272.ruler_inventory/1', 'scale': gc.kappa_scale(),
           'ruler_factor': gc.RULER_FACTOR, 'graph_sha256': sha(REPO / gc.GRAPH),
           'arms': {}}
    for arm, (sid, ckpt) in ARMS.items():
        q = gc.load_hard_q(sid, ckpt)
        full = gc.full_j(q, graph)
        out['arms'][arm] = {'sid': str(sid), 'sid_sha256': sha(sid),
                            'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt),
                            'full_j': full}
        print(arm, full['mean_j'], {k: v['mean_j'] for k, v in full['strata'].items()}, flush=True)
    (OUT / 'ruler_inventory.json').write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == '__main__':
    main()
