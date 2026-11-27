"""Issue #271: freeze J_TIGER, the pair-table sampling error and J_max before training."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
TREE = REPO / 'stage2_RQ-VAE/TIGER_global_j_rq'
sys.path.insert(0, str(TREE))
from model import global_constraint as gc

OUT = REPO / 'results/research/issue271/geometry'
OUT.mkdir(parents=True, exist_ok=True)
TIGER = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE'
K269 = REPO / 'results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result'


def weighted_values(q, sample):
    i = torch.as_tensor(sample['i'], dtype=torch.long)
    j = torch.as_tensor(sample['j'], dtype=torch.long)
    g = torch.as_tensor(sample['g'], dtype=torch.float64)
    qt = torch.as_tensor(q, dtype=torch.float64)
    d = torch.sqrt(torch.clamp(((qt[i] - qt[j]) ** 2).sum(-1), min=0.0))
    ratio = d / (gc.RULER_FACTOR * gc.kappa_scale() * g)
    err = (ratio - 1.0) ** 2
    w = torch.as_tensor(sample['weight'], dtype=torch.float64)
    return (w * err).sum().item() / w.sum().item(), err.numpy(), w.numpy()


def main():
    graph = gc.load_graph()
    table, receipt = gc.build_pair_table(graph)
    gc.save_pair_table(OUT, table, receipt)
    result = {'table': receipt, 'pairs': {k: int(len(v)) for k, v in table.items()}}
    for name, sid, ckpt in (
            ('TIGER', TIGER / 'item_sids.json', TIGER / 'rqvae_best.pth'),
            ('K269', K269 / 'item_sids.json', K269 / 'rqvae_best.pth')):
        q = gc.load_hard_q(sid, ckpt)
        np.save(OUT / f'q_{name}.npy', q.astype(np.float32))
        full = gc.full_j(q, graph)
        sample_j, err, w = weighted_values(q, table)
        rng = np.random.default_rng(271)
        n = len(err)
        draws = np.empty(500)
        for draw in range(500):
            idx = rng.integers(0, n, n)
            draws[draw] = float((w[idx] * err[idx]).sum() / w[idx].sum())
        se = float(draws.std(ddof=1))
        bias = abs(sample_j - full['mean_j'])
        result[name] = {'full_j': full, 'sample_j': sample_j, 'bootstrap_se': se,
                        'sample_bias': bias, 'ruler_factor': gc.RULER_FACTOR,
                        'kappa_scale': gc.kappa_scale()}
        print(name, 'full', full['mean_j'], 'sample', sample_j, 'se', se, 'bias', bias, flush=True)
    epsilon = max(1e-4, 2 * result['TIGER']['bootstrap_se'], result['TIGER']['sample_bias'])
    result['epsilon'] = epsilon
    result['j_max'] = result['TIGER']['full_j']['mean_j'] + epsilon
    (OUT / 'threshold.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'J_TIGER': result['TIGER']['full_j']['mean_j'], 'epsilon': epsilon,
                      'J_max': result['j_max'], 'J_269': result['K269']['full_j']['mean_j']}, indent=1))


if __name__ == '__main__':
    main()
