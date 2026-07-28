"""Task #84 baseline codebook health check — diagnostic.

Compare baseline RQ-VAE (fixed κ poincare, original hrqvae.py) codebook
utilization vs Task #137 free-curv (collapse to 1-15 unique SID).

Goal: determine if HG-Rec baseline suffers from same codebook collapse pattern.
"""
import collections
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_collision_item(all_indices_str):
    idx2ids = {}
    for i, s in enumerate(all_indices_str):
        idx2ids.setdefault(s, []).append(i)
    return [ids for s, ids in idx2ids.items() if len(ids) > 1]


def main():
    ckpt_path = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
    device = torch.device("cuda:0")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  baseline arch: num_emb_list={getattr(args, 'num_emb_list', '?')}, "
             f"e_dim={getattr(args, 'e_dim', '?')}, layers={getattr(args, 'layers', '?')}, "
             f"loss_type={getattr(args, 'loss_type', '?')}")
    log.info(f"  best_loss = {ckpt.get('best_loss', '?')}, epoch = {ckpt.get('epoch', '?')}")

    data = EmbDataset(args.data_path)

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=0.0,
        bn=False,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=getattr(args, 'kmeans_init', False),
        kmeans_iters=getattr(args, 'kmeans_iters', 100),
        sk_eps=getattr(args, 'sk_eps', [0.0] * len(args.num_emb_list)),
        sk_iters=getattr(args, 'sk_iters', 100),
        curvature_list=getattr(args, 'curvature_list', None),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=True, pin_memory=True)

    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in data_loader:
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices_str.append(str(code))

    all_indices_str = np.array(all_indices_str)
    n_unique = len(set(all_indices_str.tolist()))
    collision_rate = (len(all_indices_str) - n_unique) / len(all_indices_str)
    log.info(f"  PRE-Sinkhorn: items={len(all_indices_str)}, unique={n_unique}, "
             f"collision_rate={collision_rate:.4f}")

    # Per-layer codebook utilization
    import ast
    layer_codes = np.array([[int(c.split('_')[1].strip('>')) for c in ast.literal_eval(s)]
                            for s in all_indices_str])
    for li in range(layer_codes.shape[1]):
        unique_codes = np.unique(layer_codes[:, li])
        n_e = args.num_emb_list[li] if hasattr(args, 'num_emb_list') else '?'
        log.info(f"  Layer {li} utilization: {len(unique_codes)} / {n_e} "
                 f"({100*len(unique_codes)/int(n_e):.1f}%)")

    log.info(f"  ✅ baseline loaded; collapse check complete")


if __name__ == "__main__":
    main()
