"""Task #137 B-arm quick Stage 2 test — use current best_loss ckpt (ep 50)
to validate M=2 SID is not collapsed.

R11.3: Run on GPU 0 (R7 — B-arm Stage 1 in progress on GPU 1).
R11.3: If SID unique > 500 (M=2 expected ~9922), then full B-arm Stage 2/3/4 ready.
R11.3: If SID unique < 100, then M=2 also collapses (need deeper fix).
"""
import collections
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.utils import EmbDataset


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_collision_item(all_indices_str):
    idx2ids = {}
    for i, s in enumerate(all_indices_str):
        idx2ids.setdefault(s, []).append(i)
    return [ids for s, ids in idx2ids.items() if len(ids) > 1]


def main():
    ckpt_path = "/home/wlia0047/ar57/wenyu/GeneRec/products/task137/train/arm_B_M2/best_loss_model.pth"
    device = torch.device("cuda:0")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  M={args['M']}, kappa_max={args['kappa_max']}, num_emb_list={args['num_emb_list']}")
    log.info(f"  best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}")

    data = EmbDataset(args['data_path'])

    model = FreeCurvHRQVAE(
        in_dim=data.dim,
        num_emb_list=args['num_emb_list'],
        e_dim=args['e_dim'],
        M=args['M'],
        kappa_max=args['kappa_max'],
        layers=args['layers'],
        dropout_prob=0.0,
        bn=False,
        loss_type=args['loss_type'],
        quant_loss_weight=args['quant_loss_weight'],
        beta=args['beta'],
        kmeans_init=args['kmeans_init'],
        kmeans_iters=args['kmeans_iters'],
        sk_eps=args['sk_epsilons'],
        sk_iters=args['sk_iters'],
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    for li, vq in enumerate(model.hrq.vq_layers):
        kappas = vq.kappa_m().detach().cpu().tolist()
        log.info(f"  Layer {li} κ_m = [{', '.join(f'{k:+.4f}' for k in kappas)}]")

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=True, pin_memory=True)

    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
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

    # Single-pass Sinkhorn (max 5 iters — diagnostic only)
    all_indices_arr = np.array([eval(s) for s in all_indices_str])
    tt = 0
    while tt < 5 and not check_collision(all_indices_str):
        collision_groups = get_collision_item(all_indices_str)
        log.info(f"  Sinkhorn iter {tt}: {len(collision_groups)} groups")
        for collision_items in collision_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices_str[item] = str(code)
                all_indices_arr[item] = code
        tt += 1

    n_unique_after = len(set(all_indices_str.tolist()))
    log.info(f"  POST-Sinkhorn (max 5 iters): items={len(all_indices_str)}, "
             f"unique={n_unique_after}, collision_rate="
             f"{(len(all_indices_str) - n_unique_after) / len(all_indices_str):.4f}")

    # Per-layer codebook utilization
    layer_codes = np.array([[int(c.split('_')[1].strip('>')) for c in s]
                            for s in all_indices_str])
    for li in range(layer_codes.shape[1]):
        unique_codes = np.unique(layer_codes[:, li])
        log.info(f"  Layer {li} utilization: {len(unique_codes)} / {args['num_emb_list'][li]} "
                 f"({100*len(unique_codes)/args['num_emb_list'][li]:.1f}%)")

    # Decision
    if n_unique_after >= 500:
        log.info(f"  ✅ M=2 SID healthy — proceed with full Stage 2/3/4 after B-arm completes")
    elif n_unique_after >= 50:
        log.warning(f"  ⚠️ M=2 SID marginal ({n_unique_after} unique) — Stage 3 may have issues")
    else:
        log.error(f"  ❌ M=2 SID collapsed ({n_unique_after} unique) — need deeper fix")


if __name__ == "__main__":
    main()