"""Task #313 / Issue #35 — Stage 2 Sinkhorn 5 iter 推断 on r_l=[0.1,1,10] + s_l identity.

r_l=[0.1,1,10] (Issue #30 GO) + s_l=[1,1,1] identity (vs task312 的反向)
"""
import collections
import glob
import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.utils import EmbDataset
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def main():
    dataset = "Instruments"
    ckpt_base = "/home/wlia0047/ar57/wenyu/GeneRec/products/task313/hrqvae_issue35_rl_extreme_sl_identity"
    ckpt_pattern = f"{ckpt_base}/*/best_loss_model.pth"
    candidates = sorted(glob.glob(ckpt_pattern))
    if not candidates:
        raise FileNotFoundError(f"❌ No best_loss_model.pth in {ckpt_pattern}")
    ckpt_path = candidates[-1]

    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{dataset}/{dataset}_t5_hrqvae_issue35_rl_extreme_sl_identity.npy"
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=2 maps to cuda:0

    log = logging.getLogger("task313_issue35_gate2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  num_emb_list={ckpt_args.num_emb_list}, e_dim={ckpt_args.e_dim}")

    data = EmbDataset(ckpt_args.data_path)
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    model = HRQVAE(in_dim=data.dim,
                   num_emb_list=ckpt_args.num_emb_list,
                   e_dim=ckpt_args.e_dim,
                   layers=ckpt_args.layers,
                   dropout_prob=ckpt_args.dropout_prob,
                   bn=ckpt_args.bn,
                   loss_type=ckpt_args.loss_type,
                   quant_loss_weight=ckpt_args.quant_loss_weight,
                   beta=ckpt_args.beta,
                   kmeans_init=ckpt_args.kmeans_init,
                   kmeans_iters=ckpt_args.kmeans_iters,
                   sk_eps=ckpt_args.sk_epsilons,
                   sk_iters=ckpt_args.sk_iters)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info(f"  load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, num_workers=4, batch_size=64, shuffle=True, pin_memory=True)

    all_indices = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # Round 1: hard quantize (use_sk=False)
    for d in data_loader:
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
    all_indices_str_3 = np.array([" ".join(c) for c in all_indices])
    collision_3 = check_collision(all_indices_str_3)
    log.info(f"Round 1 (use_sk=False): {len(all_indices_str_3)} items, unique={len(set(all_indices_str_3))}, collision={1 - len(set(all_indices_str_3))/len(all_indices_str_3):.4f}")

    # Round 2: Sinkhorn 5 iter (use_sk=True)
    all_indices = []
    for d in data_loader:
        d = d.to(device)
        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
    all_indices_str_3 = np.array([" ".join(c) for c in all_indices])
    collision_3 = check_collision(all_indices_str_3)
    log.info(f"Round 2 (use_sk=True, Sinkhorn 5 iter): {len(all_indices_str_3)} items, unique={len(set(all_indices_str_3))}, collision={1 - len(set(all_indices_str_3))/len(all_indices_str_3):.4f}")

    # Dedup 4th digit
    arr = np.array([[int(s.split('_')[1].rstrip('>')) for s in code[:3]] for code in all_indices])
    N = arr.shape[0]
    fourth_col = np.zeros((N, 1), dtype=np.int64)
    seen = {}
    for i in range(N):
        key = tuple(arr[i].tolist())
        if key in seen:
            fourth_col[i, 0] = 1
        else:
            fourth_col[i, 0] = 0
            seen[key] = i
    sid = np.concatenate([arr, fourth_col], axis=1)
    log.info(f"  Final SID shape: {sid.shape}, unique={len(set(map(tuple, sid.tolist())))}")
    np.save(output_path, sid)
    log.info(f"  Saved {output_path}")


if __name__ == '__main__':
    main()