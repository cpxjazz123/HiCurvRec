"""Task #137 B-arm Stage 2 — M=2 κ_max=0.5 codebook generation.

Fork of task137_stage2_codebook_free_curv.py, hardcoded for B-arm ckpt path
(products/task137/train/arm_B_M2/) and curv0.5 output suffix.

R11.3: Use GPU 0 (R7 — C-arm Stage 1 in progress on GPU 1).
R11.3: If SID unique > 500, Stage 3 ready. If SID unique < 50, M=2 also collapses.
"""
import collections
import json
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


def get_indices_count(all_indices_str):
    cnt = collections.defaultdict(int)
    for s in all_indices_str:
        cnt[s] += 1
    return cnt


def get_collision_item(all_indices_str):
    idx2ids = {}
    for i, s in enumerate(all_indices_str):
        idx2ids.setdefault(s, []).append(i)
    return [ids for s, ids in idx2ids.items() if len(ids) > 1]


def main():
    dataset = "Instruments"
    ckpt_path = "/home/wlia0047/ar57/wenyu/GeneRec/products/task137/train/arm_B_M2/best_loss_model.pth"
    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{dataset}/{dataset}_t5_hrqvae_poincare_B_M2_curv0.5.npy"
    device = torch.device("cuda:0")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  M={args['M']}, kappa_max={args['kappa_max']}, num_emb_list={args['num_emb_list']}")
    log.info(f"  best_loss = {ckpt.get('best_loss', '?')}, epoch = {ckpt.get('epoch', '?')}")

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

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    tt = 0
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break
        collision_groups = get_collision_item(all_indices_str)
        log.info(f"  collision resolution iter {tt}: {len(collision_groups)} groups")
        for collision_items in collision_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    counts = get_indices_count(all_indices_str)
    log.info(f"  total items: {len(all_indices_str)}")
    log.info(f"  unique SIDs: {len(set(all_indices_str.tolist()))}")
    log.info(f"  max conflict: {max(counts.values())}")
    log.info(f"  collision rate: {(len(all_indices_str) - len(set(all_indices_str.tolist()))) / len(all_indices_str):.4f}")

    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
    codes = []
    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)
    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        log.info(f"  resolving {len(duplicates)} duplicate SID via append digit")
        for duplicate in duplicates:
            dup_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(dup_indices):
                codes_array[idx, -1] = i

    log.info(f"  saving to {output_path}")
    log.info(f"  first 5 codes: {codes_array[:5]}")
    np.save(output_path, codes_array)

    meta = {
        'task': 'Task #137 B-arm Stage 2 (M=2 κ_max=0.5)',
        'ckpt_path': ckpt_path,
        'output_path': output_path,
        'M': args['M'],
        'kappa_max': args['kappa_max'],
        'final_kappa_per_layer': [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers],
        'n_items': len(codes_array),
        'n_unique': len(np.unique(codes_array, axis=0)),
        'shape': list(codes_array.shape),
    }
    meta_path = output_path.replace('.npy', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  meta saved: {meta_path}")


if __name__ == "__main__":
    main()
