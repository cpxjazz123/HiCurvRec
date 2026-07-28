"""Task #207 Stage 2: SID codebook inference from best_loss checkpoint.

Usage:
    python3 scripts/task207_stage2_infer.py <arm> <ckpt_path> <output_npy>

    arm: hyp | euc
    ckpt_path: path to best_loss_model.pth
    output_npy: path to output .npy file
"""
import sys, collections, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HG-Rec'))
import torch
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from model.hrqvae import HRQVAE
from tqdm import tqdm

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <arm> <ckpt_path> <output_npy>", file=sys.stderr)
        sys.exit(1)

    arm = sys.argv[1]
    ckpt_path = sys.argv[2]
    output_path = sys.argv[3]
    data_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'

    print(f"[Stage 2 {arm}] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    args = ckpt['args']
    state_dict = ckpt['state_dict']

    print(f"[Stage 2 {arm}] Config: beta={args.beta}, num_emb_list={args.num_emb_list}, "
          f"loss_type={args.loss_type}, curvatures={getattr(args, 'curvatures', None)}")

    data = EmbDataset(data_path)
    n_emb = args.num_emb_list
    # Determine euclidean_qloss from checkpoint args
    euc_qloss = getattr(args, 'euclidean_qloss', False)

    model = HRQVAE(
        in_dim=data.dim, num_emb_list=n_emb, e_dim=args.e_dim,
        layers=args.layers, dropout_prob=0.0, bn=False,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight, beta=args.beta,
        kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons, sk_iters=args.sk_iters,
        curvature_list=(getattr(args, 'curvatures', None)
                        if getattr(args, 'curvatures', None) is not None else [1.0, 1.0, 1.0]),
        euclidean_qloss=euc_qloss,
    )
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    print(f"[Stage 2 {arm}] Model on {device}")

    loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=False, pin_memory=True)
    all_indices = []
    all_indices_str = []
    prefix = ['<a_{}>', '<b_{}>', '<c_{}>', '<d_{}>']

    for d in tqdm(loader, desc=f'Stage 2 {arm}'):
        d = d.to(device)
        with torch.no_grad():
            indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    n_total = len(all_indices_str)
    n_unique = len(set(all_indices_str))
    print(f"[Stage 2 {arm}] PRE-resolve: items={n_total}, unique={n_unique} ({n_unique/n_total*100:.1f}%)")

    # Collision resolve (use_sk=True on colliding items)
    tt = 0
    while tt < 30 and len(all_indices_str) != len(set(all_indices_str)):
        idx2ids = collections.defaultdict(list)
        for i, s in enumerate(all_indices_str):
            idx2ids[s].append(i)
        for s, ids in idx2ids.items():
            if len(ids) > 1:
                d = data[ids].to(device)
                with torch.no_grad():
                    indices = model.get_indices(d, use_sk=True)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item, index in zip(ids, indices):
                    code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                    all_indices[item] = code
                    all_indices_str[item] = str(code)
        tt += 1
    print(f"[Stage 2 {arm}] POST-resolve iterations: {tt}, "
          f"unique={len(set(all_indices_str))} ({(len(set(all_indices_str))/n_total)*100:.1f}%)")

    # Convert to int array + 4th-digit dedup
    codes = []
    for value in all_indices:
        codes.append([int(t.split('_')[1].strip('>')) for t in value])
    codes_array = np.array(codes, dtype=int)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    for duplicate in duplicates:
        dup_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(dup_indices):
            codes_array[idx, -1] = i

    print(f"[Stage 2 {arm}] Final shape: {codes_array.shape}, "
          f"unique SID: {len(np.unique(codes_array, axis=0))}")
    np.save(output_path, codes_array)
    print(f"[Stage 2 {arm}] Saved: {output_path}")

if __name__ == '__main__':
    main()
