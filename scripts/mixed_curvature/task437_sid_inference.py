"""Task #437 Stage 2.2: SID inference for HOffsetMixedCurvatureRQVAE.

Loads ckpt from task56_h_offset_train.py, runs forward pass on all items,
outputs (N, 3) int64 SID tensor (L1/L2/L3). Run `fix_sid_to_grid_format.py`
afterward to add dedup column and transpose to (4, N) format.

Stage 2.2 v2: 增加 z-score normalize (与训练 recipe 对齐, 修复坍缩).
"""
import argparse
import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

from task56_h_offset_train import HOffsetMixedCurvatureRQVAE, INPUT_DIM, N_CLUSTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument('--h-offset', type=int, required=True)
    ap.add_argument('--h-dim', type=int, required=True)
    ap.add_argument(
        '--concat-emb',
        default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt',
    )
    ap.add_argument('--out-path', required=True)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    DEVICE = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load ckpt
    print(f'[task56-inf] loading ckpt: {args.ckpt_path}', flush=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

    h_offset = ckpt.get('h_offset', args.h_offset)
    h_dim = ckpt.get('h_dim', args.h_dim)
    variant = ckpt.get('variant', f'hoff{h_offset:03d}_hdim{h_dim:03d}')
    print(f'[task56-inf] variant={variant} h_offset={h_offset} h_dim={h_dim}', flush=True)

    model = HOffsetMixedCurvatureRQVAE(
        dim=INPUT_DIM,
        h_dim=h_dim,
        h_offset=h_offset,
        n_clusters=N_CLUSTERS,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load embedding
    x_all = torch.load(args.concat_emb, map_location='cpu', weights_only=False).float()
    print(f'[task56-inf] x_all: {tuple(x_all.shape)}', flush=True)
    # Z-score normalize (must match training recipe)
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std
    print(f'[task56-inf] z-score normalized', flush=True)

    # Encode all items → 928-dim → 3-layer SID
    all_codes = torch.zeros((x_all.shape[0], 3), dtype=torch.int64)
    n = x_all.shape[0]
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            x_batch = x_all[start:end].to(DEVICE)
            z = model.encoder(model.norm_in(x_batch))
            Cs = [model.C1, model.C2, model.C3]
            for l in range(3):
                d, _, _ = model._compute_distance(z, Cs[l], l)
                idx = d.argmin(dim=1)
                all_codes[start:end, l] = idx.cpu()
                q = Cs[l][idx]
                z = z - q
            if (end // args.batch_size) % 5 == 0:
                print(f'  {end}/{n}', flush=True)

    print(
        f'[task56-inf] all_codes: {tuple(all_codes.shape)} '
        f'L1_unique={all_codes[:, 0].unique().numel()} '
        f'L2_unique={all_codes[:, 1].unique().numel()} '
        f'L3_unique={all_codes[:, 2].unique().numel()}',
        flush=True,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(all_codes, args.out_path)
    print(f'[task56-inf] saved {args.out_path}', flush=True)


if __name__ == '__main__':
    main()