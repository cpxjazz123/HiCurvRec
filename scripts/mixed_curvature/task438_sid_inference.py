#!/usr/bin/env python3
"""task57_sid_inference.py — Task #468: SID inference for PerLayerMixedCurvatureRQVAE."""
import argparse
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

from task56_per_layer_train import PerLayerMixedCurvatureRQVAE, INPUT_DIM, N_CLUSTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument(
        '--concat-emb',
        default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt',
    )
    ap.add_argument('--out-path', required=True)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    DEVICE = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    print(f'[task57-inf] loading ckpt: {args.ckpt_path}', flush=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

    h_offsets = tuple(ckpt.get('h_offsets', [768, 768, 768]))
    h_dims = tuple(ckpt.get('h_dims', [32, 32, 32]))
    variant = ckpt.get('variant', '?')
    print(f'[task57-inf] variant={variant} h_offsets={h_offsets} h_dims={h_dims}', flush=True)

    model = PerLayerMixedCurvatureRQVAE(
        dim=INPUT_DIM,
        h_offsets=h_offsets,
        h_dims=h_dims,
        n_clusters=N_CLUSTERS,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    x_all = torch.load(args.concat_emb, map_location='cpu', weights_only=False).float()
    print(f'[task57-inf] x_all: {tuple(x_all.shape)}', flush=True)
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std
    print(f'[task57-inf] z-score normalized', flush=True)

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

    print(
        f'[task57-inf] all_codes: {tuple(all_codes.shape)} '
        f'L1_unique={all_codes[:, 0].unique().numel()} '
        f'L2_unique={all_codes[:, 1].unique().numel()} '
        f'L3_unique={all_codes[:, 2].unique().numel()}',
        flush=True,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(all_codes, args.out_path)
    print(f'[task57-inf] saved {args.out_path}', flush=True)


if __name__ == '__main__':
    main()