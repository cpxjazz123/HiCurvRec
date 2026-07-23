"""Phase 2 P0.3 Stage 2.2: Joint Mixed-Curvature SID inference (custom).

P0.3 用的是 task58_p0_mixed_curvature_train.py 的 JointMixedCurvatureRQVAE,
其 ckpt 不是 GRID 标准 rqvae 格式 (state_dict 路径不同), 不能直接用 rkmeans_inference_flat。

本脚本: 加载 P0.3 ckpt, 用相同 Mixed-Curvature 距离函数 (per-sub-space H+E) 做 inference,
输出 merged_predictions_tensor.pt (12288, 3) int64 = L1/L2/L3 SID
(后续 Stage 3 推断时再加 dedup 列 → (12288, 4))
"""
import argparse
import os
import sys
import numpy as np
import torch

# 复用 train script 的 distance / lift 函数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from task58_p0_mixed_curvature_train import (
    JointMixedCurvatureRQVAE, INPUT_DIM, H_SUBSPACE_DIM, E_SUBSPACE_DIM, N_CLUSTERS
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument('--concat-emb', default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt')
    ap.add_argument('--out-path', required=True)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    DEVICE = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load ckpt
    print(f'[mixed-inf] loading {args.ckpt_path}', flush=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    model = JointMixedCurvatureRQVAE(
        dim=INPUT_DIM,
        h_dim=H_SUBSPACE_DIM,
        e_dim=E_SUBSPACE_DIM,
        n_clusters=N_CLUSTERS,
    ).to(DEVICE)
    # Load encoder/decoder
    model.encoder.load_state_dict(state['encoder'])
    model.decoder.load_state_dict(state['decoder'])
    model.norm_in.load_state_dict(state['norm_in'])
    # Load codebooks
    with torch.no_grad():
        model.C1.data = state['quantization_layer_list.0.centroids'].to(DEVICE)
        model.C2.data = state['quantization_layer_list.1.centroids'].to(DEVICE)
        model.C3.data = state['quantization_layer_list.2.centroids'].to(DEVICE)
        # radius_raw load (skip if shape mismatch - default init)
        try:
            for i, raw in enumerate(model.radius_raw):
                raw.data = state['radius_raw'][i].to(DEVICE)
        except Exception:
            pass
    model.eval()

    # Load embedding (与 train 同步: z-score normalize)
    x_all = torch.load(args.concat_emb, map_location='cpu', weights_only=False).float()
    print(f'[mixed-inf] x_all: {x_all.shape}', flush=True)
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std

    # Encode all items → 928-dim
    all_codes = torch.zeros((x_all.shape[0], 3), dtype=torch.int64)
    n = x_all.shape[0]
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            x_batch = x_all[start:end].to(DEVICE)
            z = model.encoder(model.norm_in(x_batch))
            for l in range(3):
                C = [model.C1, model.C2, model.C3][l]
                d, _, _ = model._compute_distance(z, C, l)
                idx = d.argmin(dim=1)
                all_codes[start:end, l] = idx.cpu()
                q = C[idx]
                z = z - q
            if (end // args.batch_size) % 5 == 0:
                print(f'  {end}/{n}', flush=True)
    print(f'[mixed-inf] all_codes: {all_codes.shape}, unique L1={all_codes[:,0].unique().numel()}, L2={all_codes[:,1].unique().numel()}, L3={all_codes[:,2].unique().numel()}', flush=True)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(all_codes, args.out_path)
    print(f'[mixed-inf] saved {args.out_path}', flush=True)


if __name__ == '__main__':
    main()