"""Task #180 Stage 1 — LightGCN 1 层传播 + HRQ-VAE 训练入口.

Step 1: 离线预计算 item_emb_graph = LightGCN(item_emb, adj_norm, n_layers=1)
        输出 products/task180/item_emb_graph.npy (9922, 768)
Step 2: 把 item_emb_graph 当新的 Stage 1 输入, 喂给 Phase 0 修复版 HRQ-VAE.

注意: LightGCN 1 层传播公式:
    X_graph = (X + AX) / 2
其中 A = D^{-1/2} A_raw D^{-1/2} 已经预计算 (G2_cooccurrence_adj_norm.npz)
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task180_lightgcn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_emb_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--adj_norm_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task180/graph/G2_cooccurrence_adj_norm.npz")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--output_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task180/item_emb_graph.npy")
    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1. Load item embeddings
    log.info(f"Loading item embeddings from {args.item_emb_path}")
    df = pd.read_parquet(args.item_emb_path)
    embeddings = np.stack(df['embedding'].values, axis=0)  # (N, D)
    n_items, e_dim = embeddings.shape
    log.info(f"  Shape: {embeddings.shape}, dtype={embeddings.dtype}")

    # 2. Load normalized adjacency
    log.info(f"Loading normalized adjacency from {args.adj_norm_path}")
    adj_norm = sp.load_npz(args.adj_norm_path)
    log.info(f"  Shape: {adj_norm.shape}, nnz: {adj_norm.nnz}, avg degree: {adj_norm.nnz / n_items:.2f}")

    if adj_norm.shape[0] != n_items or adj_norm.shape[1] != n_items:
        raise ValueError(f"adj_norm shape {adj_norm.shape} != embedding shape ({n_items}, {n_items})")

    # 3. Convert to torch sparse on device
    adj_norm_coo = adj_norm.tocoo()
    indices = torch.tensor(np.array([adj_norm_coo.row, adj_norm_coo.col]), dtype=torch.long)
    values = torch.tensor(adj_norm_coo.data, dtype=torch.float32)
    adj_norm_torch = torch.sparse_coo_tensor(indices, values, size=adj_norm.shape).coalesce().to(device)
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # 4. LightGCN propagation (n_layers=1): X_graph = (X + AX) / 2
    log.info(f"LightGCN propagation: n_layers={args.n_layers}, dim={e_dim}, device={device}")
    out = X
    for layer_idx in range(args.n_layers):
        out_new = torch.sparse.mm(adj_norm_torch, out)
        out = (X + out_new) / (args.n_layers + 1)  # standard LightGCN mean

    # 5. Verify shape + unit-norm optional
    log.info(f"  X shape: {X.shape}, X norm mean: {X.norm(dim=-1).mean().item():.4f}")
    log.info(f"  X_graph shape: {out.shape}, X_graph norm mean: {out.norm(dim=-1).mean().item():.4f}")

    # 6. Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, out.cpu().numpy())
    log.info(f"  Saved: {args.output_path}")


if __name__ == '__main__':
    main()