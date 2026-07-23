#!/usr/bin/env python3
"""Task #92 — RQ-VAE per-layer Gromov δ-hyperbolicity (4-point condition sampling).

Algorithm:
    For each layer i in [0, 1, ..., L-1]:
      Extract residual vectors r_i = encoded_x - sum(quant[0..i])
      Sample N random 4-tuples (a,b,c,d) from r_i
      For each tuple, compute 3 pair-sums:
        s1 = d(a,b) + d(c,d)
        s2 = d(a,c) + d(b,d)
        s3 = d(a,d) + d(b,c)
      delta_4pt = (max(s1,s2,s3) - second_largest) / 2
      Layer δ = robust statistic over N samples (95th percentile or max)

    Interpretation:
      δ ≈ 0     → tree-like (pure hierarchical structure)
      δ large   → flat / cyclic structure (non-hyperbolic)

Usage:
    python scripts/task92_rqvae_delta_hyperbolicity.py \
        --ckpt products/task73/rqvae_tiger_sas/Jul-22-2026_14-32-55/best_loss_model.pth \
        --emb ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_128.npy \
        --num_samples 5000 \
        --device cuda:0 \
        --output verdicts/task92_rqvae_delta_hyperbolicity.json
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Ensure RQ-VAE module is importable
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/RQVAE')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--emb', required=True)
    p.add_argument('--num_samples', type=int, default=5000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=2048,
                   help='Batch size for RQ-VAE forward pass')
    return p.parse_args()


def load_rqvae(ckpt_path, device):
    """Load RQ-VAE model from checkpoint, infer args from state_dict."""
    from models.rqvae import RQVAE
    from models.rq import ResidualVectorQuantizer
    from models.layers import MLPLayers
    from models.vq import VectorQuantizer

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']

    # Infer dimensions from state_dict keys/shapes
    # Encoder layers: encoder.linears.{i}.weight
    # num_emb_list from vq_layers embeddings: rq.vq_layers.{i}.embedding.weight
    num_emb_list = []
    for k, v in state.items():
        m = k.find('rq.vq_layers.')
        if m >= 0:
            # e.g. rq.vq_layers.0.embedding.weight  shape [256, 128]
            tail = k[m + len('rq.vq_layers.'):]
            idx_str = tail.split('.')[0]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            while len(num_emb_list) <= idx:
                num_emb_list.append(None)
            if 'embedding.weight' in tail:
                num_emb_list[idx] = v.shape[0]
    if any(n is None for n in num_emb_list):
        raise RuntimeError(f"Failed to infer num_emb_list from {ckpt_path}: got {num_emb_list}")

    # Infer encoder layer dims (handle both 'linears' and 'mlp_layers' naming)
    enc_keys = [k for k in state if (k.startswith('encoder.linears.') or k.startswith('encoder.mlp_layers.')) and k.endswith('.weight')]
    # Sort by numeric layer index (since keys like 'mlp_layers.1' vs 'mlp_layers.10' are out of lex order)
    enc_keys = sorted(enc_keys, key=lambda k: int(k.split('.')[-2]))
    encode_layer_dims = [state[k].shape[1] for k in enc_keys] + [state[enc_keys[-1]].shape[0]]
    decode_layer_dims = list(reversed(encode_layer_dims))

    # Build args namespace
    class Args:
        pass

    args = Args()
    args.e_dim = encode_layer_dims[-1]
    args.num_emb_list = num_emb_list
    args.layers = list(encode_layer_dims[1:-1])  # hidden layers (excluding input and e_dim)
    args.dropout_prob = 0.0
    args.bn = False
    args.loss_type = 'mse'
    args.quant_loss_weight = 1.0
    args.beta = 0.25
    args.vq_type = 'vq'
    args.sk_epsilons = [0.0] * len(num_emb_list)
    args.kmeans_init = False
    args.kmeans_iters = 100
    args.sk_iters = 100

    # Probe VectorQuantizer defaults (codebook_dim = e_dim by default)
    from models.vq import VectorQuantizer as VQ
    # Check VectorQuantizer init args — assume codebook_dim = e_dim, kmeans_init=False
    model = RQVAE(args, in_dim=encode_layer_dims[0])

    # Set state_dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # Some implementations name differently; warn but continue
        print(f"[WARN] Missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model.to(device)
    model.eval()
    return model, args


def extract_per_layer_residuals(model, embeddings_np, device, batch_size):
    """Run RQ-VAE encoder+forward, return list of (N, e_dim) residuals per layer.

    Returns:
        residuals: list of N x e_dim tensors. residuals[0] = encoded (before any quant),
                   residuals[i] = encoded - sum(quant[0..i-1]) for i >= 1.
                   Total L+1 entries (L = number of codebooks).
        codebook_assignments: list of N x 1 tensors with index per layer (L entries).
    """
    embeddings = torch.from_numpy(embeddings_np).float().to(device)
    N = embeddings.shape[0]

    # Encode all
    encoded = model.encoder(embeddings)  # (N, e_dim)
    e_dim = encoded.shape[1]

    # Run quantization layer by layer to track residuals
    residuals = [encoded.cpu().clone()]  # layer 0: raw encoded
    codebook_assignments = []

    residual = encoded
    for i, quantizer in enumerate(model.rq.vq_layers):
        x_res, loss, indices = quantizer(residual)
        residual = residual - x_res
        residuals.append(residual.cpu().clone())
        codebook_assignments.append(indices.cpu())

    return residuals, codebook_assignments


def gromov_diameter_sample(point_cloud: torch.Tensor, num_samples: int = 2000, seed: int = 0):
    """Approximate diameter via random pair sampling (avoid full N×N matrix)."""
    g = torch.Generator()
    g.manual_seed(seed)
    N, D = point_cloud.shape
    idx = torch.randint(0, N, (num_samples, 2), generator=g)
    p = point_cloud[idx[:, 0]]
    q = point_cloud[idx[:, 1]]
    dists = torch.linalg.norm(p - q, dim=1)
    return float(dists.max())


def gromov_delta_4point(point_cloud: torch.Tensor, num_samples: int = 5000, seed: int = 42):
    """Compute Gromov δ-hyperbolicity via 4-point sampling.

    Args:
        point_cloud: (N, D) tensor on CPU
        num_samples: number of 4-tuples to sample
        seed: random seed

    Returns:
        dict with delta_max, delta_95, delta_median, num_samples
    """
    g = torch.Generator()
    g.manual_seed(seed)
    N, D = point_cloud.shape

    # Sample 4 * num_samples indices with replacement (cheaper than all-pairs)
    idx = torch.randint(0, N, (num_samples, 4), generator=g)

    # Compute all pairwise distances in batch (memory-friendly)
    # Distance: Euclidean
    pc = point_cloud  # (N, D)
    a = pc[idx[:, 0]]
    b = pc[idx[:, 1]]
    c = pc[idx[:, 2]]
    d = pc[idx[:, 3]]

    dab = torch.linalg.norm(a - b, dim=1)
    dac = torch.linalg.norm(a - c, dim=1)
    dad = torch.linalg.norm(a - d, dim=1)
    dbc = torch.linalg.norm(b - c, dim=1)
    dbd = torch.linalg.norm(b - d, dim=1)
    dcd = torch.linalg.norm(c - d, dim=1)

    s1 = dab + dcd  # pairs (a,b) and (c,d)
    s2 = dac + dbd  # pairs (a,c) and (b,d)
    s3 = dad + dbc  # pairs (a,d) and (b,c)

    stacked = torch.stack([s1, s2, s3], dim=1)  # (num_samples, 3)
    sorted_, _ = torch.sort(stacked, dim=1)
    delta = (sorted_[:, -1] - sorted_[:, -2]) / 2.0  # (num_samples,)

    return {
        'delta_max': float(delta.max()),
        'delta_95': float(torch.quantile(delta, 0.95)),
        'delta_median': float(delta.median()),
        'delta_mean': float(delta.mean()),
        'num_samples': int(num_samples),
        'point_cloud_size': N,
        'point_cloud_dim': D,
        # Approximate diameter via sampled pairwise distance (avoid N×N matrix)
        'diameter_approx': float(gromov_diameter_sample(pc, num_samples=2000, seed=seed + 1)),
    }


def main():
    args = parse_args()
    print(f"[Task #92] RQ-VAE δ-hyperbolicity measurement")
    print(f"  ckpt: {args.ckpt}")
    print(f"  embedding: {args.emb}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  device: {args.device}")

    # Load embeddings
    embeddings_np = np.load(args.emb)
    print(f"  embedding shape: {embeddings_np.shape}")

    # Load RQ-VAE
    model, model_args = load_rqvae(args.ckpt, args.device)
    num_layers = len(model_args.num_emb_list)
    print(f"  num codebook layers: {num_layers}")
    print(f"  num_emb_list: {model_args.num_emb_list}")
    print(f"  e_dim: {model_args.e_dim}")

    # Extract per-layer residuals
    residuals, assignments = extract_per_layer_residuals(
        model, embeddings_np, args.device, args.batch_size
    )
    print(f"  residual shapes: {[r.shape for r in residuals]}")

    # Compute δ per layer
    layer_results = []
    for i, r in enumerate(residuals):
        label = f"layer_{i}" if i == 0 else f"layer_{i}_after_quant_{i-1}"
        if i == 0:
            label = "raw_encoded"
        else:
            label = f"residual_after_layer_{i-1}"
        print(f"\n[Layer {i}/{num_layers}] {label}, shape={tuple(r.shape)}, norm_mean={r.norm(dim=-1).mean().item():.4f}")
        t0 = time.time()
        result = gromov_delta_4point(r, num_samples=args.num_samples, seed=args.seed)
        elapsed = time.time() - t0
        result['layer'] = i
        result['label'] = label
        result['elapsed_sec'] = elapsed
        result['residual_norm_mean'] = float(r.norm(dim=-1).mean())
        result['residual_norm_std'] = float(r.norm(dim=-1).std())
        layer_results.append(result)
        print(f"  δ_max={result['delta_max']:.4f}, δ_95={result['delta_95']:.4f}, δ_median={result['delta_median']:.4f}, diameter={result['diameter_approx']:.4f} ({elapsed:.1f}s)")

    # Aggregate
    output = {
        'task': 'task92_rqvae_delta_hyperbolicity',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ckpt': args.ckpt,
        'embedding_file': args.emb,
        'num_samples': args.num_samples,
        'num_layers': num_layers,
        'num_emb_list': model_args.num_emb_list,
        'e_dim': model_args.e_dim,
        'seed': args.seed,
        'layers': layer_results,
        'interpretation': (
            'δ ≈ 0 means tree-like (hyperbolic). '
            'δ > 0 means increasingly non-hyperbolic / flat / cyclic. '
            'A growing δ across layers means residual space becomes less tree-like.'
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Task #92] Result saved to {args.output}")
    print(f"\n=== Summary ===")
    for r in layer_results:
        print(f"  Layer {r['layer']:>2} ({r['label']:<35}): δ_max={r['delta_max']:.4f}, δ_95={r['delta_95']:.4f}")


if __name__ == '__main__':
    main()