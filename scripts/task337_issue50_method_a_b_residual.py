"""
Task #337 / Issue #50 Method A + B (residual-level) — Real noise in HRQ-VAE space.

Use a loaded HG-Rec baseline (Task #84) HRQ-VAE checkpoint, extract per-layer
residuals for all 9922 items + the 159 near-duplicate pairs. Compare per-layer
pairwise L2 distances (real 'noise floor' from data) vs Task #339 Gate 0
arbitrary σ projections.

This script uses GPU 1 (idle). Does NOT touch GPU 0 (Issue #49 training).
"""
from __future__ import annotations
import sys, os, json, time, glob
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
import torch

BASE = Path('/home/wlia0047/ar57/wenyu/GeneRec')
DATA_DIR = BASE / 'HG-Rec' / 'dataset' / 'Instruments'

# Find Task #84 HRQ-VAE baseline ckpt (best_collision_model.pth = best HRQ-VAE,
# separate from T5-mini checkpoint)
CKPT_CANDIDATES = sorted(glob.glob(str(
    BASE / 'products' / 'task84' / 'ckpt' / 'Instruments' / '*' / 'best_collision_model.pth')))
if not CKPT_CANDIDATES:
    # Fallback: any healthy HRQ-VAE ckpt
    CKPT_CANDIDATES = sorted(glob.glob(str(
        BASE / 'products' / 'task194' / '*' / 'best_collision_model.pth')))
assert CKPT_CANDIDATES, 'No HRQ-VAE checkpoint available; need a baseline (Task #84) model'
CKPT_PATH = CKPT_CANDIDATES[-1]
print(f'Using checkpoint: {CKPT_PATH}')

from model.hrqvae import HRQVAE


def load_model(ckpt_path: str) -> HRQVAE:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    args = ckpt['args']  # Namespace in HRQ-VAE training ckpts
    sd = ckpt['state_dict']
    # args is Namespace; if dict, get otherwise use vars
    if not isinstance(args, dict):
        args = vars(args) if hasattr(args, '__dict__') else args
    model = HRQVAE(
        in_dim=768,
        num_emb_list=args['num_emb_list'] if isinstance(args, dict) else args.num_emb_list,
        e_dim=args['e_dim'] if isinstance(args, dict) else args.e_dim,
        layers=args['layers'] if isinstance(args, dict) else args.layers,
        dropout_prob=args['dropout_prob'] if isinstance(args, dict) else args.dropout_prob,
        bn=args['bn'] if isinstance(args, dict) else args.bn,
        loss_type=args['loss_type'] if isinstance(args, dict) else args.loss_type,
        quant_loss_weight=args['quant_loss_weight'] if isinstance(args, dict) else args.quant_loss_weight,
        beta=args['beta'] if isinstance(args, dict) else args.beta,
        kmeans_init=args['kmeans_init'] if isinstance(args, dict) else args.kmeans_init,
        kmeans_iters=args['kmeans_iters'] if isinstance(args, dict) else args.kmeans_iters,
        sk_eps=args['sk_epsilons'] if isinstance(args, dict) else args.sk_epsilons,
        sk_iters=args['sk_iters'] if isinstance(args, dict) else args.sk_iters,
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def get_residuals(model: HRQVAE, z: torch.Tensor):
    """Per-layer residuals (input before each VQ layer).

    Returns:
        sids: (N, 3) — per-layer codeword indices
        residuals: list of (N, e_dim) — residual[r] before layer r
    """
    sids = []
    residual = z.clone()
    residuals = [residual.clone()]  # input to layer 0
    for vq in model.hrq.vq_layers:
        x_res, _, indices = vq(residual, use_sk=False)
        residual = residual - x_res
        sids.append(indices)
        residuals.append(residual.clone())
    return torch.stack(sids, dim=-1), residuals


def main():
    print('=== Task #337 / Issue #50 Method A + B (HRQ-VAE residual level) ===')
    # Force GPU 1 (idle) per R7
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda:0')  # after visible devices=1, "gpu:0" = physical gpu:1

    print('Step 1: Load item embeddings')
    df = pd.read_parquet(DATA_DIR / 'item_emb.parquet')
    embeddings = np.stack([np.array(e, dtype=np.float32) for e in df['embedding']])

    print('Step 2: Load near-duplicate pairs (already done in main script)')
    with open(BASE / 'verdicts' / 'task337_issue50_method_b_near_duplicates.json') as f:
        b_result = json.load(f)
    pairs = [(p[0], p[1]) for p in b_result['sample_pairs']]
    print(f'  Loaded {len(pairs)} pairs')

    print('Step 3: Load HRQ-VAE model')
    model = load_model(CKPT_PATH)
    model.to(device)
    print(f'  Loaded, {sum(p.numel() for p in model.parameters())} params')

    print('Step 4: Extract per-layer residuals for all items')
    all_embs = torch.tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.encoder(all_embs)  # (N, e_dim)
    sids, residuals = get_residuals(model, z)
    print(f'  sids shape: {sids.shape}')
    for i, r in enumerate(residuals):
        print(f'  residual[{i}] shape: {r.shape}')

    # Save residuals per layer for downstream analysis
    out_dir = BASE / 'products' / 'task337'
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(residuals):
        np.save(out_dir / f'residual_L{i}.npy', r.cpu().numpy())
    torch.save(sids.cpu(), out_dir / 'sids_all.pt')
    print(f'  Saved to {out_dir}')

    print('Step 5: Compute pairwise residual L2 distances (Method B - HRQ-VAE level)')
    # Per-layer pairwise distances for the 159 pairs
    out_stats = {}
    for layer_idx in range(4):  # 4 residuals: input to L0, L1, L2, L3(=0)
        r = residuals[layer_idx].cpu().numpy()  # (N, e_dim)
        dists = []
        for a, b in pairs:
            dists.append(float(np.linalg.norm(r[a] - r[b])))
        dists = np.array(dists)
        out_stats[f'residual_L{layer_idx}'] = {
            'mean': float(dists.mean()) if len(dists) > 0 else None,
            'std': float(dists.std()) if len(dists) > 0 else None,
            'p5': float(np.percentile(dists, 5)) if len(dists) > 0 else None,
            'p50': float(np.percentile(dists, 50)) if len(dists) > 0 else None,
            'p95': float(np.percentile(dists, 95)) if len(dists) > 0 else None,
            'min': float(dists.min()) if len(dists) > 0 else None,
            'max': float(dists.max()) if len(dists) > 0 else None,
            'e_dim': int(r.shape[1]),
        }
        print(f'  L{layer_idx} (e_dim={r.shape[1]}): mean={dists.mean():.4f} '
              f'p5={np.percentile(dists, 5):.4f} p50={np.percentile(dists, 50):.4f} '
              f'p95={np.percentile(dists, 95):.4f}')

    # Also re-check: r[3] should be near-zero (final residual after all layers)
    r3 = residuals[3].cpu().numpy()
    print(f'\nFinal residual[3] norm (all items): '
          f'mean={np.linalg.norm(r3, axis=1).mean():.4f}, '
          f'max={np.linalg.norm(r3, axis=1).max():.4f}')

    print('\nStep 6: Save Method A+B residuals-level results')
    result = {
        'method': 'A+B - real HRQ-VAE residual pairwise L2 on near-duplicate pairs',
        'checkpoint': CKPT_PATH,
        'n_pairs': len(pairs),
        'task339_gate1_codeword_gap': {
            'L0_p5': 0.1004, 'L1_p5': 0.0630, 'L2_p5': 0.0425,
        },
        'residual_pairwise_l2_per_layer': out_stats,
        'task339_sigma_flip_rate_at_0.01': 'L2 flip 80.0%',
        'task339_sigma_flip_rate_at_0.02': 'L2 flip 94.8%',
        'task339_sigma_flip_rate_at_0.05': 'L2 flip 98.7%',
    }
    out_json = BASE / 'verdicts' / 'task337_issue50_method_a_b_residual.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {out_json}')


if __name__ == '__main__':
    main()
