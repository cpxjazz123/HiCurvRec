"""
Task #337 / Issue #50 Gate 0' — Real noise floor measurement.

Three methods to measure TRUE data-driven noise:

Method A (encoder randomness): If model has dropout/inference randomness,
  encode same item N times → measure variance.
  (T5 sentence embedding is deterministic → variance ≈ 0, expected to fail)

Method B (near-duplicate pairs — primary): Find similar products by
  brand + text Jaccard, compute pairwise residual distances in 3 layers.
  Most informative because captures real-world "same intent different SKU" noise.

Method C (text perturbation): Synonym-substitute or adjective-drop text,
  re-encode, measure propagated variance.
  Last resort if A/B inconclusive.

Pass criteria: Per-layer (mean, std, p5, p50, p95) residual distance.
Compare to Task #339 Gate 1 codeword spacing (L0: 0.1004, L1: 0.0630, L2: 0.0425)
to decide whether H2 actually holds.

Usage:
  python3 scripts/task337_issue50_gate0_real_noise_measurement.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
# torch imported lazily inside encode_via_hrqvae (needs GPU env)

BASE = Path('/home/wlia0047/ar57/wenyu/GeneRec')
DATA_DIR = BASE / 'HG-Rec' / 'dataset' / 'Instruments'
EMB_PATH = DATA_DIR / 'item_emb.parquet'


def jaccard_words(s1: str, s2: str) -> float:
    """Word-level Jaccard similarity."""
    set1 = set(str(s1).lower().split())
    set2 = set(str(s2).lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def find_near_duplicate_pairs(items: dict, max_pairs: int = 200,
                              jaccard_thresh: float = 0.85) -> list[tuple[int, int, float]]:
    """Find pairs of items with high text-overlap (same brand + similar title/description)."""
    print(f'Building brand index...')
    by_brand = defaultdict(list)
    for iid, info in items.items():
        b = info.get('brand', '').strip().lower()
        if b:
            by_brand[b].append(iid)

    print(f'  Brands with items: {len(by_brand)}')
    pairs = []
    seen = set()
    for brand, iids in by_brand.items():
        if len(iids) < 2:
            continue
        # Within brand: find pairs by title/description Jaccard
        for i in range(len(iids)):
            for j in range(i + 1, len(iids)):
                a, b = iids[i], iids[j]
                key = (min(int(a), int(b)), max(int(a), int(b)))
                if key in seen:
                    continue
                seen.add(key)
                info_a, info_b = items[str(a)], items[str(b)]
                # Title is short and reliable
                jac_title = jaccard_words(info_a.get('title', ''), info_b.get('title', ''))
                if jac_title >= jaccard_thresh:
                    pairs.append((int(a), int(b), jac_title))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def encode_via_hrqvae(model, embeddings: np.ndarray) -> list[np.ndarray]:
    """Run our HRQ-VAE on all embeddings to extract per-layer residuals.
    Returns list of (N, 32) residual arrays."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    tensors = [torch.tensor(embeddings, dtype=torch.float32)]
    ds = TensorDataset(*tensors)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_residuals = [[] for _ in range(3)]
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)  # (B, 768)
            # Get residuals from HRQ-VAE pipeline (similar to Stage 2 inference)
            z_e = model.encoder(x)
            residuals = model.get_residuals_for_noise(z_e)
            for lyr, r in enumerate(residuals):
                if r is not None:
                    all_residuals[lyr].append(r.cpu().numpy())

    return [np.concatenate(r, axis=0) if r else None for r in all_residuals]


def main():
    print('=== Task #337 / Issue #50 Gate 0\' ===')
    print('Step 1: Load 768-dim item embeddings (T5 sentence-transformers)')
    df = pd.read_parquet(EMB_PATH)
    embeddings = np.stack([np.array(e, dtype=np.float32) for e in df['embedding']])
    item_ids = df['ItemID'].values
    print(f'  Loaded {embeddings.shape[0]} items, dim={embeddings.shape[1]}')

    print('\nStep 2: Load item metadata for near-duplicate search')
    with open(DATA_DIR / 'Instruments.item.json') as f:
        items = json.load(f)
    print(f'  {len(items)} item metadata records')

    print('\nStep 3: Find near-duplicate pairs (Method B)')
    t0 = time.time()
    pairs = find_near_duplicate_pairs(items, max_pairs=300, jaccard_thresh=0.85)
    print(f'  Found {len(pairs)} pairs in {time.time()-t0:.1f}s')
    if pairs:
        for p in pairs[:5]:
            print(f'    pair ({p[0]}, {p[1]}) jaccard={p[2]:.3f}')
            print(f'      A: {items[str(p[0])]["title"][:80]}')
            print(f'      B: {items[str(p[1])]["title"][:80]}')

    print('\nStep 4: Compute pairwise L2 distances on RAW embeddings (Method B proxy)')
    raw_dists = []
    for a, b, _ in pairs:
        da = np.linalg.norm(embeddings[a] - embeddings[b])
        raw_dists.append(da)
    raw_dists = np.array(raw_dists)
    print(f'  Raw 768-dim pairwise L2: mean={raw_dists.mean():.4f}, '
          f'std={raw_dists.std():.4f}, p5={np.percentile(raw_dists, 5):.4f}, '
          f'p50={np.percentile(raw_dists, 50):.4f}, p95={np.percentile(raw_dists, 95):.4f}')
    # Compare to σ={0.01, 0.02, 0.05} σ-projected L2 in 768-d ~ σ*sqrt(768)
    print(f'  σ=0.01 in 768-d ~ L2 {0.01 * np.sqrt(768):.3f}')
    print(f'  σ=0.02 in 768-d ~ L2 {0.02 * np.sqrt(768):.3f}')
    print(f'  σ=0.05 in 768-d ~ L2 {0.05 * np.sqrt(768):.3f}')

    print('\nStep 5: Save Method B results')
    out_dir = BASE / 'verdicts'
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / 'task337_issue50_method_b_near_duplicates.json'
    result = {
        'method': 'B - near-duplicate text pairs',
        'n_pairs': len(pairs),
        'jaccard_threshold': 0.85,
        'raw_embedding_l2_stats': {
            'mean': float(raw_dists.mean()) if len(raw_dists) > 0 else None,
            'std': float(raw_dists.std()) if len(raw_dists) > 0 else None,
            'p5': float(np.percentile(raw_dists, 5)) if len(raw_dists) > 0 else None,
            'p50': float(np.percentile(raw_dists, 50)) if len(raw_dists) > 0 else None,
            'p95': float(np.percentile(raw_dists, 95)) if len(raw_dists) > 0 else None,
            'min': float(raw_dists.min()) if len(raw_dists) > 0 else None,
            'max': float(raw_dists.max()) if len(raw_dists) > 0 else None,
        },
        'sigma_projections_768d': {
            '0.01': float(0.01 * np.sqrt(768)),
            '0.02': float(0.02 * np.sqrt(768)),
            '0.05': float(0.05 * np.sqrt(768)),
        },
        'comparison_to_task339_gate1_codeword_gap': {
            'L0_NN_dist_p5': 0.1004,
            'L1_NN_dist_p5': 0.0630,
            'L2_NN_dist_p5': 0.0425,
        },
        'sample_pairs': pairs[:10],
    }
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {out_json}')

    print('\n✅ Gate 0\' Method B complete (Method A/B/C will be added)')


if __name__ == '__main__':
    main()
