"""
Task 25 — Phenomenon 2: SID Distance-Semantics Correlation

分析不同 SID 算法的"近邻保持性"：
如果两个 item 的 embedding 相似（语义相近），它们的 SID 是否也更近？

P(1) 高 → 语义相似的 item 有相近 SID → 推荐器更容易泛化
P(1) 低 → 语义相似的 item SID 被分散 → 即使 RQ-VAE 重建好，TIGER 也难学好

用法: python scripts/task10_phenom2_semantic_coherence.py [--K 1-5]
"""

import os, sys, argparse, json, time
import torch
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_sid(path, key=None):
    """Load SID tensor in (L, N) format."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        search_keys = [key] if isinstance(key, str) else (key or ["sid", "sid_tensor", "idx_lst"])
        for k in search_keys:
            if k in obj:
                obj = obj[k]
                break
        if isinstance(obj, list) and all(isinstance(v, torch.Tensor) for v in obj):
            obj = torch.stack(obj, dim=0)
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 2 and obj.shape[0] > obj.shape[1] and obj.shape[0] <= 10:
            pass  # (L, N) — good
        elif obj.dim() == 2 and obj.shape[0] > obj.shape[1]:
            obj = obj.T
        return obj.long()
    raise TypeError(f"Cannot convert {type(obj)} to tensor")


def normalize_sid_per_layer(sid):
    """Map tokens to contiguous integers per layer."""
    L, N = sid.shape
    sid_norm = sid.clone()
    for l in range(L):
        unique_vals = sid_norm[l].unique()
        mapping = {v.item(): i for i, v in enumerate(sorted(unique_vals))}
        for old_v, new_v in mapping.items():
            sid_norm[l][sid_norm[l] == old_v] = new_v
    return sid_norm


def compute_hamming_matrix(sid, max_pairs=50000, seed=42):
    """
    Compute Hamming distance for a random subset of item pairs.
    Returns (hamming_dists, cosine_sims) arrays.
    """
    N = sid.shape[1]
    rng = np.random.RandomState(seed)
    sid_np = normalize_sid_per_layer(sid).numpy()

    # Sample pairs
    n_pairs = min(max_pairs, N * (N - 1) // 2)
    idx1 = rng.choice(N, n_pairs, replace=True)
    idx2 = rng.choice(N, n_pairs, replace=True)
    # Remove self-pairs
    valid = idx1 != idx2
    idx1, idx2 = idx1[valid][:n_pairs], idx2[valid][:n_pairs]

    # Hamming distance per pair
    L = sid_np.shape[0]
    hamming = np.zeros(len(idx1), dtype=np.uint8)
    for l in range(L):
        hamming += (sid_np[l, idx1] != sid_np[l, idx2]).astype(np.uint8)

    return hamming


def compute_semantic_coherence(embeddings, sid, n_anchors=1000, n_neighbors=5, seed=42):
    """
    For anchor items, find semantic nearest neighbors (in embedding space).
    Measure: for each neighbor pair, what's the avg SID Hamming distance?
    Lower distance = better semantic coherence.
    """
    N = embeddings.shape[0]
    rng = np.random.RandomState(seed)
    anchors = rng.choice(N, min(n_anchors, N), replace=False)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / norms.clip(min=1e-8)

    sid_np = normalize_sid_per_layer(sid).numpy()
    L = sid_np.shape[0]

    results = {}
    for k in [1, 3, 5]:
        all_distances = []
        for a in anchors:
            # Cosine similarity
            sims = unit @ unit[a]
            # Top-k neighbors (excluding self)
            topk = np.argsort(-sims)[1:k + 1]
            # Hamming distance for each neighbor
            for n_idx in topk:
                h = 0
                for l in range(L):
                    if sid_np[l, a] != sid_np[l, n_idx]:
                        h += 1
                all_distances.append(h)

        avg_dist = np.mean(all_distances)
        p0 = np.mean(np.array(all_distances) == 0)
        p1 = np.mean(np.array(all_distances) == 1)
        results[k] = {
            "avg_hamming": float(avg_dist),
            "P(0)_among_neighbors": float(p0),
            "P(1)_among_neighbors": float(p1),
            "n_pairs": len(all_distances),
        }
    return results


def compute_pairwise_correlation(embeddings, sid, n_pairs=20000, seed=42):
    """
    Pearson correlation between embedding cosine similarity and SID Hamming distance.
    """
    N = embeddings.shape[0]
    rng = np.random.RandomState(seed)

    # Sample pairs
    idx1 = rng.choice(N, n_pairs, replace=True)
    idx2 = rng.choice(N, n_pairs, replace=True)
    valid = idx1 != idx2
    idx1, idx2 = idx1[valid][:n_pairs], idx2[valid][:n_pairs]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / norms.clip(min=1e-8)
    cos_sims = (unit[idx1] * unit[idx2]).sum(axis=1).astype(np.float32)

    # Hamming distance
    sid_np = normalize_sid_per_layer(sid).numpy()
    L = sid_np.shape[0]
    hamming = np.zeros(len(idx1), dtype=np.uint8)
    for l in range(L):
        hamming += (sid_np[l, idx1] != sid_np[l, idx2]).astype(np.uint8)
    hamming = hamming.astype(np.float32)

    # Spearman correlation (rank-based, robust to non-linearity)
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(cos_sims, hamming)

    # Also compute: for top-1% most similar pairs, what's avg Hamming distance?
    threshold = np.percentile(cos_sims, 99)
    top_mask = cos_sims >= threshold
    avg_h_dist_top = hamming[top_mask].mean() if top_mask.sum() > 0 else float('nan')

    # For bottom-1% least similar pairs
    b_threshold = np.percentile(cos_sims, 1)
    bottom_mask = cos_sims <= b_threshold
    avg_h_dist_bottom = hamming[bottom_mask].mean() if bottom_mask.sum() > 0 else float('nan')

    return {
        "spearman_r": float(corr),
        "p_value": float(p_value),
        "n_pairs": len(idx1),
        "avg_hamming_top1pct": float(avg_h_dist_top),
        "avg_hamming_bottom1pct": float(avg_h_dist_bottom),
        "gap": float(avg_h_dist_bottom - avg_h_dist_top),  # positive = semantically similar items have closer SIDs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--n_anchors", type=int, default=500)
    parser.add_argument("--n_pairs", type=int, default=30000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("Task 25 — Phenomenon 2: SID Distance-Semantics Correlation")
    print("=" * 60)

    # ── Load embeddings ──
    print(f"\n[1/3] Loading embeddings from {args.embeddings}...")
    embeddings = torch.load(args.embeddings, map_location="cpu").numpy()
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norms: min={np.linalg.norm(embeddings, axis=1).min():.4f}")

    # ── Define algorithms ──
    algorithms = {
        "Baseline (L=3)": {
            "path": "products/_legacy_result/task14/A_baseline_rqidx.pt",
            "key": "idx_lst",
        },
        "HRQ (L=4)": {
            "path": "products/_legacy_result/diag_joint_cos_f_radial/hrq_sid_raw.pt",
        },
        "AQ (L=4)": {
            "path": "products/_legacy_result/diag_joint_cos_f_radial/aq_sid_raw.pt",
        },
        "HHHH (L=4)": {
            "path": "products/task16/sid_tensors/task59_hhhh_l4_sid_tensor.pt",
        },
    }

    # Load prior P(h) from Task 25 Phenomenon 1
    try:
        with open("products/task10/distance_spectrum.json") as f:
            prior_spectra = json.load(f)
    except:
        prior_spectra = {}

    all_results = {}

    for name, cfg in algorithms.items():
        print(f"\n--- {name} ---")

        try:
            sid = load_sid(cfg["path"], cfg.get("key"))
            L, N = sid.shape
            print(f"  SID: {list(sid.shape)}, unique tokens={sid.unique().numel()}")

            # 1) Semantic coherence (nearest neighbors in embedding space)
            print(f"  Computing semantic coherence ({args.n_anchors} anchors)...")
            t0 = time.time()
            coherence = compute_semantic_coherence(embeddings, sid, n_anchors=args.n_anchors)
            print(f"    Done ({time.time()-t0:.1f}s)")
            print(f"    Top-1 neighbor: avg Hamming={coherence[1]['avg_hamming']:.3f}, "
                  f"P(0)={coherence[1]['P(0)_among_neighbors']:.4f}, "
                  f"P(1)={coherence[1]['P(1)_among_neighbors']:.4f}")

            # 2) Pairwise Spearman correlation
            print(f"  Computing pairwise correlation ({args.n_pairs} pairs)...")
            t0 = time.time()
            correlation = compute_pairwise_correlation(embeddings, sid, n_pairs=args.n_pairs)
            print(f"    Done ({time.time()-t0:.1f}s)")
            print(f"    Spearman r = {correlation['spearman_r']:.4f} (p={correlation['p_value']:.2e})")
            print(f"    Top-1% most similar: avg Hamming = {correlation['avg_hamming_top1pct']:.3f}")
            print(f"    Bottom-1% least similar: avg Hamming = {correlation['avg_hamming_bottom1pct']:.3f}")
            print(f"    Gap (bottom-top) = {correlation['gap']:.3f}")

            # 3) Overall Hamming spectrum (from Phenomenon 1 or recompute)
            prior_p1 = prior_spectra.get(name, {}).get("spectrum", {}).get("1", None)
            prior_p0 = prior_spectra.get(name, {}).get("spectrum", {}).get("0", None)

            all_results[name] = {
                "L": L,
                "N": N,
                "semantic_coherence": coherence,
                "pairwise_correlation": correlation,
                "prior_P0": prior_p0,
                "prior_P1": prior_p1,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary: Semantic-SID Correlation across Algorithms")
    print("=" * 60)

    header = f"{'Algorithm':>20}"
    header += f"  {'P(1)':>8}"
    header += f"  {'Top-1 avg H':>12}"
    header += f"  {'Sp r':>6}"
    header += f"  {'Gap':>6}"
    header += f"  {'R@10':>8}"
    print(header)
    print("-" * 70)

    # Known R@10 values
    known_r10 = {
        "Baseline (L=3)": 0.0973,
        "HRQ (L=4)": None,
        "AQ (L=4)": None,
        "HHHH (L=4)": 0.0284,
    }

    for name in algorithms:
        r = all_results.get(name)
        if r is None:
            continue
        p1 = r.get("prior_P1", "N/A")
        p1_str = f"{p1:.6f}" if isinstance(p1, float) else "N/A"
        avg_h = r["semantic_coherence"][1]["avg_hamming"]
        sp_r = r["pairwise_correlation"]["spearman_r"]
        gap = r["pairwise_correlation"]["gap"]
        r10 = known_r10.get(name, None)
        r10_str = f"{r10:.4f}" if r10 else "N/A"

        line = f"{name:>20}"
        line += f"  {p1_str:>8}"
        line += f"  {avg_h:>12.3f}"
        line += f"  {sp_r:>6.3f}"
        line += f"  {gap:>6.3f}"
        line += f"  {r10_str:>8}"
        print(line)

    # Save
    out_dir = "products/task10"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phenom2_semantic_coherence.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
