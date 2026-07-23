"""
Task 25 — Phenomenon 1: SID Distance Spectrum Analysis
Computes P(h) = proportion of item pairs at Hamming distance h.
Supports multi-algorithm comparison.

Usage: python scripts/task10_distance_spectrum.py
"""

import torch
import numpy as np
import json, os, time
from collections import Counter
from itertools import combinations


def load_sid_tensor(path, key=None):
    """Load SID tensor, handling dict/list wrappers."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # Try common keys
        for k in ["sid", "sid_tensor", "idx_lst"]:
            if k in obj:
                obj = obj[k]
                break
        # If idx_lst (list of tensors), stack
        if isinstance(obj, list) and all(isinstance(v, torch.Tensor) for v in obj):
            obj = torch.stack(obj, dim=0)
    # Ensure (L, N) format
    if obj.dim() == 2 and obj.shape[1] > obj.shape[0] and obj.shape[0] <= 10:
        pass  # already (L, N)
    elif obj.dim() == 2 and obj.shape[0] > obj.shape[1]:
        obj = obj.T  # (N, L) -> (L, N)
    return obj.long()


def compute_distance_spectrum(sid):
    """
    Compute P(h): pairwise Hamming distance distribution.
    sid: (L, N) tensor of SID indices
    returns: {h: proportion} dict
    """
    L, N = sid.shape
    print(f"  Computing distance spectrum: N={N}, L={L}...")

    # Vectorized approach: compute all-pairs Hamming distance
    # For each layer, compute expanded difference matrix
    # sid[l] has shape (N,) — expand to (N, N) via broadcasting
    # Total pairs = N*(N-1)/2

    # To avoid O(N²) memory, process in batches of layers
    # Hamming distance = sum over L of (sid[l,i] != sid[l,j])
    # Use numpy for memory efficiency

    sid_np = sid.numpy()  # (L, N)

    # Sample-based approach for large N: use a random subset
    # But N=11924, N²/2 ≈ 71M pairs — manageable with uint8

    # Compute pairwise comparisons layer by layer to save memory
    dist_counter = Counter()
    batch_size = 2000  # process 2000 items at a time

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = sid_np[:, start:end]  # (L, batch_size)

        # Compare each item in batch against all N items
        # For each layer: (batch_size, 1) != (1, N) -> (batch_size, N)
        layer_dists = np.zeros((end - start, N), dtype=np.uint8)
        for l in range(L):
            layer_dists += (batch[l, :, None] != sid_np[l, None, :]).astype(np.uint8)

        # Only take upper triangle (j > i) to avoid double counting
        for i in range(end - start):
            global_i = start + i
            for d in layer_dists[i, global_i + 1:]:
                dist_counter[int(d)] += 1

        if (start // batch_size) % 2 == 0:
            elapsed = time.time() - t0
            print(f"    batch {start//batch_size}+1/{N//batch_size+1} ({start}/{N} items), {elapsed:.0f}s")

    total_pairs = sum(dist_counter.values())
    spectrum = {int(h): float(c) / total_pairs for h, c in sorted(dist_counter.items())}

    elapsed = time.time() - t0
    print(f"  Done: {total_pairs} pairs, {elapsed:.1f}s")
    print(f"  Collision rate P(0) = {spectrum.get(0, 0):.6f}")
    print(f"  Near-miss rate P(1) = {spectrum.get(1, 0):.6f}")

    return spectrum


def main():
    print("=" * 60)
    print("Task 25 — Phenomenon 1: SID Distance Spectrum")
    print("=" * 60)

    # Algorithm configurations
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

    all_spectra = {}

    for name, cfg in algorithms.items():
        print(f"\n--- {name} ---")
        try:
            sid = load_sid_tensor(cfg["path"], cfg.get("key"))
            L = sid.shape[0]
            print(f"  Loaded: {list(sid.shape)}, unique tokens={sid.unique().numel()}, range=[{sid.min().item()}, {sid.max().item()}]")

            # Normalize: map tokens to contiguous integers per layer
            # This ensures fair distance computation
            sid_norm = sid.clone()
            for l in range(L):
                unique_vals = sid_norm[l].unique()
                mapping = {v.item(): i for i, v in enumerate(sorted(unique_vals))}
                for old_v, new_v in mapping.items():
                    sid_norm[l][sid_norm[l] == old_v] = new_v

            spectrum = compute_distance_spectrum(sid_norm)
            all_spectra[name] = {
                "shape": list(sid.shape),
                "L": L,
                "N": sid.shape[1],
                "unique_tokens": sid.unique().numel(),
                "spectrum": spectrum,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 60)
    print("Summary: P(h) for h=0..4")
    print("=" * 60)
    header = f"{'Algorithm':>20}"
    for h in range(5):
        header += f"  P({h})"
    print(header)
    print("-" * 60)

    for name, data in all_spectra.items():
        line = f"{name:>20}"
        for h in range(5):
            val = data["spectrum"].get(h, 0)
            line += f"  {val:.6f}"
        print(line)

    # Save
    out_dir = "products/task10"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "distance_spectrum.json")
    with open(out_path, "w") as f:
        json.dump(all_spectra, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
