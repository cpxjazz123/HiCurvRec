"""Task 418: Head/Tail AQE (Average Quantization Error) for E-E-E-E vs H-E-E-E.

Definition (this script):
  - AQE for an item i = ||x_i - x_hat_i||_2 where x_hat is the RQ sum of selected
    codewords across all layers.
  - AQE is computed in the SAME space for both variants: the original 2048-dim
    text embedding space.
  - E-E-E-E baseline: ckpt only stores 3 centroid layers (no encoder/decoder in
    state_dict). Reconstruction is the sum of the selected centroids themselves
    (since the model with no reconstruction_loss_function trains only the
    centroids, and the encoder/decoder are not saved). We sum the 3 selected
    centroids and compute ||x_norm - x_hat||_2 where x_norm = (x - x_mean) /
    x_std (the v3 H script does the same normalization on the same embeddings).
  - H-E-E-E v4: reconstruction uses C1_euclid (idx1) + C2 (idx2) + C3 (idx3),
    where idx1 is selected via Poincaré distance on C1_ball.

Bucketing:
  - Items are bucketed into Head (top-20% by training-interaction count) and
    Tail (remaining 80%). The bucket is item-level (not user-level), matching
    "Head/Tail by item popularity". Items never seen in training go to tail.

Output:
  - result/task418_aqe_head_tail/aqe.json
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Import order to break circular imports when loading Lightning ckpt
GRID_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
import src.utils.utils  # noqa: F401
import src.utils.decorators  # noqa: F401
import src.data.loading.components.iterators  # noqa: F401
import src.data.loading.components.interfaces  # noqa: F401

import tensorflow as tf  # type: ignore  # noqa: E402

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATA_DIR = "/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys"
EMB_PATH = "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt"

E_CKPT = "/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt"
H_CKPT = "/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task388v4_h_e_e_e_2026-07-14_18-44-47/checkpoints/ckpt_H_E_E_E.ckpt"

OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task418_aqe_head_tail"

HEAD_QUANTILE = 0.20

# -----------------------------------------------------------------------------
# Poincaré primitives (mirror task388 v3 / v4 training)
# -----------------------------------------------------------------------------
BALL_C = 1.0
BALL_MAX_NORM = 0.5
BALL_EPS = 1e-5


def mobius_add(x, y, c=BALL_C):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c * c * x2 * y2
    return num / denom.clamp(min=1e-15)


def project_to_ball(x, max_norm=BALL_MAX_NORM, eps=BALL_EPS):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def exp_map0(v, c=BALL_C):
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    sqrt_c = math.sqrt(c)
    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def log_map0(y, c=BALL_C):
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    sqrt_c = math.sqrt(c)
    arg = (sqrt_c * y_norm).clamp(max=1.0 - 5e-3)
    return torch.atanh(arg) * y / (sqrt_c * y_norm)


def poincare_distance(x, y, c=BALL_C):
    sqrt_c = math.sqrt(c)
    diff = mobius_add(-x, y, c=c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 - 5e-3)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)


def euclidean_to_poincare(x, c=BALL_C, target_norm=0.5):
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    v_norm_target = math.atanh(target_norm)
    v = x_unit * v_norm_target
    y = exp_map0(v, c=c)
    return project_to_ball(y, max_norm=BALL_MAX_NORM)


# -----------------------------------------------------------------------------
# Stage 1 load: text embeddings (11924, 2048) for toys
# -----------------------------------------------------------------------------
def load_embeddings():
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"missing Stage 1 embedding tensor: {EMB_PATH}")
    x = torch.load(EMB_PATH, map_location="cpu", weights_only=False).float()
    if x.dim() != 2 or x.shape[1] != 2048:
        raise ValueError(f"unexpected Stage 1 embedding shape {tuple(x.shape)}")
    return x


# -----------------------------------------------------------------------------
# Popularity: item-level count from training partitions
# -----------------------------------------------------------------------------
def _list_paritions(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"missing data folder: {folder}")
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("partition_") and f.endswith(".tfrecord.gz")
    )
    if not files:
        raise FileNotFoundError(f"no partition_*.tfrecord.gz under {folder}")
    return files


def _iter_tfrecord_rows(file_paths):
    ds = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    for raw in ds:
        ex = tf.train.Example.FromString(raw.numpy())
        f = ex.features.feature
        if "sequence_data" not in f or "user_id" not in f:
            raise KeyError(f"unexpected schema: keys={list(f.keys())}")
        seq = list(f["sequence_data"].int64_list.value)
        uid_arr = list(f["user_id"].int64_list.value)
        if not seq or not uid_arr:
            continue
        yield int(uid_arr[0]), seq


def build_item_popularity(data_dir, n_items):
    counter = Counter()
    for fp in _list_paritions(os.path.join(data_dir, "training")):
        for _uid, seq in _iter_tfrecord_rows([fp]):
            counter.update(seq)
    counts = np.zeros(n_items, dtype=np.int64)
    for iid, c in counter.items():
        if 0 <= iid < n_items:
            counts[iid] = c
    return counts


def assign_head_tail_from_popularity(counts, head_quantile=HEAD_QUANTILE):
    """Top head_quantile by count = head; rest = tail. Ties broken by iid asc."""
    n_items = counts.shape[0]
    order = sorted(range(n_items), key=lambda i: (-int(counts[i]), i))
    cutoff = max(1, int(round(head_quantile * n_items)))
    head_idx = set(order[:cutoff])
    head_mask = np.array([i in head_idx for i in range(n_items)], dtype=bool)
    tail_mask = ~head_mask
    n_head = int(head_mask.sum())
    n_tail = int(tail_mask.sum())
    if n_head == 0 or n_tail == 0:
        raise ValueError(f"empty bucket: head={n_head}, tail={n_tail}")
    return head_mask, tail_mask, n_head, n_tail


# -----------------------------------------------------------------------------
# E-E-E-E forward: assign per layer via Euclidean nearest-centroid, sum centroids
# -----------------------------------------------------------------------------
def eeee_reconstruct(x_norm, centroids_l0, centroids_l1, centroids_l2):
    """x_norm: (N, D), already mean/std normalized.

    No encoder/decoder available (ckpt only stores centroids). Use direct
    sum-of-centroids reconstruction in the normalized 2048-dim space.
    """
    d0 = torch.cdist(x_norm, centroids_l0, p=2)
    idx0 = d0.argmin(dim=1)
    q0 = centroids_l0[idx0]
    r1 = x_norm - q0
    d1 = torch.cdist(r1, centroids_l1, p=2)
    idx1 = d1.argmin(dim=1)
    q1 = centroids_l1[idx1]
    r2 = r1 - q1
    d2 = torch.cdist(r2, centroids_l2, p=2)
    idx2 = d2.argmin(dim=1)
    q2 = centroids_l2[idx2]
    x_hat = q0 + q1 + q2
    aqe = (x_norm - x_hat).norm(dim=-1)  # L2 residual per item
    return aqe, torch.stack([idx0, idx1, idx2], dim=1)


# -----------------------------------------------------------------------------
# H-E-E-E forward: L1 = Poincaré distance → C1_euclid; L2/L3 = Euclidean
# -----------------------------------------------------------------------------
def heee_reconstruct(x_norm, C1_ball, C1_euclid, C2, C3, target_norm=0.5):
    x_ball = euclidean_to_poincare(x_norm, c=BALL_C, target_norm=target_norm)
    d1 = poincare_distance(
        x_ball.unsqueeze(1), C1_ball.unsqueeze(0), c=BALL_C
    )
    idx1 = d1.argmin(dim=1)
    q1_e = C1_euclid[idx1]
    r1 = x_norm - q1_e
    d2 = torch.cdist(r1, C2, p=2)
    idx2 = d2.argmin(dim=1)
    q2 = C2[idx2]
    r2 = r1 - q2
    d3 = torch.cdist(r2, C3, p=2)
    idx3 = d3.argmin(dim=1)
    q3 = C3[idx3]
    x_hat = q1_e + q2 + q3
    aqe = (x_norm - x_hat).norm(dim=-1)
    return aqe, torch.stack([idx1, idx2, idx3], dim=1)


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def mean_std(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty array passed to mean_std")
    return float(arr.mean()), float(arr.std(ddof=0))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--emb_path", default=EMB_PATH)
    parser.add_argument("--e_ckpt", default=E_CKPT)
    parser.add_argument("--h_ckpt", default=H_CKPT)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--head_quantile", type=float, default=HEAD_QUANTILE)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load embeddings (raw, 2048-dim)
    print(f"[task418] loading embeddings: {args.emb_path}")
    x_all = load_embeddings()
    n_items, dim = x_all.shape
    print(f"[task418] x_all shape: {tuple(x_all.shape)}, dtype={x_all.dtype}")

    # 2) Compute per-item popularity (training interactions)
    print(f"[task418] building item popularity from {args.data_dir}")
    t0 = time.time()
    counts = build_item_popularity(args.data_dir, n_items)
    print(
        f"[task418] popularity built: total_items={n_items}, "
        f"distinct_with_count={int((counts > 0).sum())}, "
        f"t={time.time() - t0:.1f}s"
    )

    # 3) Assign head/tail buckets
    head_mask, tail_mask, n_head, n_tail = assign_head_tail_from_popularity(
        counts, head_quantile=args.head_quantile
    )
    print(
        f"[task418] head_quantile={args.head_quantile}: "
        f"head_n={n_head}, tail_n={n_tail}"
    )

    # 4) Load E-E-E-E baseline ckpt and compute AQE
    print(f"[task418] loading E-E-E-E baseline ckpt: {args.e_ckpt}")
    if not os.path.exists(args.e_ckpt):
        raise FileNotFoundError(f"missing E-E-E-E ckpt: {args.e_ckpt}")
    e_ckpt = torch.load(args.e_ckpt, map_location="cpu", weights_only=False)
    e_sd = e_ckpt["state_dict"]
    centroids_l0 = e_sd["quantization_layer_list.0.centroids"].float()
    centroids_l1 = e_sd["quantization_layer_list.1.centroids"].float()
    centroids_l2 = e_sd["quantization_layer_list.2.centroids"].float()
    # Sanity: ckpt hp says n_layers=3, input_dim=2048
    e_hp = e_ckpt.get("hyper_parameters", {})
    if e_hp.get("input_dim") != 2048:
        raise ValueError(f"E-E-E-E input_dim mismatch: {e_hp.get('input_dim')}")
    if e_hp.get("n_layers") != 3:
        raise ValueError(f"E-E-E-E n_layers mismatch: {e_hp.get('n_layers')}")
    print(
        f"[task418] E centroids: L0={tuple(centroids_l0.shape)}, "
        f"L1={tuple(centroids_l1.shape)}, L2={tuple(centroids_l2.shape)}"
    )

    # Normalize (x - mean) / std (same scheme as H-E-E-E v3/v4 training script)
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_norm = (x_all - x_mean) / x_std

    # E-E-E-E reconstruction
    print("[task418] running E-E-E-E forward...")
    t0 = time.time()
    e_aqe, e_codes = eeee_reconstruct(x_norm, centroids_l0, centroids_l1, centroids_l2)
    e_aqe_np = e_aqe.numpy()
    print(
        f"[task418] E-E-E-E AQE done. global mean={float(e_aqe_np.mean()):.4f}, "
        f"std={float(e_aqe_np.std()):.4f}, t={time.time() - t0:.1f}s"
    )

    # 5) Load H-E-E-E v4 ckpt and compute AQE
    print(f"[task418] loading H-E-E-E v4 ckpt: {args.h_ckpt}")
    if not os.path.exists(args.h_ckpt):
        raise FileNotFoundError(f"missing H-E-E-E ckpt: {args.h_ckpt}")
    h_ckpt = torch.load(args.h_ckpt, map_location="cpu", weights_only=False)
    h_sd = h_ckpt["state_dict"]
    if "C1_ball" not in h_sd:
        raise KeyError(
            "H-E-E-E ckpt missing C1_ball; this script requires v3+ (dual codebook)."
        )
    C1_ball = h_sd["C1_ball"].float()
    C1_euclid = h_sd["quantization_layer_list.0.centroids"].float()
    C2 = h_sd["quantization_layer_list.1.centroids"].float()
    C3 = h_sd["quantization_layer_list.2.centroids"].float()
    print(
        f"[task418] H centroids: C1_ball={tuple(C1_ball.shape)}, "
        f"C1_euclid={tuple(C1_euclid.shape)}, C2={tuple(C2.shape)}, C3={tuple(C3.shape)}"
    )

    # H-E-E-E reconstruction
    print("[task418] running H-E-E-E forward...")
    t0 = time.time()
    h_aqe, h_codes = heee_reconstruct(x_norm, C1_ball, C1_euclid, C2, C3)
    h_aqe_np = h_aqe.numpy()
    print(
        f"[task418] H-E-E-E AQE done. global mean={float(h_aqe_np.mean()):.4f}, "
        f"std={float(h_aqe_np.std()):.4f}, t={time.time() - t0:.1f}s"
    )

    # 6) Bucket aggregation
    e_head_mean, e_head_std = mean_std(e_aqe_np[head_mask])
    e_tail_mean, e_tail_std = mean_std(e_aqe_np[tail_mask])
    h_head_mean, h_head_std = mean_std(h_aqe_np[head_mask])
    h_tail_mean, h_tail_std = mean_std(h_aqe_np[tail_mask])

    e_delta_ht = e_head_mean - e_tail_mean
    h_delta_ht = h_head_mean - h_tail_mean
    head_aqe_diff = h_head_mean - e_head_mean  # H - E on head
    tail_aqe_diff = h_tail_mean - e_tail_mean  # H - E on tail
    head_tail_delta_gap = head_aqe_diff - tail_aqe_diff

    summary = {
        "task": "task418_head_tail_aqe",
        "data_dir": args.data_dir,
        "embedding_path": args.emb_path,
        "n_items": int(n_items),
        "embedding_dim": int(dim),
        "head_quantile": args.head_quantile,
        "bucket_sizes": {"head": int(n_head), "tail": int(n_tail)},
        "normalization": {
            "type": "per-feature (x - mean) / std",
            "computed_from": args.emb_path,
        },
        "e_ckpt": args.e_ckpt,
        "h_ckpt": args.h_ckpt,
        "e_eee": {
            "head_mean_aqe": e_head_mean,
            "head_std_aqe": e_head_std,
            "tail_mean_aqe": e_tail_mean,
            "tail_std_aqe": e_tail_std,
            "delta_head_tail": e_delta_ht,
        },
        "h_eee": {
            "head_mean_aqe": h_head_mean,
            "head_std_aqe": h_head_std,
            "tail_mean_aqe": h_tail_mean,
            "tail_std_aqe": h_tail_std,
            "delta_head_tail": h_delta_ht,
        },
        "diffs": {
            "head_aqe_diff_h_minus_e": head_aqe_diff,
            "tail_aqe_diff_h_minus_e": tail_aqe_diff,
            "head_tail_delta_gap": head_tail_delta_gap,
        },
        "hypothesis_check": {
            "rule": (
                "hypothesis_supported if (tail_aqe_diff < 0) AND "
                "(head_aqe_diff <= small positive tolerance)"
            ),
            "tail_aqe_lower_for_h": bool(tail_aqe_diff < 0),
            "head_aqe_not_much_higher_for_h": bool(head_aqe_diff <= 0.10),
        },
    }

    out_json = os.path.join(args.out_dir, "aqe.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[task418] saved: {out_json}")

    # Also save per-item AQE for downstream analysis
    np.save(os.path.join(args.out_dir, "e_eee_aqe_per_item.npy"), e_aqe_np)
    np.save(os.path.join(args.out_dir, "h_eee_aqe_per_item.npy"), h_aqe_np)
    np.save(os.path.join(args.out_dir, "item_popularity.npy"), counts)
    np.save(os.path.join(args.out_dir, "head_mask.npy"), head_mask.astype(np.int8))
    np.save(os.path.join(args.out_dir, "tail_mask.npy"), tail_mask.astype(np.int8))

    print("\n=== task418 summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()