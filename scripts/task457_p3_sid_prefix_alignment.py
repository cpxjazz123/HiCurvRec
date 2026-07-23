#!/usr/bin/env python3
"""task457_p3_sid_prefix_alignment.py — Phase 2 P3 因果链 4 验证

验证: SID 前缀层 vs 用户行为对齐度 (Mixed-Curvature P0.3 vs Euclidean-Concat P0.2)

方法:
  1. 采样 N=2000 用户 (seed=42)
  2. 提取每个用户的 history item list (限 max_seq_len=120)
  3. 对每对 (i,j) 计算:
     - behavior_jaccard(i,j) = |H_i ∩ H_j| / |H_i ∪ H_j|
     - sid_L*_jaccard(i,j)   = |S*_i ∩ S*_j| / |S*_i ∪ S*_j|  * ∈ {1,2,3}
  4. 算两组 jaccard 矩阵的 Spearman ρ
  5. 比较 P0.2 vs P0.3

预期: Mixed-Curvature (P0.3) 的 SID 前缀层 ρ 应高于 Euclidean-Concat (P0.2)
"""
import os
import sys
import json
import gzip
import random
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys"
SID_P0_2 = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s22/pickle/merged_predictions_tensor.pt"
SID_P0_3 = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt"

OUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp456")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_USERS = 2000
MAX_SEQ_LEN = 120
SEED = 42


def load_tfrecord_user_histories(data_dir: str, n_users: int, seed: int) -> dict:
    """Read Toys training tfrecords, return {user_id: [item_id_seq]}.

    Uses raw tfrecord parsing (avoid GRID's SequenceDataModule wrapper which needs Hydra config).
    """
    import tensorflow as tf

    train_dir = Path(data_dir) / "training"
    tf_files = sorted([str(p) for p in train_dir.glob("*.tfrecord.gz")])

    print(f"[load] {len(tf_files)} tfrecord.gz files in {train_dir}")

    random.seed(seed)
    random.shuffle(tf_files)

    user_histories = {}
    for fpath in tf_files:
        if len(user_histories) >= n_users:
            break
        try:
            ds = tf.data.TFRecordDataset([fpath], compression_type="GZIP")
            for raw in ds:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw.numpy())
                    feats = example.features.feature
                    user_id = None
                    item_seq = []
                    for key, val in feats.items():
                        if key == "sequence_data":
                            item_seq = [int(x) for x in val.int64_list.value]
                        elif key == "user_id":
                            uid_list = list(val.int64_list.value)
                            user_id = uid_list[0] if uid_list else None
                    if user_id is None:
                        # fall back to using a generated ID
                        user_id = len(user_histories)
                    user_histories[user_id] = item_seq[:MAX_SEQ_LEN]
                    if len(user_histories) >= n_users:
                        break
                except Exception as e:
                    print(f"[warn] skip record in {fpath}: {e}")
                    continue
        except Exception as e:
            print(f"[warn] skip file {fpath}: {e}")
            continue
    return user_histories


def compute_jaccard_matrix(user_sids: dict) -> np.ndarray:
    """Compute pairwise Jaccard similarity matrix given {user_id: set_of_sids}."""
    uids = sorted(user_sids.keys())
    n = len(uids)
    sets = [user_sids[u] for u in uids]
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sets[j]
            if not sj:
                continue
            inter = len(si & sj)
            union = len(si | sj)
            if union > 0:
                mat[i, j] = mat[j, i] = inter / union
    return mat, uids


def main():
    print(f"[P3] Sampling N={N_USERS} users, MAX_SEQ_LEN={MAX_SEQ_LEN}, seed={SEED}")

    # Step 1: Load user histories
    user_histories = load_tfrecord_user_histories(DATA_DIR, N_USERS, SEED)
    print(f"[P3] Loaded {len(user_histories)} user histories")
    if len(user_histories) < 100:
        print(f"[P3] ERROR: too few users ({len(user_histories)}), abort")
        return

    # Step 2: Load SID tensors
    sid_p0_2 = torch.load(SID_P0_2, map_location="cpu", weights_only=False).long()  # (4, N_items)
    sid_p0_3 = torch.load(SID_P0_3, map_location="cpu", weights_only=False).long()
    n_items = sid_p0_2.shape[1]
    print(f"[P3] SID shape: {tuple(sid_p0_2.shape)} (P0.2) / {tuple(sid_p0_3.shape)} (P0.3)")
    assert sid_p0_2.shape == sid_p0_3.shape == (4, n_items), "SID shape mismatch"

    # Step 3: For each user, derive SID-L1, L2, L3 sets (union over history)
    sid_sets = {"P0.2": {1: {}, 2: {}, 3: {}}, "P0.3": {1: {}, 2: {}, 3: {}}}
    valid_users = []
    skipped = 0
    for uid, items in user_histories.items():
        items = [i for i in items if 0 <= i < n_items]
        if len(items) < 3:
            skipped += 1
            continue
        valid_users.append(uid)
        for variant_name, sid_tensor in [("P0.2", sid_p0_2), ("P0.3", sid_p0_3)]:
            for layer in (1, 2, 3):
                sids_in_layer = set()
                for it in items:
                    code = int(sid_tensor[layer - 1, it].item())
                    sids_in_layer.add(code)
                sid_sets[variant_name][layer][uid] = sids_in_layer
    print(f"[P3] Valid users (>=3 items): {len(valid_users)}, skipped: {skipped}")

    # Step 4: Compute behavior Jaccard
    behavior_sets = {uid: set(items) for uid, items in user_histories.items() if uid in valid_users}
    beh_mat, uids = compute_jaccard_matrix(behavior_sets)
    print(f"[P3] behavior matrix: {beh_mat.shape}, mean off-diag = {beh_mat[beh_mat > 0].mean():.4f}")

    # Step 5: For each (variant, layer), compute SID Jaccard and Spearman ρ vs behavior
    results = {}
    for variant in ("P0.2", "P0.3"):
        results[variant] = {}
        for layer in (1, 2, 3):
            sid_sets_aligned = {uid: sid_sets[variant][layer][uid] for uid in uids}
            sid_mat, _ = compute_jaccard_matrix(sid_sets_aligned)
            iu = np.triu_indices_from(beh_mat, k=1)
            rho, pval = spearmanr(beh_mat[iu], sid_mat[iu])
            mean_sid_jaccard = sid_mat[iu].mean()
            std_sid_jaccard = sid_mat[iu].std()
            results[variant][f"L{layer}"] = {
                "spearman_rho": float(rho),
                "pvalue": float(pval),
                "mean_sid_jaccard_offdiag": float(mean_sid_jaccard),
                "std_sid_jaccard_offdiag": float(std_sid_jaccard),
            }
            print(f"[P3] {variant} L{layer}: ρ={rho:.4f} p={pval:.2e} mean_offdiag={mean_sid_jaccard:.4f}")

    # Step 6: P0.3 vs P0.2 ρ diff
    print(f"\n[P3] === P0.3 - P0.2 ρ diff ===")
    for layer in (1, 2, 3):
        rho_p0_2 = results["P0.2"][f"L{layer}"]["spearman_rho"]
        rho_p0_3 = results["P0.3"][f"L{layer}"]["spearman_rho"]
        diff = rho_p0_3 - rho_p0_2
        print(f"[P3] L{layer}: P0.2={rho_p0_2:.4f} | P0.3={rho_p0_3:.4f} | diff={diff:+.4f}")

    # Step 7: Save
    out = {
        "n_users_sampled": N_USERS,
        "n_users_valid": len(valid_users),
        "n_items": n_items,
        "max_seq_len": MAX_SEQ_LEN,
        "seed": SEED,
        "behavior_mean_offdiag_jaccard": float(beh_mat[beh_mat > 0].mean()),
        "results": results,
    }
    out_path = OUT_DIR / "task457_p3_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[P3] Saved results to {out_path}")
    return out


if __name__ == "__main__":
    main()